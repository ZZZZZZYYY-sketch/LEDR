import numpy as np
import os
import torch
from .layers import *
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel
from sklearn.cluster import KMeans
from utils.utils import SCL


class CloserModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,domain_num, dropout, dataset, alpha):
        super(CloserModel, self).__init__()
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64}
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
            # self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
            # self.bert = RobertaModel.from_pretrained('roberta-base',cache_dir='./pretrained_model').requires_grad_(False)


        for name,param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
                
        self.num_expert = 5   
        self.domain_num = domain_num
        self.alpha = alpha
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)
        
        self.all_feature = {}
        # self.em=self.bert.embeddings
        
        self.convs1 = cnn_extractor(feature_kernel, emb_dim)
        self.convs2 = cnn_extractor(feature_kernel, emb_dim)
        # self.convs2 = cnn_extractor(feature_kernel, emb_dim)

        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        
        #self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.gate_feature = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], mlp_input_shape))
        self.together_1 = nn.Sequential(nn.Linear(2*mlp_input_shape,mlp_input_shape)
                                  )
        self.together_2 = nn.Sequential(nn.Linear(2*mlp_input_shape,mlp_input_shape)
                                  )
        # self.gate_value = nn.Sequential(nn.Linear(mlp_input_shape, mlp_dims[-1]),
        #                           nn.ReLU(),
        #                           nn.Linear(mlp_dims[-1], self.num_expert),
        #                           nn.Softmax(dim=1))
        
        self.classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], self.domain_num))
        self.domain_center = []
        self.domain_loss_fn = nn.CrossEntropyLoss()
        self.scl = SCL(temperature=0.1)
        
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        comment = kwargs['comments']
        comment_masks = kwargs['comments_masks']
        label = kwargs['label']
        expert = kwargs['aug_comments']
        expert_masks = kwargs['aug_comments_masks']
        lossfun = torch.nn.BCELoss()
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        content_feature=self.convs1(bert_feature)

        comment = self.bert(comment, attention_mask=comment_masks).last_hidden_state
        comment_feature=self.convs1(comment)
        
        # expert = self.bert(expert, attention_mask=expert_masks).last_hidden_state
        # expert_feature=self.convs2(expert)

        common_feature = torch.cat((content_feature,comment_feature),dim=1)
        common_feature = self.together_1(common_feature)
        
        # # common_feature = torch.cat((content_feature,comment_feature,expert_feature),dim=1)
        # # common_feature = self.together_1(common_feature)
        
        #after fusion 
        category = kwargs['category']
        idxs = category.long().view(-1, 1).to(self.domain_embedder.weight.device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        #domain_feature = self.gate_feature(domain_embedding)
        
        #final fusion
        gate_feature = self.gate_feature(domain_embedding)
        loss2 = self.scl(gate_feature,gate_feature,category)
        
        shared_feature = torch.cat((common_feature,gate_feature),dim=1)
        shared_feature = self.together_2(shared_feature)
        # # for i in range(self.num_expert):
        #     tmp_feature = self.expert[i](bert_feature)
        # #     shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))
        # shared_feature = common_feature
        # #loss2
        # domain_center = self.domain_center.expand(content_feature.size(0), -1)
        domain_logits = self.domain_classifier(shared_feature)
        #loss2 = self.domain_loss_fn(domain_logits, category)
        
        out = []
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        loss1 = lossfun(output,label.float())

        loss = (1-self.alpha)*loss1 + self.alpha*loss2
        #+ self.alpha*loss2
        out.append(logits)         #0
        out.append(output)         #1
        out.append(shared_feature) #2
        out.append(loss) 
        return out    
            
    
