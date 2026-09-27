import os
import torch
import tqdm
import torch.nn as nn
import datetime
import numpy as np
from models.layers import *
from models.mdfend import MultiDomainFENDModel as MDFENDModel
from models.textcnn_u import StudentModel as TextCNN_U
from models.student_ad import StudentModel as StudentADModel
from models.bigru import BiGRUModel
from models.bert import BertFNModel
from models.m3fend import M3FENDModel
from models.dblk import CloserModel
from utils.utils import data2gpu, Averager, metrics, Recorder

from torch.nn import functional as F

def euclidean_dist(shared_feature):
    trans=shared_feature.T
    dist_matrix=torch.cdist(trans,trans)
    dist_matrix=dist_matrix.T
    return dist_matrix

def distillation(student_scores,teacher_scores,temp):
    loss_soft=F.kl_div(F.log_softmax(student_scores/temp,dim=1),F.softmax(teacher_scores/temp,dim=1),reduction="batchmean")
    return loss_soft*temp*temp

class Trainer():
    def __init__(self,
                 model_name,
                 emb_dim,
                 mlp_dims,
                 usemul,
                 logits_shape,
                 use_cuda,
                 dataset,
                 lr,
                 dropout,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 early_stop,
                 epoches,
                 train_loader,
                 val_loader,
                 test_loader,
                 domain_num,
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.usemul = usemul
        self.logits_shape = logits_shape

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.domain_num = domain_num
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self):
        # if self.modelname1 == 'mdfend':
        #     self.teacher0=MDFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset,logits_shape=self.logits_shape)
        # elif self.modelname1 == 'm3fend':
        #     self.teacher0 = M3FENDModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
        #                            self.style_num, self.lnn_dim, len(self.category_dict), dataset=self.dataset,logits_shape=self.logits_shape)
        # self.teacher1 =StudentADModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, dataset=self.dataset, logits_shape=self.logits_shape)
        # if self.modelname2 == 'textcnn-u':
        #     self.model = StudentModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout,
        #                           dataset=self.dataset, logits_shape=self.logits_shape)
        # elif self.modelname2=='bigru-u':
        #     self.model = BiGRUModel(self.emb_dim, 1, self.mlp_dims, self.dropout, self.dataset)
        self.model = CloserModel(self.emb_dim, self.mlp_dims, self.domain_num, self.dropout, self.dataset, 0.15) 

        if self.use_cuda:
            self.model = self.model.cuda()
        lossfun = torch.nn.BCELoss()
        loss_fn2=torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        for epoch in range(self.epoches):
            self.model.train()
            avg_loss = Averager()
            train_data_iter = tqdm.tqdm(self.train_loader)
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                out = self.model(**batch_data)
                loss = out[3]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results = self.test(self.val_loader, 0)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter' + 'scl1'+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir,'parameter'+
                                        'scl1'+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl')))
        results = self.test(self.test_loader, 1)
        print(results)
        with open('scl1.txt', 'w', encoding='utf-8') as f:
            f.write(str(results))
        return results, os.path.join(self.save_param_dir, 'parameter' +'scl1'+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl')

    def test(self, dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.model(**batch_data)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())

        result = metrics(label, pred, category, self.category_dict)
        if testorval == 1:
            os.makedirs('recodertestpkl', exist_ok=True)
            torch.save(self.model,
                       'recodertestpkl/' +'scl1'+ self.dataset + '.pkl')

        return result

    def testteacher0(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher0.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher0(**batch_data)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result
    def testteacher1(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher1.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher1(**batch_data, alpha=-1)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result
