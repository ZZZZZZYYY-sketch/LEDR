import numpy as np
import torch
import tqdm
from sklearn import  metrics as  metr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F
import os
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F
import os
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class SCL(torch.nn.Module):
    def __init__(self, temperature=0.1, total_epochs=50, ema_decay=0.9):
        super(SCL, self).__init__()
        self.temperature = temperature
        self.total_epochs = total_epochs

    def forward(self, inrep_1, inrep_2, domain_labels):
        bs_1 = int(inrep_1.shape[0])  # 获取批次大小
        bs_2 = int(inrep_2.shape[0])
        normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
        normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)  # 输入特征向量归一化处理/-
        cosine_similarity = torch.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

        diag = torch.diag(cosine_similarity)  # 对角线元素
        cos_diag = torch.diag_embed(diag)  # bs,bs

        label = torch.unsqueeze(domain_labels,-1)

        # 标签矩阵
        if label.shape[0] == 1:
            cos_loss = torch.zeros(1)
        else:
            for i in range(label.shape[0] - 1):
                if i == 0:
                    label_mat = torch.cat((label, label), -1)
                else:
                    label_mat = torch.cat((label_mat, label), -1)  # bs, bs
            domain_mask = (domain_labels.unsqueeze(1) == domain_labels.unsqueeze(0)).float()
            mid_mat_ = (label_mat.eq(label_mat.t()))
            mid_mat = mid_mat_.float()  # 标签矩阵的相等性矩阵
            cosine_similarity = (cosine_similarity - cos_diag)
            cosine_similarity[domain_mask == 0] = cosine_similarity[domain_mask == 0] 
            cosine_similarity = cosine_similarity / self.temperature  # the diag is 0计算对比学习损失
            mid_diag = torch.diag_embed(torch.diag(mid_mat))
            mid_mat = mid_mat - mid_diag

            cosine_similarity = cosine_similarity.masked_fill_(mid_diag.bool(), -float(
                'inf'))  # mask the diag将对角线元素掩盖为无穷大 确保softmax中对角线元素不会产生影响

            cos_loss = torch.log(
                torch.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1
            # softmax计算每个样本和其他样本的相似度
            cos_loss = cos_loss * mid_mat

            cos_loss = torch.sum(cos_loss, dim=1) / (torch.sum(mid_mat, dim=1) + 1e-10)  # bs
            
        cos_loss = -torch.mean(cos_loss, dim=0)
        return cos_loss
        
def plot(feature, category):
    """可视化特征（仅修改刻度字体为 Times New Roman）"""
    # 转换输入数据
    X = torch.tensor(feature)
    y = torch.tensor(category)

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', 'v', '^', 'p', '*', 'h', 'H', '+']
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
              '#edc948', '#b07aa1', '#ff9da7', '#9c755f']
    
    for i, (marker, color) in enumerate(zip(markers, colors)):
        mask = (y == i)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   marker=marker, color=color, s=120,
                   alpha=0.9, edgecolors='w', label=f'Category {i}')

    # 获取当前坐标轴
    ax = plt.gca()

    # 设置刻度标签字体为 Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(23)  # 同时设置字号

    plt.savefig('output.png', dpi=300, bbox_inches='tight')
    plt.show()
class GridRecorder(): 

    def __init__(self, early_step):
        self.max = {'f1': 0,'FPED': 1}
        self.cur = {'f1': 0,'FNED': 1}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['f1'] > self.max['f1']:
            # if (self.cur['f1'] - self.max['f1']>=0.002) or ((self.cur['FNED']+self.cur['FPED'])<(self.max['FNED']+self.max['FPED'])):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)
class Recorder(): 

    def __init__(self, early_step):
        self.max = {'f1': 0,'FPED': 1}
        self.cur = {'f1': 0,'FNED': 1}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['f1'] > self.max['f1']:
            # if (self.cur['f1'] - self.max['f1']>=0.002) or ((self.cur['FNED']+self.cur['FPED'])<(self.max['FNED']+self.max['FPED'])):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)
def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': metr.roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

    metrics_by_category['overallauc'] = metr.roc_auc_score(y_true, y_pred, average='macro').round(4),

    metrics_by_category['overallauc'] = list(metrics_by_category['overallauc'])[0]

    y_pred = np.around(np.array(y_pred)).astype(int)

    metrics_by_category['f1'] = metr.f1_score(y_true, y_pred, average='macro').round(4)

    allcm = metr.confusion_matrix(y_true, y_pred)
    print(allcm)

    tn, fp, fn, tp = allcm[0][0], allcm[0][1], allcm[1][0], allcm[1][1]
    metrics_by_category['overallFNR'] = (fn / (tp + fn)).round(4)
    metrics_by_category['overallFPR'] = (fp / (fp + tn)).round(4)
    # metrics_by_category['overallFNR'] = list(metrics_by_category['overallFNR'])[0]
    # metrics_by_category['overallFPR'] = list(metrics_by_category['overallFPR'])[0]

    for c, res in res_by_category.items():
        # try:
        metrics_by_category[c]['auc'] = metr.roc_auc_score(res['y_true'], res['y_pred'], average='macro').round(4),
        metrics_by_category[c]['auc'] = list(metrics_by_category[c]['auc'])[0]

        metrics_by_category[c]['f1'] = metr.f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                                     average='macro').round(4).tolist(),
        metrics_by_category[c]['f1'] = list(metrics_by_category[c]['f1'])[0]

        cm = metr.confusion_matrix(res['y_true'], np.around(np.array(res['y_pred'])).astype(int))
        tn1, fp1, fn1, tp1 = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        metrics_by_category[c]['FNR'] = (fn1 / (tp1 + fn1)).round(4)
        metrics_by_category[c]['FPR'] = (fp1 / (fp1 + tn1)).round(4)

    metrics_by_category['FNED'] = 0
    metrics_by_category['FPED'] = 0

    for k, v in category_dict.items():
        metrics_by_category['FNED'] += abs(metrics_by_category['overallFNR'] - metrics_by_category[k]['FNR'])
        metrics_by_category['FPED'] += abs(metrics_by_category['overallFPR'] - metrics_by_category[k]['FPR'])
        metrics_by_category['FPED'] = metrics_by_category['FPED'].round(4)
        metrics_by_category['FNED'] = metrics_by_category['FNED'].round(4)

    return metrics_by_category


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'comments': batch[2].cuda(),
            'comments_masks': batch[3].cuda(),
            'aug_comments': batch[4].cuda(),
            'aug_comments_masks': batch[5].cuda(),
            'content_emotion': batch[6].cuda(),
            'comments_emotion': batch[7].cuda(),
            'emotion_gap': batch[8].cuda(),
            'style_feature': batch[9].cuda(),
            'label': batch[10].cuda(),
            'category': batch[11].cuda(),
            'categoryonehot': batch[12].cuda()
            }
        # batch_data = {
        #     'content': batch[0].cuda(),
        #     'content_masks': batch[1].cuda(),
        #     'comments': batch[2].cuda(),
        #     'comments_masks': batch[3].cuda(),
        #     'content_emotion': batch[4].cuda(),
        #     'comments_emotion': batch[5].cuda(),
        #     'emotion_gap': batch[6].cuda(),
        #     'style_feature': batch[7].cuda(),
        #     'label': batch[8].cuda(),
        #     'category': batch[9].cuda(),
        #     'categoryonehot': batch[10].cuda()
        #     }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'comments': batch[2],
            'comments_masks': batch[3],
            'aug_comments': batch[4],
            'aug_comments_masks': batch[5],
            'content_emotion': batch[6],
            'comments_emotion': batch[7],
            'emotion_gap': batch[8],
            'style_feature': batch[9],
            'label': batch[10],
            'category': batch[11],
            'categoryonehot': batch[12]
            }
        # batch_data = {
        #     'content': batch[0],
        #     'content_masks': batch[1],
        #     'comments': batch[2],
        #     'comments_masks': batch[3],
        #     'content_emotion': batch[4],
        #     'comments_emotion': batch[5],
        #     'emotion_gap': batch[6],
        #     'style_feature': batch[7],
        #     'label': batch[8],
        #     'category': batch[9],
        #     'categoryonehot': batch[10]
        #     }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
