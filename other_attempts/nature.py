#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import shutil
# import matplotlib.pyplot as plt


# # 一些想法
# 1. 改channels参数（Copy1在跑）
# 2. SODA-refine
# 3. 改成nature的模型（0329改）

# In[2]:


# 指定上传或者本地跑模型
UPLOAD = True
if UPLOAD:
    data_train_dir = '/tcdata/enso_round1_train_20210201/'
    data_test_dir = '/tcdata/enso_final_test_data_B/'
else:
    data_train_dir = '/data/anonym5/zrk/AIEarth/train/'
    data_test_dir = '/data/anonym5/zrk/AIEarth/test/'


# In[3]:


def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

set_seed(427)


# In[4]:


class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):   
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), self.data['label'][idx]

    
def load_data2():
    # CMIP data    
    train = xr.open_dataset(data_train_dir + 'CMIP_train.nc')
    label = xr.open_dataset(data_train_dir + 'CMIP_label.nc')
   
    train_sst = train['sst'][:, :12].values  # (4645, 12, 24, 72)
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    train_label = label['nino'][:, :].values

    train_ua = np.nan_to_num(train_ua)
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    # SODA data    
    train2 = xr.open_dataset(data_train_dir + 'SODA_train.nc')
    label2 = xr.open_dataset(data_train_dir + 'SODA_label.nc')
    
    train_sst2 = train2['sst'][:, :12].values  # (4645, 12, 24, 72)
    train_t3002 = train2['t300'][:, :12].values
    train_ua2 = train2['ua'][:, :12].values
    train_va2 = train2['va'][:, :12].values
    train_label2 = label2['nino'][:, :].values

    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(train_label2)))
    
    dict_train = {
        'sst':train_sst,
        't300':train_t300,
        'ua':train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst':train_sst2,
        't300':train_t3002,
        'ua':train_ua2,
        'va': train_va2,
        'label': train_label2}
    train_dataset = EarthDataSet(dict_train)
    valid_dataset = EarthDataSet(dict_valid)
    return train_dataset, valid_dataset


fit_params = {
    'n_epochs' : 5000,
    'learning_rate' : 8e-6,
    'batch_size' : 64,
}

train_dataset, valid_dataset = load_data2()
train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])
refine_loader = DataLoader(valid_dataset,batch_size=100)


# # CNN + LSTM
# 
# **gz 3月29日晚修改**
# - 按照nature的网络结构，在pool2之后新增一个Conv2d层；
# - 按照nature的网络结构，去掉batchnorm

# In[11]:


class simpleSpatailTimeNN(nn.Module):
    def __init__(self):
        super(simpleSpatailTimeNN, self).__init__()

#         self.layers_sst = self._define_layers_for_one_feature(kernels)
#         self.layers_t300 = self._define_layers_for_one_feature(kernels)
#         self.layers_ua = self._define_layers_for_one_feature(kernels)
#         self.layers_va = self._define_layers_for_one_feature(kernels)
#         self.linear = nn.Linear(96, 24)
#         self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=30, kernel_size=(4, 8)) # [batch, 30, 24, 72]
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(2, 4))
        self.pool2 = nn.AvgPool2d(kernel_size=(2,2))
#         self.conv3 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(2, 4))
        self.batch_norm = nn.BatchNorm1d(30, affine=False)
#         lstm = nn.LSTM(108, 8, 2, bidirectional=True, batch_first=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 108))
#         lstm = nn.LSTM(32, 1, 2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(108, 24)
#         self.linear1 = nn.Linear(30*108, 50)
        self.tanh = nn.Tanh()
#         self.linear2 = nn.Linear(50, 24)
    
    def forward(self, sst, t300):
        """
        layer 1 (padding + conv1)             [b, 6, 24, 72] --> [b, 30, 24, 72]
        layer 2 (pool1)                       [b, 30, 24, 72] --> [b, 30, 12, 36]
        layer 3 (padding + conv2)             [b, 30, 12, 36] --> [b, 30, 12, 36]
        layer 4 (pool2)                       [b, 30, 12, 36] --> [b, 30, 6,  18]
        layer 5 (flatten + batch_norm)        [b, 30, 6,  18] --> [b, 30, 108]
        layer 6 (pool3)                       [b, 30, 108]     --> [b, 1, 108]
        layer 7 (flatten + linear + tanh)     [b, 108]      --> [b, 24]
        """
        inp = torch.cat([sst, t300], dim=1)
#         print(inp.shape)
    
        # 1: padding + conv1:  [b, 12, 24, 72] --> [b*12, 1, 24, 72] --> [b*12, 30, 24, 72]
        out1 = F.pad(inp, [4, 3, 2, 1])  # 先padding，后卷积
        out1 = self.conv1(out1)
        
        # 2: pool1:  [b*12, 30, 24, 72] --> [b*12, 30, 12, 36]
        out2 = self.pool1(out1)
        
        # 3: padding + conv2:  [b*12, 30, 12, 36] --> [b*12, 1, 12, 36]
        out3 = F.pad(out2, [2, 1, 1, 0])
        out3 = self.conv2(out3)
        
        # 4: pool2:  [b*12, 1, 12, 36] --> [b*12, 1, 6, 18]
        out4 = self.pool2(out3)
        
        # 5: flatten + batch_norm:  [b, 12, 6, 18] --> [b, 12, 108]
        out5 = torch.flatten(out4, start_dim=2)
        out5 = self.batch_norm(out5)
        
#         out6 = lstm(out5)[0]
        out6 = self.pool3(out5) #[b, 1, 108]
        
        # 7: flatten + linear + tanh:  [b, 108]  --> [b, 24]
        out7 = torch.reshape(out6,(-1,108))
        out7 = self.linear(out7)
        out7 = self.tanh(out7)        
    
        return out7


# model1 = simpleSpatailTimeNN(kernels=kernels)
models = []
for i in range(10):
    models.append(simpleSpatailTimeNN())

device = 'cuda' if torch.cuda.is_available() else 'cpu'   
# optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
# loss_fn = nn.MSELoss()


# In[12]:


def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean)**2) * sum((y - y_mean)**2)
    return c1/np.sqrt(c2)

def rmse(preds, y):
    return np.sqrt(sum((preds - y)**2)/preds.shape[0])

def eval_score(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    acskill = 0
    RMSE = 0
    a = 0
    a = [1.5]*4 + [2]*7 + [3]*7 + [4]*6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])
    
        acskill += a[i] * np.log(i+1) * cor
    return 2/3 * acskill - RMSE


# In[13]:


Valid_Score = np.zeros([10])
for j in range(10):
    model = models[j]
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
    loss_fn = nn.MSELoss()
    model.to(device)
    loss_fn.to(device)

    scores = []
    for i in range(fit_params['n_epochs']):
        model.train()
        for step, ((sst, t300, ua, va), label) in enumerate(train_loader):  
            sst = sst[:,j:j+3,:,:]
            t300 = t300[:,j:j+3,:,:]
            label = label[:,j+3:j+27]
#             label = label[:,j+3:j+27]
#             print(label.shape)
            sst = sst.to(device).float()
            t300 = t300.to(device).float()
            optimizer.zero_grad()
            label = label.to(device).float()
            preds = model(sst, t300)
            loss = loss_fn(preds, label)
            loss.backward()
            optimizer.step()
    #         print('Step: {}, Train Loss: {}'.format(step, loss))

        model.eval()
        y_true, y_pred = [], []
        for step, ((sst, t300, ua, va), label) in enumerate(valid_loader):
            sst = sst[:,j:j+3,:,:]
            t300 = t300[:,j:j+3,:,:]
            label = label[:,j+3:j+27]
            sst = sst.to(device).float()
            t300 = t300.to(device).float()
            label = label.to(device).float()
            preds = model(sst, t300,)

            y_pred.append(preds)
            y_true.append(label)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        score = eval_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        scores.append(score)
        if i % 10 == 0:
            print(f'Epoch: {i+1}, Valid Score {score}')

        if len(scores) > 1:
            if score <= scores[-2]:
                print(f"Model_{j}: Early Stopping with score {score}")
                break
        Valid_Score[j] = score
    # # torch.save(self.model.state_dict(), '../user_data/ref.pkl')
#     torch.save(model, '/data/anonym5/zrk/AIEarth/model_baseline/model_baseline.pkl')
#     print('Model saved successfully')


# # 测试

# In[ ]:


num_models = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                       10, 10, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])  # 13~36月中每个月涉及到的模型的个数。最后一个月（36）涉及的模型数为1
Valid_Score[Valid_Score < 0] = 0

def predict(sst, t300):
    # 根据10个模型，(b, 12, 24, 72) * 4 的输入，计算(b, 24)的输出
    y_pred = np.zeros(shape=[sst.shape[0], 24])
    for i in range(10):
        # 第i个模型对应的是(i):(i+3)的输入，(i+3):(i+27)的预测。
        model = models[i]
        model.eval()
        y_pred_temp = model(sst[:, i:i+3, :, :], t300[:, i:i+3, :, :])
        y_pred_temp = y_pred_temp.detach().cpu().numpy()
        start = 12  # 有用预测的开始月
        end = i + 27  # 有用预测的结束月
        # 因为y_pred中是从第13月开始的，所以start和end要再减12
        start = start - 12
        end = end - 12
        num_valid_month = end - start  # 真正有用的预测月份个数
        
        # 权重
        weights = np.zeros([1, num_valid_month])
        for j in range(num_valid_month):
            num_model = num_models[j]  # 预测第12+j月的有num_model个模型
            weights[0, j] = Valid_Score[i] / Valid_Score[-num_model:].sum()
            if j == num_valid_month - 1:
                print(weights)
        y_pred[:, :num_valid_month] = y_pred[:, :num_valid_month] + weights * y_pred_temp[:, start:end]
    return y_pred

# y_true, y_pred = [], []
# for step, ((sst, t300, ua, va), label) in enumerate(valid_loader):
#     sst = sst[:, :, :, :].to(device).float()
#     t300 = t300[:,:,:,:].to(device).float()
# #     ua = ua[:,:,:,:].to(device).float()
# #     va = va[:,:,:,:].to(device).float()
    
#     label = label[:, 12:].cpu().numpy()
#     preds = predict(sst, t300)

#     y_pred.append(preds)
#     y_true.append(label)
# y_pred = np.concatenate(y_pred, axis=0)
# y_true = np.concatenate(y_true, axis=0)
# print("Final score on SODA:", eval_score(y_true, y_pred))


# In[ ]:


# Valid_Score


# In[ ]:


# teX = np.load('/data/anonym5/zrk/AIEarth/test/test_0144-01-12.npy','r')
# teY = np.load('/data/anonym5/zrk/AIEarth/test/label/test_0144-01-12.npy','r')
# for j in range(10):
#     model = models[j]
#     te_sst = torch.from_numpy(teX[:, :, :, 0].reshape(1, 12, 24, 72)).to(device).float()
#     te_t300 = torch.from_numpy(teX[:, :, :, 1].reshape(1, 12, 24, 72)).to(device).float()
# #     te_ua = torch.from_numpy(teX[:, :, :, 2].reshape(1, 12, 24, 72)).to(device).float()
# #     te_va = torch.from_numpy(teX[:, :, :, 3].reshape(1, 12, 24, 72)).to(device).float()

#     te_pred = predict(te_sst, te_t300)

#     te_pred = te_pred[0]

# import matplotlib.pyplot as plt
# plt.plot(te_pred)
# plt.plot(teY)


# # 上传
# 
# 如果不上传docker，不要运行下面的代码
# 
# 上传代码的话，还需要修改对应的test文件、train文件路径。

# In[10]:


if UPLOAD:
    res_dir = '/result/'

    # predict and save to './result/' directory
    for filename in os.listdir(data_test_dir):
        teX = np.load(data_test_dir + filename)
        te_sst = torch.from_numpy(teX[:, :, :, 0].reshape(1, 12, 24, 72)).to(device).float()
        te_t300 = torch.from_numpy(teX[:, :, :, 1].reshape(1, 12, 24, 72)).to(device).float()
        te_ua = torch.from_numpy(teX[:, :, :, 2].reshape(1, 12, 24, 72)).to(device).float()
        te_va = torch.from_numpy(teX[:, :, :, 3].reshape(1, 12, 24, 72)).to(device).float()

        prY = predict(te_sst, te_t300)
        np.save(res_dir + filename, prY)
    
    # save to an archive:
    shutil.make_archive('result', 'zip', res_dir)


# In[ ]:




