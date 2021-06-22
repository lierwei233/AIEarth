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


# In[2]:


# 指定上传或者本地跑模型
UPLOAD = True
if UPLOAD:
    data_train_dir = './tcdata/enso_round1_train_20210201/'
    data_test_dir = './tcdata/enso_round1_test_20210201/'
else:
    data_train_dir = '/data/anonym5/zrk/AIEarth/train/'
    data_test_dir = '/data/anonym5/zrk/AIEarth/test/'


# In[3]:


def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

set_seed()


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
    train_label = label['nino'][:, 12:36].values

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
    train_label2 = label2['nino'][:, 12:36].values

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
    'n_epochs' : 30,
    'learning_rate' : 8e-5,
    'batch_size' : 64,
}

train_dataset, valid_dataset = load_data2()
train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])


# # CNN + LSTM

# In[5]:


class simpleSpatailTimeNN(nn.Module):
    def __init__(self, kernels:list, n_lstm_units:int=64):
        super(simpleSpatailTimeNN, self).__init__()
#         self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernels]) 
#         self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernels])
#         self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernels])
#         self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernels])
        self.conv_sst = self._define_CNN_layer_for_one_feature(kernels)
        self.conv_t300 = self._define_CNN_layer_for_one_feature(kernels)
        self.conv_ua = self._define_CNN_layer_for_one_feature(kernels)
        self.conv_va = self._define_CNN_layer_for_one_feature(kernels)
        
        self.pool1 = nn.AdaptiveAvgPool2d((22, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 70))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(108*4, n_lstm_units, 2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, 24)
    
    def _define_CNN_layer_for_one_feature(self, kernels):
        conv1 = nn.Conv2d(in_channels=12, out_channels=30, kernel_size=kernels[0])
        pool1 = nn.MaxPool2d(kernel_size=kernels[1])
        conv2 = nn.Conv2d(in_channels=30, out_channels=12, kernel_size=kernels[2])
        pool2 = nn.MaxPool2d(kernel_size=kernels[3])
        
        return nn.ModuleList([conv1, pool1, conv2, pool2])

    def forward(self, sst, t300, ua, va):
        for i, conv in enumerate(self.conv_sst):
            if i == 0:
                sst = F.pad(sst, [4, 3, 2, 1])  # 此时，kernel为(4, 8)
            elif i == 2:
                sst = F.pad(sst, [2, 1, 1, 0])  # 此时，kernel为(2, 4)
            sst = conv(sst)  # batch * 12 * 6 * 18
        for i, conv in enumerate(self.conv_t300):
            if i == 0:
                t300 = F.pad(t300, [4, 3, 2, 1])  # 此时，kernel为(4, 8)
            elif i == 2:
                t300 = F.pad(t300, [2, 1, 1, 0])  # 此时，kernel为(2, 4)
            t300 = conv(t300)  # batch * 12 * 6 * 18
        for i, conv in enumerate(self.conv_ua):
            if i == 0:
                ua = F.pad(ua, [4, 3, 2, 1])  # 此时，kernel为(4, 8)
            elif i == 2:
                ua = F.pad(ua, [2, 1, 1, 0])  # 此时，kernel为(2, 4)
            ua = conv(ua)  # batch * 12 * 6 * 18
        for i, conv in enumerate(self.conv_va):
            if i == 0:
                va = F.pad(va, [4, 3, 2, 1])  # 此时，kernel为(4, 8)
            elif i == 2:
                va = F.pad(va, [2, 1, 1, 0])  # 此时，kernel为(2, 4)
            va = conv(va)  # batch * 12 * 6 * 18

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 108
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)
            
        x = torch.cat([sst, t300, ua, va], dim=-1)  # batch * 12 * 432
        x = self.batch_norm(x)  # batch * 12 * 432
        x, _ = self.lstm(x)  # batch * 12 * 64
        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)
        return x

kernels = [(4, 8), (2, 2), (2, 4), (2, 2)]  # C --> MP --> C --> MP
model = simpleSpatailTimeNN(kernels=kernels, n_lstm_units=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'   
optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
loss_fn = nn.MSELoss()


# In[6]:


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


# In[9]:


model.to(device)
loss_fn.to(device)

scores = []
for i in range(fit_params['n_epochs']):
    model.train()
    for step, ((sst, t300, ua, va), label) in enumerate(train_loader):                
        sst = sst.to(device).float()
        t300 = t300.to(device).float()
        ua = ua.to(device).float()
        va = va.to(device).float()
        optimizer.zero_grad()
        label = label.to(device).float()
        preds = model(sst, t300, ua, va)
        loss = loss_fn(preds, label)
        loss.backward()
        optimizer.step()
#         print('Step: {}, Train Loss: {}'.format(step, loss))

    model.eval()
    y_true, y_pred = [], []
    for step, ((sst, t300, ua, va), label) in enumerate(valid_loader):
        sst = sst.to(device).float()
        t300 = t300.to(device).float()
        ua = ua.to(device).float()
        va = va.to(device).float()
        label = label.to(device).float()
        preds = model(sst, t300, ua, va)

        y_pred.append(preds)
        y_true.append(label)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    score = eval_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    scores.append(score)
    print('Epoch: {}, Valid Score {}'.format(i + 1, score))
    
    if len(scores) > 1:
        if score <= scores[-2]:
            print("Early Stopping")
            break

# torch.save(self.model.state_dict(), '../user_data/ref.pkl')
# torch.save(model, '/data/anonym5/zrk/AIEarth/model_baseline/model_baseline.pkl')
# print('Model saved successfully')


# # 测试

# In[10]:


# teX = np.load('/data/anonym5/zrk/AIEarth/test/test_0144-01-12.npy','r')
# teY = np.load('/data/anonym5/zrk/AIEarth/test/label/test_0144-01-12.npy','r')
# te_sst = torch.from_numpy(teX[:, :, :, 0].reshape(1, 12, 24, 72)).to(device).float()
# te_t300 = torch.from_numpy(teX[:, :, :, 1].reshape(1, 12, 24, 72)).to(device).float()
# te_ua = torch.from_numpy(teX[:, :, :, 2].reshape(1, 12, 24, 72)).to(device).float()
# te_va = torch.from_numpy(teX[:, :, :, 3].reshape(1, 12, 24, 72)).to(device).float()

# model.eval()
# te_pred = model(te_sst, te_t300, te_ua, te_va)

# te_pred = te_pred.detach().cpu().numpy()[0]

# import matplotlib.pyplot as plt
# plt.plot(te_pred)
# plt.plot(teY)


# # 上传
# 
# 如果不上传docker，不要运行下面的代码
# 
# 上传代码的话，还需要修改对应的test文件、train文件路径。

# In[ ]:


if UPLOAD:
    res_dir = './result/'

    # Clear './result/' directory
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir, ignore_errors=True)
    os.makedirs(res_dir)

    # predict and save to './result/' directory
    for filename in os.listdir(data_test_dir):
        teX = np.load(data_test_dir + filename)
        te_sst = torch.from_numpy(teX[:, :, :, 0].reshape(1, 12, 24, 72)).to(device).float()
        te_t300 = torch.from_numpy(teX[:, :, :, 1].reshape(1, 12, 24, 72)).to(device).float()
        te_ua = torch.from_numpy(teX[:, :, :, 2].reshape(1, 12, 24, 72)).to(device).float()
        te_va = torch.from_numpy(teX[:, :, :, 3].reshape(1, 12, 24, 72)).to(device).float()

        model.eval()
        prY = model(te_sst, te_t300, te_ua, te_va)
        prY = prY.detach().cpu().numpy()[0]
        np.save(res_dir + filename, prY)
    # save to an archive:

    shutil.make_archive('result', 'zip', res_dir)

