import scipy.io as scio
import torchvision
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
import time
import numpy as np 
import pandas as pd
import random
import math
from torch.nn import functional as F
import csv

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename="./ref_traj/"+'E'+"_reftraj.mat"

t_data=[]
y_data=[]
sigma_data=[]
file_data=scio.loadmat(filename)['refTraj'][0]
for data in file_data:
    t_data.append([data[0][0][0]-1.0])
    y_data.append([data[1][0][0],data[1][1][0],data[1][2][0],data[1][3][0]])
    sigma_data.append(data[2])

t_data=np.array(t_data)
y_data=np.array(y_data)
sigma_data=np.array(sigma_data)

batch_size_K = 40
batch_size_outer = 100
meta_lambda=0.03
n_epochs = 200

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.params = [
                    torch.Tensor(128, 8).uniform_(-1./math.sqrt(8), 1./math.sqrt(8)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(4, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(4).zero_().requires_grad_(),
                ]

    def dense(self, x, params):
        x = F.linear(x, params[0], params[1])
        x = F.relu(x)

        x = F.linear(x, params[2], params[3])
        x = F.relu(x)

        x = F.linear(x, params[4], params[5])
        x = F.relu(x)

        x = F.linear(x, params[6], params[7])
        return x

    def input_process(self, x):
        x2=torch.pow(x, 2)
        x3=torch.pow(x, 3)
        x4=torch.pow(x, 4)
        x_sin=torch.sin(x*3.14)
        x_cos=torch.cos(x*3.14)
        x_sin_2=torch.sin(2*x*3.14)
        x_cos_2=torch.cos(2*x*3.14)
        return torch.cat((x,x2,x3,x4,x_sin,x_cos,x_sin_2,x_cos_2), 1)

    def forward(self, x, params):
        return self.dense(self.input_process(x), params)*10.0

def my_mse_loss(outputs, Q, Sigma):
    a=outputs - Q 
    a=torch.reshape(a,(-1,4,1))
    b=torch.reshape(a,(-1,1,4))
    #print(a.shape)
    #print(b.shape)
    #print(Sigma.shape)
    return torch.mean(torch.matmul(torch.matmul(b,torch.inverse(Sigma)),a))

def adjust_learning_rate(optimizer, epoch, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 


model = Model()
model = model.to(device)

learning_rate=0.003
optimizer0 = torch.optim.Adam(model.params,lr=learning_rate,weight_decay=0.01)

data_loader_train = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data).float(),torch.tensor(y_data).float(),torch.tensor(sigma_data).float()),shuffle = True, batch_size = batch_size_outer)

model.train() # 可加可不加，默认是有的
optimizer=optimizer0

for epoch in range(n_epochs):
    loss_train_sum = 0.0

    for step_train, data_train_now in enumerate(data_loader_train):
        (features, labels, sigmas)=data_train_now
        features = features.to(device)
        labels = labels.to(device)
        sigmas=sigmas.to(device)
        outputs = model(features, model.params)
        loss_train = my_mse_loss(outputs, labels,sigmas)
        optimizer.zero_grad()
        # 反向传播求梯度
        loss_train.backward()
        optimizer.step()

        loss_train_sum += loss_train.item()
        if (step_train+1) % 4 == 0:
            print(f'step = {step_train+1}, loss = {loss_train_sum / 4:.6f}')
            loss_train_sum=0

outputs=[]
inputss=[]
labelss=[]
with torch.no_grad():
    for i, (inputs, labels, sigma) in enumerate(data_loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs, model.params)
        outputs = outputs+output.cpu().numpy().tolist()
        inputss=inputss+inputs.cpu().numpy().tolist()
        labelss=labelss+labels.cpu().numpy().tolist()
outputs=np.array(outputs)
inputss=np.array(inputss)
labelss=np.array(labelss)
print(inputss.shape)
print(outputs.shape)
x=inputss[:,0]
y=outputs[:,0]
z=outputs[:,1]
y_label=labelss[:,0]
z_label=labelss[:,1]
#plt.scatter(x, y, 1, alpha=0.8)
#plt.show()
#plt.scatter(x, z, 1, alpha=0.8)
#plt.show()
plt.scatter(y, z, alpha=0.4)
plt.scatter(y_label, z_label, alpha=0.4)
plt.show()

