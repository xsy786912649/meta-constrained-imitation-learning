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

filename_list=["./ref_traj/"+i+"_reftraj.mat" for i in [chr(x) for x in range(ord('A'), ord('W') + 1)]]
test_file_name_list=["./ref_traj/"+i+"_reftraj.mat" for i in [chr(x) for x in range(ord('X'), ord('Z') + 1)]]


t_data_list=[]
y_data_list=[]
sigma_data_list=[]

t_data_test_list=[]
y_data_test_list=[]
sigma_data_test_list=[]

whole_task_num=len(filename_list) 
task_test_num=len(test_file_name_list)

for filename in filename_list:
    t_data=[]
    y_data=[]
    sigma_data=[]
    file_data=scio.loadmat(filename)['refTraj'][0]
    for data in file_data:
        t_data.append([data[0][0][0]-1.0])
        y_data.append([data[1][0][0],data[1][1][0],data[1][2][0],data[1][3][0]])
        sigma_data.append(data[2]+0.001*np.identity(4))
    t_data_list.append(np.array(t_data))
    y_data_list.append(np.array(y_data))
    sigma_data_list.append(np.array(sigma_data))

for filename in test_file_name_list:
    t_data=[]
    y_data=[]
    sigma_data=[]
    file_data=scio.loadmat(filename)['refTraj'][0]
    for data in file_data:
        t_data.append([data[0][0][0]-1.0])
        y_data.append([data[1][0][0],data[1][1][0],data[1][2][0],data[1][3][0]])
        sigma_data.append(data[2]+0.001*np.identity(4))
    t_data_test_list.append(np.array(t_data))
    y_data_test_list.append(np.array(y_data))
    sigma_data_test_list.append(np.array(sigma_data))

batch_size_K = 40
batch_size_outer = 80
meta_lambda=0.00005
n_epochs = 400
n_inner_level_epochs=80
    
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

                    torch.Tensor(2, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(2).zero_().requires_grad_(),
                ]

    def dense(self, x, params):
        y = F.linear(x, params[0], params[1])
        y = F.relu(y)

        y = F.linear(y, params[2], params[3])
        y = F.relu(y)

        y = F.linear(y, params[4], params[5])
        y = F.relu(y)

        y = F.linear(y, params[6], params[7])

        return y

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
        v = torch.ones(x.shape,dtype=torch.float) 
        position=self.dense(self.input_process(x), params)*10.0
        position1,position2=position.split([1,1],dim=1)
        vel1=torch.autograd.grad(position1,x,v,retain_graph=True, create_graph=True)[0]
        vel2=torch.autograd.grad(position2,x,v,retain_graph=True, create_graph=True)[0]
        return torch.cat((position1,position2,vel1,vel2), 1)
    
    def forward1(self, x, params):
        position=self.dense(self.input_process(x), params)*10.0
        return position

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
optimizer0 = torch.optim.Adam(model.params,lr=learning_rate,weight_decay=0.0000)
learning_rate=0.0003
optimizer1 = torch.optim.SGD(model.params,lr=learning_rate,weight_decay=0.0000)

for epoch in range(n_epochs):

    if epoch<n_epochs-50:
        task_num=23
    elif epoch>=n_epochs-50:
        task_num=23 #len(x_train_list)
    number_list=random.sample(range(whole_task_num),task_num)
    t_data_list_thisepoch=[]
    y_data_list_thisepoch=[]
    sigma_data_list_thisepoch=[]
    for i in number_list:
        #print(filename_list[i])
        t_data_list_thisepoch.append(t_data_list[i])
        y_data_list_thisepoch.append(y_data_list[i])
        sigma_data_list_thisepoch.append(sigma_data_list[i])
    t_data_list_thisepoch=t_data_list_thisepoch+t_data_test_list
    y_data_list_thisepoch=y_data_list_thisepoch+y_data_test_list
    sigma_data_list_thisepoch=sigma_data_list_thisepoch+sigma_data_test_list

    data_loader_train_list=[]
    data_loader_train_list_outer=[]
    for i in range(len(t_data_list_thisepoch)):
        data_loader_train = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_list_thisepoch[i]).float().requires_grad_(),torch.tensor(y_data_list_thisepoch[i]).float(),torch.tensor(sigma_data_list_thisepoch[i]).float()),shuffle = True, batch_size = batch_size_K)
        data_loader_train_list.append(data_loader_train)
        data_loader_train1 = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_list_thisepoch[i]).float().requires_grad_(),torch.tensor(y_data_list_thisepoch[i]).float(),torch.tensor(sigma_data_list_thisepoch[i]).float()),shuffle = True, batch_size = batch_size_outer)
        data_loader_train_list_outer.append(data_loader_train1)

    data_train=zip(*data_loader_train_list,*data_loader_train_list_outer)
    #data_train1=zip(*data_loader_train_list,*data_loader_train_list_outer)
    #print("From "+str(sum(1 for _ in data_train1))+" select")

    model.train()
    loss_train_sum = 0.0
    loss_no_grad_sum= 0.0
    loss_test_sum= 0.0
    loss_no_grad_sum_test=0
    optimizer=optimizer0

    print(f'Epoch {epoch + 1}/{n_epochs}'.center(40,'-'))

    for step_train, data_train_now in enumerate(data_train):
        theta_prime_list=[]
        loss_meta_train_tensor=[]
        loss_meta_test_tensor=[]
        loss_no_grad=[]
        loss_no_grad_test=[]
        
        for number, data_loader_train_now in enumerate(data_train_now):
            
            if number < task_num + task_test_num:
                task_now=number
                (features, labels, sigmas)=data_loader_train_now
                features = features.to(device)
                labels = labels.to(device)
                sigmas=sigmas.to(device)
                outputs = model(features, model.params)
                loss_train = my_mse_loss(outputs, labels, sigmas)
                if number<task_num:
                    loss_no_grad.append(loss_train.item())
                else:
                    loss_no_grad_test.append(loss_train.item())
                # 梯度清零
                optimizer.zero_grad()
                grads = torch.autograd.grad(loss_train, model.params, create_graph=True, retain_graph=True)
                [(grads[i].retain_grad()) for i in range(len(model.params))]
                theta_prime = [(model.params[i] - meta_lambda*grads[i]) for i in range(len(model.params))]
                [(theta_prime[i].retain_grad()) for i in range(len(model.params))]
                theta_prime_list.append(theta_prime)

            elif number>=task_num+task_test_num and number<2*task_num+task_test_num:
                task_now=number-(task_num+task_test_num)
                (features1, labels1,sigmas1)=data_loader_train_now
                features1 = features1.to(device)
                labels1 = labels1.to(device)
                sigmas1= sigmas1.to(device)
                outputs1=model(features1, theta_prime_list[task_now])
                current_loss=my_mse_loss(outputs1, labels1,sigmas1)
                loss_meta_train_tensor.append(current_loss)
                
            
            elif number>=2*task_num+task_test_num:
                task_now=number-(task_num+task_test_num)
                (features1, labels1,sigmas1)=data_loader_train_now
                features1 = features1.to(device)
                labels1 = labels1.to(device)
                sigmas1= sigmas1.to(device)
                outputs1=model(features1, theta_prime_list[task_now])
                current_loss=my_mse_loss(outputs1, labels1,sigmas1)
                loss_meta_test_tensor.append(current_loss)
                

        loss_meta_train=sum(loss_meta_train_tensor)/float(task_num)
        
        optimizer.zero_grad()
        loss_meta_train.backward()
        optimizer.step()

        loss_meta_test=sum(loss_meta_test_tensor)/float(task_test_num)

        loss_train_sum += loss_meta_train.item()
        loss_test_sum += loss_meta_test.item()
        loss_no_grad_sum += sum(loss_no_grad)/(task_num)
        loss_no_grad_sum_test += sum(loss_no_grad_test)/(task_test_num)


        if (step_train+1) % 4 == 0:
            print(f'step = {step_train+1}, loss = {loss_train_sum / 4:.6f}')
            print(f'step = {step_train+1}, test_loss = {loss_test_sum / 4:.6f}')
            print('loss_no_grad:'+str(loss_no_grad_sum/ 4))
            print('loss_no_grad_test:'+str(loss_no_grad_sum_test/ 4))
            loss_train_sum=0
            loss_no_grad_sum=0
            loss_test_sum=0
            loss_no_grad_sum_test=0
       
    print("meta lambda:   "+str(meta_lambda))

################ save model ################
torch.save(model, 'model_meta.pkl') 

