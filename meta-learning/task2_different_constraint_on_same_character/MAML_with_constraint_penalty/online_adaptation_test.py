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



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(200)


filename_list_whole=["../../ref_traj/"+'A'+"_reftraj.mat" ]*100
center_list_whole=np.random.normal(0, 1, [len(filename_list_whole),2])

redius=2.0
less=False
weight=500.0
softplus_para=200.0

batch_size_K = 400
meta_lambda=0.00006
n_epochs = 40
    
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

def constraint_voilations(outputs, center=[0.0,0.0], less=less, redius=redius, weight=weight):
    position,vel=outputs.split([2,2],dim=1)
    center_tensor=torch.tensor(center, dtype= torch.float)
    constraint_voilations=0.0
    if less:
        constraint_voilations= (F.softplus((torch.norm(position-center_tensor,dim=1)- redius),softplus_para)-0.001)*weight
    else:
        constraint_voilations= (F.softplus((-torch.norm(position-center_tensor,dim=1)+ redius),softplus_para)-0.001)*weight
    return torch.mean(constraint_voilations)

def adjust_learning_rate(optimizer, epoch, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 


def test_result(round):

    t_data_test_list=[]
    y_data_test_list=[]
    sigma_data_test_list=[]

    test_file_name_list=filename_list_whole[round:round+1]
    task_test_num=len(test_file_name_list)

    #center_list_train=center_list_whole[0:round,:]
    center_list_test=center_list_whole[round:round+1,:]

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

    for num_task in range(len(t_data_test_list)):
        model = torch.load('./pkl/model_meta_'+str(round)+'.pkl') 
        model = model.to(device)

        data_loader_train = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_test_list[num_task]).float().requires_grad_(),torch.tensor(y_data_test_list[num_task]).float(),torch.tensor(sigma_data_test_list[num_task]).float()),shuffle = True, batch_size = batch_size_K)
        data_loader_test = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_test_list[num_task]).float().requires_grad_(),torch.tensor(y_data_test_list[num_task]).float(),torch.tensor(sigma_data_test_list[num_task]).float()),shuffle = False, batch_size = 400)
        (step_train, data_train_now) = list(enumerate(data_loader_train))[0]
        (step_test, data_test_now) = list(enumerate(data_loader_test))[0]

        for epoch in range(n_epochs):

            (features, labels, sigmas)=data_train_now
            features = features.to(device)
            labels = labels.to(device)
            sigmas=sigmas.to(device)
            outputs = model(features, model.params)
            loss_train = my_mse_loss(outputs, labels, sigmas)

            (features_constraint, labels_constraint, sigmas_constraint)=data_test_now
            features_constraint = features_constraint.to(device)
            outputs1 = model(features_constraint, model.params)
            loss_train+=constraint_voilations(outputs1,center=center_list_test[num_task])

            grads = torch.autograd.grad(loss_train, model.params, create_graph=False, retain_graph=False)
            theta_prime = [(model.params[i] - meta_lambda*grads[i]) for i in range(len(model.params))]
            model.params= theta_prime

            (features, labels, sigmas)=data_test_now
            features = features.to(device)
            labels = labels.to(device)
            sigmas=sigmas.to(device)
            outputs = model(features, model.params)
            loss_test = my_mse_loss(outputs, labels,sigmas)
            constraint_test=constraint_voilations(outputs,center=center_list_test[num_task])

        print(f"round = {round}")
        print(f'epoch = {epoch+1}, step = {step_test+1}, train loss = {loss_train.item():.6f}, test mse loss = {loss_test.item():.6f}, test constraint loss = {constraint_test.item()  / 1:.6f}')

    return loss_test.item(),constraint_test.item()

if __name__ == "__main__":
    round=99
    with open("result.csv", "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(round):
            loss_test,constraint_test=test_result(i+1)
            writer.writerow([str(loss_test),str(constraint_test)])