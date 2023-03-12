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
import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import hypergrad as hg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename_list=["../../ref_traj/"+'A'+"_reftraj.mat" ]*100
test_file_name_list=["../../ref_traj/"+'A'+"_reftraj.mat" ]*3

t_data_list=[]
y_data_list=[]
sigma_data_list=[]

t_data_test_list=[]
y_data_test_list=[]
sigma_data_test_list=[]

whole_task_num=len(filename_list) 
task_test_num=len(test_file_name_list)

center_list_train=np.random.normal(0, 1, [whole_task_num,2])
center_list_test=np.random.normal(0, 1, [task_test_num,2])
print(center_list_test)

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

batch_size_K = 400
batch_size_outer = 400
meta_lambda=1.0
n_epochs = 100
n_inner_level_epochs=100

redius=2.0
less=False
weight=500.0
#center=[0.0,0.0]
softplus_para=200.0
    
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

def bias_reg(params,meta_parameter, lambada=meta_lambda):
    theta_prime = [(params[i] - meta_parameter[i]) for i in range(len(params))]
    bias_reg_loss=0.0
    for i in range(len(params)):
        bias_reg_loss+=torch.norm(theta_prime[i])*torch.norm(theta_prime[i])
    return bias_reg_loss

def adjust_learning_rate(optimizer, epoch, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 


def inner_loop(hparams, params, optim, n_steps=50, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

    return params_history

model = Model()
model = model.to(device)

learning_rate=0.001
optimizer0 = torch.optim.Adam(model.params,lr=learning_rate,weight_decay=0.0000)

#learning_rate=0.00006
#optimizer0 = torch.optim.SGD(model.params,lr=learning_rate,weight_decay=0.0000)

for epoch in range(n_epochs):

    task_num=5
    #if epoch<n_epochs-50:
    #    task_num=5
    #elif epoch>=n_epochs-50:
    #    task_num=10 #len(x_train_list)

    number_list=random.sample(range(whole_task_num),task_num)
    t_data_list_thisepoch=[]
    y_data_list_thisepoch=[]
    sigma_data_list_thisepoch=[]
    center_list_train_thisepoch=[]
    for i in number_list:
        #print(filename_list[i])
        t_data_list_thisepoch.append(t_data_list[i])
        y_data_list_thisepoch.append(y_data_list[i])
        sigma_data_list_thisepoch.append(sigma_data_list[i])
        center_list_train_thisepoch.append(center_list_train[i])
    t_data_list_thisepoch=t_data_list_thisepoch+t_data_test_list
    y_data_list_thisepoch=y_data_list_thisepoch+y_data_test_list
    sigma_data_list_thisepoch=sigma_data_list_thisepoch+sigma_data_test_list
    center_list_train_thisepoch.extend(center_list_test)

    data_loader_train_list=[]
    data_loader_train_list_outer=[]
    data_loader_train_list_constraint=[]
    for i in range(len(t_data_list_thisepoch)):
        data_loader_train = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_list_thisepoch[i]).float().requires_grad_(),torch.tensor(y_data_list_thisepoch[i]).float(),torch.tensor(sigma_data_list_thisepoch[i]).float()),shuffle = True, batch_size = batch_size_K)
        data_loader_train_list.append(data_loader_train)
        data_loader_train1 = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_list_thisepoch[i]).float().requires_grad_(),torch.tensor(y_data_list_thisepoch[i]).float(),torch.tensor(sigma_data_list_thisepoch[i]).float()),shuffle = True, batch_size = batch_size_outer)
        data_loader_train_list_outer.append(data_loader_train1)
        data_loader_train2 = torch.utils.data.DataLoader(TensorDataset(torch.tensor(t_data_list_thisepoch[i]).float().requires_grad_(),torch.tensor(y_data_list_thisepoch[i]).float()),shuffle = False, batch_size = 400)
        data_loader_train_list_constraint.append(data_loader_train2)

    data_train=zip(*data_loader_train_list,*data_loader_train_list_outer)
    data_train_constraint=zip(*data_loader_train_list_constraint)
    data_train_constraint_now=list(data_train_constraint)[0]

    model.train()
    loss_train_sum = 0.0
    loss_no_grad_sum= 0.0
    loss_test_sum= 0.0
    loss_no_grad_sum_test=0
    loss_constraint_sum_test=0
    optimizer=optimizer0

    print(f'Epoch {epoch + 1}/{n_epochs}'.center(40,'-'))

    for step_train, data_train_now in enumerate(data_train):
        theta_prime_list=[]
        loss_meta_train_tensor=[]
        loss_meta_test_tensor=[]
        loss_no_grad=[]
        loss_no_grad_test=[]
        loss_constraint_meta_test_tensor=[]

        data_train_now_same=[[data_xy.clone().requires_grad_() for data_xy in  data_loader_train_new] for data_loader_train_new in data_train_now]

        optimizer.zero_grad()

        for number, data_loader_train_now in enumerate(data_train_now):
            
            if number < task_num + task_test_num:
                task_now=number
                
                (features, labels, sigmas)=data_loader_train_now
                features = features.to(device)
                labels = labels.to(device)
                sigmas=sigmas.to(device)
                outputs = model(features, model.params)

                (features_constraint, labels_constraint)=data_train_constraint_now[task_now]
                features_constraint = features_constraint.to(device)
                outputs1 = model(features_constraint, model.params)

                loss_train = my_mse_loss(outputs, labels, sigmas)
                if number < task_num:
                    loss_train+=constraint_voilations( outputs1, center=center_list_train_thisepoch[task_now] )
                else:
                    loss_train+=constraint_voilations( outputs1, center=center_list_test[task_now-task_num] )

                if number<task_num:
                    loss_no_grad.append(loss_train.item())
                else:
                    loss_no_grad_test.append(loss_train.item())

                def loss_train_call(params, hparams):
                    return my_mse_loss(model(features, params), labels, sigmas)+bias_reg(params,hparams)+ constraint_voilations( model(features_constraint, params), center=center_list_train_thisepoch[task_now] )
                
                #(features_test, label_test,sigma_test)=data_train_now_same[task_now + task_num + task_test_num]
                (features_test, label_test,sigma_test)=data_loader_train_now

                features_test = features_test.to(device)
                label_test=label_test.to(device)
                sigma_test=sigma_test.to(device)
                
                def loss_val_call(params, hparams):
                    return my_mse_loss(model(features_test, params), label_test, sigma_test)
                
                inner_opt_class = hg.GradientDescent
                inner_opt_kwargs = {'step_size': 0.00006}
                inner_opt=inner_opt_class(loss_train_call, **inner_opt_kwargs)
                inner_epoch=100

                if number<task_num:
                    theta_tem = [p.detach().clone().requires_grad_(True) for p in model.params] 
                    theta_prime = inner_loop(model.params, theta_tem, inner_opt, inner_epoch)[-1]
                    theta_prime_list.append(theta_prime)

                    cg_fp_map = hg.GradientDescent(loss_f=loss_train_call, step_size=1.)  
                    hg.CG(theta_prime, list(model.params), K=5, fp_map=cg_fp_map, outer_loss=loss_val_call) 
                else:
                    theta_tem = [p.detach().clone().requires_grad_(True) for p in model.params] 
                    theta_prime = inner_loop(model.params, theta_tem, inner_opt, inner_epoch)[-1]
                    theta_prime_list.append(theta_prime)

                
            elif number>=task_num+task_test_num and number<2*task_num+task_test_num:
                task_now=number-(task_num+task_test_num)
                (features1, labels1,sigmas1)=data_loader_train_now
                features1 = features1.to(device)
                labels1 = labels1.to(device)
                sigmas1= sigmas1.to(device)
                outputs1=model(features1, theta_prime_list[task_now])

                (features_constraint, labels_constraint)=data_train_constraint_now[task_now]
                features_constraint = features_constraint.to(device)
                outputs2 = model(features_constraint, theta_prime_list[task_now])
                
                current_loss=my_mse_loss(outputs1, labels1,sigmas1)+constraint_voilations(outputs2, center=center_list_train_thisepoch[task_now])
                loss_meta_train_tensor.append(current_loss)
            
            elif number>=2*task_num+task_test_num:
                task_now=number-(task_num+task_test_num)
                (features1, labels1,sigmas1)=data_loader_train_now
                features1 = features1.to(device)
                labels1 = labels1.to(device)
                sigmas1= sigmas1.to(device)
                outputs1=model(features1, theta_prime_list[task_now])

                (features_constraint, labels_constraint)=data_train_constraint_now[task_now]
                features_constraint = features_constraint.to(device)
                outputs2 = model(features_constraint, theta_prime_list[task_now])

                current_loss=my_mse_loss(outputs1, labels1,sigmas1)+constraint_voilations(outputs2, center=center_list_train_thisepoch[task_now] )
                loss_meta_test_tensor.append(current_loss)
                loss_constraint_meta_test_tensor.append(constraint_voilations(outputs2, center=center_list_train_thisepoch[task_now] ))
                
        loss_meta_train=sum(loss_meta_train_tensor)/float(task_num)
        #loss_meta_train.backward(retain_graph=  False)
        optimizer.step()

        loss_meta_test=sum(loss_meta_test_tensor)/float(task_test_num)
        loss_constraint_meta_test=sum(loss_constraint_meta_test_tensor)/float(task_test_num)

        loss_train_sum += loss_meta_train.item()
        loss_test_sum += loss_meta_test.item()
        loss_constraint_sum_test +=loss_constraint_meta_test.item()
        loss_no_grad_sum += sum(loss_no_grad)/(task_num)
        loss_no_grad_sum_test += sum(loss_no_grad_test)/(task_test_num)


        if (step_train+1) % 1 == 0:
            print(f'step = {step_train+1}, loss = {loss_train_sum / 1:.6f}')
            print(f'step = {step_train+1}, test_loss = {loss_test_sum / 1:.6f}')
            print('loss_no_grad:'+str(loss_no_grad_sum/ 1))
            print('loss_no_grad_test:'+str(loss_no_grad_sum_test/ 1))
            print('loss_constraint_test:'+str(loss_constraint_sum_test/ 1))       
            loss_train_sum=0
            loss_no_grad_sum=0
            loss_test_sum=0
            loss_no_grad_sum_test=0
            loss_constraint_sum_test=0

    print("meta lambda:   "+str(meta_lambda))

################ save model ################
torch.save(model, 'model_meta.pkl') 

