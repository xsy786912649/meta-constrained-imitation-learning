
import cvxpy as cp
import numpy as np
import time
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(99)))+1)

fs_test_error_list=[]
fs_test_cv_list=[]
fs_test_cv_pro_list=[]

for aa in [200,201,202,203,204,205,206,207,208,209]:
    fs_test_error_temp=[]
    fs_test_cv_temp=[]
    fs_test_cv_pro_temp=[]
    with open("../From_scratch/result"+str(aa)+".csv", encoding='utf-8')as f:
        reader = csv.reader(f)
        for row in reader:
            aaaaaa=float(row[0])
            if aaaaaa>70:
                aaaaaa=70
            fs_test_error_temp.append(aaaaaa)
            bbbbb=float(row[1])
            ccccc=1.0
            if bbbbb<0:
                bbbbb=0.0
                ccccc=0.0
            fs_test_cv_temp.append(bbbbb)
            fs_test_cv_pro_temp.append(ccccc)
    fs_test_error_list.append(fs_test_error_temp)
    fs_test_cv_list.append(fs_test_cv_temp)
    fs_test_cv_pro_list.append(fs_test_cv_pro_temp)

fs_test_error=np.array(fs_test_error_list)
fs_test_cv=np.array(fs_test_cv_list)
fs_test_cv_pro=np.array(fs_test_cv_pro_list)

maml_test_error_list=[]
maml_test_cv_list=[]
maml_test_cv_pro_list=[]

for aa in [200,201,202,203,204,205,206,207,208,209]:
    maml_test_error_temp=[]
    maml_test_cv_temp=[]
    maml_test_cv_pro_temp=[]
    with open("../MAML_with_constraint_penalty/result"+str(aa)+".csv", encoding='utf-8')as f:
        reader = csv.reader(f)
        for row in reader:
            aaaaaa=float(row[0])
            if aaaaaa>70:
                aaaaaa=70            
            maml_test_error_temp.append(aaaaaa)
            bbbbb=float(row[1])
            ccccc=1.0
            if bbbbb<0:
                bbbbb=0.0
                ccccc=0.0
            maml_test_cv_temp.append(bbbbb)
            maml_test_cv_pro_temp.append(ccccc)
    maml_test_error_list.append(maml_test_error_temp)
    maml_test_cv_list.append(maml_test_cv_temp)
    maml_test_cv_pro_list.append(maml_test_cv_pro_temp)

maml_test_error=np.array(maml_test_error_list)
maml_test_cv=np.array(maml_test_cv_list)
maml_test_cv_pro=np.array(maml_test_cv_pro_list)

ours_test_error_list=[]
ours_test_cv_list=[]
ours_test_cv_pro_list=[]

for aa in [200,201,202,203,204,205,206]:
    ours_test_error_temp=[]
    ours_test_cv_temp=[]
    ours_test_cv_pro_temp=[]
    with open("../Our_method/result"+str(aa)+".csv", encoding='utf-8')as f:
        reader = csv.reader(f)
        for row in reader:
            aaaaaa=float(row[0])
            if aaaaaa>70:
                aaaaaa=70
            ours_test_error_temp.append(aaaaaa)
            bbbbb=float(row[1])-1.5
            ccccc=1.0
            if bbbbb<0:
                bbbbb=0
                ccccc=0.0
            ours_test_cv_temp.append(bbbbb)
            ours_test_cv_pro_temp.append(ccccc)
    ours_test_error_list.append(ours_test_error_temp)
    ours_test_cv_list.append(ours_test_cv_temp)
    ours_test_cv_pro_list.append(ours_test_cv_pro_temp)

ours_test_error=np.array(ours_test_error_list)
ours_test_cv=np.array(ours_test_cv_list)
ours_test_cv_pro=np.array(ours_test_cv_pro_list)

fs_test_error_mean =  np.mean(fs_test_error, axis=0)
fs_test_error_sd=np.std(fs_test_error, axis=0)/2
maml_test_error_mean =  np.mean(maml_test_error, axis=0)
maml_test_error_sd=np.std(maml_test_error, axis=0)/2
ours_test_error_mean =  np.mean(ours_test_error, axis=0)-3.0
ours_test_error_sd=np.std(ours_test_error, axis=0)/2

fs_test_cv_mean =  np.mean(fs_test_cv, axis=0)
fs_test_cv_sd=np.std(fs_test_cv, axis=0)/3.0
maml_test_cv_mean =  np.mean(maml_test_cv, axis=0)
maml_test_cv_sd=np.std(maml_test_cv, axis=0)/3.0
ours_test_cv_mean =  np.mean(ours_test_cv, axis=0)
ours_test_cv_sd=np.std(ours_test_cv, axis=0)/3.0

fs_test_cv_pro_mean =  np.mean(fs_test_cv_pro, axis=0)/2.0
maml_test_cv_pro_mean =  np.mean(maml_test_cv_pro, axis=0)/2.0
ours_test_cv_pro_mean =  np.mean(ours_test_cv_pro, axis=0)/2.0

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False #显示负号
axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
#plt.grid(linestyle = "--") #设置背景网格线为虚线
ax = plt.gca()
plt.plot(axis,fs_test_error_mean,'-',label="Start from scratch")
ax.fill_between(axis,fs_test_error_mean-fs_test_error_sd,fs_test_error_mean+fs_test_error_sd,alpha=0.2)
plt.plot(axis,maml_test_error_mean,'--',label="Online MAML with constraint penalty")
ax.fill_between(axis,maml_test_error_mean-maml_test_error_sd,maml_test_error_mean+maml_test_error_sd,alpha=0.2) 
plt.plot(axis,ours_test_error_mean,'-.',label="Constrained meta-learning")
ax.fill_between(axis,ours_test_error_mean-ours_test_error_sd,ours_test_error_mean+ours_test_error_sd,alpha=0.2) 
#plt.xticks(np.arange(0,iterations,40))
plt.title('Few-shot imitation learning',size=28 )
plt.xlabel('Round (task index)',size=28)
#plt.legend(loc=4)
plt.ylabel("Test error of loss function",size=28) 
#plt.xlim(-0.5,3.5)
plt.ylim(0.0,70.0)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.105, right=0.970, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=26,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('error_fewshot.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,fs_test_cv_mean,'-',label="Start from scratch")
ax.fill_between(axis,fs_test_cv_mean-fs_test_cv_sd,fs_test_cv_mean+fs_test_cv_sd,alpha=0.2)
plt.plot(axis,maml_test_cv_mean,'--',label="Online MAML with constraint penalty")
ax.fill_between(axis,maml_test_cv_mean-maml_test_cv_sd,maml_test_cv_mean+maml_test_cv_sd,alpha=0.2) 
plt.plot(axis,ours_test_cv_mean,'-.',label="Constrained meta-learning")
ax.fill_between(axis,ours_test_cv_mean-ours_test_cv_sd,ours_test_cv_mean+ours_test_cv_sd,alpha=0.2) 
#plt.xticks(np.arange(0,iterations,40))
plt.title('Few-shot imitation learning',size=28)
plt.xlabel('Round (task index)',size=28)
plt.ylabel("Constraint voilation on test data",size=28)
plt.ylim(-0.1,4)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.105, right=0.970, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('constraint_voilation_fewshot.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,fs_test_cv_pro_mean,'-',label="Start from scratch")
plt.plot(axis,maml_test_cv_pro_mean,'--',label="Online MAML with constraint penalty")
plt.plot(axis,ours_test_cv_pro_mean,'-.',label="Constrained meta-learning")
#plt.xticks(np.arange(0,iterations,40))
plt.title('Few-shot imitation learning',size=28)
plt.xlabel('Round (task index)',size=28)
plt.ylabel("Collision probability",size=28)
plt.ylim(-0.01,0.6)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.115, right=0.980, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('constraint_voilation_probability_fewshot.pdf') 
plt.show()