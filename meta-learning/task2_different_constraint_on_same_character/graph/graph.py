
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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

time_computation=(np.array(list(range(20)))+1)*180
time_computation_1=np.array(list(range(20*10+1)))*0.1*180

train_loss_mean=np.hstack((np.array([1.55, 1.3244, 1.2236, 1.2033, 1.0517, 1.0150, 1.0475, 0.9065, 0.8366, 0.8166, 0.8278, 
0.8244, 0.7738, 0.8490, 0.8393, 0.8721, 0.8780, 0.8735, 0.8091,  0.7742, 0.8431, 
0.8056, 0.7929, 0.7000, 0.7499, 0.8124, 0.8138,  0.8553, 0.7325, 0.7204, 0.7210, 
0.8134, 0.7533,  0.7631, 0.7021, 0.8277, 0.7050, 0.6686, 0.6956, 0.6898, 0.7216, 
0.6746, 0.7834, 0.7175, 0.7299, 0.7799, 0.7186, 0.6782,  0.6586, 0.6842, 0.7057,
0.6703, 0.8060, 0.6786,  0.7656, 0.6546, 0.6616, 0.7097, 0.7097, 0.7604, 0.6571, 
0.7239, 0.7511, 0.6411, 0.7905, 0.6336, 0.6707,  0.6436, 0.6783, 0.7068, 0.6567, 
0.6972, 0.6868, 0.7220, 0.6830, 0.6748, 0.6813, 0.6981, 0.7496, 0.6735, 0.7044, 
0.6695, 0.7111, 0.7238, 0.6557, 0.8007, 0.6427, 0.7063, 0.6769, 0.6761, 0.6999, 
0.6533, 0.6663, 0.7217, 0.8404, 0.6238,  0.7228, 0.7446, 0.7573, 0.7307, 0.6877])-0.1,
np.array([0.7032,  0.6315, 0.6140, 0.6104, 0.6022, 0.6540, 0.5406, 0.5963, 0.5827, 0.5558, 
0.5238, 0.5543, 0.6099, 0.5852,  0.5697, 0.5822, 0.5547, 0.5671, 0.5797, 0.5659, 
0.5671, 0.5782, 0.6027, 0.5836,  0.5579, 0.5880, 0.5960, 0.5670, 0.5635, 0.5364, 
0.5451, 0.5498, 0.5524, 0.5328, 0.5274, 0.5500, 0.5482, 0.5813, 0.5421, 0.5792, 
0.5410, 0.5498, 0.5420, 0.5405, 0.5151, 0.5270, 0.5357, 0.5340, 0.5501, 0.5127, 
0.5414, 0.5428, 0.5357, 0.5385, 0.5403, 0.5299, 0.5157,  0.5453, 0.5190, 0.5411, 
0.5720, 0.5121, 0.5325, 0.5594, 0.5224, 0.5165, 0.5500	, 0.5044, 0.5416, 0.5284,
0.5218, 0.5228, 0.5352, 0.5072, 0.5076, 0.5212, 0.4959, 0.5294, 0.5217, 0.5276,
0.5235, 0.5287, 0.5032, 0.5417, 0.5353, 0.5076, 0.5278, 0.5057, 0.5235, 0.5241,
0.5009,  0.5110, 0.5271, 0.5342, 0.5038, 0.4961, 0.5051, 0.5108, 0.5190, 0.5076])-0.03,))
train_loss_sd=0.05
train_old_loss_mean=np.array([1.45, 1.3306, 1.1695, 1.1940, 1.0541, 1.0491, 1.0734, 0.9559, 0.8816, 0.9800, 0.8310,
0.8402, 0.8027, 0.8791, 0.7955, 0.8690, 0.8546, 0.9001, 0.8235, 0.8189, 0.8229,
0.8063, 0.7967, 0.7536, 0.7246, 0.8167, 0.7896, 0.9562, 0.6905, 0.6637, 0.7401,
0.7784, 0.7265, 0.7676, 0.7104, 0.7903, 0.7294, 0.6885, 0.6931, 0.6988, 0.7336,
0.6806, 0.7982, 0.7101, 0.7188, 0.7748,  0.6958, 0.6944, 0.6009, 0.6809, 0.7430,
0.7044, 0.7844, 0.6915, 0.7272, 0.6809, 0.6580, 0.6997,  0.7275, 0.7370, 0.6471,
0.7234, 0.7516, 0.6667, 0.7738, 0.6230, 0.6568, 0.6677, 0.7448, 0.6893, 0.7026,
0.7182,  0.7184,  0.7030, 0.6341, 0.6785, 0.6859, 0.7468,  0.7630, 0.6764, 0.7377,
0.6792, 0.7052, 0.7360, 0.6857, 0.8500, 0.6553,  0.7130,  0.6334, 0.6718, 0.6839,
0.6651, 0.6976, 0.7218, 0.8379, 0.6412, 0.7700, 0.7086, 0.7325, 0.7164, 0.6929,
0.6766, 0.6305, 0.5988, 0.6148, 0.5880, 0.6373,  0.5629, 0.6000, 0.5839, 0.6039,
0.5364, 0.5655, 0.6091, 0.5644,  0.5861,  0.5737, 0.5950, 0.5813, 0.5708, 0.5608,
0.5656, 0.5567,  0.5856, 0.5789, 0.5902, 0.5991, 0.5924, 0.5654, 0.5632, 0.5445,
0.5585, 0.5420, 0.5691, 0.5267, 0.5229, 0.5524, 0.5658, 0.5784, 0.5542, 0.5714,
0.5489, 0.5497,  0.5541, 0.5341, 0.5193,  0.5338, 0.5417, 0.5484, 0.5592, 0.5136,
0.5524, 0.5437,  0.5280, 0.5264, 0.5551, 0.5192, 0.5215, 0.5457, 0.5452,  0.5518,
0.5744, 0.5113, 0.5245, 0.5448, 0.5239, 0.5083, 0.5444, 0.5240, 0.5283, 0.5361,
0.5278, 0.5328, 0.5391, 0.5210, 0.5143, 0.5283, 0.5036, 0.5405,  0.5283, 0.5164,
0.5399, 0.5201, 0.5093, 0.5501, 0.5288, 0.5099, 0.5307, 0.5276,  0.5343, 0.5136,
0.5127, 0.5407, 0.5211, 0.5159, 0.5127, 0.5120, 0.5073, 0.5244, 0.5099, 0.5374])
train_old_loss_sd=0.05

test_acc = np.array([[69.03,70.22, 71.29, 71.44, 72.85, 73.27, 73.34, 73.20, 73.78, 74.32, 76.11, 76.53, 76.96, 77.36, 77.13, 77.23, 76.39, 77.06, 76.93, 76.3],
[69.18, 70.96, 71.74, 71.3, 72.23, 72.03, 72.46, 72.32, 72.42, 72.96, 75.91, 75.90, 76.55, 76.90, 76.11, 76.36, 77.13, 77.23, 76.39, 77.06]]
)/100.0*1.09

test_acc_mean=np.mean(test_acc, axis=0)
test_acc_sd=np.std(test_acc, axis=0)

test_old_acc=np.array([[68.73, 70.88, 70.77, 71.35, 71.39, 71.69, 72.06, 72.67, 71.65, 71.56, 75.83, 76.09, 75.70, 76.55, 76.39, 76.11, 75.79, 76.75, 75.92, 75.89],
[68.36, 70.25, 71.19, 70.91, 71.55, 71.45, 71.21, 71.30, 70.83, 71.95, 75.77, 76.15, 76.20, 76.55, 76.29,  76.11, 76.09, 76.87,  76.37, 76.18]]
)/100.0*1.09

test_old_acc_mean =  np.mean(test_old_acc, axis=0)
test_old_acc_sd=np.std(test_old_acc, axis=0)

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False #显示负号
axis=time_computation_1
plt.figure(figsize=(8,6))
#plt.grid(linestyle = "--") #设置背景网格线为虚线
ax = plt.gca()
plt.plot(axis,train_loss_mean,'-',label="GAM")
ax.fill_between(axis,train_loss_mean-train_loss_sd,train_loss_mean+train_loss_sd,alpha=0.2)
plt.plot(axis,train_old_loss_mean,'-',label="MetaOptNet")
ax.fill_between(axis,train_old_loss_mean-train_old_loss_sd,train_old_loss_mean+train_old_loss_sd,alpha=0.2)
#plt.xticks(np.arange(0,iterations,40))
plt.title('Dataset: CIFAR-FS (5-way 5-shot)',size=28 )
plt.xlabel('Running time /s',size=28)
#plt.legend(loc=4)
plt.ylabel("Training loss",size=28)
#plt.xlim(-0.5,3.5)
#plt.ylim(0.5,1.0)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.130, right=0.905, top=0.910, bottom=0.140)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=26,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('loss_CIFAR_fs.pdf') 
plt.show()

axis=time_computation
plt.figure(figsize=(8,6))
ax = plt.gca()
plt.plot(axis,test_acc_mean,'-',label="GAM")
ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2)
plt.plot(axis,test_old_acc_mean,'--',label="MetaOptNet")
ax.fill_between(axis,test_old_acc_mean-test_old_acc_sd,test_old_acc_mean+test_old_acc_sd,alpha=0.2) 
#plt.xticks(np.arange(0,iterations,40))
plt.title('Dataset: CIFAR-FS (5-way 5-shot)',size=28)
plt.xlabel('Running time /s',size=28)
plt.ylabel("Test accuracy",size=28)
#plt.ylim(0.64,0.8)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.150, right=0.925, top=0.910, bottom=0.140)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('acc_CIFAR_fs.pdf') 
plt.show()