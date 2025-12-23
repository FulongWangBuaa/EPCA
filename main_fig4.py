# %%
'''
EPCA方法的总体性能
'''

# %%
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib auto
import pickle
import pandas as pd
import os.path as op

# %%
from my_code.datasets.mybenchmarkdataset import MyBenchmarkDataset
from my_code.utils.benchmarkpreprocess import preprocess as benchmarkpreprocess
from my_code.utils.benchmarkpreprocess import filterbank as benchmarkfilterbank

from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.utils.megpreprocess import preprocess as megpreprocess
from my_code.utils.megpreprocess import filterbank as megfilterbank


data_path = r"D:\科研\代码\工作\5、EPCA\EPCA\datasets\Benchmark"
Benchmark = MyBenchmarkDataset(path=data_path)
Benchmark.regist_preprocess(benchmarkpreprocess)
Benchmark.regist_filterbank(benchmarkfilterbank)

data_path = r"D:\科研\代码\工作\5、EPCA\EPCA\datasets\OPMMEG"
MEG = MyMEGDataset(path=data_path)
MEG.regist_preprocess(megpreprocess)
MEG.regist_filterbank(megfilterbank)

datasets = {'Benchmark':Benchmark, 'MEG': MEG}


# %%
ch_nums = [2,3,4,5,6,7,8,9]
num_trains = [1,2,3,4,5]
tw_seqs = [0.5,1]

import pickle
with open('./result/fig_4/acc_all_benchmark.pkl','rb') as f:
    acc_all_benchmark = pickle.load(f)

with open('./result/fig_4/itr_all_benchmark.pkl','rb') as f:
    itr_all_benchmark = pickle.load(f)

with open('./result/fig_4/acc_all_meg.pkl','rb') as f:
    acc_all_meg = pickle.load(f)

with open('./result/fig_4/itr_all_meg.pkl','rb') as f:
    itr_all_meg = pickle.load(f)

# %%
acc_all_benchmark = acc_all_benchmark*100
acc_all_meg = acc_all_meg*100



# %%
'''
Benchmark数据集结果
'''
# %%
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,2,figsize=(5, 4),constrained_layout=True)
# plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)
axes = axes.flatten()

colors = ['#18748E','w','#FC5454']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

num_trains = [1,2,3,4,5]
ch_nums = [2,3,4,5,6,7,8,9]

titles = ['EPCA, 0.5 s','EPCA, 1 s', 'eEPCA, 0.5 s', 'eEPCA, 1 s']
acc_all_benchmark_mean = np.mean(acc_all_benchmark,axis=2)
for k,ax in enumerate(axes.flatten()):
    im = ax.imshow(acc_all_benchmark_mean[:,k//2,k%2,:],cmap=cmap,vmin=0,vmax=100)
    for i in range(len(ch_nums)):
        for j in range(len(num_trains)):
            text = ax.text(j, i, "{:.2f}%".format(acc_all_benchmark_mean[:,k//2,k%2,:][i,j]),
                        ha="center", va="center", color="k",fontsize=7)
    ax.set_title(titles[k],fontsize=9,fontweight='bold')
    ax.set_xticks(np.arange(len(num_trains)))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(ch_nums)))
    ax.set_yticklabels([str(i) for i in ch_nums])
    ax.set_aspect(0.55)

    ax.tick_params(axis='both', which='both',  direction='out',length=2.5,width=1,color='k',labelsize=9,
                   bottom=True,top=False,left=True,right=False )

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.03)
cbar.set_label("Accuracy (%)", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0)
plt.show()

# fig.savefig('./fig/fig_4/fig_4_benchmark.png',dpi=600)
# fig.savefig('./fig/fig_4/fig_4_benchmark.svg')
# fig.savefig('./fig/fig_4/fig_4_benchmark.pdf')

'''
Benchmark数据集统计检验结果
'''
# %%
from scipy.stats import ttest_rel
t1 = np.zeros((len(num_trains),len(ch_nums)))
p1 = np.zeros((len(num_trains),len(ch_nums)))
for train_idx in range(len(num_trains)):
    for ch_idx in range(len(ch_nums)):
        acc = acc_all_benchmark[ch_idx,:,:,0,train_idx]
        t1[train_idx,ch_idx],p1[train_idx,ch_idx] = ttest_rel(acc[1],acc[0])

t2 = np.zeros((len(num_trains),len(ch_nums)))
p2 = np.zeros((len(num_trains),len(ch_nums)))
for train_idx in range(len(num_trains)):
    for ch_idx in range(len(ch_nums)):
        acc = acc_all_benchmark[ch_idx,:,:,1,train_idx]
        t2[train_idx,ch_idx],p2[train_idx,ch_idx] = ttest_rel(acc[1],acc[0])

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(1,2,figsize=(5, 2),constrained_layout=True)
# plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)
axes = axes.flatten()


colors = ['#ffb9df', '#ff1493']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
ts = [t1,t2]
ps = [p1,p2]
titles = ['eEPCA vs. EPCA, 0.5 s','eEPCA vs. EPCA, 1 s']
for idx in range(2):
    ax = axes.flatten()[idx]
    im = ax.imshow(ts[idx].T,cmap=cmap,origin='lower',vmin=0,vmax=15)
    for i in range(len(num_trains)):
        for j in range(len(ch_nums)):
            if ps[idx][i,j] > 0.05:
                rect1 = Rectangle((i-0.5, j-0.5), 1, 1, linewidth=2, edgecolor='w', facecolor='w', alpha=1)
                ax.add_patch(rect1)
            text = ax.text(i,j, "{:.2f}".format(ts[idx][i,j]),
                        ha="center", va="center", color="k",fontsize=8)
    ax.set_ylim(7.5,-0.5)
    ax.set_title(titles[idx],fontsize=9,fontweight='bold')
    ax.set_xticks(np.arange(len(num_trains)))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(ch_nums)))
    ax.set_yticklabels([str(i) for i in ch_nums])
    ax.set_aspect(0.55)

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.03,aspect=11)
cbar.set_label("t-value", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0)
plt.show()

# fig.savefig('./fig/fig_4/fig_4_benchmark_sta.png',dpi=600)
# fig.savefig('./fig/fig_4/fig_4_benchmark_sta.svg')
# fig.savefig('./fig/fig_4/fig_4_benchmark_sta.pdf')





# %%
'''
MEG数据集结果
'''
# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,2,figsize=(5, 4),constrained_layout=True)
# plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)
axes = axes.flatten()

colors = ['#18748E','w','#FC5454']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

num_trains = [1,2,3,4,5]
ch_nums = [2,3,4,5,6,7,8,9]

titles = ['EPCA, 0.5 s','EPCA, 1 s', 'eEPCA, 0.5 s', 'eEPCA, 1 s']
acc_all_meg_mean = np.mean(acc_all_meg,axis=2)
for k,ax in enumerate(axes.flatten()):
    im = ax.imshow(acc_all_meg_mean[:,k//2,k%2,:],cmap=cmap,vmin=0,vmax=100)
    for i in range(len(ch_nums)):
        for j in range(len(num_trains)):
            text = ax.text(j, i, "{:.2f}%".format(acc_all_meg_mean[:,k//2,k%2,:][i,j]),
                        ha="center", va="center", color="k",fontsize=7)
    ax.set_title(titles[k],fontsize=9,fontweight='bold')
    ax.set_xticks(np.arange(len(num_trains)))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(ch_nums)))
    ax.set_yticklabels([str(i) for i in ch_nums])
    ax.set_aspect(0.55)

    ax.tick_params(axis='both', which='both',  direction='out',length=2.5,width=1,color='k',labelsize=9,
                   bottom=True,top=False,left=True,right=False )

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.03)
cbar.set_label("Accuracy (%)", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0)
plt.show()

# fig.savefig('./fig/fig_4/fig_4_meg.png',dpi=600)
# fig.savefig('./fig/fig_4/fig_4_meg.svg')
# fig.savefig('./fig/fig_4/fig_4_meg.pdf')


# %%
'''
MEG数据集结果统计检验结果
'''
# %%
from scipy.stats import ttest_rel
t3 = np.zeros((len(num_trains),len(ch_nums)))
p3 = np.zeros((len(num_trains),len(ch_nums)))
for train_idx in range(len(num_trains)):
    for ch_idx in range(len(ch_nums)):
        acc = acc_all_meg[ch_idx,:,:,0,train_idx]
        t3[train_idx,ch_idx],p3[train_idx,ch_idx] = ttest_rel(acc[1],acc[0])

t4 = np.zeros((len(num_trains),len(ch_nums)))
p4 = np.zeros((len(num_trains),len(ch_nums)))
for train_idx in range(len(num_trains)):
    for ch_idx in range(len(ch_nums)):
        acc = acc_all_meg[ch_idx,:,:,1,train_idx]
        t4[train_idx,ch_idx],p4[train_idx,ch_idx] = ttest_rel(acc[1],acc[0])


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(1,2,figsize=(5, 2),constrained_layout=True)
# plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)
axes = axes.flatten()

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
colors = ['#ffb9df', '#ff1493']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
ts = [t3,t4]
ps = [p3,p4]
titles = ['eEPCA vs. EPCA, 0.5 s','eEPCA vs. EPCA, 1 s']
for idx in range(2):
    ax = axes.flatten()[idx]
    im = ax.imshow(ts[idx].T,cmap=cmap,origin='lower',vmin=0,vmax=15)
    for i in range(len(num_trains)):
        for j in range(len(ch_nums)):
            if ps[idx][i,j] > 0.05:
                rect1 = Rectangle((i-0.5, j-0.5), 1, 1, linewidth=2, edgecolor='w', facecolor='w', alpha=1)
                ax.add_patch(rect1)
            text = ax.text(i,j, "{:.2f}".format(ts[idx][i,j]),
                        ha="center", va="center", color="k",fontsize=8)
    ax.set_ylim(7.5,-0.5)
    ax.set_title(titles[idx],fontsize=9,fontweight='bold')
    ax.set_xticks(np.arange(len(num_trains)))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(ch_nums)))
    ax.set_yticklabels([str(i) for i in ch_nums])
    ax.set_aspect(0.55)

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.03,aspect=12)
cbar.set_label("t-value", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0)
plt.show()

# fig.savefig('./fig/fig_4/fig_4_meg_sta.png',dpi=600)
# fig.savefig('./fig/fig_4/fig_4_meg_sta.svg')
# fig.savefig('./fig/fig_4/fig_4_meg_sta.pdf')

# %%
