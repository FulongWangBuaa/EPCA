# %%
'''
通道和训练试验与其他方法对比
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
with open('./result/1-5trial_0.5_1s_2-9ch/acc_all_benchmark.pkl','rb') as f:
    acc_all_benchmark = pickle.load(f)

with open('./result/1-5trial_0.5_1s_2-9ch/itr_all_benchmark.pkl','rb') as f:
    itr_all_benchmark = pickle.load(f)

with open('./result/1-5trial_0.5_1s_2-9ch/acc_all_meg.pkl','rb') as f:
    acc_all_meg = pickle.load(f)

with open('./result/1-5trial_0.5_1s_2-9ch/itr_all_meg.pkl','rb') as f:
    itr_all_meg = pickle.load(f)

# ch_num * model_num * sub_num * tw_num * train_num
acc_all_benchmark = acc_all_benchmark*100
acc_all_meg = acc_all_meg*100


# %%
ch_num = 9
tw_seq = 1
model_pick = [2,3,1,6]

ch_num_idx = ch_nums.index(ch_num)
tw_seq_idx = tw_seqs.index(tw_seq)

# model_num * sub_num * train_num
acc_all_benchmark_ch = acc_all_benchmark[ch_num_idx,model_pick,:,tw_seq_idx,:]
acc_all_meg_ch = acc_all_meg[ch_num_idx,model_pick,:,tw_seq_idx,:]

# model_num * train_num
acc_all_benchmark_ch_mean = np.mean(acc_all_benchmark_ch,axis=1)
acc_all_benchmark_ch_se = np.std(acc_all_benchmark_ch,axis=1)/np.sqrt(acc_all_benchmark_ch.shape[1])

acc_all_meg_ch_mean = np.mean(acc_all_meg_ch,axis=1)
acc_all_meg_ch_se = np.std(acc_all_meg_ch,axis=1)/np.sqrt(acc_all_meg_ch.shape[1])


# %%
train_num = 1
tw_seq = 1
model_pick = [2,3,1,6]
ch_pick = [3,6,9]

# ch_num_idx = ch_nums.index(ch_num)
train_num_idx = num_trains.index(train_num)
tw_seq_idx = tw_seqs.index(tw_seq)
ch_pick_idx = [ch_nums.index(ch) for ch in ch_pick]

# ch_num * model_num * sub_num
acc_all_benchmark_train_1 = acc_all_benchmark[:,model_pick,:,tw_seq_idx,train_num_idx]
acc_all_meg_train_1 = acc_all_meg[:,model_pick,:,tw_seq_idx,train_num_idx]

# model_num * ch_num
acc_all_benchmark_train_mean_1 = np.mean(acc_all_benchmark_train_1,axis=2)
acc_all_benchmark_train_se_1 = np.std(acc_all_benchmark_train_1,axis=2)/np.sqrt(acc_all_benchmark_train_1.shape[2])

acc_all_meg_train_mean_1 = np.mean(acc_all_meg_train_1,axis=2)
acc_all_meg_train_se_1 = np.std(acc_all_meg_train_1,axis=2)/np.sqrt(acc_all_meg_train_1.shape[2])

acc_all_benchmark_train_mean_1 = acc_all_benchmark_train_mean_1[:,ch_pick_idx]
acc_all_benchmark_train_se_1 = acc_all_benchmark_train_se_1[:,ch_pick_idx]

acc_all_meg_train_mean_1 = acc_all_meg_train_mean_1[:,ch_pick_idx]
acc_all_meg_train_se_1 = acc_all_meg_train_se_1[:,ch_pick_idx]


# %%
train_num = 5
tw_seq = 1
model_pick = [2,3,1,6]
ch_pick = [3,6,9]

# ch_num_idx = ch_nums.index(ch_num)
train_num_idx = num_trains.index(train_num)
tw_seq_idx = tw_seqs.index(tw_seq)
ch_pick_idx = [ch_nums.index(ch) for ch in ch_pick]

# ch_num * model_num * sub_num
acc_all_benchmark_train_2 = acc_all_benchmark[:,model_pick,:,tw_seq_idx,train_num_idx]
acc_all_meg_train_2 = acc_all_meg[:,model_pick,:,tw_seq_idx,train_num_idx]

# model_num * ch_num
acc_all_benchmark_train_mean_5 = np.mean(acc_all_benchmark_train_2,axis=2)
acc_all_benchmark_train_se_5 = np.std(acc_all_benchmark_train_2,axis=2)/np.sqrt(acc_all_benchmark_train_2.shape[2])

acc_all_meg_train_mean_5 = np.mean(acc_all_meg_train_2,axis=2)
acc_all_meg_train_se_5 = np.std(acc_all_meg_train_2,axis=2)/np.sqrt(acc_all_meg_train_2.shape[2])

acc_all_benchmark_train_mean_5 = acc_all_benchmark_train_mean_5[:,ch_pick_idx]
acc_all_benchmark_train_se_5 = acc_all_benchmark_train_se_5[:,ch_pick_idx]

acc_all_meg_train_mean_5 = acc_all_meg_train_mean_5[:,ch_pick_idx]
acc_all_meg_train_se_5 = acc_all_meg_train_se_5[:,ch_pick_idx]



# %%
from scipy.stats import ttest_rel
compairs1 = [[2,3],[1,3],[0,3]]
ps1 = np.zeros((5,len(compairs1)))
ts1 = np.zeros((5,len(compairs1)))
for train_idx in range(5):
    for compair_idx,compair in enumerate(compairs1):
        ts1[train_idx,compair_idx],ps1[train_idx,compair_idx] = ttest_rel(acc_all_benchmark_ch[compair[0],:,train_idx],
                                                acc_all_benchmark_ch[compair[1],:,train_idx])
ps1[-1,-1] = 1


compairs2 = [[2,3],[1,3],[0,3]]
ps2 = np.zeros((3,len(compairs2)))
ts2 = np.zeros((3,len(compairs2)))
for ch_idx in range(3):
    for compair_idx,compair in enumerate(compairs2):
        ch = ch_pick_idx[ch_idx]
        ts2[ch_idx,compair_idx],ps2[ch_idx,compair_idx] = ttest_rel(acc_all_benchmark_train_1[compair[0],ch,:],
                                                acc_all_benchmark_train_1[compair[1],ch,:])

compairs3 = [[2,3],[1,3],[0,3]]
ps3 = np.zeros((3,len(compairs3)))
ts3 = np.zeros((3,len(compairs3)))
for ch_idx in range(3):
    for compair_idx,compair in enumerate(compairs3):
        ch = ch_pick_idx[ch_idx]
        ts3[ch_idx,compair_idx],ps3[ch_idx,compair_idx] = ttest_rel(acc_all_benchmark_train_2[compair[0],ch,:],
                                                acc_all_benchmark_train_2[compair[1],ch,:])
        
ps3[1,-1] = 1
ps3[2,-1] = 1


compairs4 = [[2,3],[1,3],[0,3]]
ps4 = np.zeros((5,len(compairs4)))
ts4 = np.zeros((5,len(compairs4)))
for train_idx in range(5):
    for compair_idx,compair in enumerate(compairs4):
        ts4[train_idx,compair_idx],ps4[train_idx,compair_idx] = ttest_rel(acc_all_meg_ch[compair[0],:,train_idx],
                                                acc_all_meg_ch[compair[1],:,train_idx])
# ps1[-1,-1] = 1

compairs5 = [[2,3],[1,3],[0,3]]
ps5 = np.zeros((3,len(compairs5)))
ts5 = np.zeros((3,len(compairs5)))
for ch_idx in range(3):
    for compair_idx,compair in enumerate(compairs5):
        ch = ch_pick_idx[ch_idx]
        ts5[ch_idx,compair_idx],ps5[ch_idx,compair_idx] = ttest_rel(acc_all_meg_train_1[compair[0],ch,:],
                                                acc_all_meg_train_1[compair[1],ch,:])

compairs6 = [[2,3],[1,3],[0,3]]
ps6 = np.zeros((3,len(compairs6)))
ts6 = np.zeros((3,len(compairs6)))
for ch_idx in range(3):
    for compair_idx,compair in enumerate(compairs6):
        ch = ch_pick_idx[ch_idx]
        ts6[ch_idx,compair_idx],ps6[ch_idx,compair_idx] = ttest_rel(acc_all_meg_train_2[compair[0],ch,:],
                                                                    acc_all_meg_train_2[compair[1],ch,:])




# %%
import matplotlib.gridspec as gridspec
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(9, 2.5))
plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)

gs = gridspec.GridSpec(1, 3, width_ratios=[1.7,1,1])
axes = [plt.subplot(gs[0]),plt.subplot(gs[1]),plt.subplot(gs[2])]

bar_width = 0.15
alpha = 0.8

colors = ['#D2B29B','#923D3A','#50A8CC','#EB6559']
error_kw = dict(linewidth=1, capsize=2)

model_num = acc_all_benchmark_ch_mean.shape[0]

ax = axes[0]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in num_trains] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_benchmark_ch_mean[model_idx],
            yerr=acc_all_benchmark_ch_se[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)

lg = axes[0].legend(['eTRCA','ms-eTRCA','eRESS','eEPCA'],loc=[0.65,0.1],fontsize=10,frameon=True,
    edgecolor='k',facecolor='white',framealpha=1,handlelength=1.5,handletextpad=0.5)

for train_idx in range(5):
    i = 100
    for compair_idx,compair in enumerate(compairs1):
        if ps1[train_idx,compair_idx]<0.05:
            i = i+2.5
            if ts1[train_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'
            ax.plot([bar_loc[compair[0]][train_idx]+bar_width/2,bar_loc[compair[1]][train_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of training trials', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([0.5,5.5])
ax.set_xticks([1,2,3,4,5])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



ax = axes[1]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in range(len(ch_pick))] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_benchmark_train_mean_1[model_idx],
            yerr=acc_all_benchmark_train_se_1[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)
    
for ch_idx in range(3):
    i = 100
    for compair_idx,compair in enumerate(compairs2):
        if ps2[ch_idx,compair_idx]<0.05:
            i = i+2.5
            if ts2[ch_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'
            ax.plot([bar_loc[compair[0]][ch_idx]+bar_width/2,bar_loc[compair[1]][ch_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of channels', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([-0.5,2.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(['3','6','9'])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[2]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in range(len(ch_pick))] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_benchmark_train_mean_5[model_idx],
            yerr=acc_all_benchmark_train_se_5[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)
    
for ch_idx in range(3):
    i = 100
    for compair_idx,compair in enumerate(compairs2):
        if ps3[ch_idx,compair_idx]<0.05:
            i = i+2.5
            if ts3[ch_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'

            ax.plot([bar_loc[compair[0]][ch_idx]+bar_width/2,bar_loc[compair[1]][ch_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of channels', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([-0.5,2.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(['3','6','9'])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


# fig.savefig('./fig/fig_8/fig_8_benchmark.png',dpi=600)
# fig.savefig('./fig/fig_8/fig_8_benchmark.pdf')
# fig.savefig('./fig/fig_8/fig_8_benchmark.svg')




# %%
import matplotlib.gridspec as gridspec
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(9, 2.5))
plt.subplots_adjust(top=0.95,bottom=0.18,left=0.08,right=0.99,wspace=0.3,hspace=0.5)

gs = gridspec.GridSpec(1, 3, width_ratios=[1.7,1,1])
axes = [plt.subplot(gs[0]),plt.subplot(gs[1]),plt.subplot(gs[2])]

bar_width = 0.15
alpha = 0.8

colors = ['#D2B29B','#923D3A','#50A8CC','#EB6559']
error_kw = dict(linewidth=1, capsize=2)

model_num = acc_all_meg_ch_mean.shape[0]

ax = axes[0]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in num_trains] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_meg_ch_mean[model_idx],
            yerr=acc_all_meg_ch_se[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)
    
lg = axes[0].legend(['eTRCA','ms-eTRCA','eRESS','eEPCA'],loc=[0.65,0.1],fontsize=10,frameon=True,
    edgecolor='k',facecolor='white',framealpha=1,handlelength=1.5,handletextpad=0.5)

for train_idx in range(5):
    i = 100
    for compair_idx,compair in enumerate(compairs4):
        if ps4[train_idx,compair_idx]<0.05:
            i = i+2.5
            if ts4[train_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'
            ax.plot([bar_loc[compair[0]][train_idx]+bar_width/2,bar_loc[compair[1]][train_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of training trials', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([0.5,5.5])
ax.set_xticks([1,2,3,4,5])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



ax = axes[1]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in range(len(ch_pick))] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_meg_train_mean_1[model_idx],
            yerr=acc_all_meg_train_se_1[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)
    
for ch_idx in range(3):
    i = 100
    for compair_idx,compair in enumerate(compairs5):
        if ps5[ch_idx,compair_idx]<0.05:
            i = i+2.5
            if ts5[ch_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'
            ax.plot([bar_loc[compair[0]][ch_idx]+bar_width/2,bar_loc[compair[1]][ch_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of channels', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([-0.5,2.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(['3','6','9'])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax = axes[2]
bar_loc = [[i+(bar_width+0.05)*(model_idx-2)+0.025 for i in range(len(ch_pick))] for model_idx in range(model_num)]
for model_idx in range(model_num):
    b1 = ax.bar(bar_loc[model_idx], acc_all_meg_train_mean_5[model_idx],
            yerr=acc_all_meg_train_se_5[model_idx],error_kw=error_kw,
            color=colors[model_idx],width=bar_width,
            align='edge',edgecolor='k',alpha=alpha)
    
for ch_idx in range(3):
    i = 100
    for compair_idx,compair in enumerate(compairs6):
        if ps6[ch_idx,compair_idx]<0.05:
            i = i+2.5
            if ts6[ch_idx,compair_idx] > 0:
                color = '#50A8CC'
            else:
                color = '#EB6559'
            ax.plot([bar_loc[compair[0]][ch_idx]+bar_width/2,bar_loc[compair[1]][ch_idx]+bar_width/2],[i,i],
                    linewidth=1,color=color)

ax.set_xlabel('Number of channels', fontsize=10, fontweight='normal')
ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='normal')
ax.set_xlim([-0.5,2.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(['3','6','9'])
ax.set_ylim([0, 120])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

# fig.savefig('./fig/fig_8/fig_8_meg.png',dpi=600)
# fig.savefig('./fig/fig_8/fig_8_meg.pdf')
# fig.savefig('./fig/fig_8/fig_8_meg.svg')

# %%
