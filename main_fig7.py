# %%
'''
与其他方法对比
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


# %%
from my_code.datasets.mybenchmarkdataset import MyBenchmarkDataset
from my_code.utils.benchmarkpreprocess import preprocess as benchmarkpreprocess
from my_code.utils.benchmarkpreprocess import filterbank as benchmarkfilterbank

from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.utils.megpreprocess import preprocess as megpreprocess
from my_code.utils.megpreprocess import filterbank as megfilterbank


dataset_key = 'MEG'
if dataset_key == 'Benchmark':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA\datasets\Benchmark"
    dataset = MyBenchmarkDataset(path=data_path)
    dataset.regist_preprocess(benchmarkpreprocess)
    dataset.regist_filterbank(benchmarkfilterbank)
elif dataset_key == 'MEG':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA\datasets\OPMMEG"
    dataset = MyMEGDataset(path=data_path)
    dataset.regist_preprocess(megpreprocess)
    dataset.regist_filterbank(megfilterbank)

# %%
from SSVEPAnalysisToolbox.evaluator import BaseEvaluator
evaluator = BaseEvaluator()
if dataset_key == 'Benchmark':
    evaluator.load(r'.\result\1-5trial_0.2-2s\evaluator_1-5trial_0.2-2s.pkl')
elif dataset_key == 'MEG':
    evaluator.load(r'.\result\1-5trial_0.2-2s\evaluator_MEG.pkl')



# %%
model_container = evaluator.model_container
trial_container = evaluator.trial_container
num_trains = [1,2,3,4,5]
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

# %%
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
n_models = len(model_container)
n_subs = len(dataset.subjects)
n_tws = len(tw_seqs)
n_trains = len(num_trains)

performance_container = evaluator.performance_container
acc_store = dict()
itr_store = dict()
for trialinfo,performance in zip(trial_container,performance_container):
    tw = trialinfo[0].tw
    idx_tw = tw_seqs.index(tw)
    idx_sub = trialinfo[0].sub_idx[0]
    train_block_idx = tuple(trialinfo[0].block_idx[0])
    idx_train = len(train_block_idx)

    t_latency = dataset.default_t_latency
    n_targs = len(trialinfo[0].trial_idx[0])

    if idx_train not in acc_store.keys():
        acc_store[idx_train] = dict()
        itr_store[idx_train] = dict()
    if train_block_idx not in acc_store[idx_train].keys():
        acc_store[idx_train][train_block_idx] = np.zeros((n_models, n_subs, n_tws))
        itr_store[idx_train][train_block_idx] = np.zeros((n_models, n_subs, n_tws))

    for idx_model,model_performance in enumerate(performance):

        Y_test = model_performance.true_label_test
        Y_pred = model_performance.pred_label_test

        acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
        itr = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = t_latency,
                      t_comp = 0,N = n_targs, acc = acc)

        acc_store[idx_train][train_block_idx][idx_model, idx_sub, idx_tw] = acc
        itr_store[idx_train][train_block_idx][idx_model, idx_sub, idx_tw] = itr

# %%
acc_mean = np.zeros((n_models, n_subs, n_tws, n_trains))
itr_mean = np.zeros((n_models, n_subs, n_tws, n_trains))
for idx_train,key1 in enumerate(acc_store.keys()):
    a = []
    i = []
    for key2 in acc_store[key1].keys():
        a.append(acc_store[key1][key2])
        i.append(itr_store[key1][key2])
    acc_mean[:,:,:,idx_train] = np.mean(a,axis=0)
    itr_mean[:,:,:,idx_train] = np.mean(i,axis=0)

# %%
# import pickle
# with open('acc_mean.pkl','wb') as f:
#     pickle.dump(acc_mean,f)

# with open('itr_mean.pkl','wb') as f:
#     pickle.dump(itr_mean,f)

# %%
num_trains = [1,2,3,4,5]
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

import pickle

with open('./result/1-5trial_0.2-2s/acc_mean_benchmark.pkl','rb') as f:
    acc_mean_benchmark = pickle.load(f)

with open('./result/1-5trial_0.2-2s/itr_mean_benchmark.pkl','rb') as f:
    itr_mean_benchmark = pickle.load(f)

with open('./result/1-5trial_0.2-2s/acc_mean_meg.pkl','rb') as f:
    acc_mean_meg = pickle.load(f)

with open('./result/1-5trial_0.2-2s/itr_mean_meg.pkl','rb') as f:
    itr_mean_meg = pickle.load(f)

# %%
acc_all_benchmark_mean = np.mean(acc_mean_benchmark,axis=1)*100
acc_all_benchmark_stde = np.std(acc_mean_benchmark,axis=1)*100/np.sqrt(acc_mean_benchmark.shape[1])

itr_all_benchmark_mean = np.mean(itr_mean_benchmark,axis=1)
itr_all_benchmark_stde = np.std(itr_mean_benchmark,axis=1)/np.sqrt(itr_mean_benchmark.shape[1])

acc_all_meg_mean = np.mean(acc_mean_meg,axis=1)*100
acc_all_meg_stde = np.std(acc_mean_meg,axis=1)*100/np.sqrt(acc_mean_meg.shape[1])

itr_all_meg_mean = np.mean(itr_mean_meg,axis=1)
itr_all_meg_stde = np.std(itr_mean_meg,axis=1)/np.sqrt(itr_mean_meg.shape[1])


# %%
'''
----------------------------------fig7-------------------------------------
'''

# %%
from my_code.utils.utils import print_significance
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,2,figsize=(5.5, 5),constrained_layout=False)
plt.subplots_adjust(top=0.92,bottom=0.09,left=0.1,right=0.99,wspace=0.3,hspace=0.5)
capsize = 3
ms = 3
linewidth = 0.8
markers = ['o','^','d','v','p','s']
ms = [3,3,3,3,3,3]
colors = ['#F08080','#4169E1','#48D1CC','#252C38','#619B35','#FF69B4']
method_IDs = ['ePRCA','eRESS','eTRCA','ms-eTRCA','TDCA','eEPCA']

train_num = 1

# ax = axes[0,0]

method_idx = [1,2,3,5]
train_num = [1,3]
itr_ylims = [(0,160),(0,200)]
itr_yticks = [[0,40,80,120,160],[0,40,80,120,160,200]]
marksr_y = [160,213]

for idx,j in enumerate(train_num):

    p_acc_benchmark = []
    p_itr_benchmark = []
    for i in range(acc_mean_benchmark.shape[2]):
        t_stat, p_value = ttest_rel(acc_mean_benchmark[1,:,i,j], acc_mean_benchmark[5,:,i,j])
        p_acc_benchmark.append(p_value)

        t_stat, p_value = ttest_rel(itr_mean_benchmark[1,:,i,j], itr_mean_benchmark[5,:,i,j])
        p_itr_benchmark.append(p_value)

    rejected, p_acc_benchmark, _, _ = multipletests(p_acc_benchmark, method='fdr_bh')
    rejected, p_itr_benchmark, _, _ = multipletests(p_itr_benchmark, method='fdr_bh')

    ax = axes[idx,0]
    for i in method_idx:
        ax.plot(tw_seqs,acc_all_benchmark_mean[i,:,j-1],label=method_IDs[i],lw=linewidth,color=colors[i],
                marker=markers[i],markersize=ms[i])
        ax.errorbar(tw_seqs,acc_all_benchmark_mean[i,:,j-1],yerr=acc_all_benchmark_stde[i,:,j-1],c=colors[i],lw=linewidth,
                    elinewidth=linewidth,capsize=capsize)
    for x,p in zip(tw_seqs,p_acc_benchmark):
        p = print_significance(p)
        if p != 'ns':
            for k in range(len(p)):
                ax.text(x,101-k*3,'*',ha='center')
    ax.hlines(y=90,xmin=0.15,xmax=2.05,color='k',linestyle='--',lw=linewidth)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
    ax.set_ylim(0,100)
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks([0,20,40,60,80,100])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.set_title(' ')


    ax = axes[idx,1]
    for i in method_idx:
        ax.plot(tw_seqs,itr_all_benchmark_mean[i,:,j-1],label=method_IDs[i],lw=linewidth,color=colors[i],
                marker=markers[i],markersize=ms[i])
        ax.errorbar(tw_seqs,itr_all_benchmark_mean[i,:,j-1],yerr=itr_all_benchmark_stde[i,:,j-1],c=colors[i],lw=linewidth,
                    elinewidth=linewidth,capsize=capsize)

    for x,p in zip(tw_seqs,p_itr_benchmark):
        p = print_significance(p)
        if p != 'ns':
            for k in range(len(p)):
                ax.text(x,marksr_y[idx]-k*marksr_y[idx]*0.03,'*',ha='center')
    ax.set_ylabel('ITR (bits/min)')
    ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
    ax.set_ylim(itr_ylims[idx])
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks(itr_yticks[idx])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.set_title(' ')


lg = axes[1,0].legend(loc=(0.5,0.15),fontsize=8)
lg.set_frame_on(False)

fig.text(0.5, 0.98, r'Benchmark $\regular{N_{t}}$=1', ha='center', va='center', fontsize=12,fontweight='bold')
fig.text(0.5, 0.48, r'Benchmark $\regular{N_{t}}$=3', ha='center', va='center', fontsize=12,fontweight='bold')
# fig.text(0.005,0.97,'A',fontweight='bold',fontsize=15)
# fig.text(0.005,0.47,'B',fontweight='bold',fontsize=15)

plt.show()

# fig.savefig('./fig/fig_7/fig_7_benchmark.png',dpi=600)
# fig.savefig('./fig/fig_7/fig_7_benchmark.pdf')
# fig.savefig('./fig/fig_7/fig_7_benchmark.svg')


# %%
from my_code.utils.utils import print_significance
from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,2,figsize=(5.5, 5),constrained_layout=False)
plt.subplots_adjust(top=0.92,bottom=0.09,left=0.1,right=0.99,wspace=0.3,hspace=0.5)
capsize = 3
ms = 3
linewidth = 0.8
markers = ['o','^','d','v','p','s']
ms = [3,3,3,3,3,3]
colors = ['#F08080','#4169E1','#48D1CC','#252C38','#619B35','#FF69B4']
method_IDs = ['ePRCA','eRESS','eTRCA','ms-eTRCA','TDCA','eEPCA']

train_num = 1

ax = axes[0,0]

method_idx = [1,2,3,5]
train_num = [1,3]
itr_ylims = [(0,80),(0,120)]
itr_yticks = [[0,20,40,60,80],[0,20,40,60,80,100,120]]
marker_y = [80,120]

for idx,j in enumerate(train_num):

    p_acc_meg = []
    p_itr_meg = []
    for i in range(acc_mean_meg.shape[2]):
        t_stat, p_value = ttest_rel(acc_mean_meg[1,:,i,j], acc_mean_meg[5,:,i,j])
        p_acc_meg.append(p_value)

        t_stat, p_value = ttest_rel(itr_mean_meg[1,:,i,j], itr_mean_meg[5,:,i,j])
        p_itr_meg.append(p_value)

    rejected, p_acc_meg, _, _ = multipletests(p_acc_meg, method='fdr_bh')
    rejected, p_itr_meg, _, _ = multipletests(p_itr_meg, method='fdr_bh')


    ax = axes[idx,0]
    for i in method_idx:
        ax.plot(tw_seqs,acc_all_meg_mean[i,:,j-1],label=method_IDs[i],lw=linewidth,color=colors[i],
                marker=markers[i],markersize=ms[i])
        ax.errorbar(tw_seqs,acc_all_meg_mean[i,:,j-1],yerr=acc_all_meg_stde[i,:,j-1],c=colors[i],lw=linewidth,
                    elinewidth=linewidth,capsize=capsize)
    for x,p in zip(tw_seqs,p_acc_meg):
        p = print_significance(p)
        if p != 'ns':
            for k in range(len(p)):
                ax.text(x,101-k*3,'*',ha='center')
    ax.hlines(y=90,xmin=0.15,xmax=2.05,color='k',linestyle='--',lw=linewidth)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
    ax.set_ylim(0,100)
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks([0,20,40,60,80,100])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(' ')


    ax = axes[idx,1]
    for i in method_idx:
        ax.plot(tw_seqs,itr_all_meg_mean[i,:,j-1],label=method_IDs[i],lw=linewidth,color=colors[i],
                marker=markers[i],markersize=ms[i])
        ax.errorbar(tw_seqs,itr_all_meg_mean[i,:,j-1],yerr=itr_all_meg_stde[i,:,j-1],c=colors[i],lw=linewidth,
                    elinewidth=linewidth,capsize=capsize)

    for x,p in zip(tw_seqs,p_itr_meg):
        p = print_significance(p)
        if p != 'ns':
            for k in range(len(p)):
                ax.text(x,marker_y[idx]-k*marker_y[idx]*0.03,'*',ha='center')
    ax.set_ylabel('ITR (bits/min)')
    ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
    ax.set_ylim(itr_ylims[idx])
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks(itr_yticks[idx])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(' ')


lg = axes[1,0].legend(loc=(0.5,0.15),fontsize=8)
lg.set_frame_on(False)

fig.text(0.5, 0.98, r'MEG $\regular{N_{t}}$=1', ha='center', va='center', fontsize=12,fontweight='bold')
fig.text(0.5, 0.48, r'MEG $\regular{N_{t}}$=3', ha='center', va='center', fontsize=12,fontweight='bold')
# fig.text(0.005,0.97,'A',fontweight='bold',fontsize=15)
# fig.text(0.005,0.47,'B',fontweight='bold',fontsize=15)

plt.show()

# fig.savefig('./fig/fig_7/fig_7_meg.png',dpi=600)
# fig.savefig('./fig/fig_7/fig_7_meg.pdf')
# fig.savefig('./fig/fig_7/fig_7_meg.svg')



















# %%
'''
----------------------------------其他-------------------------------------
'''

# %%
from matplotlib.colors import LinearSegmentedColormap
# colors = ['#c6eada', '#42b983', '#143827']
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# colors = ['#c6eada', '#42b983', '#143827']
colors = ['#18748E','w','#FC5454']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
n_models, n_tws, n_trains = acc_all_benchmark_mean.shape

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,3,figsize=(6, 4),constrained_layout=False)
plt.subplots_adjust(top=0.96,bottom=0.07,left=0.08,right=0.9,wspace=0.3,hspace=0.3)

titles = ['eTRCA','eRESS','eEPCA']

tw_pick_idx = [0,2,4,6,8]

for m,model_idx in enumerate([2,1,5]):
    ax = axes.flatten()[m]
    im = ax.imshow(acc_all_benchmark_mean[model_idx,tw_pick_idx,:],cmap=cmap,origin='lower',vmin=0,vmax=100)
    for idx,i in enumerate(tw_pick_idx):
        for j in range(n_trains):
            text = ax.text(j, idx, "{:.2f}".format(acc_all_benchmark_mean[model_idx, i, j]),
                        ha="center", va="center", color="k",fontsize=8)
    ax.set_title(titles[m],fontsize=10,fontweight='bold')
    ax.set_xticks(np.arange(n_trains))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(tw_pick_idx)))
    ax.set_yticklabels([str(tw_seqs[i]) for i in tw_pick_idx])
    ax.set_aspect(1)

fig.text(0.485,0.525,r'Number of training trials',ha='center',va='center',
         fontname='Times New Roman',fontsize=12,fontweight='bold')
fig.text(0.015,0.75,r'Data length (s)',ha='center',va='center',rotation='vertical',
         fontname='Times New Roman',fontsize=12,fontweight='bold')

cbar = plt.colorbar(im, ax=axes[0,:], orientation="vertical", fraction=0.014, pad=0.02)
cbar.set_label("Accuracy (%)", fontsize=10)



from scipy.stats import ttest_rel
model_idx = [2,1]
t_trca_ress = np.zeros((len(num_trains),len(tw_pick_idx)))
p_trca_ress = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_benchmark[model_idx,:,tw_idx,train_idx]
        t_trca_ress[train_idx,idx],p_trca_ress[train_idx,idx] = ttest_rel(acc[1],acc[0])

model_idx = [2,5]
t_trca_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
p_trca_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_benchmark[model_idx,:,tw_idx,train_idx]
        t_trca_epca[train_idx,idx],p_trca_epca[train_idx,idx] = ttest_rel(acc[1],acc[0])

model_idx = [1,5]
t_ress_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
p_ress_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_benchmark[model_idx,:,tw_idx,train_idx]
        t_ress_epca[train_idx,idx],p_ress_epca[train_idx,idx] = ttest_rel(acc[1],acc[0])

import matplotlib.patches as patches

titles = ['eRESS vs. eTRCA','eEPCA vs. eTRCA','eEPCA vs. eRESS']

colors = ['#ffb9df', '#ff1493']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

ts = [t_trca_ress,t_trca_epca,t_ress_epca]
ps = [p_trca_ress,p_trca_epca,p_ress_epca]

for idx in range(3):
    ax = axes.flatten()[idx+3]
    im = ax.imshow(ts[idx].T,cmap=cmap,origin='lower',vmin=0,vmax=20)
    for i in range(len(tw_pick_idx)):
        for j in range(n_trains):
            if ps[idx][j, i] > 0.05:
                rect1 = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='w', facecolor='w', alpha=1)
                ax.add_patch(rect1)
            text = ax.text(j, i, "{:.2f}".format(ts[idx][j, i]),
                        ha="center", va="center", color="k",fontsize=8)
            

    
    ax.set_title(titles[idx],fontsize=10,fontweight='bold')
    ax.set_xticks(np.arange(n_trains))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(tw_pick_idx)))
    ax.set_yticklabels([str(tw_seqs[i]) for i in tw_pick_idx])
    ax.set_aspect(1)


fig.text(0.485,0.02,r'Number of training trials',ha='center',va='center',
         fontname='Times New Roman',fontsize=12,fontweight='bold')
fig.text(0.015,0.25,r'Data length (s)',ha='center',va='center',rotation='vertical',
         fontname='Times New Roman',fontsize=12,fontweight='bold')

cbar = fig.colorbar(im, ax=axes[1,:], orientation="vertical", fraction=0.014, pad=0.02)
cbar.set_label("t-value", fontsize=10)

plt.show()

# fig.savefig('./fig/fig_5/fig_5_benchmark1.png',dpi=600)
# fig.savefig('./fig/fig_5/fig_5_benchmark1.pdf')
# fig.savefig('./fig/fig_5/fig_5_benchmark1.svg')


# %%
from matplotlib.colors import LinearSegmentedColormap
# colors = ['#c6eada', '#42b983', '#143827']
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# colors = ['#c6eada', '#42b983', '#143827']
colors = ['#18748E','w','#FC5454']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
n_models, n_tws, n_trains = acc_all_meg_mean.shape

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,3,figsize=(6, 4),constrained_layout=False)
plt.subplots_adjust(top=0.96,bottom=0.07,left=0.08,right=0.9,wspace=0.3,hspace=0.3)

titles = ['eTRCA','eRESS','eEPCA']

tw_pick_idx = [0,2,4,6,8]

for m,model_idx in enumerate([2,1,5]):
    ax = axes.flatten()[m]
    im = ax.imshow(acc_all_meg_mean[model_idx,tw_pick_idx,:],cmap=cmap,origin='lower',vmin=0,vmax=100)
    for idx,i in enumerate(tw_pick_idx):
        for j in range(n_trains):
            text = ax.text(j, idx, "{:.2f}".format(acc_all_meg_mean[model_idx, i, j]),
                        ha="center", va="center", color="k",fontsize=8)
    ax.set_title(titles[m],fontsize=10,fontweight='bold')
    ax.set_xticks(np.arange(n_trains))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(tw_pick_idx)))
    ax.set_yticklabels([str(tw_seqs[i]) for i in tw_pick_idx])
    ax.set_aspect(1)

fig.text(0.485,0.525,r'Number of training trials',ha='center',va='center',
         fontname='Times New Roman',fontsize=12,fontweight='bold')
fig.text(0.015,0.75,r'Data length (s)',ha='center',va='center',rotation='vertical',
         fontname='Times New Roman',fontsize=12,fontweight='bold')

cbar = plt.colorbar(im, ax=axes[0,:], orientation="vertical", fraction=0.014, pad=0.02)
cbar.set_label("Accuracy (%)", fontsize=10)



from scipy.stats import ttest_rel
model_idx = [2,1]
t_trca_ress = np.zeros((len(num_trains),len(tw_pick_idx)))
p_trca_ress = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_meg[model_idx,:,tw_idx,train_idx]
        t_trca_ress[train_idx,idx],p_trca_ress[train_idx,idx] = ttest_rel(acc[1],acc[0])

model_idx = [2,5]
t_trca_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
p_trca_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_meg[model_idx,:,tw_idx,train_idx]
        t_trca_epca[train_idx,idx],p_trca_epca[train_idx,idx] = ttest_rel(acc[1],acc[0])

model_idx = [1,5]
t_ress_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
p_ress_epca = np.zeros((len(num_trains),len(tw_pick_idx)))
for train_idx in range(len(num_trains)):
    for idx,tw_idx in enumerate(tw_pick_idx):
        acc = acc_mean_meg[model_idx,:,tw_idx,train_idx]
        t_ress_epca[train_idx,idx],p_ress_epca[train_idx,idx] = ttest_rel(acc[1],acc[0])

import matplotlib.patches as patches

titles = ['eRESS vs. eTRCA','eEPCA vs. eTRCA','eEPCA vs. eRESS']

colors = ['#ffb9df', '#ff1493']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

ts = [t_trca_ress,t_trca_epca,t_ress_epca]
ps = [p_trca_ress,p_trca_epca,p_ress_epca]

for idx in range(3):
    ax = axes.flatten()[idx+3]
    im = ax.imshow(ts[idx].T,cmap=cmap,origin='lower',vmin=0,vmax=20)
    for i in range(len(tw_pick_idx)):
        for j in range(n_trains):
            if ps[idx][j, i] > 0.05:
                rect1 = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='w', facecolor='w', alpha=1)
                ax.add_patch(rect1)
            text = ax.text(j, i, "{:.2f}".format(ts[idx][j, i]),
                        ha="center", va="center", color="k",fontsize=8)
            

    
    ax.set_title(titles[idx],fontsize=10,fontweight='bold')
    ax.set_xticks(np.arange(n_trains))
    ax.set_xticklabels([str(i) for i in num_trains])
    ax.set_yticks(np.arange(len(tw_pick_idx)))
    ax.set_yticklabels([str(tw_seqs[i]) for i in tw_pick_idx])
    ax.set_aspect(1)


fig.text(0.485,0.02,r'Number of training trials',ha='center',va='center',
         fontname='Times New Roman',fontsize=12,fontweight='bold')
fig.text(0.015,0.25,r'Data length (s)',ha='center',va='center',rotation='vertical',
         fontname='Times New Roman',fontsize=12,fontweight='bold')

cbar = fig.colorbar(im, ax=axes[1,:], orientation="vertical", fraction=0.014, pad=0.02)
cbar.set_label("t-value", fontsize=10)

plt.show()



# %%
