# %%
'''
数据长度的影响
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
tw_seq = [0.4,0.8,1.2,1.6,2]
tw_idx = [tw_seqs.index(i) for i in tw_seq]
model_idx = 5

num_train = 1
trian_idx = num_trains.index(num_train)
acc_mean_benchmark_train_1 = acc_mean_benchmark[model_idx,:,tw_idx,trian_idx]*100
acc_mean_meg_train_1 = acc_mean_meg[model_idx,:,tw_idx,trian_idx]*100

num_train = 5
trian_idx = num_trains.index(num_train)
acc_mean_benchmark_train_5 = acc_mean_benchmark[model_idx,:,tw_idx,trian_idx]*100
acc_mean_meg_train_5 = acc_mean_meg[model_idx,:,tw_idx,trian_idx]*100


# %%
import pingouin as pg

data = {'sub':list(np.tile(np.arange(0,35),5)),
        'tw':list(np.repeat(tw_seq,35)),
        'acc':list(np.reshape(acc_mean_benchmark_train_1,-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='tw', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='tw', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)


print('\n\n-------------------------')
data = {'sub':list(np.tile(np.arange(0,35),5)),
        'tw':list(np.repeat(tw_seq,35)),
        'acc':list(np.reshape(acc_mean_benchmark_train_5,-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='tw', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='tw', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)


print('\n\n-------------------------')
data = {'sub':list(np.tile(np.arange(0,13),5)),
        'tw':list(np.repeat(tw_seq,13)),
        'acc':list(np.reshape(acc_mean_meg_train_1,-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='tw', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='tw', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)



print('\n\n-------------------------')
data = {'sub':list(np.tile(np.arange(0,13),5)),
        'tw':list(np.repeat(tw_seq,13)),
        'acc':list(np.reshape(acc_mean_meg_train_5,-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='tw', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='tw', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)




# %%
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# compairs = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]]
# ps = []
# for i, compair in enumerate(compairs):
#     t,p = ttest_rel(acc_mean_benchmark_train_1[compair[0]],acc_mean_benchmark_train_1[compair[1]])
#     ps.append(p)

# rejected, ps, _, _ = multipletests(ps, method='fdr_bh')

import seaborn as sns
from my_code.utils.utils import print_significance
from pypalettes import load_cmap
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 8
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.5),constrained_layout=False)
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.1,right=0.99,wspace=0.3,hspace=0.5)
# colors = ['#398BD1','#E99E74','#CF5149','#6D4EA4','#A52F9E','#619B35','#0E4F69','#357887']
cmap = load_cmap("pink_material") # Doughton pink_material purple_material
colors = cmap.colors[5:]
ylims = [0,100]

ax = axes[0]
acc_mean_benchmark_train_1 = [acc_mean_benchmark_train_1[i] for i in range(len(tw_seq))]
sns.barplot(acc_mean_benchmark_train_1,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_benchmark_train_1,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_benchmark_train_1,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels([str(tw_seq[i]) for i in range(len(tw_seq))])

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax = axes[1]
acc_mean_benchmark_train_5 = [acc_mean_benchmark_train_5[i] for i in range(len(tw_seq))]
sns.barplot(acc_mean_benchmark_train_5,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_benchmark_train_5,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_benchmark_train_5,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels([str(tw_seq[i]) for i in range(len(tw_seq))])

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

# fig.savefig('./fig/fig_6/fig_6_benchmark.png',dpi=600)
# fig.savefig('./fig/fig_6/fig_6_benchmark.pdf')
# fig.savefig('./fig/fig_6/fig_6_benchmark.svg')




# %%
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# compairs = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]]
# ps = []
# for i, compair in enumerate(compairs):
#     t,p = ttest_rel(acc_mean_benchmark_train_1[compair[0]],acc_mean_benchmark_train_1[compair[1]])
#     ps.append(p)

# rejected, ps, _, _ = multipletests(ps, method='fdr_bh')

import seaborn as sns
from my_code.utils.utils import print_significance
from pypalettes import load_cmap
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 8
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.5),constrained_layout=False)
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.1,right=0.99,wspace=0.3,hspace=0.5)
# colors = ['#398BD1','#E99E74','#CF5149','#6D4EA4','#A52F9E','#619B35','#0E4F69','#357887']
cmap = load_cmap("purple_material") # Doughton pink_material purple_material
colors = cmap.colors[5:]
ylims = [0,100]

ax = axes[0]
acc_mean_meg_train_1 = [acc_mean_meg_train_1[i] for i in range(len(tw_seq))]
sns.barplot(acc_mean_meg_train_1,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_meg_train_1,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_meg_train_1,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels([str(tw_seq[i]) for i in range(len(tw_seq))])

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax = axes[1]
acc_mean_meg_train_5 = [acc_mean_meg_train_5[i] for i in range(len(tw_seq))]
sns.barplot(acc_mean_meg_train_5,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_meg_train_5,capsize=0.1,ax=ax,palette=colors,width=0.4,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_meg_train_5,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels([str(tw_seq[i]) for i in range(len(tw_seq))])

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

# fig.savefig('./fig/fig_6/fig_6_meg.png',dpi=600)
# fig.savefig('./fig/fig_6/fig_6_meg.pdf')
# fig.savefig('./fig/fig_6/fig_6_meg.svg')

# %%
