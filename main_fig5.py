# %%
'''
训练试验数量影响
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
tw_seq = 1
tw_seq_idx = tw_seqs.index(tw_seq)

acc_mean_benchmark_epca = acc_mean_benchmark[-1,:,tw_seq_idx,:]*100
acc_mean_benchmark_trca = acc_mean_benchmark[2,:,tw_seq_idx,-2]*100

acc_mean_benchmark_epca = [acc_mean_benchmark_epca[:,i] for i in range(acc_mean_benchmark_epca.shape[1])]
acc_mean_benchmark_epca.insert(0, acc_mean_benchmark_trca)

import seaborn as sns
from my_code.utils.utils import print_significance

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 2),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.2,left=0.1,right=0.99,wspace=0.3,hspace=0.5)

colors = ['#398BD1','#E99E74','#CF5149','#6D4EA4','#A52F9E','#619B35']
ylims = [0,100]

ax = axes[0]
sns.barplot(acc_mean_benchmark_epca[1:],capsize=0.1,ax=ax,palette=colors[1:],width=0.5,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_benchmark_epca[1:],capsize=0.1,ax=ax,palette=colors[1:],width=0.5,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_benchmark_epca[1:],ax=ax,palette=colors[1:],size=1.5,alpha=0.8,jitter=0.2,zorder=1)



ax.set_xticks([0,1,2,3,4])
ax.set_xlabel('Number of training trials')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Benchmark',fontsize=10,fontweight='bold')

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# xlim = ax.get_xlim()
# ax.hlines(90,xlim[0],xlim[1],linestyles='dashed',colors='r',linewidth=0.8)

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection
ts,ps = [],[]
pair = [[0,1],[1,2],[2,3],[3,4],[0,2],[1,3],[2,4],[0,3],[1,4],[2,4]]
for i in range(len(pair)):
    a,b = pair[i]
    t,p = ttest_rel(acc_mean_benchmark_epca[a],acc_mean_benchmark_epca[b])
    ts.append(t)
    ps.append(p)
rejected, ps, _, _ = multipletests(ps, method='fdr_bh')
rejected1, ps1 = fdrcorrection(ps, alpha=0.05, method='indep')




ax = axes[1]
acc_mean_meg_epca = acc_mean_meg[-1,:,tw_seq_idx,:]*100
acc_mean_meg_trca = acc_mean_meg[2,:,tw_seq_idx,-2]*100

acc_mean_meg_epca = [acc_mean_meg_epca[:,i] for i in range(acc_mean_meg_epca.shape[1])]
acc_mean_meg_epca.insert(0, acc_mean_meg_trca)

sns.barplot(acc_mean_meg_epca[1:],capsize=0.1,ax=ax,palette=colors[1:],width=0.5,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_meg_epca[1:],capsize=0.1,ax=ax,palette=colors[1:],width=0.5,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_meg_epca[1:],ax=ax,palette=colors[1:],size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4])
ax.set_xlabel('Number of training trials')
ax.set_ylabel('Accuracy (%)')
ax.set_title('MEG',fontsize=10,fontweight='bold')

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection
ts,ps = [],[]
pair = [[0,1],[1,2],[2,3],[3,4],[0,2],[1,3],[2,4],[0,3],[1,4],[2,4]]
for i in range(len(pair)):
    a,b = pair[i]
    t,p = ttest_rel(acc_mean_meg_epca[a],acc_mean_meg_epca[b])
    ts.append(t)
    ps.append(p)
rejected, ps, _, _ = multipletests(ps, method='fdr_bh')
rejected1, ps1 = fdrcorrection(ps, alpha=0.05, method='indep')

fig.text(0.01,0.92,'A',fontsize=12,fontweight='bold')
fig.text(0.51,0.92,'B',fontsize=12,fontweight='bold')

plt.show()


# fig.savefig('./fig/fig_5/fig_5.png',dpi=600)
# fig.savefig('./fig/fig_5/fig_5.pdf')
# fig.savefig('./fig/fig_5/fig_5.svg')


# %%
import pingouin as pg
data = {'sub':list(np.tile(np.arange(0,35),5)),
        'trails':list(np.repeat(num_trains,35)),
        'acc':list(np.reshape(np.array(acc_mean_benchmark_epca[1:]),-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='trails', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='trails', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)



import pingouin as pg
data = {'sub':list(np.tile(np.arange(0,13),5)),
        'trails':list(np.repeat(num_trains,13)),
        'acc':list(np.reshape(np.array(acc_mean_meg_epca[1:]),-1))
}
df = pd.DataFrame(data)
# 重复测量方差分析
anova_result = pg.rm_anova(data=df, dv='acc', within='trails', subject='sub', detailed=True)
print(anova_result)
print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

posthoc = pg.pairwise_tests(dv='acc', within='trails', subject='sub', data=df, padjust='fdr_bh')
print(posthoc)


























# %%
import textwrap
tw_seq = 1
tw_seq_idx = tw_seqs.index(tw_seq)

acc_mean_benchmark_epca = acc_mean_benchmark[-1,:,tw_seq_idx,:]*100
acc_mean_benchmark_trca = acc_mean_benchmark[2,:,tw_seq_idx,-2]*100

acc_mean_benchmark_epca = [acc_mean_benchmark_epca[:,i] for i in range(acc_mean_benchmark_epca.shape[1])]
acc_mean_benchmark_epca.insert(0, acc_mean_benchmark_trca)

import seaborn as sns
from my_code.utils.utils import print_significance

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.15,left=0.2,right=0.99,wspace=0.3,hspace=0.5)

colors = ['#398BD1','#E99E74','#CF5149','#6D4EA4','#A52F9E','#619B35']
ylims = [0,100]

sns.barplot(acc_mean_benchmark_epca,capsize=0.1,ax=ax,palette=colors,width=0.5,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_benchmark_epca,capsize=0.1,ax=ax,palette=colors,width=0.5,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_benchmark_epca,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

# for i, bar in enumerate(ax.patches):
#     bar.set_edgecolor(colors[i % len(colors)])
#     bar.set_linewidth(1.5)

ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels([])

ticks1 = ['eTRCA'] + ['eEPCA']*5
ticks2 = [r'$\regular{N_{t}}=5$',r'$\regular{N_{t}}=1$',r'$\regular{N_{t}}=2$',r'$\regular{N_{t}}=3$',
          r'$\regular{N_{t}}=4$',r'$\regular{N_{t}}=5$']
for i in range(6):
    if i == 0:
        fontweight = 'bold'
        color='r'
    else:
        color='k'
        fontweight = 'normal'
    fig.text(0.268+i*0.131,0.085,ticks1[i],color=color,ha='center',va='center',fontsize=8,fontweight=fontweight)
    fig.text(0.268+i*0.131,0.03,ticks2[i],color=color,ha='center',va='center',fontsize=8,fontweight=fontweight)

ax.set_ylabel('Accuracy (%)')

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests
ts,ps = [],[]
for i in range(1,len(acc_mean_benchmark_epca)):
    t,p = ttest_rel(acc_mean_benchmark_epca[0],acc_mean_benchmark_epca[i])
    ts.append(t)
    ps.append(p)
rejected, ps, _, _ = multipletests(ps, method='fdr_bh')

x_offset = 0.09
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.31, x_offset+0.31], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.31], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.245,0.83,print_significance(ps[0]),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.44, x_offset+0.44], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.44], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.32,0.89,print_significance(ps[1]),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.92, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.57, x_offset+0.57], [0.92, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.57], [0.94, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.40,0.95,print_significance(ps[2]),ha='center',va='center',fontsize=10,fontweight='bold')

plt.show()

# fig.savefig('./fig/fig_5/fig_5_benchmark.png',dpi=600)
# fig.savefig('./fig/fig_5/fig_5_benchmark.pdf')
# fig.savefig('./fig/fig_5/fig_5_benchmark.svg')


# %%
tw_seq = 1
tw_seq_idx = tw_seqs.index(tw_seq)

acc_mean_meg_epca = acc_mean_meg[-1,:,tw_seq_idx,:]*100
acc_mean_meg_trca = acc_mean_meg[2,:,tw_seq_idx,-2]*100

acc_mean_meg_epca = [acc_mean_meg_epca[:,i] for i in range(acc_mean_meg_epca.shape[1])]
acc_mean_meg_epca.insert(0, acc_mean_meg_trca)

import seaborn as sns
from my_code.utils.utils import print_significance

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.15,left=0.2,right=0.99,wspace=0.3,hspace=0.5)

colors = ['#398BD1','#E99E74','#CF5149','#6D4EA4','#A52F9E','#619B35']
ylims = [0,100]

sns.barplot(acc_mean_meg_epca,capsize=0.1,ax=ax,palette=colors,width=0.5,alpha=1,linewidth=1.5,fill=False,
            errorbar=None,zorder=0)
sns.barplot(acc_mean_meg_epca,capsize=0.1,ax=ax,palette=colors,width=0.5,alpha=0.2,linewidth=1.5,
            err_kws={'linewidth': 1,'color':'k'},zorder=2)
sns.stripplot(acc_mean_meg_epca,ax=ax,palette=colors,size=1.5,alpha=0.8,jitter=0.2,zorder=1)

ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels([])

ticks1 = ['eTRCA'] + ['eEPCA']*5
ticks2 = [r'$\regular{N_{t}}=5$',r'$\regular{N_{t}}=1$',r'$\regular{N_{t}}=2$',r'$\regular{N_{t}}=3$',
          r'$\regular{N_{t}}=4$',r'$\regular{N_{t}}=5$']
for i in range(6):
    if i == 0:
        fontweight = 'bold'
        color='r'
    else:
        color='k'
        fontweight = 'normal'
    fig.text(0.268+i*0.131,0.085,ticks1[i],color=color,ha='center',va='center',fontsize=8,fontweight=fontweight)
    fig.text(0.268+i*0.131,0.03,ticks2[i],color=color,ha='center',va='center',fontsize=8,fontweight=fontweight)

ax.set_ylabel('Accuracy (%)')

ax.set_ylim(ylims)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests
ts,ps = [],[]
for i in range(1,len(acc_mean_meg_epca)):
    t,p = ttest_rel(acc_mean_meg_epca[0],acc_mean_meg_epca[i])
    ts.append(t)
    ps.append(p)
rejected, ps, _, _ = multipletests(ps, method='fdr_bh')

x_offset = 0.09
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.31, x_offset+0.31], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.31], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.245,0.83,print_significance(ps[0]),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.44, x_offset+0.44], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.44], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.32,0.89,print_significance(ps[1]),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.175, x_offset+0.175], [0.92, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.57, x_offset+0.57], [0.92, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.175, x_offset+0.57], [0.94, 0.94], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.40,0.95,print_significance(ps[2]),ha='center',va='center',fontsize=10,fontweight='bold')

plt.show()

# fig.savefig('./fig/fig_5/fig_5_meg.png',dpi=600)
# fig.savefig('./fig/fig_5/fig_5_meg.pdf')
# fig.savefig('./fig/fig_5/fig_5_meg.svg')

# %%
