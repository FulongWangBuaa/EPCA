# %%
'''
分类准确率与刺激频率之间的关系
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


dataset_key = 'Benchmark'
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
evaluator_path = r"F:\科研\5、EPCA\EPCA\result\1-5trial_0.2-2s(EPRCA,ERESS,RTRCA,EEPCA)\evaluator_1-5trial_0.2-2s.pkl"
evaluator.load(evaluator_path)


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
n_stims = dataset.stim_info['stim_num']

performance_container = evaluator.performance_container
Y_test = dict()
Y_pred = dict()
for trialinfo,performance in zip(trial_container,performance_container):
    tw = trialinfo[0].tw
    idx_tw = tw_seqs.index(tw)
    idx_sub = trialinfo[0].sub_idx[0]
    train_block_idx = tuple(trialinfo[0].block_idx[0])
    idx_train = len(train_block_idx)

    t_latency = dataset.default_t_latency
    n_targs = len(trialinfo[0].trial_idx[0])

    if idx_train not in Y_test.keys():
        Y_test[idx_train] = dict()
        Y_pred[idx_train] = dict()
    if idx_tw not in Y_test[idx_train].keys():
        Y_test[idx_train][idx_tw] = dict()
        Y_pred[idx_train][idx_tw] = dict()
    if idx_sub not in Y_test[idx_train][idx_tw].keys():
        Y_test[idx_train][idx_tw][idx_sub] = []
        Y_pred[idx_train][idx_tw][idx_sub] = []

    for idx_model,model_performance in enumerate(performance):

        Y_test_0 = model_performance.true_label_test
        Y_pred_0 = model_performance.pred_label_test

        # print(len(Y_test_0))
        # acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
        # itr = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = t_latency,
        #               t_comp = 0,N = n_targs, acc = acc)

        Y_test[idx_train][idx_tw][idx_sub].append(Y_test_0)
        Y_pred[idx_train][idx_tw][idx_sub].append(Y_pred_0)


# %%
# with open('./result/fig_9/Y_test_benchmark.pkl','wb') as f:
#     pickle.dump(Y_test,f)
# with open('./result/fig_9/Y_pred_benchmark.pkl','wb') as f:
#     pickle.dump(Y_pred,f)



# %%
acc_all = np.zeros((n_stims, n_trains, n_tws, n_subs))
itr_all = np.zeros((n_stims, n_trains, n_tws, n_subs))
for train_idx, key1 in enumerate(Y_test.keys()):
    for tw_idx, key2 in enumerate(Y_test[key1].keys()):
        for sub_idx, key3 in enumerate(Y_test[key1][key2].keys()):
            Y_test_0 = [item2 for item1 in Y_test[key1][key2][key3] for item2 in item1]
            Y_pred_0 = [item2 for item1 in Y_pred[key1][key2][key3] for item2 in item1]
            for stim_idx in range(n_stims):
                stim_test_idx = [index for index, value in enumerate(Y_test_0) if value == stim_idx]
                Y_test_stim = [Y_test_0[idx] for idx in stim_test_idx]
                Y_pred_stim = [Y_pred_0[idx] for idx in stim_test_idx]
                acc = cal_acc(Y_true = Y_test_stim, Y_pred = Y_pred_stim)
                itr = cal_itr(tw = tw_seqs[key2], t_break = dataset.t_break, t_latency = dataset.default_t_latency,
                            t_comp = 0,N = n_stims, acc = acc)
                
                acc_all[stim_idx,train_idx,tw_idx,sub_idx] = acc*100
                itr_all[stim_idx,train_idx,tw_idx,sub_idx] = itr

# %%
# with open('./result/fig_9/acc_all_benchmark.pkl','wb') as f:
#     pickle.dump(acc_all,f)
# with open('./result/fig_9/itr_all_benchmark.pkl','wb') as f:
#     pickle.dump(itr_all,f)


# %%
with open('./result/fig_9/acc_all_benchmark.pkl','rb') as f:
    acc_all = pickle.load(f)
with open('./result/fig_9/itr_all_benchmark.pkl','rb') as f:
    itr_all = pickle.load(f)

num_trains = [1,2,3,4,5]
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]


# %%
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(3, 2),constrained_layout=False)
plt.subplots_adjust(top=0.92,bottom=0.20,left=0.18,right=0.99,wspace=0.3,hspace=0.5)

stim_freqs = dataset.stim_info['freqs']
idx = np.argsort(stim_freqs)
stim_freqs_sort = [stim_freqs[i] for i in idx]

capsize = 2
linewidth = 0.8
markers = ['o','^','s']
ms = [2,2,2]
colors = ['#4169E1','#48D1CC','#F08080','#252C38','#619B35','#FF69B4']
method_IDs = ['0.6 s','1 s','2 s']

tws = [0.6,1,2]
train_num = 1

for i,tw in enumerate(tws):
    tw_idx = tw_seqs.index(tw)

    tw_idx = tw_seqs.index(tw)
    train_idx = num_trains.index(train_num)

    acc_pick = acc_all[:,train_idx,tw_idx,:]

    acc_pick_sort = acc_pick[idx,:]
    acc_pick_sort_mean = np.mean(acc_pick_sort,axis=1)
    acc_pick_sort_stde = np.std(acc_pick_sort,axis=1)/np.sqrt(acc_pick_sort.shape[1])

    ax.plot(stim_freqs_sort,acc_pick_sort_mean,label=method_IDs[i],lw=linewidth,color=colors[i],
                    marker=markers[i],markersize=ms[i])
    ax.errorbar(stim_freqs_sort,acc_pick_sort_mean,yerr=acc_pick_sort_stde,c=colors[i],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)
    
    import pingouin as pg

    data = {'sub':list(np.tile(np.arange(0,35),40)),
            'freq':list(np.repeat(stim_freqs_sort,35)),
            'acc':list(np.reshape(acc_pick_sort,-1))
    }
    df = pd.DataFrame(data)
    # 重复测量方差分析
    anova_result = pg.rm_anova(data=df, dv='acc', within='freq', subject='sub', detailed=True)
    print(anova_result,'\r\n\r\n\r\n')
    # print(anova_result['DF'][0]*anova_result['eps'][0], anova_result['DF'][1]*anova_result['eps'][0])

    # posthoc = pg.pairwise_tests(dv='acc', within='freq', subject='sub', data=df, padjust='fdr_bh')
    # print(posthoc)


ax.set_xlim(stim_freqs_sort[0]-0.1,stim_freqs_sort[-1]+0.1)
ax.set_xticks(stim_freqs_sort)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

ax.set_ylim(25,100)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Stimulus Frequency (Hz)')
ax.set_ylabel('Accuracy (%)')

lg = ax.legend(loc=(0.02,0.85),frameon=False,ncol=3,fontsize=10)

plt.show()

# fig.savefig('./fig/fig_9/fig9_acc.png',dpi=900)
# fig.savefig('./fig/fig_9/fig9_acc.pdf')
# fig.savefig('./fig/fig_9/fig9_acc.svg')



# %%
