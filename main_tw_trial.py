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

from my_code.datasets.mybetadataset import MyBetaDataset
from my_code.utils.betapreprocess import preprocess as betapreprocess
from my_code.utils.betapreprocess import filterbank as betafilterbank

from my_code.datasets.myeldbetadataset import MyeldBetaDataset
from my_code.utils.eldbetapreprocess import preprocess as eldbetapreprocess
from my_code.utils.eldbetapreprocess import filterbank as eldbetafilterbank

from my_code.datasets.myguhfdataset import MyGuHFDataset
from my_code.utils.guhfpreprocess import preprocess as guhfpreprocess
from my_code.utils.guhfpreprocess import filterbank as guhffilterbank


# dataset_keys = ['Benchmark', 'MEG', 'BETA', 'eldBETA', 'GuHF]

dataset_key = 'GuHF'

if dataset_key == 'Benchmark':
    data_path = r".\datasets\Benchmark"
    dataset = MyBenchmarkDataset(path=data_path)
    dataset.regist_preprocess(benchmarkpreprocess)
    dataset.regist_filterbank(benchmarkfilterbank)
elif dataset_key == 'MEG':
    data_path = r".\datasets\OPMMEG"
    dataset = MyMEGDataset(path=data_path)
    dataset.regist_preprocess(megpreprocess)
    dataset.regist_filterbank(megfilterbank)
elif dataset_key == 'BETA':
    data_path = r".\datasets\BETA"
    dataset = MyBetaDataset(path=data_path)
    dataset.regist_preprocess(betapreprocess)
    dataset.regist_filterbank(betafilterbank)
elif dataset_key == 'eldBETA':
    data_path = r".\datasets\eldBETA"
    dataset = MyeldBetaDataset(path=data_path)
    dataset.regist_preprocess(eldbetapreprocess)
    dataset.regist_filterbank(eldbetafilterbank)
elif dataset_key == 'GuHF':
    data_path = r".\datasets\GuHF"
    dataset = MyGuHFDataset(path=data_path)
    dataset.regist_preprocess(guhfpreprocess)
    dataset.regist_filterbank(guhffilterbank)

# %%
from my_code.utils.benchmarkpreprocess import benchmark_suggested_ch
from my_code.utils.benchmarkpreprocess import suggested_weights_filterbank as benchmark_weights_filterbank

from my_code.utils.megpreprocess import meg_suggested_ch
from my_code.utils.megpreprocess import suggested_weights_filterbank as meg_weights_filterbank

from my_code.utils.betapreprocess import beta_suggested_ch
from my_code.utils.betapreprocess import suggested_weights_filterbank as beta_weights_filterbank

from my_code.utils.eldbetapreprocess import eldbeta_suggested_ch
from my_code.utils.eldbetapreprocess import suggested_weights_filterbank as eldbeta_weights_filterbank

from my_code.utils.guhfpreprocess import guhf_suggested_ch
from my_code.utils.guhfpreprocess import suggested_weights_filterbank as guhff_weights_filterbank


from my_code.algorithms.cca import FBCCA
from my_code.algorithms.trca import ETRCA, MSETRCA
from my_code.algorithms.epca import EEPCA
from my_code.algorithms.ress import ERESS
from my_code.algorithms.tdca import TDCA

from my_code.evaluator.MyBaseEvaluator import BaseEvaluator

if dataset_key == 'Benchmark':
    ch_used = benchmark_suggested_ch(9)
    weights_filterbank = benchmark_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'MEG':
    ch_used = meg_suggested_ch(9)
    weights_filterbank = meg_weights_filterbank()
    harmonic_num = 4
elif dataset_key == 'BETA':
    ch_used = beta_suggested_ch(9)
    weights_filterbank = beta_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'eldBETA':
    ch_used = eldbeta_suggested_ch(9)
    weights_filterbank = eldbeta_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'GuHF':
    ch_used = guhf_suggested_ch(9)
    weights_filterbank = guhff_weights_filterbank()
    harmonic_num = 5

stim_freqs = dataset.stim_info['freqs']
srate=dataset.srate

if dataset.block_num > 5:
    num_trains = [1,2,3,4,5]
else:
    num_trains = np.arange(1, dataset.block_num)
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

for num_train in num_trains:
    for tw_seq in tw_seqs:
        evaluator_file = f'./result_R1/evaluator_tw_trial/evaluator_tw_trial_{dataset_key}_{num_train}_{tw_seq}.pkl'
        if os.path.exists(evaluator_file):
            print(f'num_train:{num_train},tw_seq:{tw_seq} already exists')
            pass
        else:
            print(f'num_train:{num_train},tw_seq:{tw_seq}')

            trial_container = dataset.gen_trials_leave_out(tw_seq = [tw_seq],
                                                        trains = [num_train],
                                                        harmonic_num = harmonic_num,
                                                        ch_used = ch_used)

            model_container = [
                FBCCA(weights_filterbank=weights_filterbank),
                ETRCA(weights_filterbank=weights_filterbank),
                MSETRCA(n_neighbor=2, weights_filterbank = weights_filterbank),
                TDCA(n_component=8, padding_len=5),
                ERESS(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=dataset.srate,
                        ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
                EEPCA(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate),
                ]
            evaluator = BaseEvaluator(dataset_container = [dataset],
                                    model_container = model_container,
                                    trial_container = trial_container,
                                    save_model = False,
                                    disp_processbar = True)

            evaluator.run(n_jobs = 20,eval_train = False)

            evaluator.save(evaluator_file)



# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
evaluator = BaseEvaluator()
evaluator.load('./result_R1/evaluator_tw_trial/evaluator_tw_trial_BETA_3_2.pkl')


# %%
from SSVEPAnalysisToolbox.evaluator import (
    cal_performance_onedataset_individual_diffsiglen,
    cal_confusionmatrix_onedataset_individual_diffsiglen
)
# acc_store shape: (num_models, num_subs, num_tw)
acc_store, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = [2],
                                                                         train_or_test = 'test')



# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA', 'GuHF','MEG']
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
num_models = 6
tw_trial_all = dict()
for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    tw_trial_all[dataset_key] = dict()
    acc_iter = np.zeros((num_models, num_subs, len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_models, num_subs, len(trials[dataset_idx]), len(tw_seqs)))
    for trial_idx, trial in enumerate(trials[dataset_idx]):
        for tw_idx, tw in enumerate(tw_seqs):
            evaluator = BaseEvaluator()
            evaluator.load(f'G:/EPCA\EPCA-R1/result_R1/evaluator_tw_trial/evaluator_tw_trial_{dataset_key}_{trial}_{tw}.pkl')
            trial_container = evaluator.trial_container
            performance_container = evaluator.performance_container
            
            for trialinfo,performance in zip(trial_container,performance_container):
                sub_idx = trialinfo[0].sub_idx[0]
                for model_idx,model_performance in enumerate(performance):
                    Y_test = model_performance.true_label_test
                    Y_pred = model_performance.pred_label_test

                    acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                    itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                  t_comp = 0, N = num_stims, acc = acc)
                    acc_iter[model_idx,sub_idx,trial_idx,tw_idx] = acc
                    itr_iter[model_idx,sub_idx,trial_idx,tw_idx] = itr
    tw_trial_all[dataset_key]['acc'] = acc_iter
    tw_trial_all[dataset_key]['itr'] = itr_iter
    tw_trial_all[dataset_key]['tws'] = tw_seqs
    tw_trial_all[dataset_key]['trials'] = trials[dataset_idx]


# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA', 'GuHF','MEG']
tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
num_models = 6
tw_trial_all = dict()
for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    tw_trial_all[dataset_key] = dict()
    acc_iter = np.zeros((num_models, num_subs, len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_models, num_subs, len(trials[dataset_idx]), len(tw_seqs)))
    for trial_idx, trial in enumerate(trials[dataset_idx]):
        for tw_idx, tw in enumerate(tw_seqs):
            evaluator = BaseEvaluator()
            evaluator.load(f'G:/EPCA\EPCA-R1/result_R1/evaluator_tw_trial/evaluator_tw_trial_{dataset_key}_{trial}_{tw}.pkl')
            trial_container = evaluator.trial_container
            performance_container = evaluator.performance_container
            
            for model_idx in range(num_models):
                for sub_idx in range(num_subs):
                    acc_sub = []
                    itr_sub = []
                    for trialinfo,performance in zip(trial_container,performance_container):
                        model_performance = performance[model_idx]
                        if sub_idx == trialinfo[0].sub_idx[0]:
                            Y_test = model_performance.true_label_test
                            Y_pred = model_performance.pred_label_test

                            acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                            itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                        t_comp = 0, N = num_stims, acc = acc)
                            acc_sub.append(acc)
                            itr_sub.append(itr)

                    acc_iter[model_idx,sub_idx,trial_idx,tw_idx] = np.mean(acc_sub)
                    itr_iter[model_idx,sub_idx,trial_idx,tw_idx] = np.mean(itr_sub)
    tw_trial_all[dataset_key]['acc'] = acc_iter
    tw_trial_all[dataset_key]['itr'] = itr_iter
    tw_trial_all[dataset_key]['tws'] = tw_seqs
    tw_trial_all[dataset_key]['trials'] = trials[dataset_idx]

# %%
# import pickle
# with open('tw_trial_all_true.pkl','wb') as f:
#     pickle.dump(tw_trial_all,f)


# %%
with open('./result_R1/tw_trial_all.pkl','rb') as f:
    tw_trial_all = pickle.load(f)

# %%
# -------------------Benchmark-------------------
acc_benchmark = tw_trial_all['Benchmark']['acc']*100 # num_models * num_subs * num_trials * num_tws
itr_benchmark = tw_trial_all['Benchmark']['itr']

acc_benchmark_mean = np.mean(acc_benchmark,axis=1)
acc_benchmark_se = np.std(acc_benchmark,axis=1)/np.sqrt(acc_benchmark.shape[1])
itr_benchmark_mean = np.mean(itr_benchmark,axis=1)
itr_benchmark_se = np.std(itr_benchmark,axis=1)/np.sqrt(itr_benchmark.shape[1])

# -------------------BETA-------------------
acc_beta = tw_trial_all['BETA']['acc']*100
itr_beta = tw_trial_all['BETA']['itr']

acc_beta_mean = np.mean(acc_beta,axis=1)
acc_beta_se = np.std(acc_beta,axis=1)/np.sqrt(acc_beta.shape[1])
itr_beta_mean = np.mean(itr_beta,axis=1)
itr_beta_se = np.std(itr_beta,axis=1)/np.sqrt(itr_beta.shape[1])

# -------------------eldBETA-------------------
acc_eldbeta = tw_trial_all['eldBETA']['acc']*100
itr_eldbeta = tw_trial_all['eldBETA']['itr']

acc_eldbeta_mean = np.mean(acc_eldbeta,axis=1)
acc_eldbeta_se = np.std(acc_eldbeta,axis=1)/np.sqrt(acc_eldbeta.shape[1])
itr_eldbeta_mean = np.mean(itr_eldbeta,axis=1)
itr_eldbeta_se = np.std(itr_eldbeta,axis=1)/np.sqrt(itr_eldbeta.shape[1])


# -------------------GuHF-------------------
acc_guhf = tw_trial_all['GuHF']['acc']*100
itr_guhf = tw_trial_all['GuHF']['itr']

acc_guhf_mean = np.mean(acc_guhf,axis=1)
acc_guhf_se = np.std(acc_guhf,axis=1)/np.sqrt(acc_guhf.shape[1])
itr_guhf_mean = np.mean(itr_guhf,axis=1)
itr_guhf_se = np.std(itr_guhf,axis=1)/np.sqrt(itr_guhf.shape[1])

# -------------------MEG-------------------
acc_meg = tw_trial_all['MEG']['acc']*100
itr_meg = tw_trial_all['MEG']['itr']

acc_meg_mean = np.mean(acc_meg,axis=1)
acc_meg_se = np.std(acc_meg,axis=1)/np.sqrt(acc_meg.shape[1])
itr_meg_mean = np.mean(itr_meg,axis=1)
itr_meg_se = np.std(itr_meg,axis=1)/np.sqrt(itr_meg.shape[1])


# %%
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
method_IDs = ['FBCCA','eTRCA','ms-eTRCA','TDCA','eRESS','eEPCA']

tws = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]


dataset = 'Benchmark'
acc_dataset = tw_trial_all[dataset]['acc']*100
itr_dataset = tw_trial_all[dataset]['itr']
acc_dataset_mean = np.mean(acc_dataset,axis=1)
acc_dataset_se = np.std(acc_dataset,axis=1)/np.sqrt(acc_dataset.shape[1])
itr_dataset_mean = np.mean(itr_dataset,axis=1)
itr_dataset_se = np.std(itr_dataset,axis=1)/np.sqrt(itr_dataset.shape[1])


trials = 1
# -------------------acc-------------------
ax = axes[0,0]
for i in range(len(method_IDs)):
    ax.plot(tws,acc_dataset_mean[i,trials-1,:],label=method_IDs[i],lw=linewidth,color=colors[i],
            marker=markers[i],markersize=ms[i])
    ax.errorbar(tws,acc_dataset_mean[i,trials-1,:],yerr=acc_dataset_se[i,trials-1,:],c=colors[i],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)
    
# ax.hlines(y=90,xmin=0.15,xmax=2.05,color='k',linestyle='--',lw=linewidth)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
ax.set_ylim(0,100)
ax.set_xlim(0.15,2.05)
ax.set_xticks(tws)
ax.set_yticks([0,20,40,60,80,100])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# -------------------itr-------------------
ax = axes[0,1]
for i in range(len(method_IDs)):
    ax.plot(tws,itr_dataset_mean[i,trials-1,:],label=method_IDs[i],lw=linewidth,color=colors[i],
            marker=markers[i],markersize=ms[i])
    ax.errorbar(tws,itr_dataset_mean[i,trials-1,:],yerr=itr_dataset_se[i,trials-1,:],c=colors[i],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
ax.set_ylim(0,200)
ax.set_xlim(0.15,2.05)
ax.set_xticks(tws)
ax.set_yticks([0,40,80,120,160,200])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)




# dataset = 'GuHF'
# acc_dataset = tw_trial_all[dataset]['acc']*100
# itr_dataset = tw_trial_all[dataset]['itr']
# acc_dataset_mean = np.mean(acc_dataset,axis=1)
# acc_dataset_se = np.std(acc_dataset,axis=1)/np.sqrt(acc_dataset.shape[1])
# itr_dataset_mean = np.mean(itr_dataset,axis=1)
# itr_dataset_se = np.std(itr_dataset,axis=1)/np.sqrt(itr_dataset.shape[1])


trials = 3

# -------------------trials=1, acc-------------------
ax = axes[1,0]
for i in range(len(method_IDs)):
    ax.plot(tws,acc_dataset_mean[i,trials-1,:],label=method_IDs[i],lw=linewidth,color=colors[i],
            marker=markers[i],markersize=ms[i])
    ax.errorbar(tws,acc_dataset_mean[i,trials-1,:],yerr=acc_dataset_se[i,trials-1,:],c=colors[i],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)

# ax.hlines(y=90,xmin=0.15,xmax=2.05,color='k',linestyle='--',lw=linewidth)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
ax.set_ylim(0,100)
ax.set_xlim(0.15,2.05)
ax.set_xticks(tws)
ax.set_yticks([0,20,40,60,80,100])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# -------------------trials=1, itr-------------------
ax = axes[1,1]
for i in range(len(method_IDs)):
    ax.plot(tws,itr_dataset_mean[i,trials-1,:],label=method_IDs[i],lw=linewidth,color=colors[i],
            marker=markers[i],markersize=ms[i])
    ax.errorbar(tws,itr_dataset_mean[i,trials-1,:],yerr=itr_dataset_se[i,trials-1,:],c=colors[i],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'Data length (s)',fontname='Times New Roman')
ax.set_ylim(0,200)
ax.set_xlim(0.15,2.05)
ax.set_xticks(tws)
ax.set_yticks([0,40,80,120,160,200])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()








# %%







# %%
