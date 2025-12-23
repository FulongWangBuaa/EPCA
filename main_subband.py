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

from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.utils.megpreprocess import preprocess as megpreprocess

from my_code.datasets.mybetadataset import MyBetaDataset
from my_code.utils.betapreprocess import preprocess as betapreprocess

from my_code.datasets.myeldbetadataset import MyeldBetaDataset
from my_code.utils.eldbetapreprocess import preprocess as eldbetapreprocess

from my_code.datasets.myguhfdataset import MyGuHFDataset
from my_code.utils.guhfpreprocess import preprocess as guhfpreprocess

from my_code.utils.myfilterbank import filterbank_1,filterbank_2,filterbank_3,filterbank_4,filterbank_5,filterbank_6


# dataset_keys = ['Benchmark', 'MEG', 'BETA', 'eldBETA', 'GuHF]

dataset_key = 'Benchmark'

if dataset_key == 'Benchmark':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\Benchmark"
    dataset = MyBenchmarkDataset(path=data_path)
    dataset.regist_preprocess(benchmarkpreprocess)
elif dataset_key == 'MEG':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\OPMMEG"
    dataset = MyMEGDataset(path=data_path)
    dataset.regist_preprocess(megpreprocess)
elif dataset_key == 'BETA':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\BETA"
    dataset = MyBetaDataset(path=data_path)
    dataset.regist_preprocess(betapreprocess)
elif dataset_key == 'eldBETA':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\eldBETA"
    dataset = MyeldBetaDataset(path=data_path)
    dataset.regist_preprocess(eldbetapreprocess)
elif dataset_key == 'GuHF':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\GuHF"
    dataset = MyGuHFDataset(path=data_path)
    dataset.regist_preprocess(guhfpreprocess)

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

filterbanks = [filterbank_1,filterbank_2,filterbank_3,filterbank_4,filterbank_5,filterbank_6]

stim_freqs = dataset.stim_info['freqs']
srate=dataset.srate

if dataset.block_num > 5:
    num_trains = [1,2,3,4,5]
else:
    num_trains = np.arange(1, dataset.block_num)
tw_seqs = [0.5,1,1.5,2]

for num_subband in [1,2,3,4,5,6]:
    dataset.regist_filterbank(filterbanks[num_subband-1])
    
    if dataset_key == 'Benchmark':
        ch_used = benchmark_suggested_ch(9)
        weights_filterbank = benchmark_weights_filterbank(num_subbands=num_subband)
        # harmonic_num = 5
    elif dataset_key == 'MEG':
        ch_used = meg_suggested_ch(9)
        weights_filterbank = meg_weights_filterbank(num_subbands=num_subband)
        # harmonic_num = 4
    elif dataset_key == 'BETA':
        ch_used = beta_suggested_ch(9)
        weights_filterbank = beta_weights_filterbank(num_subbands=num_subband)
        # harmonic_num = 5
    elif dataset_key == 'eldBETA':
        ch_used = eldbeta_suggested_ch(9)
        weights_filterbank = eldbeta_weights_filterbank(num_subbands=num_subband)
        # harmonic_num = 5
    elif dataset_key == 'GuHF':
        ch_used = guhf_suggested_ch(9)
        weights_filterbank = guhff_weights_filterbank(num_subbands=num_subband)
        # harmonic_num = 5
    harmonic_num = num_subband

    evaluator_file = f'./result_R1/evaluator_subband/evaluator_subband_{dataset_key}_{num_subband}.pkl'
    if os.path.exists(evaluator_file):
        print(f'num_subband:{num_subband} already exists')
        pass
    else:
        print(f'num_subband:{num_subband}')

        trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seqs,
                                                    trains = num_trains,
                                                    harmonic_num = harmonic_num,
                                                    ch_used = ch_used)

        model_container = [
            # FBCCA(weights_filterbank=weights_filterbank),
            # ETRCA(weights_filterbank=weights_filterbank),
            # MSETRCA(n_neighbor=2, weights_filterbank = weights_filterbank),
            # TDCA(n_component=8, padding_len=5),
            # ERESS(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=dataset.srate,
            #         ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
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
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA', 'GuHF','MEG']
num_subband = [1,2,3,4,5,6]
tw_seqs = [0.5,1,1.5,2]
trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
subband_trial_tw_all = dict()
for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    subband_trial_tw_all[dataset_key] = dict()
    acc_iter = np.zeros((num_subs, len(num_subband), len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_subs, len(num_subband), len(trials[dataset_idx]), len(tw_seqs)))

    for band_idx in range(len(num_subband)):
        num_band = num_subband[band_idx]
        evaluator = BaseEvaluator()
        evaluator.load(f'G:/EPCA/EPCA-R1/result_R1/evaluator_subband/evaluator_subband_{dataset_key}_{num_band}.pkl')
        trial_container = evaluator.trial_container
        performance_container = evaluator.performance_container
        
        for sub_idx in range(num_subs):
            for tw_idx in range(len(tw_seqs)):
                tw = tw_seqs[tw_idx]
                for trial_idx in range(len(trials[dataset_idx])):
                    num_trains = trials[dataset_idx][trial_idx]
                    acc_sub_tw_trial = []
                    itr_sub_tw_trial = []
                    for trialinfo,performance in zip(trial_container,performance_container):
                        model_performance = performance[0]
                        if sub_idx==trialinfo[0].sub_idx[0] and tw==trialinfo[0].tw and num_trains==len(trialinfo[0].block_idx[0]):
                            Y_test = model_performance.true_label_test
                            Y_pred = model_performance.pred_label_test

                            acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                            itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                            t_comp = 0, N = num_stims, acc = acc)
                            
                            acc_sub_tw_trial.append(acc)
                            itr_sub_tw_trial.append(itr)

                    acc_iter[sub_idx,band_idx,trial_idx,tw_idx] = np.mean(acc_sub_tw_trial)
                    itr_iter[sub_idx,band_idx,trial_idx,tw_idx] = np.mean(itr_sub_tw_trial)
    subband_trial_tw_all[dataset_key]['acc'] = acc_iter
    subband_trial_tw_all[dataset_key]['itr'] = itr_iter
    subband_trial_tw_all[dataset_key]['tws'] = tw_seqs
    subband_trial_tw_all[dataset_key]['trials'] = trials[dataset_idx]


# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA', 'GuHF','MEG']
num_subband = [1,2,3,4,5,6]
tw_seqs = [0.5,1,1.5,2]
trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
subband_trial_tw_all = dict()
for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    subband_trial_tw_all[dataset_key] = dict()
    acc_iter = np.zeros((num_subs, len(num_subband), len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_subs, len(num_subband), len(trials[dataset_idx]), len(tw_seqs)))

    for band_idx in range(len(num_subband)):
        num_band = num_subband[band_idx]
        evaluator = BaseEvaluator()
        evaluator.load(f'G:/EPCA/EPCA-R1/result_R1/evaluator_subband/evaluator_subband_{dataset_key}_{num_band}.pkl')
        trial_container = evaluator.trial_container
        performance_container = evaluator.performance_container
        
        for sub_idx in range(num_subs):
            for tw_idx in range(len(tw_seqs)):
                tw = tw_seqs[tw_idx]
                for trial_idx in range(len(trials[dataset_idx])):
                    num_trains = trials[dataset_idx][trial_idx]
                    acc_sub_tw_trial = []
                    itr_sub_tw_trial = []
                    for trialinfo,performance in zip(trial_container,performance_container):
                        model_performance = performance[0]
                        if sub_idx==trialinfo[0].sub_idx[0] and tw==trialinfo[0].tw and num_trains==len(trialinfo[0].block_idx[0]):
                            Y_test = model_performance.true_label_test
                            Y_pred = model_performance.pred_label_test

                            acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                            itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                            t_comp = 0, N = num_stims, acc = acc)
                            
                            acc_sub_tw_trial.append(acc)
                            itr_sub_tw_trial.append(itr)

                    acc_iter[sub_idx,band_idx,trial_idx,tw_idx] = np.mean(acc_sub_tw_trial)
                    itr_iter[sub_idx,band_idx,trial_idx,tw_idx] = np.mean(itr_sub_tw_trial)
    subband_trial_tw_all[dataset_key]['acc'] = acc_iter
    subband_trial_tw_all[dataset_key]['itr'] = itr_iter
    subband_trial_tw_all[dataset_key]['tws'] = tw_seqs
    subband_trial_tw_all[dataset_key]['trials'] = trials[dataset_idx]

# %%
# import pickle
# with open('subband_trial_tw_all_true.pkl','wb') as f:
#     pickle.dump(subband_trial_tw_all,f)




# %%
