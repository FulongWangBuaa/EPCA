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

from my_code.datasets.mybetadataset import MyBetaDataset
from my_code.utils.betapreprocess import preprocess as betapreprocess
from my_code.utils.betapreprocess import filterbank as betafilterbank

from my_code.datasets.myeldbetadataset import MyeldBetaDataset
from my_code.utils.eldbetapreprocess import preprocess as eldbetapreprocess
from my_code.utils.eldbetapreprocess import filterbank as eldbetafilterbank

from my_code.datasets.myguhfdataset import MyGuHFDataset
from my_code.utils.guhfpreprocess import preprocess as guhfpreprocess
from my_code.utils.guhfpreprocess import filterbank as guhffilterbank


data_path = r".\datasets\Benchmark"
Benchmark = MyBenchmarkDataset(path=data_path)
Benchmark.regist_preprocess(benchmarkpreprocess)
Benchmark.regist_filterbank(benchmarkfilterbank)

data_path = r".\datasets\BETA"
BETA = MyBetaDataset(path=data_path)
BETA.regist_preprocess(betapreprocess)
BETA.regist_filterbank(betafilterbank)

data_path = r".\datasets\eldBETA"
eldBETA = MyeldBetaDataset(path=data_path)
eldBETA.regist_preprocess(eldbetapreprocess)
eldBETA.regist_filterbank(eldbetafilterbank)

data_path = r".\datasets\GuHF"
GuHF = MyGuHFDataset(path=data_path)
GuHF.regist_preprocess(guhfpreprocess)
GuHF.regist_filterbank(guhffilterbank)

data_path = r".\datasets\OPMMEG"
MEG = MyMEGDataset(path=data_path)
MEG.regist_preprocess(megpreprocess)
MEG.regist_filterbank(megfilterbank)

datasets = {'Benchmark':Benchmark,'BETA':BETA ,'eldBETA':eldBETA,'GuHF':GuHF,'MEG': MEG}

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

num_chs = [2,3,4,5,6,7,8,9]
tw_seqs = [0.5,1]

for dataset_key in datasets.keys():
    dataset = datasets[dataset_key]

    if dataset.block_num > 5:
        num_trains = [1,2,3,4,5]
    else:
        num_trains = np.arange(1, dataset.block_num)

    for num_ch in num_chs:
        for num_trial in num_trains:
            print('dataset:',dataset_key,'num_ch:',num_ch,'num_trial:',num_trial)
            evaluator_file = f'./result_R1/evaluator_ch_trial/evaluator_ch_trial_{dataset_key}_{num_ch}_{num_trial}.pkl'
            if os.path.exists(evaluator_file):
                print(f'{evaluator_file} already exists')
                continue
            else:
                if dataset_key == 'Benchmark':
                    ch_used = benchmark_suggested_ch(num_ch)
                    weights_filterbank = benchmark_weights_filterbank()
                    harmonic_num = 5
                elif dataset_key == 'BETA':
                    ch_used = beta_suggested_ch(num_ch)
                    weights_filterbank = beta_weights_filterbank()
                    harmonic_num = 5
                elif dataset_key == 'eldBETA':
                    ch_used = eldbeta_suggested_ch(num_ch)
                    weights_filterbank = eldbeta_weights_filterbank()
                    harmonic_num = 5
                elif dataset_key == 'GuHF':
                    ch_used = guhf_suggested_ch(num_ch)
                    weights_filterbank = guhff_weights_filterbank()
                    harmonic_num = 5
                elif dataset_key == 'MEG':
                    ch_used = meg_suggested_ch(num_ch)
                    weights_filterbank = meg_weights_filterbank()
                    harmonic_num = 4
                stim_freqs = dataset.stim_info['freqs']
                srate=dataset.srate

                trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seqs,
                                                            trains = [num_trial],
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
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA','GuHF','MEG']
tw_seqs = [0.5,1]
num_chs = [2,3,4,5,6,7,8,9]

trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
num_models = 6
ch_trial_all = dict()

for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    ch_trial_all[dataset_key] = dict()
    acc_iter = np.zeros((num_models, num_subs, len(num_chs), len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_models, num_subs, len(num_chs), len(trials[dataset_idx]), len(tw_seqs)))

    for ch_idx, ch in enumerate(num_chs):
        for trial_idx, trial in enumerate(trials[dataset_idx]):
        
            evaluator = BaseEvaluator()
            evaluator.load(f'G:/EPCA\EPCA-R1/result_R1/evaluator_ch_trial/evaluator_ch_trial_{dataset_key}_{ch}_{trial}.pkl')
            trial_container = evaluator.trial_container
            performance_container = evaluator.performance_container

            for trialinfo,performance in zip(trial_container,performance_container):
                sub_idx = trialinfo[0].sub_idx[0]
                tw = trialinfo[0].tw
                tw_idx = tw_seqs.index(tw)
                for model_idx,model_performance in enumerate(performance):
                    Y_test = model_performance.true_label_test
                    Y_pred = model_performance.pred_label_test

                    acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                    itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                  t_comp = 0, N = num_stims, acc = acc)
                    acc_iter[model_idx,sub_idx,ch_idx,trial_idx,tw_idx] = acc
                    itr_iter[model_idx,sub_idx,ch_idx,trial_idx,tw_idx] = itr

    ch_trial_all[dataset_key]['acc'] = acc_iter
    ch_trial_all[dataset_key]['itr'] = itr_iter
    ch_trial_all[dataset_key]['chs'] = num_chs
    ch_trial_all[dataset_key]['trials'] = trials[dataset_idx]
    ch_trial_all[dataset_key]['tws'] = tw_seqs



# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA','GuHF','MEG']
tw_seqs = [0.5,1]
num_chs = [2,3,4,5,6,7,8,9]

trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
num_models = 6
ch_trial_all = dict()

for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    ch_trial_all[dataset_key] = dict()
    acc_iter = np.zeros((num_models, num_subs, len(num_chs), len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_models, num_subs, len(num_chs), len(trials[dataset_idx]), len(tw_seqs)))

    for ch_idx, ch in enumerate(num_chs):
        for trial_idx, trial in enumerate(trials[dataset_idx]):
        
            evaluator = BaseEvaluator()
            evaluator.load(f'G:/EPCA\EPCA-R1/result_R1/evaluator_ch_trial/evaluator_ch_trial_{dataset_key}_{ch}_{trial}.pkl')
            trial_container = evaluator.trial_container
            performance_container = evaluator.performance_container

            for model_idx in range(num_models):
                for sub_idx in range(num_subs):
                    for tw_idx in range(len(tw_seqs)):
                        acc_tw = []
                        itr_tw = []
                        tw = tw_seqs[tw_idx]
                        for trialinfo,performance in zip(trial_container,performance_container):
                            model_performance = performance[model_idx]
                            if sub_idx == trialinfo[0].sub_idx[0] and tw == trialinfo[0].tw:
                                Y_test = model_performance.true_label_test
                                Y_pred = model_performance.pred_label_test

                                acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                                itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                            t_comp = 0, N = num_stims, acc = acc)
                                acc_tw.append(acc)
                                itr_tw.append(itr)

                        acc_iter[model_idx,sub_idx,ch_idx,trial_idx,tw_idx] = np.mean(acc_tw)
                        itr_iter[model_idx,sub_idx,ch_idx,trial_idx,tw_idx] = np.mean(itr_tw)

    ch_trial_all[dataset_key]['acc'] = acc_iter
    ch_trial_all[dataset_key]['itr'] = itr_iter
    ch_trial_all[dataset_key]['chs'] = num_chs
    ch_trial_all[dataset_key]['trials'] = trials[dataset_idx]
    ch_trial_all[dataset_key]['tws'] = tw_seqs


# %%
import pickle
with open('ch_trial_all_true.pkl','wb') as f:
    pickle.dump(ch_trial_all,f)

# %%
with open('./result_R1/ch_trial_all.pkl','rb') as f:
    ch_trial_all = pickle.load(f)


# %%
