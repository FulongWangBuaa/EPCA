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


dataset_key = 'Benchmark'

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

tw = [1]
num_trains = [1]
num_ch = 9

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

trial_container = dataset.gen_trials_leave_out(tw_seq = tw,
                                            trains = num_trains,
                                            harmonic_num = harmonic_num,
                                            ch_used = ch_used)

models = [
    FBCCA(weights_filterbank=weights_filterbank),
    ETRCA(weights_filterbank=weights_filterbank),
    MSETRCA(n_neighbor=2, weights_filterbank = weights_filterbank),
    TDCA(n_component=8, padding_len=5),
    ERESS(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=dataset.srate,
            ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
    EEPCA(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate),
]

# %%
import time
methods_name = ['FBCCA','ETRCA','MSETRCA','TDCA','ERESS','EEPCA']
for i in range(len(models)):
    tic = time.time()
    model_container = [models[i]]
    evaluator = BaseEvaluator(dataset_container = [dataset],
                            model_container = model_container,
                            trial_container = trial_container,
                            save_model = False,
                            disp_processbar = True)

    evaluator.run(n_jobs = 20,eval_train = False)
    
    save_file = f'./result_R1/computational_complexity/evaluator_{methods_name[i]}.pkl'
    evaluator.save(save_file)

# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
tw  = 1
trial = 1
dataset_key = 'eldBETA'
evaluator = BaseEvaluator()
evaluator.load(f'G:/EPCA/EPCA-R1/result_R1/evaluator_tw_trial/evaluator_tw_trial_{dataset_key}_{trial}_{tw}.pkl')

# %%
trial_containers = evaluator.trial_container
performance_containers = evaluator.performance_container

methods_name = ['FBCCA','ETRCA','MSETRCA','TDCA','ERESS','EEPCA']
if dataset_key == 'Benchmark':
    num_subs = 35
    num_blocks = 6
elif dataset_key == 'BETA':
    num_subs = 70
    num_blocks = 3
elif dataset_key == 'eldBETA':
    num_subs = 100
    num_blocks = 7
elif dataset_key == 'GuHF':
    num_subs = 30
    num_blocks = 6
elif dataset_key == 'MEG':
    num_subs = 13
    num_blocks = 6

train_time = np.zeros((num_subs, 6, num_blocks)) # num_sub, num_method, num_trials
train_trials = np.zeros((num_subs, 6, num_blocks)) # num_sub, num_method, num_trials
test_time = np.zeros((num_subs, 6, num_blocks)) # num_sub, num_method, num_trials
test_trials = np.zeros((num_subs, 6, num_blocks)) # num_sub, num_method, num_trials

for trial_container,performance_container in zip(trial_containers, performance_containers):
    for model_idx in range(6):
        performance_container_nodel = performance_container[model_idx]
        sub_idx = trial_container[0].sub_idx[0]

        train_block_idx = trial_container[0].block_idx[0]
        test_block_idx = trial_container[1].block_idx[0]

        train_trial = len(trial_container[0].trial_idx[0]) * len(train_block_idx)
        test_trial = len(trial_container[1].trial_idx[0]) * len(test_block_idx)
        
        train_time[sub_idx,model_idx,train_block_idx[0]] = performance_container[model_idx].train_time[0]
        train_trials[sub_idx,model_idx,train_block_idx[0]] = train_trial

        test_time[sub_idx,model_idx,train_block_idx[0]] = performance_container[model_idx].test_time_test[0]
        test_trials[sub_idx,model_idx,train_block_idx[0]] = test_trial

# %%
mean_train_time = np.sum(np.mean(train_time,axis=2),axis=0)

mean_test_time = np.mean(test_time,axis=(0,2))/test_trials[0,0,0]


# %%
