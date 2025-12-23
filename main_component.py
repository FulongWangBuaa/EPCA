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

dataset_key = 'Benchmark'

if dataset_key == 'Benchmark':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\Benchmark"
    dataset = MyBenchmarkDataset(path=data_path)
    dataset.regist_preprocess(benchmarkpreprocess)
    dataset.regist_filterbank(benchmarkfilterbank)
elif dataset_key == 'MEG':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\OPMMEG"
    dataset = MyMEGDataset(path=data_path)
    dataset.regist_preprocess(megpreprocess)
    dataset.regist_filterbank(megfilterbank)
elif dataset_key == 'BETA':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\BETA"
    dataset = MyBetaDataset(path=data_path)
    dataset.regist_preprocess(betapreprocess)
    dataset.regist_filterbank(betafilterbank)
elif dataset_key == 'eldBETA':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\eldBETA"
    dataset = MyeldBetaDataset(path=data_path)
    dataset.regist_preprocess(eldbetapreprocess)
    dataset.regist_filterbank(eldbetafilterbank)
elif dataset_key == 'GuHF':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\GuHF"
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
from my_code.algorithms.epca_n import EPCA,EEPCA
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
num_trains = [5]
# tw_seqs = [0.5,1,1.5,2]
tw_seqs = [1]

for num_component in [1]:
    for trains in num_trains:
        for tw in tw_seqs:
            evaluator_file = f'./result_R1/evaluator_component/evaluator_component_{dataset_key}_{num_component}_{trains}_{tw}.pkl'
            if os.path.exists(evaluator_file):
                print(f'{dataset_key} num_component:{num_component} trains:{trains} tw:{tw} already exists')
                pass
            else:
                print(f'{dataset_key} num_component:{num_component} trains:{trains} tw:{tw}')

                trial_container = dataset.gen_trials_leave_out(tw_seq = [tw],
                                                            trains = [trains],
                                                            harmonic_num = harmonic_num,
                                                            ch_used = ch_used)

                model_container = [
                    # FBCCA(weights_filterbank=weights_filterbank),
                    # ETRCA(weights_filterbank=weights_filterbank),
                    # MSETRCA(n_neighbor=2, weights_filterbank = weights_filterbank),
                    # TDCA(n_component=8, padding_len=5),
                    # ERESS(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=dataset.srate,
                    #         ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
                    EPCA(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate,n_component=num_component),
                    EEPCA(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate,n_component=num_component),
                    ]
                evaluator = BaseEvaluator(dataset_container = [dataset],
                                        model_container = model_container,
                                        trial_container = trial_container,
                                        save_model = False,
                                        disp_processbar = True)

                evaluator.run(n_jobs = 20,eval_train = False)

                evaluator.save(evaluator_file)

# %%
from SSVEPAnalysisToolbox.evaluator import (
    cal_performance_onedataset_individual_diffsiglen,
    cal_confusionmatrix_onedataset_individual_diffsiglen
)
# acc_store shape: (num_models, num_subs, num_tw)
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
evaluator = BaseEvaluator()
evaluator.load(r"D:\科研\代码\工作\5、EPCA\EPCA-R1\result_R1\evaluator_component\evaluator_component_Benchmark_1_5_1.pkl")
acc_store1, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = [1],
                                                                         train_or_test = 'test')

evaluator = BaseEvaluator()
evaluator.load(r"D:\科研\代码\工作\5、EPCA\EPCA-R1\result_R1\evaluator_component\evaluator_component_Benchmark_2_5_1.pkl")
acc_store2, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = [1],
                                                                         train_or_test = 'test')

evaluator = BaseEvaluator()
evaluator.load(r"D:\科研\代码\工作\5、EPCA\EPCA-R1\result_R1\evaluator_component\evaluator_component_Benchmark_3_5_1.pkl")
acc_store3, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = [1],
                                                                         train_or_test = 'test')

# %%
from my_code.evaluator.MyBaseEvaluator import BaseEvaluator
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
from tqdm import tqdm
datasets = ['Benchmark','BETA','eldBETA', 'GuHF','MEG']
components = [1,2,3,4,5,6]
tw_seqs = [0.5,1,1.5,2]
trials = [[1,2,3,4,5],[1,2,3],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
subs = [35, 70, 100, 30, 13]
stims = [40, 40, 9, 24, 9]
models = ['EPCA','EEPCA']
component_trial_tw_all = dict()
for dataset_idx, dataset_key in enumerate(tqdm(datasets)):
    num_stims = stims[dataset_idx]
    num_subs = subs[dataset_idx]

    component_trial_tw_all[dataset_key] = dict()
    acc_iter = np.zeros((num_subs, len(models), len(components), len(trials[dataset_idx]), len(tw_seqs)))
    itr_iter = np.zeros((num_subs, len(models), len(components), len(trials[dataset_idx]), len(tw_seqs)))

    for component_idx in range(len(components)):
        num_component = components[component_idx]
        for trial_idx in range(len(trials[dataset_idx])):
            num_trains = trials[dataset_idx][trial_idx]
            for tw_idx in range(len(tw_seqs)):
                tw = tw_seqs[tw_idx]

                evaluator = BaseEvaluator()
                evaluator.load(f'G:/EPCA/EPCA-R1/result_R1/evaluator_component/evaluator_component_{dataset_key}_{num_component}_{num_trains}_{tw}.pkl')
                trial_container = evaluator.trial_container
                performance_container = evaluator.performance_container
                for sub_idx in range(num_subs):
                    for model_idx in range(len(models)):
                        acc_sub = []
                        itr_sub = []
                        for trialinfo,performance in zip(trial_container,performance_container):
                            model_performance = performance[model_idx]
                            if sub_idx==trialinfo[0].sub_idx[0]:
                                Y_test = model_performance.true_label_test
                                Y_pred = model_performance.pred_label_test

                                acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
                                itr = cal_itr(tw = tw, t_break = 0.5, t_latency = 0.14,
                                                t_comp = 0, N = num_stims, acc = acc)
                                
                                acc_sub.append(acc)
                                itr_sub.append(itr)
                        acc_iter[sub_idx,model_idx,component_idx,trial_idx,tw_idx] = np.mean(acc_sub)
                        itr_iter[sub_idx,model_idx,component_idx,trial_idx,tw_idx] = np.mean(itr_sub)

    component_trial_tw_all[dataset_key]['acc'] = acc_iter
    component_trial_tw_all[dataset_key]['itr'] = itr_iter
    component_trial_tw_all[dataset_key]['tws'] = tw_seqs
    component_trial_tw_all[dataset_key]['trials'] = trials[dataset_idx]


# %%
import pickle
with open('component_trial_tw_all_true.pkl','wb') as f:
    pickle.dump(component_trial_tw_all,f)


# %%
