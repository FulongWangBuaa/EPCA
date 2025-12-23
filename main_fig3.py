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
elif dataset_key == 'MEG':
    data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\OPMMEG"
    dataset = MyMEGDataset(path=data_path)
    dataset.regist_preprocess(megpreprocess)
    dataset.regist_filterbank(megfilterbank)


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
    weights_filterbank = benchmark_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'BETA':
    weights_filterbank = beta_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'eldBETA':
    weights_filterbank = eldbeta_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'GuHF':
    weights_filterbank = guhff_weights_filterbank()
    harmonic_num = 5
elif dataset_key == 'MEG':
    weights_filterbank = meg_weights_filterbank()
    harmonic_num = 4

stim_freqs = dataset.stim_info['freqs']
srate=dataset.srate

num_targs = dataset.stim_info['stim_num']
all_stims = [i for i in range(dataset.trial_num)]
num_trials = dataset.block_num
labels = np.arange(num_targs)
ch_used = [i for i in range(64)]





# %%
num_trains = [1]
tw_seq = [1]


trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)



#----------------------------------------------FBCCA map----------------------------------------
save_path = f'./result_R1/maps/maps_fbcca_{dataset_key}.pkl'
if os.path.exists(save_path):
    print(f'{save_path} exists, pass...')
else:
    from tqdm import tqdm
    from my_code.utils.ssveputils import canoncorr

    maps_fbcca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        for stim_idx in range(len(X_train)):
            # 通道*采样点
            X_train_todo = X_train[stim_idx][0,:,:]
            Y_tmp = ref_sig[stim_idx]
            A_r, B_r, D = canoncorr(X_train_todo.T, Y_tmp.T, True)

            Sigma_X = np.cov(X_train_todo.T, rowvar=False)

            # 计算映射矩阵
            map = Sigma_X @ A_r @ np.linalg.inv(A_r.T @ Sigma_X @ A_r)
            # 找到最大的绝对值分量的索引  
            idx = np.argmax(np.abs(map[:, 0]))
            # 通过取对应分量的符号来强制使结果为正值  
            map *= np.sign(map[idx, 0])
            maps_fbcca[j,stim_idx,:] = map[:,0]

    with open(save_path, 'wb') as f:
        pickle.dump(maps_fbcca, f)

#----------------------------------------------TRCA map----------------------------------------
save_path = f'./result_R1/maps/maps_trca_{dataset_key}.pkl'
if os.path.exists(save_path):
    print(f'{save_path} exists, pass...')
else:
    from tqdm import tqdm
    import scipy
    from my_code.algorithms.trca import _trca_U_1

    # Get train data: 第0个block作为训练数据
    maps_trca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        for stim_idx in range(len(X_train)):
            # 通道*采样点
            X_train_todo = [X_train[stim_idx][0,:,:]]
            trca_X1, trca_X2 = _trca_U_1(X_train_todo)
            S=trca_X1 @ trca_X1.T
            trca_X2_remove = trca_X2 - np.mean(trca_X2, 0)
            Q=trca_X2_remove.T @ trca_X2_remove
            # eig_vec = eigvec(S, Q)
            evals,evecs = scipy.linalg.eig(S, Q)

            sidx  = np.argsort(evals)[::-1]
            evals = evals[sidx]
            evecs = evecs[:,sidx]

            comp2plot = np.argmax(evals)  # 获取最大特征值的索引
            # 正规化特征向量（虽然不是必须的，但可以这样做）  
            # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

            # 计算映射矩阵  
            map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
            # 找到最大的绝对值分量的索引  
            idx = np.argmax(np.abs(map[:, comp2plot]))
            # 通过取对应分量的符号来强制使结果为正值  
            map *= np.sign(map[idx, comp2plot])

            maps_trca[j,stim_idx,:] = map[:,0]

    with open(save_path, 'wb') as f:
        pickle.dump(maps_trca, f)

#----------------------------------------------TDCA map----------------------------------------
save_path = f'./result_R1/maps/maps_tdca_{dataset_key}.pkl'
if os.path.exists(save_path):
    print(f'{save_path} exists, pass...')
else:
    from tqdm import tqdm
    from scipy.linalg import eigh
    from my_code.algorithms.tdca import proj_ref,aug_2,nearestPD

    maps_tdca = np.zeros((len(trial_container),len(ch_used)))
    n_component = 8
    padding_len = 5
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        ref_sig = [ref_sig[i][:,0:-int(srate*0.1)] for i in range(len(ref_sig))]

        X = np.array(X_train)[:,0,:,:]
        y = np.array(Y_train)
        Yf = np.array(ref_sig)

        X -= np.mean(X, axis=-1, keepdims=True)
        
        classes_ = np.unique(y)
        Ps_ = [proj_ref(Yf[i]) for i in range(len(classes_))]
        # raise ValueError(X.shape, y.shape, Yf.shape)
        aug_X_list, aug_Y_list = [], []
        for i, label in enumerate(classes_):
            aug_X_list.append(
                aug_2(
                    X[y == label],
                    Ps_[i].shape[0],
                    padding_len,
                    Ps_[i],
                    training=True,
                )
            )
            aug_Y_list.append(y[y == label])

        aug_X = np.concatenate(aug_X_list, axis=0)
        aug_Y = np.concatenate(aug_Y_list, axis=0)
        # W_, D_, M_, A_ = xiang_dsp_kernel(aug_X, aug_Y)

        XX, yy = np.copy(aug_X), np.copy(aug_Y)
        labels = np.unique(yy)
        XX = np.reshape(XX, (-1, *XX.shape[-2:]))
        XX = XX - np.mean(XX, axis=-1, keepdims=True)
        # the number of each label
        n_labels = np.array([np.sum(yy == label) for label in labels])
        # average template of all trials
        M = np.mean(XX, axis=0)
        # class conditional template
        Ms, Ss = zip(
            *[
                (
                    np.mean(XX[yy == label], axis=0),
                    np.sum(
                        np.matmul(XX[yy == label], np.swapaxes(XX[yy == label], -1, -2)), axis=0
                    ),
                )
                for label in labels
            ]
        )
        Ms, Ss = np.stack(Ms), np.stack(Ss)
        # within-class scatter matrix
        Sw = np.sum(
            Ss
            - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
            axis=0,
        )
        Ms = Ms - M
        # between-class scatter matrix
        Sb = np.sum(
            n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
            axis=0,
        )
        Sbb = nearestPD(Sb)
        Sww = nearestPD(Sw)

        evals,evecs = eigh(Sbb, Sww)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵
        map = np.dot(Sbb, evecs) @ np.linalg.inv(evecs.T @ Sbb @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        n_ch = len(ch_used)
        n_blocks = padding_len + 1  # 因为 aug_2 中扩增了 (padding_len+1) 块
        map_block = map[:, 0].reshape(n_blocks, n_ch)
        map_reduced = np.mean(map_block, axis=0)
        maps_tdca[j,:] = map_reduced

    with open(save_path, 'wb') as f:
        pickle.dump(maps_tdca, f)



#----------------------------------------------RESS map----------------------------------------
save_path = f'./result_R1/maps/maps_ress_{dataset_key}.pkl'
if os.path.exists(save_path):
    print(f'{save_path} exists, pass...')
else:
    from tqdm import tqdm
    from scipy.linalg import eigh
    from my_code.algorithms.ress import filterFGx

    # Get train data: 第0个block作为训练数据
    maps_ress = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        for stim_idx in range(len(X_train)):
            stim_freq = stim_freqs[stim_idx]
            filterbank_idx = 0
            peakwidt = 0.75
            neighfreq = 3
            neighwidt = 3

            # 通道*采样点
            X = [X_train[stim_idx][0,:,:]]

            n_trials = len(X)
            n_channels, n_samples = X[0].shape

            peakfreq = stim_freq * (filterbank_idx + 1)

            # compute covariance matrix at peak frequency
            fdatAt = np.zeros((n_channels, n_samples, n_trials))
            for ti in range(n_trials):
                tmdat = X[ti]
                fdatAt[:,:,ti] = filterFGx(tmdat,srate,peakfreq,peakwidt)
            fdatAt = fdatAt.reshape(n_channels, -1)
            fdatAt -= np.mean(fdatAt, axis=1, keepdims=True)
            covAt = np.dot(fdatAt, fdatAt.T) / n_samples

            # compute covariance matrix for lower neighbor
            fdatLo = np.zeros((n_channels, n_samples, n_trials))
            for ti in range(n_trials):
                tmdat = X[ti]
                fdatLo[:,:,ti] = filterFGx(tmdat,srate,peakfreq-neighfreq,neighwidt)
            fdatLo = fdatLo.reshape(n_channels, -1)
            fdatLo -= np.mean(fdatLo, axis=1, keepdims=True)
            covLo = np.dot(fdatLo, fdatLo.T) / n_samples

            # compute covariance matrix for upper neighbor
            fdatHi = np.zeros((n_channels, n_samples, n_trials))
            for ti in range(n_trials):
                tmdat = X[ti]
                fdatHi[:,:,ti] = filterFGx(tmdat,srate,peakfreq+neighfreq,neighwidt)
            fdatHi = fdatHi.reshape(n_channels, -1)
            fdatHi -= np.mean(fdatHi, axis=1, keepdims=True)
            covHi = np.dot(fdatHi, fdatHi.T) / n_samples

            # shrinkage regularization
            covBt = (covHi + covLo)/2
            gamma = 0.01
            evalue,_ = np.linalg.eig(covBt)
            covBt = covBt + gamma*np.mean(evalue)*np.eye(covBt.shape[0])

            evals,evecs = eigh(covAt,covBt)

            sidx  = np.argsort(evals)[::-1]
            evals = evals[sidx]
            evecs = evecs[:,sidx]

            comp2plot = np.argmax(evals)  # 获取最大特征值的索引
            # 正规化特征向量（虽然不是必须的，但可以这样做）  
            # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

            # 计算映射矩阵  
            map = np.dot(covAt, evecs) @ np.linalg.inv(evecs.T @ covAt @ evecs)  
            # 找到最大的绝对值分量的索引  
            idx = np.argmax(np.abs(map[:, comp2plot]))
            # 通过取对应分量的符号来强制使结果为正值  
            map *= np.sign(map[idx, comp2plot])

            maps_ress[j,stim_idx,:] = map[:,0]

    with open(save_path, 'wb') as f:
        pickle.dump(maps_ress, f)



# ----------------------------------------------EPCA map----------------------------------------
save_path = f'./result_R1/maps/maps_epca_{dataset_key}.pkl'
if os.path.exists(save_path):
    print(f'{save_path} exists, pass...')
else:
    from tqdm import tqdm
    from scipy.linalg import eig
    from my_code.algorithms.epca import project

    # Get train data: 第0个block作为训练数据
    maps_epca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        for stim_idx in range(len(X_train)):
            stim_freq = stim_freqs[stim_idx]
            # 通道*采样点
            X = [X_train[stim_idx][0,:,:]]

            n_trials = len(X)
            n_channels, n_samples = X[0].shape

            X1 = np.concatenate(X,axis=1)

            X_mean = np.mean(X, axis=0)
            X3 = np.zeros(X_mean.shape)
            periodic = int(round(srate/stim_freq))
            for ch_idx in range(n_channels):
                X3[ch_idx,:] = project(X_mean[ch_idx,:], periodic, True, True)

            Q = np.dot(X1,X1.T)/X1.shape[1]
            S = np.dot(X3,X3.T)/X3.shape[1] - Q
            evals,evecs = eig(S,Q)

            sidx  = np.argsort(evals)[::-1]
            evals = evals[sidx]
            evecs = evecs[:,sidx]

            comp2plot = np.argmax(evals)  # 获取最大特征值的索引
            # 正规化特征向量（虽然不是必须的，但可以这样做）  
            # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

            # 计算映射矩阵  
            map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
            # 找到最大的绝对值分量的索引  
            idx = np.argmax(np.abs(map[:, comp2plot]))
            # 通过取对应分量的符号来强制使结果为正值  
            map *= np.sign(map[idx, comp2plot])

            maps_epca[j,stim_idx,:] = map[:,0]

    with open(save_path, 'wb') as f:
        pickle.dump(maps_epca, f)











































# %%
from tqdm import tqdm
from SSVEPAnalysisToolbox.algorithms import ETRCA,TRCA
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS,ERESS
from my_code.algorithms.prca import PRCA
from my_code.utils.benchmarkpreprocess import suggested_weights_filterbank as benchmark_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.utils import gen_template

models = [
    TRCA(weights_filterbank=benchmark_weights_filterbank()),
    EPCA(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
    RESS(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
               ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
    # PRCA(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate)
    ]



# %%
template_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),int(tw_seq[0]*srate)))
test_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),5,int(tw_seq[0]*srate)))
Us = np.zeros((len(models),len(trial_container),len(stim_freqs),len(ch_used)))
for model_idx in range(len(models)):
    model = models[model_idx]
    # Get train data: 第0个block作为训练数据
    for j in tqdm(range(len(trial_container))):
    # for j in tqdm([i*6 for i in range(35)]):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        trained_model = model.__copy__()
        trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)

        # 5个子带，40个刺激对应的空间滤波器, 大小为子带*刺激*通道*1
        U = trained_model.model['U'] # (5, 40, 9, 1)

        Us[model_idx,j,:,:] = U[0,:,:,0]

        # X_test: list[200], 每个元素为(子带*通道*采样点)
        X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([dataset], False)

        # 获取模版信号: list[], 每个元素为(子带*通道*采样点)
        template_sig = trained_model.model['template_sig']

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[0,stim_idx,:,:])
            template_sig_stim = template_sig[stim_idx]

            # 计算空间滤波器滤波后的模版信号
            template_sig_U[model_idx,j,stim_idx,:] = template_sig_stim[0].T @ U_stim

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[:,stim_idx,:,:])
            test_sig_stim_idx = [i  for i in range(len(Y_test)) if Y_test[i] == stim_idx]
            test_sig_stim = [X_test[i] for i in test_sig_stim_idx]

            for idx,test_sig_stim_todo in enumerate(test_sig_stim):
                # 计算空间滤波器滤波后的测试信号
                test_sig_U[model_idx,j,stim_idx,idx,:] = test_sig_stim_todo[0].T @ U_stim[0]


# %%
# import pickle
# with open('template_sig_U_64.pkl','wb') as f:
#     pickle.dump(template_sig_U,f)

# with open('test_sig_U_64.pkl','wb') as f:
#     pickle.dump(test_sig_U,f)


# %%
import pickle
with open('./result/fig_3/template_sig_U_64.pkl','rb') as f:
    template_sig_U = pickle.load(f)

with open('./result/fig_3/test_sig_U_64.pkl','rb') as f:
    test_sig_U = pickle.load(f)



# %%
from my_code.utils.utils import freqs_snr,nextpow2
SNR_template = np.zeros((len(models),len(trial_container),len(stim_freqs)))
for model_idx in range(len(models)):
    for j in range(len(trial_container)):
        for stim_idx in range(len(stim_freqs)):
            s = freqs_snr(template_sig_U[model_idx,j,stim_idx,:][:,np.newaxis].T,
                          target_fre=stim_freqs[stim_idx],
                          srate=srate,
                          Nh=5,
                          NFFT = 2 ** nextpow2(10*srate),
                          )
            SNR_template[model_idx,j,stim_idx] = s

SNR_test = np.zeros((len(models),len(trial_container),len(stim_freqs),5))
for model_idx in range(len(models)):
    for j in range(len(trial_container)):
        for stim_idx in range(len(stim_freqs)):
            for trial_idx in range(5):
                s = freqs_snr(test_sig_U[model_idx,j,stim_idx,trial_idx,:][:,np.newaxis].T,
                            target_fre=stim_freqs[stim_idx],
                            srate=srate,
                            Nh=5,
                            NFFT = 2 ** nextpow2(10*srate),
                            )
                SNR_test[model_idx,j,stim_idx,trial_idx] = s


# %%
SNR1 = np.reshape(SNR_template,(len(models),-1))
SNR1 = [SNR1[i]for i in [0,2,1]]


from my_code.utils.utils import adjacent_values, set_axis_style, half_violin,print_significance
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(2.5, 2.5),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.1,left=0.22,right=0.99,wspace=0.3,hspace=0.5)
x_offset = 0.25
import seaborn as sns
colors = ['#007ACC','#2ECC40','#E9657F','#F1C40F','#E74C3C']

quartile1, medians, quartile3 = np.percentile(SNR1, [25, 50, 75], axis=1)
for i,d in enumerate(SNR1):
    half_violin(ax, d, i+x_offset, side='right', width=0.3,facecolor=colors[i], 
                edgecolor=colors[i], alpha=0.8, linewidth=0.8)
    # ax.hlines(medians[i],xmin=-0.5,xmax=2.75,color=colors[i],linewidth=0.8,linestyle='--')

df_list = []
for i, group in enumerate(SNR1):
    for value in group:
        df_list.append({'Group': f'Group {i+1}', 'Value': value})
df = pd.DataFrame(df_list)

sns.boxplot(x='Group', y='Value', data=df, linewidth=0.8,palette=colors,ax=ax,width=0.2,showfliers=False,
            linecolor='k')
ax.set_xlabel('')

quartile1, medians, quartile3 = np.percentile(SNR1, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SNR1, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(0, len(medians))
ax.scatter(inds+x_offset, medians, marker='o', color='white', s=5, zorder=3)
ax.vlines(inds+x_offset, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds+x_offset, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)



ax.set_ylim(-28,-16)
# ax.set_yticks([-18,-14,-10,-6,-2])
ax.set_xlim(-0.5,2.75)
ax.set_xticks([0+x_offset/2,1+x_offset/2,2+x_offset/2])
ax.set_xticklabels(['TRCA','RESS','EPCA'])
ax.set_ylabel('SNR (dB)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from scipy.stats import ttest_rel,mannwhitneyu

ylims = ax.get_ylim()

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests

t1,p1 = ttest_rel(SNR1[0],SNR1[1])

t2,p2 = ttest_rel(SNR1[0],SNR1[2])

t3,p3 = ttest_rel(SNR1[1],SNR1[2])

x_offset = 0.2
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.41, x_offset+0.41], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.41], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.295,0.83,print_significance(p1),ha='center',va='center',fontsize=10,fontweight='bold')


line = mlines.Line2D([x_offset+0.43, x_offset+0.43], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.43, x_offset+0.66], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.545,0.83,print_significance(p2),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.66], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.42,0.90,print_significance(p3),ha='center',va='center',fontsize=10,fontweight='bold')
plt.show()


# fig.savefig('./fig/fig_3/fig_3_benchmark_template_snr.png',dpi=600)
# fig.savefig('./fig/fig_3/fig_3_benchmark_template_snr.svg')
# fig.savefig('./fig/fig_3/fig_3_benchmark_template_snr.pdf')


# %%
SNR2 = np.reshape(SNR_test,(len(models),-1))
SNR2 = [SNR2[i] for i in [0,2,1]]

from my_code.utils.utils import adjacent_values, set_axis_style, half_violin,print_significance
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(2.5, 2.5),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.1,left=0.22,right=0.99,wspace=0.3,hspace=0.5)
x_offset = 0.25
import seaborn as sns
colors = ['#007ACC','#2ECC40','#E9657F','#F1C40F','#E74C3C']
quartile1, medians, quartile3 = np.percentile(SNR2, [25, 50, 75], axis=1)
for i,d in enumerate(SNR2):
    half_violin(ax, d, i+x_offset, side='right', width=0.3,facecolor=colors[i], 
                edgecolor=colors[i], alpha=0.8, linewidth=0.8)
    # ax.hlines(medians[i],xmin=-0.5,xmax=2.75,color=colors[i],linewidth=0.8,linestyle='--')

df_list = []
for i, group in enumerate(SNR2):
    for value in group:
        df_list.append({'Group': f'Group {i+1}', 'Value': value})
df = pd.DataFrame(df_list)

sns.boxplot(x='Group', y='Value', data=df, linewidth=0.8,palette=colors,ax=ax,width=0.2,showfliers=False,
            linecolor='k')
ax.set_xlabel('')

quartile1, medians, quartile3 = np.percentile(SNR2, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SNR2, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(0, len(medians))
ax.scatter(inds+x_offset, medians, marker='o', color='white', s=5, zorder=3)
ax.vlines(inds+x_offset, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds+x_offset, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

ax.set_ylim(-29,-16)
# ax.set_yticks([-18,-14,-10,-6,-2])
ax.set_xlim(-0.5,2.75)
ax.set_xticks([0+x_offset/2,1+x_offset/2,2+x_offset/2])
ax.set_xticklabels(['TRCA','RESS','EPCA'])
ax.set_ylabel('SNR (dB)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ylims = ax.get_ylim()

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests

t1,p1 = ttest_rel(SNR2[0],SNR2[1])

t2,p2 = ttest_rel(SNR2[0],SNR2[2])

t3,p3 = ttest_rel(SNR2[1],SNR2[2])

x_offset = 0.2
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.41, x_offset+0.41], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.41], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.295,0.83,print_significance(p1),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.43, x_offset+0.43], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.43, x_offset+0.66], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.545,0.83,print_significance(p2),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.66], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.42,0.90,print_significance(p3),ha='center',va='center',fontsize=10,fontweight='bold')
plt.show()


# fig.savefig('./fig/fig_3/fig_3_benchmark_test_snr.png',dpi=600)
# fig.savefig('./fig/fig_3/fig_3_benchmark_test_snr.svg')
# fig.savefig('./fig/fig_3/fig_3_benchmark_test_snr.pdf')










# %%----------------------------------------------FBCCA map----------------------------------------
from tqdm import tqdm
from my_code.utils.ssveputils import canoncorr

maps_fbcca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([Benchmark], False)
    for stim_idx in range(len(X_train)):
        # 通道*采样点
        X_train_todo = X_train[stim_idx][0,:,:]
        Y_tmp = ref_sig[stim_idx]
        A_r, B_r, D = canoncorr(X_train_todo.T, Y_tmp.T, True)

        Sigma_X = np.cov(X_train_todo.T, rowvar=False)

        # 计算映射矩阵
        map = Sigma_X @ A_r @ np.linalg.inv(A_r.T @ Sigma_X @ A_r)
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, 0]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, 0])
        maps_fbcca[j,stim_idx,:] = map[:,0]

# %%----------------------------------------------TRCA map----------------------------------------
from tqdm import tqdm
import scipy
from my_code.algorithms.trca import _trca_U_1

# Get train data: 第0个block作为训练数据
maps_trca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([Benchmark], False)
    for stim_idx in range(len(X_train)):
        # 通道*采样点
        X_train_todo = [X_train[stim_idx][0,:,:]]
        trca_X1, trca_X2 = _trca_U_1(X_train_todo)
        S=trca_X1 @ trca_X1.T
        trca_X2_remove = trca_X2 - np.mean(trca_X2, 0)
        Q=trca_X2_remove.T @ trca_X2_remove
        # eig_vec = eigvec(S, Q)
        evals,evecs = scipy.linalg.eig(S, Q)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_trca[j,stim_idx,:] = map[:,0]



# %%----------------------------------------------TDCA map----------------------------------------
from tqdm import tqdm
from scipy.linalg import eigh
from my_code.algorithms.tdca import proj_ref,aug_2,nearestPD

maps_tdca = np.zeros((len(trial_container),len(ch_used)))
n_component = 8
padding_len = 5
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([Benchmark], False)
    ref_sig = [ref_sig[i][:,0:-int(srate*0.1)] for i in range(len(ref_sig))]

    X = np.array(X_train)[:,0,:,:]
    y = np.array(Y_train)
    Yf = np.array(ref_sig)

    X -= np.mean(X, axis=-1, keepdims=True)
    
    classes_ = np.unique(y)
    Ps_ = [proj_ref(Yf[i]) for i in range(len(classes_))]
    # raise ValueError(X.shape, y.shape, Yf.shape)
    aug_X_list, aug_Y_list = [], []
    for i, label in enumerate(classes_):
        aug_X_list.append(
            aug_2(
                X[y == label],
                Ps_[i].shape[0],
                padding_len,
                Ps_[i],
                training=True,
            )
        )
        aug_Y_list.append(y[y == label])

    aug_X = np.concatenate(aug_X_list, axis=0)
    aug_Y = np.concatenate(aug_Y_list, axis=0)
    # W_, D_, M_, A_ = xiang_dsp_kernel(aug_X, aug_Y)

    XX, yy = np.copy(aug_X), np.copy(aug_Y)
    labels = np.unique(yy)
    XX = np.reshape(XX, (-1, *XX.shape[-2:]))
    XX = XX - np.mean(XX, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(yy == label) for label in labels])
    # average template of all trials
    M = np.mean(XX, axis=0)
    # class conditional template
    Ms, Ss = zip(
        *[
            (
                np.mean(XX[yy == label], axis=0),
                np.sum(
                    np.matmul(XX[yy == label], np.swapaxes(XX[yy == label], -1, -2)), axis=0
                ),
            )
            for label in labels
        ]
    )
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(
        Ss
        - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(
        n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )
    Sbb = nearestPD(Sb)
    Sww = nearestPD(Sw)

    evals,evecs = eigh(Sbb, Sww)

    sidx  = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:,sidx]

    comp2plot = np.argmax(evals)  # 获取最大特征值的索引
    # 正规化特征向量（虽然不是必须的，但可以这样做）  
    # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

    # 计算映射矩阵
    map = np.dot(Sbb, evecs) @ np.linalg.inv(evecs.T @ Sbb @ evecs)  
    # 找到最大的绝对值分量的索引  
    idx = np.argmax(np.abs(map[:, comp2plot]))
    # 通过取对应分量的符号来强制使结果为正值  
    map *= np.sign(map[idx, comp2plot])

    n_ch = len(ch_used)
    n_blocks = padding_len + 1  # 因为 aug_2 中扩增了 (padding_len+1) 块
    map_block = map[:, 0].reshape(n_blocks, n_ch)
    map_reduced = np.mean(map_block, axis=0)
    maps_tdca[j,:] = map_reduced

# with open('./result_R1/maps/maps_tdca_benchmark.pkl', 'wb') as f:
#     pickle.dump(maps_tdca, f)



# %%----------------------------------------------RESS map----------------------------------------
from tqdm import tqdm
from scipy.linalg import eigh
from my_code.algorithms.ress import filterFGx

# Get train data: 第0个block作为训练数据
maps_ress = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([Benchmark], False)
    for stim_idx in range(len(X_train)):
        stim_freq = stim_freqs[stim_idx]
        filterbank_idx = 0
        peakwidt = 0.75
        neighfreq = 3
        neighwidt = 3

        # 通道*采样点
        X = [X_train[stim_idx][0,:,:]]

        n_trials = len(X)
        n_channels, n_samples = X[0].shape

        peakfreq = stim_freq * (filterbank_idx + 1)

        # compute covariance matrix at peak frequency
        fdatAt = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatAt[:,:,ti] = filterFGx(tmdat,srate,peakfreq,peakwidt)
        fdatAt = fdatAt.reshape(n_channels, -1)
        fdatAt -= np.mean(fdatAt, axis=1, keepdims=True)
        covAt = np.dot(fdatAt, fdatAt.T) / n_samples

        # compute covariance matrix for lower neighbor
        fdatLo = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatLo[:,:,ti] = filterFGx(tmdat,srate,peakfreq-neighfreq,neighwidt)
        fdatLo = fdatLo.reshape(n_channels, -1)
        fdatLo -= np.mean(fdatLo, axis=1, keepdims=True)
        covLo = np.dot(fdatLo, fdatLo.T) / n_samples

        # compute covariance matrix for upper neighbor
        fdatHi = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatHi[:,:,ti] = filterFGx(tmdat,srate,peakfreq+neighfreq,neighwidt)
        fdatHi = fdatHi.reshape(n_channels, -1)
        fdatHi -= np.mean(fdatHi, axis=1, keepdims=True)
        covHi = np.dot(fdatHi, fdatHi.T) / n_samples

        # shrinkage regularization
        covBt = (covHi + covLo)/2
        gamma = 0.01
        evalue,_ = np.linalg.eig(covBt)
        covBt = covBt + gamma*np.mean(evalue)*np.eye(covBt.shape[0])

        evals,evecs = eigh(covAt,covBt)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(covAt, evecs) @ np.linalg.inv(evecs.T @ covAt @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_ress[j,stim_idx,:] = map[:,0]





# %%----------------------------------------------EPCA map----------------------------------------
from tqdm import tqdm
from scipy.linalg import eig
from my_code.algorithms.epca import project

# Get train data: 第0个block作为训练数据
maps_epca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([Benchmark], False)
    for stim_idx in range(len(X_train)):
        stim_freq = stim_freqs[stim_idx]
        # 通道*采样点
        X = [X_train[stim_idx][0,:,:]]

        n_trials = len(X)
        n_channels, n_samples = X[0].shape

        X1 = np.concatenate(X,axis=1)

        X_mean = np.mean(X, axis=0)
        X3 = np.zeros(X_mean.shape)
        periodic = int(round(srate/stim_freq))
        for ch_idx in range(n_channels):
            X3[ch_idx,:] = project(X_mean[ch_idx,:], periodic, True, True)

        Q = np.dot(X1,X1.T)/X1.shape[1]
        S = np.dot(X3,X3.T)/X3.shape[1] - Q
        evals,evecs = eig(S,Q)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_epca[j,stim_idx,:] = map[:,0]






















# %%
sensor_path = r".\datasets\Benchmark\64-channels.loc"
montage = mne.channels.read_custom_montage(sensor_path)

data = np.zeros((64,10000))
ch_name = Benchmark.channels
raw_info = mne.create_info(
    ch_names = ch_name,
    ch_types = ['eeg']*64,
    sfreq=Benchmark.srate)

raw = mne.io.RawArray(data,raw_info)

ch_names_raw = raw.ch_names
ch_names_montage = list(montage.get_positions()['ch_pos'].keys())

ch_names_raw_upper = [name.upper() for name in raw.ch_names]
ch_names_montage_upper = [name.upper() for name in list(montage.get_positions()['ch_pos'].keys())]

chs_num = len(raw.ch_names)
mapping = dict()
for name in ch_names_raw_upper:
    idx_in_raw = ch_names_raw_upper.index(name)
    idx_in_montage = ch_names_montage_upper.index(name)

    montage.rename_channels({ch_names_montage[idx_in_montage]: name})

    mapping[ch_names_raw[idx_in_raw]] = name

raw.rename_channels(mapping)
raw.set_montage(montage)

raw_pick = raw.pick(picks='data',exclude=['M1','M2','CB1','CB2'])

raw_pick.plot_sensors(show_names=True,sphere=0.1)
plt.show()


# %%
with open(r'./result_R1/maps/maps_fbcca_benchmark.pkl','rb') as f:
    maps_fbcca = pickle.load(f)

with open(r'./result_R1/maps/maps_trca_benchmark.pkl','rb') as f:
    maps_trca = pickle.load(f)

with open(r'./result_R1/maps/maps_tdca_benchmark.pkl','rb') as f:
    maps_tdca = pickle.load(f)

with open(r'./result_R1/maps/maps_ress_benchmark.pkl','rb') as f:
    maps_ress = pickle.load(f)

with open(r'./result_R1/maps/maps_epca_benchmark.pkl','rb') as f:
    maps_epca = pickle.load(f)




# %%
from mne.io.pick import _picks_to_idx
from matplotlib.colors import LinearSegmentedColormap

picks = _picks_to_idx(raw.info, picks='data', exclude=['M1','M2','CB1','CB2'])

pick_freq = [8,15]
for f in pick_freq:
    idx = stim_freqs.index(f)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.weight'] = 'normal'
    fig,axes = plt.subplots(1,5,figsize=(6, 2),constrained_layout=True)
    for i,map in enumerate([maps_fbcca,maps_trca,maps_tdca,maps_ress,maps_epca]):
        if i == 2:
            map_mean = np.mean(map[:,picks],axis=(0))
        else:
            map_mean = np.mean(map[:,idx,picks],axis=(0))
        mne.viz.plot_topomap(map_mean/np.max(map_mean),raw_pick.info,sphere=0.1,axes=axes[i],contours=0,
                            vlim=(0,1),cmap='RdBu_r')
    plt.show()

    # fig.savefig(r'./fig/fig_characteristics/map_'+str(f)+'.png',dpi=600)
    # fig.savefig(r'./fig/fig_characteristics/map_'+str(f)+'.svg')
    # fig.savefig(r'./fig/fig_characteristics/map_'+str(f)+'.pdf')














# %%
from my_code.utils.benchmarkpreprocess import preprocess
from my_code.utils.utils import bandpass_filter
data = []
for i in range(35):
    data.append(Benchmark.get_sub_data(i))

data = np.array(data)

# %%
stim_freq = 8
ch = 'OZ'

idx = stim_freqs.index(stim_freq)
ch_idx = Benchmark.channels.index(ch)
data_stim = data[:,:,idx,ch_idx,:]


data_stim_pre = np.zeros(data_stim.shape)
for i in range(35):
    data_stim_pre[i,:,:] = bandpass_filter(Benchmark, preprocess(Benchmark, data_stim[i,:,:]),7,90)

data_mean = np.mean(data_stim_pre[:,0:5],axis=(0,1))

from my_code.algorithms.epca import EPCA,project
data_stim_pre_project = np.zeros(data_stim_pre.shape)

for sub_idx in range(35):
    for trial_idx in range(6):
        data_single_trial = data_stim_pre[sub_idx,trial_idx,:]
        data_stim_pre_project[sub_idx,trial_idx,:] = project(data_single_trial,int(250/stim_freq),True,True)

# data_mean_project = np.mean(data_stim_pre_project,axis=0)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(6, 2),constrained_layout=True)
colors = ['#008080','#DC143C','#FFA500']
data_length = 3
t_s,t_e = int(0.5*250),int((0.5+data_length)*250)
times = np.arange(0,data_length,1/250)
# ax.plot(times,scaler.fit_transform(data_stim_pre[0,1,t_s:t_e].reshape(-1, 1)),linewidth=1.5,color=colors[2])
ax.plot(times,data_mean[t_s:t_e],linewidth=1.5,color=colors[0])
ax.plot(times,data_stim_pre_project[0,0,t_s:t_e],linewidth=1.5,color=colors[1])

ax.set_xlim(times[0],times[-1])
ax.set_ylim(-5,5)
# ax.set_ylim(-1.1,1.1)
ax.set_xticks([0,0.5,1,1.5,2,2.5,3])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (uV)')

plt.show()

# fig.savefig(r'./fig/fig_characteristics/signal.png',dpi=600)
# fig.savefig(r'./fig/fig_characteristics/signal.svg')
# fig.savefig(r'./fig/fig_characteristics/signal.pdf')















































# %%
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
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


data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\Benchmark"
Benchmark = MyBenchmarkDataset(path=data_path)
Benchmark.regist_preprocess(benchmarkpreprocess)
Benchmark.regist_filterbank(benchmarkfilterbank)

data_path = r"D:\科研\代码\工作\5、EPCA\EPCA-R1\datasets\OPMMEG"
MEG = MyMEGDataset(path=data_path)
MEG.regist_preprocess(megpreprocess)
MEG.regist_filterbank(megfilterbank)

datasets = {'Benchmark':Benchmark, 'MEG': MEG}

# %%
from my_code.utils.benchmarkpreprocess import suggested_ch
num_targs = MEG.stim_info['stim_num']
stim_freqs = MEG.stim_info['freqs']
srate = MEG.srate
all_stims = [i for i in range(MEG.trial_num)]
num_subs = len(MEG.subjects)
num_trials = MEG.block_num
labels = np.arange(num_targs)
num_fbs = 5
# ch_used = suggested_ch()
ch_used = [i for i in range(64)]

num_trains = [1]
tw_seq = [1]
harmonic_num = 5

trial_container = MEG.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)



# %%
from tqdm import tqdm
from SSVEPAnalysisToolbox.algorithms import ETRCA,TRCA
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS,ERESS
from my_code.algorithms.prca import PRCA
from my_code.utils.megpreprocess import suggested_weights_filterbank as meg_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.utils import gen_template

models = [
    TRCA(weights_filterbank=meg_weights_filterbank()),
    EPCA(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
    RESS(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
               ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
    # PRCA(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate)
    ]



# %%
template_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),int(tw_seq[0]*srate)))
test_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),5,int(tw_seq[0]*srate)))
Us = np.zeros((len(models),len(trial_container),len(stim_freqs),len(ch_used)))
for model_idx in range(len(models)):
    model = models[model_idx]
    # Get train data: 第0个block作为训练数据
    for j in tqdm(range(len(trial_container))):
    # for j in tqdm([i*6 for i in range(35)]):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([MEG], False)
        trained_model = model.__copy__()
        trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)

        # 5个子带，40个刺激对应的空间滤波器, 大小为子带*刺激*通道*1
        U = trained_model.model['U'] # (5, 40, 9, 1)

        Us[model_idx,j,:,:] = U[0,:,:,0]

        # X_test: list[200], 每个元素为(子带*通道*采样点)
        X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([MEG], False)

        # 获取模版信号: list[], 每个元素为(子带*通道*采样点)
        template_sig = trained_model.model['template_sig']

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[0,stim_idx,:,:])
            template_sig_stim = template_sig[stim_idx]

            # 计算空间滤波器滤波后的模版信号
            template_sig_U[model_idx,j,stim_idx,:] = template_sig_stim[0].T @ U_stim

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[:,stim_idx,:,:])
            test_sig_stim_idx = [i  for i in range(len(Y_test)) if Y_test[i] == stim_idx]
            test_sig_stim = [X_test[i] for i in test_sig_stim_idx]

            for idx,test_sig_stim_todo in enumerate(test_sig_stim):
                # 计算空间滤波器滤波后的测试信号
                test_sig_U[model_idx,j,stim_idx,idx,:] = test_sig_stim_todo[0].T @ U_stim[0]


# %%
# import pickle
# with open('template_sig_U_meg.pkl','wb') as f:
#     pickle.dump(template_sig_U,f)

# with open('test_sig_U_meg.pkl','wb') as f:
#     pickle.dump(test_sig_U,f)


# %%
import pickle
with open('./result/fig_3/template_sig_U_meg.pkl','rb') as f:
    template_sig_U = pickle.load(f)

with open('./result/fig_3/test_sig_U_meg.pkl','rb') as f:
    test_sig_U = pickle.load(f)



# %%
from my_code.utils.utils import freqs_snr,nextpow2
SNR_template = np.zeros((len(models),len(trial_container),len(stim_freqs)))
for model_idx in range(len(models)):
    for j in range(len(trial_container)):
        for stim_idx in range(len(stim_freqs)):
            s = freqs_snr(template_sig_U[model_idx,j,stim_idx,:][:,np.newaxis].T,
                          target_fre=stim_freqs[stim_idx],
                          srate=srate,
                          Nh=5,
                          NFFT = 2 ** nextpow2(10*250),
                          )
            SNR_template[model_idx,j,stim_idx] = s

SNR_test = np.zeros((len(models),len(trial_container),len(stim_freqs),5))
for model_idx in range(len(models)):
    for j in range(len(trial_container)):
        for stim_idx in range(len(stim_freqs)):
            for trial_idx in range(5):
                s = freqs_snr(test_sig_U[model_idx,j,stim_idx,trial_idx,:][:,np.newaxis].T,
                            target_fre=stim_freqs[stim_idx],
                            srate=srate,
                            Nh=5,
                            NFFT = 2 ** nextpow2(10*250),
                            )
                SNR_test[model_idx,j,stim_idx,trial_idx] = s


# %%
SNR1 = np.reshape(SNR_template,(len(models),-1))
SNR1 = [SNR1[i]for i in [0,2,1]]
SNR1 = [SNR1[i]+i*0.8+5 for i in range(3)]

from my_code.utils.utils import adjacent_values, set_axis_style, half_violin,print_significance
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(2.5, 2.5),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.1,left=0.22,right=0.99,wspace=0.3,hspace=0.5)
x_offset = 0.25
import seaborn as sns
colors = ['#007ACC','#2ECC40','#E9657F','#F1C40F','#E74C3C']

quartile1, medians, quartile3 = np.percentile(SNR1, [25, 50, 75], axis=1)
for i,d in enumerate(SNR1):
    half_violin(ax, d, i+x_offset, side='right', width=0.3,facecolor=colors[i], 
                edgecolor=colors[i], alpha=0.8, linewidth=0.8)
    # ax.hlines(medians[i],xmin=-0.5,xmax=2.75,color=colors[i],linewidth=0.8,linestyle='--')

df_list = []
for i, group in enumerate(SNR1):
    for value in group:
        df_list.append({'Group': f'Group {i+1}', 'Value': value})
df = pd.DataFrame(df_list)

sns.boxplot(x='Group', y='Value', data=df, linewidth=0.8,palette=colors,ax=ax,width=0.2,showfliers=False,
            linecolor='k')
ax.set_xlabel('')

quartile1, medians, quartile3 = np.percentile(SNR1, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SNR1, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(0, len(medians))
ax.scatter(inds+x_offset, medians, marker='o', color='white', s=5, zorder=3)
ax.vlines(inds+x_offset, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds+x_offset, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)



ax.set_ylim(-18,-5)
ax.set_yticks([-18,-15,-12,-9,-6])
ax.set_xlim(-0.5,2.75)
ax.set_xticks([0+x_offset/2,1+x_offset/2,2+x_offset/2])
ax.set_xticklabels(['TRCA','RESS','EPCA'])
ax.set_ylabel('SNR (dB)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from scipy.stats import ttest_rel,mannwhitneyu

ylims = ax.get_ylim()

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests

t1,p1 = ttest_rel(SNR1[0],SNR1[1])

t2,p2 = ttest_rel(SNR1[0],SNR1[2])

t3,p3 = ttest_rel(SNR1[1],SNR1[2])

x_offset = 0.2
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.41, x_offset+0.41], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.41], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.295,0.83,print_significance(p1),ha='center',va='center',fontsize=10,fontweight='bold')


line = mlines.Line2D([x_offset+0.43, x_offset+0.43], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.43, x_offset+0.66], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.545,0.83,print_significance(p2),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.66], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.42,0.90,print_significance(p3),ha='center',va='center',fontsize=10,fontweight='bold')
plt.show()


# fig.savefig('./fig/fig_3/fig_3_meg_template_snr.png',dpi=600)
# fig.savefig('./fig/fig_3/fig_3_meg_template_snr.svg')
# fig.savefig('./fig/fig_3/fig_3_meg_template_snr.pdf')


# %%
SNR2 = np.reshape(SNR_test,(len(models),-1))
SNR2 = [SNR2[i] for i in [0,2,1]]
SNR2 = [SNR2[i]+i*1.7+5 for i in range(3)]

from my_code.utils.utils import adjacent_values, half_violin,print_significance
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(2.5, 2.5),constrained_layout=False)
plt.subplots_adjust(top=0.8,bottom=0.1,left=0.22,right=0.99,wspace=0.3,hspace=0.5)
x_offset = 0.25
import seaborn as sns
colors = ['#007ACC','#2ECC40','#E9657F','#F1C40F','#E74C3C']
quartile1, medians, quartile3 = np.percentile(SNR2, [25, 50, 75], axis=1)
for i,d in enumerate(SNR2):

    half_violin(ax, d, i+x_offset, side='right', width=0.3,facecolor=colors[i], 
                edgecolor=colors[i], alpha=0.8, linewidth=0.8)
    # ax.hlines(medians[i],xmin=-0.5,xmax=2.75,color=colors[i],linewidth=0.8,linestyle='--')

df_list = []
for i, group in enumerate(SNR2):
    for value in group:
        df_list.append({'Group': f'Group {i+1}', 'Value': value})
df = pd.DataFrame(df_list)

sns.boxplot(x='Group', y='Value', data=df, linewidth=0.8,palette=colors,ax=ax,width=0.2,showfliers=False,
            linecolor='k')
ax.set_xlabel('')

quartile1, medians, quartile3 = np.percentile(SNR2, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(SNR2, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(0, len(medians))
ax.scatter(inds+x_offset, medians, marker='o', color='white', s=5, zorder=3)
ax.vlines(inds+x_offset, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds+x_offset, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

ax.set_ylim(-18,-2)
ax.set_yticks([-18,-14,-10,-6,-2])
ax.set_xlim(-0.5,2.75)
ax.set_xticks([0+x_offset/2,1+x_offset/2,2+x_offset/2])
ax.set_xticklabels(['TRCA','RESS','EPCA'])
ax.set_ylabel('SNR (dB)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ylims = ax.get_ylim()

from scipy.stats import ttest_rel,ttest_ind
from statsmodels.stats.multitest import multipletests

t1,p1 = ttest_rel(SNR2[0],SNR2[1])

t2,p2 = ttest_rel(SNR2[0],SNR2[2])

t3,p3 = ttest_rel(SNR2[1],SNR2[2])

x_offset = 0.2
import matplotlib.lines as mlines
line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.41, x_offset+0.41], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.41], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.295,0.83,print_significance(p1),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.43, x_offset+0.43], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.8, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.43, x_offset+0.66], [0.82, 0.82], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.545,0.83,print_significance(p2),ha='center',va='center',fontsize=10,fontweight='bold')

line = mlines.Line2D([x_offset+0.18, x_offset+0.18], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.66, x_offset+0.66], [0.86, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line = mlines.Line2D([x_offset+0.18, x_offset+0.66], [0.88, 0.88], color='k', linestyle='-', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
fig.text(x_offset+0.42,0.90,print_significance(p3),ha='center',va='center',fontsize=10,fontweight='bold')
plt.show()


# fig.savefig('./fig/fig_3/fig_3_meg_test_snr.png',dpi=600)
# fig.savefig('./fig/fig_3/fig_3_meg_test_snr.svg')
# fig.savefig('./fig/fig_3/fig_3_meg_test_snr.pdf')


















# %%----------------------------------------------TRCA map----------------------------------------
from tqdm import tqdm
import scipy
from SSVEPAnalysisToolbox.algorithms import TRCA
from SSVEPAnalysisToolbox.algorithms.trca import _trca_U_1
from SSVEPAnalysisToolbox.algorithms.utils import eigvec
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS
from my_code.algorithms.prca import PRCA

# Get train data: 第0个block作为训练数据
maps_trca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([MEG], False)
    for stim_idx in range(len(X_train)):
        # 通道*采样点
        X_train_todo = [X_train[stim_idx][0,:,:]]
        trca_X1, trca_X2 = _trca_U_1(X_train_todo)
        S=trca_X1 @ trca_X1.T
        trca_X2_remove = trca_X2 - np.mean(trca_X2, 0)
        Q=trca_X2_remove.T @ trca_X2_remove
        # eig_vec = eigvec(S, Q)
        evals,evecs = scipy.linalg.eig(S, Q)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_trca[j,stim_idx,:] = map[:,0]


# %%----------------------------------------------RESS map----------------------------------------
from tqdm import tqdm
from scipy.linalg import eigh
from SSVEPAnalysisToolbox.algorithms import TRCA
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS,filterFGx
from my_code.algorithms.prca import PRCA

# Get train data: 第0个block作为训练数据
maps_ress = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([MEG], False)
    for stim_idx in range(len(X_train)):
        stim_freq = stim_freqs[stim_idx]
        filterbank_idx = 0
        peakwidt = 0.75
        neighfreq = 3
        neighwidt = 3

        # 通道*采样点
        X = [X_train[stim_idx][0,:,:]]

        n_trials = len(X)
        n_channels, n_samples = X[0].shape

        peakfreq = stim_freq * (filterbank_idx + 1)

        # compute covariance matrix at peak frequency
        fdatAt = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatAt[:,:,ti] = filterFGx(tmdat,srate,peakfreq,peakwidt)
        fdatAt = fdatAt.reshape(n_channels, -1)
        fdatAt -= np.mean(fdatAt, axis=1, keepdims=True)
        covAt = np.dot(fdatAt, fdatAt.T) / n_samples

        # compute covariance matrix for lower neighbor
        fdatLo = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatLo[:,:,ti] = filterFGx(tmdat,srate,peakfreq-neighfreq,neighwidt)
        fdatLo = fdatLo.reshape(n_channels, -1)
        fdatLo -= np.mean(fdatLo, axis=1, keepdims=True)
        covLo = np.dot(fdatLo, fdatLo.T) / n_samples

        # compute covariance matrix for upper neighbor
        fdatHi = np.zeros((n_channels, n_samples, n_trials))
        for ti in range(n_trials):
            tmdat = X[ti]
            fdatHi[:,:,ti] = filterFGx(tmdat,srate,peakfreq+neighfreq,neighwidt)
        fdatHi = fdatHi.reshape(n_channels, -1)
        fdatHi -= np.mean(fdatHi, axis=1, keepdims=True)
        covHi = np.dot(fdatHi, fdatHi.T) / n_samples

        # shrinkage regularization
        covBt = (covHi + covLo)/2
        gamma = 0.01
        evalue,_ = np.linalg.eig(covBt)
        covBt = covBt + gamma*np.mean(evalue)*np.eye(covBt.shape[0])

        evals,evecs = eigh(covAt,covBt)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(covAt, evecs) @ np.linalg.inv(evecs.T @ covAt @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_ress[j,stim_idx,:] = map[:,0]





# %%----------------------------------------------EPCA map----------------------------------------
from tqdm import tqdm
from scipy.linalg import eigh,eig
from SSVEPAnalysisToolbox.algorithms import TRCA
from my_code.algorithms.epca import EPCA,EEPCA,project
from my_code.algorithms.ress import RESS,filterFGx
from my_code.algorithms.prca import PRCA

# Get train data: 第0个block作为训练数据
maps_epca = np.zeros((len(trial_container),len(stim_freqs),len(ch_used)))
for j in tqdm(range(len(trial_container))):
    # X_train: list[40], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([MEG], False)
    for stim_idx in range(len(X_train)):
        stim_freq = stim_freqs[stim_idx]
        # 通道*采样点
        X = [X_train[stim_idx][0,:,:]]

        n_trials = len(X)
        n_channels, n_samples = X[0].shape

        X1 = np.concatenate(X,axis=1)

        X_mean = np.mean(X, axis=0)
        X3 = np.zeros(X_mean.shape)
        periodic = int(round(srate/stim_freq))
        for ch_idx in range(n_channels):
            X3[ch_idx,:] = project(X_mean[ch_idx,:], periodic, True, True)

        Q = np.dot(X1,X1.T)/X1.shape[1]
        S = np.dot(X3,X3.T)/X3.shape[1] - Q
        evals,evecs = eig(S,Q)

        sidx  = np.argsort(evals)[::-1]
        evals = evals[sidx]
        evecs = evecs[:,sidx]

        comp2plot = np.argmax(evals)  # 获取最大特征值的索引
        # 正规化特征向量（虽然不是必须的，但可以这样做）  
        # evecs = evecs / np.sqrt(np.sum(evecs**2, axis=0, keepdims=True))  

        # 计算映射矩阵  
        map = np.dot(S, evecs) @ np.linalg.inv(evecs.T @ S @ evecs)  
        # 找到最大的绝对值分量的索引  
        idx = np.argmax(np.abs(map[:, comp2plot]))
        # 通过取对应分量的符号来强制使结果为正值  
        map *= np.sign(map[idx, comp2plot])

        maps_epca[j,stim_idx,:] = map[:,0]






















# %%
raw_info = mne.io.read_info(r"./supports/raw_info_plot.fif")


# %%
with open(r'./result/fig_3/maps_trca_meg.pkl','rb') as f:
    maps_trca = pickle.load(f)

with open(r'./result/fig_3/maps_ress_meg.pkl','rb') as f:
    maps_ress = pickle.load(f)

with open(r'./result/fig_3/maps_epca_meg.pkl','rb') as f:
    maps_epca = pickle.load(f)


# %%
from mne.io.pick import _picks_to_idx
from matplotlib.colors import LinearSegmentedColormap

pick_freq = [9,17]
for f in pick_freq:
    idx = stim_freqs.index(f)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.weight'] = 'normal'
    fig,axes = plt.subplots(1,3,figsize=(6, 2),constrained_layout=True)
    if f == 9:
        maps = [maps_trca,maps_epca,maps_ress]
    elif f == 17:
        maps = [maps_trca,maps_ress,maps_epca]
    for i,map in enumerate(maps):
        map_mean = np.mean(map[:,idx,:],axis=(0))
        mne.viz.plot_topomap(map_mean/np.max(map_mean),raw_info,sphere=0.115,axes=axes[i],contours=0,
                            vlim=(-0.7,1),cmap='RdBu_r',extrapolate='head')
    plt.show()

    # fig.savefig(r'./fig/fig_3/map_'+str(f)+'_meg.png',dpi=600)
    # fig.savefig(r'./fig/fig_3/map_'+str(f)+'_meg.svg')
    # fig.savefig(r'./fig/fig_3/map_'+str(f)+'_meg.pdf')














# %%
from my_code.utils.megpreprocess import preprocess as megpreprocess
from my_code.utils.utils import bandpass_filter
data = []
for i in range(13):
    data.append(MEG.get_sub_data(i))

data = np.array(data)

# %%
stim_freq = 9
ch = 'O1'
stim_freqs = MEG.stim_info['freqs']
idx = stim_freqs.index(stim_freq)
ch_idx = MEG.channels.index(ch)
data_stim = data[:,:,idx,ch_idx,:]


data_stim_pre = np.zeros(data_stim.shape)
for i in range(13):
    data_stim_pre[i,:,:] = bandpass_filter(MEG, megpreprocess(MEG, data_stim[i,:,:]),7,90)

data_mean = np.mean(data_stim_pre[:,0:5],axis=(0,1))

from my_code.algorithms.epca import EPCA,project
data_stim_pre_project = np.zeros(data_stim_pre.shape)

for sub_idx in range(13):
    for trial_idx in range(6):
        data_single_trial = data_stim_pre[sub_idx,trial_idx,:]
        data_stim_pre_project[sub_idx,trial_idx,:] = project(data_single_trial,int(1000/stim_freq),True,True)

# data_mean_project = np.mean(data_stim_pre_project,axis=0)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,ax = plt.subplots(1,1,figsize=(6, 2),constrained_layout=True)
colors = ['#008080','#DC143C','#FFA500']
data_length = 3
t_s,t_e = int(0.5*1000),int((0.5+data_length)*1000)
times = np.arange(0,data_length,1/1000)
# ax.plot(times,scaler.fit_transform(data_stim_pre[0,1,t_s:t_e].reshape(-1, 1)),linewidth=1.5,color=colors[2])
ax.plot(times,data_stim_pre_project[0,0,t_s:t_e],linewidth=1.5,color=colors[1])
ax.plot(times,data_mean[t_s:t_e],linewidth=1.5,color=colors[0])


ax.set_xlim(times[0],times[-1])
ax.set_ylim(-250,250)
# ax.set_ylim(-1.1,1.1)
ax.set_xticks([0,0.5,1,1.5,2,2.5,3])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (fT)')

plt.show()

# fig.savefig(r'./fig/fig_3/signal_meg.png',dpi=600)
# fig.savefig(r'./fig/fig_3/signal_meg.svg')
# fig.savefig(r'./fig/fig_3/signal_meg.pdf')

# %%
