# %%
# t-SNE可视化
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib auto
import pickle
import pandas as pd
from tqdm import tqdm

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

dataset_key = 'MEG'

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

num_targs = dataset.stim_info['stim_num']
stim_freqs = dataset.stim_info['freqs']
srate = dataset.srate
all_stims = [i for i in range(dataset.trial_num)]
num_subs = len(dataset.subjects)
num_trials = dataset.block_num
labels = np.arange(num_targs)

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














# %%
# FBCCA
num_trains = [num_trials-1]
tw_seq = [0.6]
harmonic_num = 5

trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)

from my_code.algorithms.cca import FBCCA
sig_U_fbcca = np.zeros((1,len(trial_container),len(stim_freqs),int(tw_seq[0]*srate)))
y_fbcca = np.zeros((1,len(trial_container),len(stim_freqs)))
model = FBCCA(weights_filterbank=weights_filterbank,force_output_UV=True,n_jobs=-1)
for j in tqdm(range(len(trial_container))):
    # X_train: list[], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
    trained_model = model.__copy__()
    trained_model.fit(ref_sig=ref_sig)

    # X_train: list[], 每个元素为(子带*通道*采样点)
    X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([dataset], False)

    trained_model.predict(X_test)

    Us = trained_model.model['U'] # List[], filterbank_num*stimulus_num*channel_num*n_component

    for idx,(test_tode,y) in enumerate(zip(X_test,Y_test)):
        U_todo = Us[idx][0,y,:,:] # channel_num * n_component
        sig_U_fbcca[0,j,y,:] = U_todo.T @ test_tode[0,:,:]
        y_fbcca[0,j,y] = y


# %%
import pickle
with open(f'./result_R1/t-sne/{dataset_key}_fbcca_06.pkl','wb')as f:
    pickle.dump([sig_U_fbcca,y_fbcca],f)

# %%
# import pickle
# with open('./result_R1/t-sne/MEG_fbcca.pkl','rb')as f:
#     sig_U_fbcca,y_fbcca = pickle.load(f)

# %%
from sklearn.manifold import TSNE
# 创建TSNE对象
tsne_FBCCA = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_FBCCA = tsne_FBCCA.fit_transform(sig_U_fbcca[0,:,:].reshape(sig_U_fbcca.shape[1] * sig_U_fbcca.shape[2],sig_U_fbcca.shape[3]))
y_fbcca = y_fbcca[0].reshape(y_fbcca.shape[1] * y_fbcca.shape[2])

# %%
# 绘制散点图
cmap = plt.get_cmap("jet")
num_colors = sig_U_fbcca.shape[2]
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(3, 3))
plt.subplots_adjust(top=0.95,bottom=0.05,left=0.04,right=0.995,wspace=0.3,hspace=0.3)

gs = gridspec.GridSpec(1, 1, width_ratios=[1])
ax = plt.subplot(gs[0])

s = 1.5
for i in range(num_colors):
    sc = ax.scatter(X_tsne_FBCCA[y_fbcca == i, 0], X_tsne_FBCCA[y_fbcca == i, 1], color=colors[i],
                s=s, linewidths=0)
plt.show()

















# %%
# TDCA
from my_code.algorithms.tdca import TDCA

num_trains = [1]
tw_seq = [1]
harmonic_num = 5

trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)

sig_U_tdca = []
y_tdca = np.zeros((1,len(trial_container),len(stim_freqs)))
model = TDCA(n_component=8, padding_len=5)
for j in tqdm(range(len(trial_container))):
    # X_train: list[], 每个元素为(子带*通道*采样点)
    X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
    y_tdca[0,j,:] = Y_train

    ref_sig = [ref_sig[i][:,0:-int(srate*0.1)] for i in range(len(ref_sig))]
    trained_model = model.__copy__()
    trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig)
    templates_U = trained_model.templates_[:,0:8,:] # (n_trials, n_component, n_samples)
    templates_U1 = []
    for t in templates_U:
        templates_U1.append(np.reshape(t, (-1)))
    templates_U1 = np.array(templates_U1)
    sig_U_tdca.append(templates_U1)

sig_U_tdca = np.array(sig_U_tdca)
sig_U_tdca = np.expand_dims(sig_U_tdca, axis=0)


# %%
import pickle
with open(f'./result_R1/t-sne/{dataset_key}_tdca.pkl','wb')as f:
    pickle.dump([sig_U_tdca,y_tdca],f)




# %%
from sklearn.manifold import TSNE
# 创建TSNE对象
tsne_TDCA= TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_tdca = tsne_TDCA.fit_transform(sig_U_tdca[0,:,:].reshape(sig_U_tdca.shape[1] * sig_U_tdca.shape[2],sig_U_tdca.shape[3]))
y_tdca = y_tdca[0].reshape(y_tdca.shape[1] * y_tdca.shape[2])


# %%
# 绘制散点图
cmap = plt.get_cmap("jet")
num_colors = sig_U_tdca.shape[2]
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(3, 3))
plt.subplots_adjust(top=0.95,bottom=0.05,left=0.04,right=0.995,wspace=0.3,hspace=0.3)

gs = gridspec.GridSpec(1, 1, width_ratios=[1])
ax = plt.subplot(gs[0])

s = 1.5
for i in range(num_colors):
    sc = ax.scatter(X_tsne_tdca[y_tdca == i, 0], X_tsne_tdca[y_tdca == i, 1], color=colors[i],
                s=s, linewidths=0)
plt.show()


















# %%
from tqdm import tqdm
from my_code.algorithms.trca import TRCA, ETRCA, MSETRCA
from my_code.algorithms.epca import EPCA, EEPCA
from my_code.algorithms.ress import RESS, ERESS


num_trains = [1]
tw_seq = [1]
harmonic_num = 5

trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)


models = [
    TRCA(weights_filterbank=weights_filterbank,n_jobs=-1),
    RESS(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate,
            ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3},n_jobs=-1),
    EPCA(weights_filterbank=weights_filterbank,stim_freqs=stim_freqs,srate=srate,n_jobs=-1),
]


sig_U_trca_ress_epca = np.zeros((len(models),len(trial_container),len(stim_freqs),int(tw_seq[0]*srate)))
y_trca_ress_epca = np.zeros((len(models),len(trial_container),len(stim_freqs)))
# test_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),5,int(tw_seq[0]*srate)))
for model_idx in range(len(models)):
    model = models[model_idx]
    # Get train data: 第0个block作为训练数据
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)

        y_trca_ress_epca[model_idx,j,:] = Y_train

        trained_model = model.__copy__()
        trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)

        # 5个子带，40个刺激对应的空间滤波器, 大小为子带*刺激*通道*1
        U = trained_model.model['U'] # (5, 40, 9, 1)

        # Us[model_idx,j,:,:] = U[0,:,:,0]

        # X_train: list[200], 每个元素为(子带*通道*采样点)
        X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([dataset], False)

        # 获取模版信号: list[], 每个元素为(子带*通道*采样点)
        template_sig = trained_model.model['template_sig']

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[:,stim_idx,:,:])
            template_sig_stim = template_sig[stim_idx]

            # 计算空间滤波器滤波后的模版信号
            sig_U_trca_ress_epca[model_idx,j,stim_idx,:] = template_sig_stim[0].T @ U_stim[0]

        # for stim_idx in range(len(stim_freqs)):
        #     U_stim = np.squeeze(U[:,stim_idx,:,:])
        #     test_sig_stim_idx = [i  for i in range(len(Y_test)) if Y_test[i] == stim_idx]
        #     test_sig_stim = [X_test[i] for i in test_sig_stim_idx]

        #     for idx,test_sig_stim_todo in enumerate(test_sig_stim):
        #         # 计算空间滤波器滤波后的测试信号
        #         test_sig_U[model_idx,j,stim_idx,idx,:] = test_sig_stim_todo[0].T @ U_stim[0]

# %%
import pickle
with open(f'./result_R1/t-sne/{dataset_key}_trca_ress_epca.pkl','wb')as f:
    pickle.dump([sig_U_trca_ress_epca,y_trca_ress_epca],f)


# %%
