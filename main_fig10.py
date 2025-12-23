# %%
'''
t-SNE可视化特征向量
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
from my_code.utils.benchmarkpreprocess import benchmark_suggested_ch
from my_code.utils.megpreprocess import meg_suggested_ch
num_targs = dataset.stim_info['stim_num']
stim_freqs = dataset.stim_info['freqs']
srate = dataset.srate
all_stims = [i for i in range(dataset.trial_num)]
num_subs = len(dataset.subjects)
num_trials = dataset.block_num
labels = np.arange(num_targs)
num_fbs = 5
if dataset_key == 'Benchmark':
    ch_used = benchmark_suggested_ch(9)
elif dataset_key == 'MEG':
    ch_used = meg_suggested_ch(9)

num_trains = [1]
tw_seq = [1]
harmonic_num = 5

trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)

# %%
from tqdm import tqdm
from SSVEPAnalysisToolbox.algorithms import TRCA
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS
from my_code.algorithms.prca import PRCA

from my_code.utils.benchmarkpreprocess import suggested_weights_filterbank as benchmark_weights_filterbank


from my_code.utils.megpreprocess import suggested_weights_filterbank as meg_weights_filterbank

if dataset_key == 'Benchmark':
    models = [
        TRCA(weights_filterbank=benchmark_weights_filterbank()),
        EPCA(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
        RESS(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
                ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
        ]
elif dataset_key == 'MEG':
    models = [
        TRCA(weights_filterbank=meg_weights_filterbank()),
        EPCA(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
        RESS(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
             ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3})
    ]

# %%
template_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),int(tw_seq[0]*srate)))
test_sig_U = np.zeros((len(models),len(trial_container),len(stim_freqs),5,int(tw_seq[0]*srate)))
Us = np.zeros((len(models),len(trial_container),len(stim_freqs),len(ch_used)))
rs = np.zeros((len(models),len(trial_container),len(stim_freqs)))
for model_idx in range(len(models)):
    model = models[model_idx]
    # Get train data: 第0个block作为训练数据
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        trained_model = model.__copy__()
        trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)

        # 5个子带，40个刺激对应的空间滤波器, 大小为子带*刺激*通道*1
        U = trained_model.model['U'] # (5, 40, 9, 1)

        Us[model_idx,j,:,:] = U[0,:,:,0]

        # X_train: list[200], 每个元素为(子带*通道*采样点)
        X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([dataset], False)

        # 获取模版信号: list[], 每个元素为(子带*通道*采样点)
        template_sig = trained_model.model['template_sig']

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[:,stim_idx,:,:])
            template_sig_stim = template_sig[stim_idx]

            # 计算空间滤波器滤波后的模版信号
            template_sig_U[model_idx,j,stim_idx,:] = template_sig_stim[0].T @ U_stim[0]

        for stim_idx in range(len(stim_freqs)):
            U_stim = np.squeeze(U[:,stim_idx,:,:])
            test_sig_stim_idx = [i  for i in range(len(Y_test)) if Y_test[i] == stim_idx]
            test_sig_stim = [X_test[i] for i in test_sig_stim_idx]

            for idx,test_sig_stim_todo in enumerate(test_sig_stim):
                # 计算空间滤波器滤波后的测试信号
                test_sig_U[model_idx,j,stim_idx,idx,:] = test_sig_stim_todo[0].T @ U_stim[0]


# %%
# import pickle
# with open('./result/fig_10/波形/Us_meg.pkl','wb') as f:
#     pickle.dump(Us,f)

# with open('./result/fig_10/波形/template_sig_U_meg.pkl','wb') as f:
#     pickle.dump(template_sig_U,f)

# %%
# with open('./result/fig_10/Us.pkl','rb') as f:
#     Us = pickle.load(f)

with open('./result/fig_10/波形/template_sig_U.pkl','rb') as f:
    template_sig_U = pickle.load(f)

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = np.zeros((len(models),len(trial_container)*len(stim_freqs),int(tw_seq[0]*srate)))
Y = np.zeros((len(models),len(trial_container)*len(stim_freqs)))
for model_idx in range(len(models)):
    for j in range(len(trial_container)):
        for stim_idx in range(len(stim_freqs)):
            X[model_idx,j*len(stim_freqs)+stim_idx,:] = template_sig_U[model_idx,j,stim_idx,:]
            Y[model_idx,j*len(stim_freqs)+stim_idx] = stim_idx


# 创建TSNE对象
tsne_TRCA = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_TRCA = tsne_TRCA.fit_transform(X[0,:,:])

tsne_EPCA = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_EPCA = tsne_EPCA.fit_transform(X[1,:,:])

tsne_RESS = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_RESS = tsne_RESS.fit_transform(X[2,:,:])


# %%
# with open('./result/fig_10/波形/X_tsne_meg1.pkl','wb') as f:
#     pickle.dump([X_tsne_TRCA_meg,X_tsne_RESS_meg, X_tsne_EPCA_meg, Y_meg],f)

# with open('./result/fig_10/波形/X_tsne_benchmark1.pkl','wb') as f:
#     pickle.dump([X_tsne_TRCA_benchmark,X_tsne_RESS_benchmark, X_tsne_EPCA_benchmark, Y_benchmark],f)

# %%
with open('./result/fig_10/波形/X_tsne_meg.pkl','rb') as f:
    [X_tsne_TRCA_meg,X_tsne_RESS_meg, X_tsne_EPCA_meg, Y_meg] = pickle.load(f)

with open('./result/fig_10/波形/X_tsne_benchmark.pkl','rb') as f:
    [X_tsne_TRCA_benchmark,X_tsne_RESS_benchmark, X_tsne_EPCA_benchmark, Y_benchmark] = pickle.load(f)

# with open('./result/fig_10/波形/Y_benchmark.pkl','rb') as f:
#     Y_benchmark = pickle.load(f)

# %%
# 绘制散点图
cmap = plt.get_cmap("jet")
num_colors = 40
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(8, 5))
plt.subplots_adjust(top=0.95,bottom=0.05,left=0.04,right=0.995,wspace=0.3,hspace=0.3)

gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.1])
axes = [plt.subplot(gs[0,0]),plt.subplot(gs[0,1]),plt.subplot(gs[0,2])]

X_tsnes = [X_tsne_TRCA_benchmark,X_tsne_EPCA_benchmark,X_tsne_RESS_benchmark]
for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    ax = axes[tsne_idx]
    s = 1.5
    for i in range(40):
        sc = ax.scatter(X_tsne[Y_benchmark[tsne_idx,:] == i, 0], X_tsne[Y_benchmark[tsne_idx,:] == i, 1], color=colors[i],
                   s=s, linewidths=0)
    ax.set_xlim(-65,65)
    ax.set_ylim(-65,65)
    ax.set_xticks([-60,-30,0,30,60])
    ax.set_yticks([-60,-30,0,30,60])
    

ax = plt.subplot(gs[0,3])
ax.axis('off')
cax = fig.add_axes([0.92, 0.56, 0.01, 0.393])
sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=8, vmax=15.8))
cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
cbar.set_label('Stimulus Frequency (Hz)')
cbar.set_ticks([8,10,12,14,15.8])


fig.text(0.045,0.93,'eTRCA',va='center',ha='left',fontweight='bold')
fig.text(0.355,0.93,'eRESS',va='center',ha='left',fontweight='bold')
fig.text(0.665,0.93,'eEPCA',va='center',ha='left',fontweight='bold')


axes = [plt.subplot(gs[1,0]),plt.subplot(gs[1,1]),plt.subplot(gs[1,2])]
cmap = plt.get_cmap("jet")
num_colors = 9
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
X_tsnes = [X_tsne_TRCA_meg,X_tsne_RESS_meg,X_tsne_EPCA_meg]
for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    ax = axes[tsne_idx]
    s = 3
    for i in range(9):
        sc = ax.scatter(X_tsne[Y_meg[tsne_idx,:] == i, 0], X_tsne[Y_meg[tsne_idx,:] == i, 1], color=colors[i],
                   s=s, linewidths=0)
    ax.set_xlim(-35,35)
    ax.set_ylim(-35,35)
    ax.set_xticks([-30,-15,0,15,30])
    ax.set_yticks([-30,-15,0,15,30])

ax = plt.subplot(gs[1,3])
ax.axis('off')
cax = fig.add_axes([0.92, 0.05, 0.01, 0.393])
sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=9, vmax=17))
cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
cbar.set_label('Stimulus Frequency (Hz)')
cbar.set_ticks([9,11,13,15,17])

fig.text(0.045,0.42,'eTRCA',va='center',ha='left',fontweight='bold')
fig.text(0.355,0.42,'eRESS',va='center',ha='left',fontweight='bold')
fig.text(0.665,0.42,'eEPCA',va='center',ha='left',fontweight='bold')


fig.text(0.48,0.98,'Benchmark',va='center',ha='center',fontweight='bold',fontsize=11)
fig.text(0.48,0.47,'MEG',va='center',ha='center',fontweight='bold',fontsize=11)

fig.text(0.02,0.97,'A',va='center',ha='center',fontweight='bold',fontsize=14)
fig.text(0.02,0.46,'B',va='center',ha='center',fontweight='bold',fontsize=14)



plt.show()

# fig.savefig('./fig/fig_10/波形/fig_10.png',dpi=900)
# fig.savefig('./fig/fig_10/波形/fig_10.svg')
# fig.savefig('./fig/fig_10/波形/fig_10.pdf')


# %%
# 绘制散点图
cmap = plt.get_cmap("jet")
num_colors = 40
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(4.5, 6))
# plt.subplots_adjust(top=0.95,bottom=0.05,left=0.04,right=0.995,wspace=0.3,hspace=0.3) # 字体为10
plt.subplots_adjust(top=0.96,bottom=0.05,left=0.06,right=0.98,wspace=0.3,hspace=0.2) # 字体为13

gs = gridspec.GridSpec(3, 2, width_ratios=[1,1])
axes = [plt.subplot(gs[0,0]),plt.subplot(gs[1,0]),plt.subplot(gs[2,0])]

X_tsnes = [X_tsne_TRCA_benchmark,X_tsne_EPCA_benchmark,X_tsne_RESS_benchmark]
for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    ax = axes[tsne_idx]
    s = 2
    for i in range(40):
        sc = ax.scatter(X_tsne[Y_benchmark[tsne_idx,:] == i, 0], X_tsne[Y_benchmark[tsne_idx,:] == i, 1], color=colors[i],
                   s=s, linewidths=0)
    ax.set_xlim(-65,65)
    ax.set_ylim(-65,65)
    ax.set_xticks([-60,-30,0,30,60])
    ax.set_yticks([-60,-30,0,30,60])
    

# ax = plt.subplot(gs[:,1])
# ax.axis('off')
# cax = fig.add_axes([0.4, 0.56, 0.01, 0.393])
# sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=8, vmax=15.8))
# cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
# cbar.set_label('Stimulus Frequency (Hz)')
# cbar.set_ticks([8,10,12,14,15.8])


fig.text(0.07,0.945,'eTRCA',va='center',ha='left',fontweight='bold')
fig.text(0.07,0.62,'eRESS',va='center',ha='left',fontweight='bold')
fig.text(0.07,0.3,'eEPCA',va='center',ha='left',fontweight='bold')


axes = [plt.subplot(gs[0,1]),plt.subplot(gs[1,1]),plt.subplot(gs[2,1])]
cmap = plt.get_cmap("jet")
num_colors = 9
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
X_tsnes = [X_tsne_TRCA_meg,X_tsne_RESS_meg,X_tsne_EPCA_meg]
for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    ax = axes[tsne_idx]
    s = 3
    for i in range(9):
        sc = ax.scatter(X_tsne[Y_meg[tsne_idx,:] == i, 0], X_tsne[Y_meg[tsne_idx,:] == i, 1], color=colors[i],
                   s=s, linewidths=0)
    ax.set_xlim(-35,35)
    ax.set_ylim(-35,35)
    ax.set_xticks([-30,-15,0,15,30])
    ax.set_yticks([-30,-15,0,15,30])

# ax = plt.subplot(gs[:,3])
# ax.axis('off')
# cax = fig.add_axes([0.91, 0.05, 0.01, 0.393])
# sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=9, vmax=17))
# cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
# cbar.set_label('Stimulus Frequency (Hz)')
# cbar.set_ticks([9,11,13,15,17])

fig.text(0.59,0.945,'eTRCA',va='center',ha='left',fontweight='bold')
fig.text(0.59,0.62,'eRESS',va='center',ha='left',fontweight='bold')
fig.text(0.59,0.3,'eEPCA',va='center',ha='left',fontweight='bold')


fig.text(0.26,0.98,'Dataset Ⅰ (Benchmark)',va='center',ha='center',fontweight='bold',fontsize=11)
fig.text(0.78,0.98,'Dataset Ⅱ (MEG)',va='center',ha='center',fontweight='bold',fontsize=11)

fig.text(0.02,0.98,'A',va='center',ha='center',fontweight='bold',fontsize=14)
fig.text(0.52,0.98,'B',va='center',ha='center',fontweight='bold',fontsize=14)



plt.show()

# fig.savefig('./fig/fig_10/波形/fig_10_1.png',dpi=900)
# fig.savefig('./fig/fig_10/波形/fig_10_1.svg')
# fig.savefig('./fig/fig_10/波形/fig_10_1.pdf')


# %%
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    # **KMeans 聚类**
    kmeans = KMeans(n_clusters=40, random_state=42, n_init=100)
    labels_pred = kmeans.fit_predict(X_tsne)

    # **计算聚类指标**
    chi = calinski_harabasz_score(X_tsne, labels_pred)
    sc = silhouette_score(X_tsne, labels_pred)
    dbi = davies_bouldin_score(X_tsne, labels_pred)

    print(f'KMeans 聚类指标: CHI={chi:.3f}, SC={sc:.3f}, DBI={dbi:.3f}')































# %%
from my_code.datasets.mybenchmarkdataset import MyBenchmarkDataset
from my_code.utils.benchmarkpreprocess import preprocess as benchmarkpreprocess
from my_code.utils.benchmarkpreprocess import filterbank as benchmarkfilterbank

from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.utils.megpreprocess import preprocess as megpreprocess
from my_code.utils.megpreprocess import filterbank as megfilterbank


dataset_key = 'MEG'
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
from my_code.utils.benchmarkpreprocess import benchmark_suggested_ch
from my_code.utils.megpreprocess import meg_suggested_ch
num_targs = dataset.stim_info['stim_num']
stim_freqs = dataset.stim_info['freqs']
srate = dataset.srate
all_stims = [i for i in range(dataset.trial_num)]
num_subs = len(dataset.subjects)
num_trials = dataset.block_num
labels = np.arange(num_targs)
num_fbs = 5
if dataset_key == 'Benchmark':
    ch_used = benchmark_suggested_ch(9)
elif dataset_key == 'MEG':
    ch_used = meg_suggested_ch(9)

num_trains = [1]
tw_seq = [0.8]
harmonic_num = 5

trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                               trains = num_trains,
                                               harmonic_num = harmonic_num,
                                               ch_used = ch_used)

# %%
from tqdm import tqdm
from SSVEPAnalysisToolbox.algorithms import TRCA
from my_code.algorithms.epca import EPCA,EEPCA
from my_code.algorithms.ress import RESS
from my_code.algorithms.prca import PRCA

from my_code.utils.benchmarkpreprocess import suggested_weights_filterbank as benchmark_weights_filterbank

from my_code.utils.megpreprocess import suggested_weights_filterbank as meg_weights_filterbank

if dataset_key == 'Benchmark':
    models = [
        TRCA(weights_filterbank=benchmark_weights_filterbank()),
        EPCA(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
        RESS(weights_filterbank=benchmark_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
                ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
        ]
elif dataset_key == 'MEG':
    models = [
        TRCA(weights_filterbank=meg_weights_filterbank()),
        EPCA(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate),
        RESS(weights_filterbank=meg_weights_filterbank(),stim_freqs=stim_freqs,srate=srate,
             ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3})
    ]

# %%
if dataset_key == 'Benchmark':
    weights_filterbank = benchmark_weights_filterbank()
elif dataset_key == 'MEG':
    weights_filterbank = meg_weights_filterbank()

rs = []
Ys = []
for model_idx in range(len(models)):
    model = models[model_idx]
    # Get train data: 第0个block作为训练数据
    Y = []
    R = []
    for j in tqdm(range(len(trial_container))):
        # X_train: list[40], 每个元素为(子带*通道*采样点)
        X_train, Y_train, ref_sig, freqs = trial_container[j][0].get_data([dataset], False)
        trained_model = model.__copy__()
        trained_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)

        # X_train: list[200], 每个元素为(子带*通道*采样点)
        X_test, Y_test, ref_sig, _ = trial_container[j][1].get_data([dataset], False)

        Y_pred, r = trained_model.predict(X_test)

        R += [weights_filterbank @ r_tmp for r_tmp in r]
        Y += Y_test

    rs.append(R)
    Ys.append(Y)

# %%
# with open('./result/fig_10/特征/rs_1trials_meg.pkl','wb') as f:
#     pickle.dump(rs,f)

# with open('./result/fig_10/特征/Ys_1trials_meg.pkl','wb') as f:
#     pickle.dump(Ys,f)

# %%
from sklearn.manifold import TSNE
# 创建TSNE对象
tsne_TRCA = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_TRCA = tsne_TRCA.fit_transform(np.array(rs[0]))

tsne_EPCA = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_EPCA = tsne_EPCA.fit_transform(np.array(rs[1]))

tsne_RESS = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
X_tsne_RESS = tsne_RESS.fit_transform(np.array(rs[2]))

# %%
# with open('./result/fig_10/特征/X_tsne_1trials_meg.pkl','wb') as f:
#     pickle.dump([X_tsne_TRCA,X_tsne_EPCA,X_tsne_RESS],f)



# %%
cmap = plt.get_cmap("jet")
num_colors = len(stim_freqs)
colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

# 绘制散点图
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(8, 2.5))
plt.subplots_adjust(top=0.95,bottom=0.1,left=0.04,right=0.995,wspace=0.3,hspace=0.3)

gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,0.1])
axes = [plt.subplot(gs[0]),plt.subplot(gs[1]),plt.subplot(gs[2])]

X_tsnes = [X_tsne_TRCA,X_tsne_RESS,X_tsne_EPCA]
for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    ax = axes[tsne_idx]
    if dataset_key == 'Benchmark':
        ax.set_xlim(-75,75)
        ax.set_ylim(-75,75)
        ax.set_xticks([-50,0,50])
        ax.set_yticks([-50,0,50])
        s = 1.5
    elif dataset_key == 'MEG':
        ax.set_xlim(-65,65)
        ax.set_ylim(-65,65)
        ax.set_xticks([-40,0,40])
        ax.set_yticks([-40,0,40])
        s = 3
    for i in range(len(stim_freqs)):
        sc = ax.scatter(X_tsne[np.array(Ys[tsne_idx]) == i, 0], X_tsne[np.array(Ys[tsne_idx]) == i, 1], color=colors[i],
                    s=s, linewidths=0)

ax = plt.subplot(gs[3])
ax.axis('off')
cax = fig.add_axes([0.92, 0.1, 0.01, 0.85])
if dataset_key == 'Benchmark':
    sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=40))
    cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
    cbar.set_label('Stimulus Frequency (Hz)')
    cbar.set_ticks([0,10,20,30,40])

elif dataset_key == 'MEG':
    sm = ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=9))
    cbar = fig.colorbar(sm, orientation="vertical", cax=cax)
    cbar.set_label('Stimulus Frequency (Hz)')
    cbar.set_ticks([0,3,6,9])


fig.text(0.045,0.9,'eTRCA',va='center',ha='left',fontweight='bold')
fig.text(0.355,0.9,'eRESS',va='center',ha='left',fontweight='bold')
fig.text(0.665,0.9,'eEPCA',va='center',ha='left',fontweight='bold')

plt.show()


# fig.savefig('./fig/fig_10/特征/fig_10_1trials_meg.png',dpi=900)
# fig.savefig('./fig/fig_10/特征/fig_10_1trials_meg.svg')
# fig.savefig('./fig/fig_10/特征/fig_10_1trials_meg.pdf')


# %%
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

for tsne_idx in range(len(X_tsnes)):
    X_tsne = X_tsnes[tsne_idx]
    # **KMeans 聚类**
    kmeans = KMeans(n_clusters=40, random_state=42, n_init=100)
    labels_pred = kmeans.fit_predict(X_tsne)

    # **计算聚类指标**
    chi = calinski_harabasz_score(X_tsne, labels_pred)
    sc = silhouette_score(X_tsne, labels_pred)
    dbi = davies_bouldin_score(X_tsne, labels_pred)

    print(f'KMeans 聚类指标: CHI={chi:.3f}, SC={sc:.3f}, DBI={dbi:.3f}')

# %%
