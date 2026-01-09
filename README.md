# EPCA

## 1. Overview
**Exactly Periodic Component Analysis (EPCA)** is a signal processing method designed to enhance **individual calibration and classification performance** in **SSVER-based brain–computer interfaces (BCIs)**.

This repository provides the **implementation code** for EPCA and several representative SSVER decoding methods used for comparison, enabling reproducibility of the experimental results reported in the associated paper.

---

## 2. Requirements
This project depends on the **SSVEPAnalysisToolbox** Python package.

Please follow the official installation instructions:
https://ssvep-analysis-toolbox.readthedocs.io/en/latest/Installation_pages/Installation.html

---

## 3. Datasets
The following datasets are used in this study:

- **Benchmark dataset (Dataset I)**: Wang Y, Chen X, Gao X, et al. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10): 1746-1752.
  
  URL: https://bci.med.tsinghua.edu.cn/download.html

- **BETA dataset (Dataset II)**: Liu B, Huang X, Wang Y, et al. BETA: A Large Benchmark Database Toward SSVEP-BCI Application[J]. Frontiers in Neuroscience, 2020, 14.
  
  URL: https://bci.med.tsinghua.edu.cn/download.html

- **eldBETA dataset (Dataset III)**: Liu B, Wang Y, Gao X, et al. eldBETA: A Large Eldercare-oriented Benchmark Database of SSVEP-BCI for the Aging Population[J]. Scientific Data, 2022, 9(1): 252.
  
  URL: https://doi.org/10.6084/m9.figshare.18032669

- **GuHF dataset (Dataset IV)**: Gu M, Pei W, Gao X, et al. An open dataset for human SSVEPs in the frequency range of 1-60 Hz[J]. Scientific Data, 2024, 11(1): 196.
  
  URL: https://doi.org/10.6084/m9.figshare.23641092

- **OPMMEG dataset (Dataset V)**  
  The OPMMEG dataset will be made available by the corresponding author upon reasonable request.

---

## 4. Notes on Implementations
This repository includes Python implementations of several SSVER decoding methods:

- **./my_code/algorithms/cca.py**  
  Implementation of FBCCA (Filter Bank Canonical Correlation Analysis)[1].

- **./my_code/algorithms/trca.py**  
  Implementation of TRCA (Task-Related Component Analysis) [2], ms-TRCA [3], eTRCA, and ms-eTRCA.

- **./my_code/algorithms/ress.py**  
  Implementation of RESS (Rhythmic Entrainment Source Separation) [4][5] and eRESS.

- **./my_code/algorithms/tdca.py**  
  Implementation of TDCA (Task Discriminant Component Analysis) [6].

- **./my_code/algorithms/epca.py**  
  Implementation of EPCA (Exactly Periodic Component Analysis) and eEPCA.

- **Deep learning methods**
  CCNN [7], ConvCA [8], FB-tCNN [9], SSVEPformer [10], SSVEPNet [11]
  

---

## 5. Usage Examples

### Standard Methods (FBCCA / EPCA / RESS / TDCA / TRCA)
```python
from cca import FBCCA
from ress import RESS
from prca import TDCA
from trca import TRCA
from epca import EPCA

model = FBCCA(weights_filterbank=weights_filterbank)

model = EPCA(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank
)

model = RESS(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank,
    ress_param={'peakwidt': 0.75, 'neighfreq': 3, 'neighwidt': 3}
)

model = TDCA(n_component=8, padding_len=5)

model = TRCA(weights_filterbank=weights_filterbank)

# X_train: list of trials
# y_train: list of labels
model.fit(X_train, y_train)

# X_test: list of trials
Y_pred = model.predict(X_test)
```

### Enhanced Methods (eEPCA / eRESS / eTRCA / ms-eTRCA)
```python
from epca import EEPCA
from ress import ERESS
from trca import ETRCA, MSETRCA

model = EEPCA(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank
)

model = ERESS(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank,
    ress_param={'peakwidt': 0.75, 'neighfreq': 3, 'neighwidt': 3}
)

model = MSETRCA(n_neighbor=2, weights_filterbank = weights_filterbank)
model = ETRCA(weights_filterbank=weights_filterbank)

model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
```

## 6. Cite

If you use any part of the code, please cite the following publication:

- Wang F, Cao F, Yang J, et al. Enhancing Individual Calibration Classification in SSVER Based BCI with Exactly Periodic Component Analysis[J]. IEEE Transactions on Industrial Informatics, 2026.

## 7. Acknowledgements
- [1] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface[J]. Journal of neural engineering, 2015, 12(4): 046008.
- [2] Nakanishi M, Wang Y, Chen X, et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2018, 65(1): 104-112.[DOI: 10.1109/TBME.2017.2694818](10.1109/TBME.2017.2694818)
- [3] Wong C M, Wan F, Wang B, et al. Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs[J]. Journal of Neural Engineering, 2020, 17(1): 016026.[DOI: 10.1088/1741-2552/ab2373](10.1088/1741-2552/ab2373)
- [4] Xu W, Ke Y, Ming D. Improving the Performance of Individually Calibrated SSVEP Classification by Rhythmic Entrainment Source Separation[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024: 1-1.[DOI: 10.1109/TNSRE.2024.3503772](10.1109/TNSRE.2024.3503772)
- [5] Cohen M X, Gulbinaite R. Rhythmic entrainment source separation: Optimizing analyses of neural responses to rhythmic sensory stimulation[J]. NeuroImage, 2017, 147: 43-56.[DOI: 10.1016/j.neuroimage.2016.11.036](10.1016/j.neuroimage.2016.11.036)
- [6] Liu B, Chen X, Shi N, et al. Improving the Performance of Individually Calibrated SSVEP-BCI by Task- Discriminant Component Analysis[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021, 29: 1998-2007.
- [7] Waytowich N, Lawhern V J, Garcia J O, et al. Compact convolutional neural networks for classification of asynchronous steady-state visual evoked potentials[J]. Journal of Neural Engineering, 2018, 15(6): 066031.
- [8] Li Y, Xiang J, Kesavadas T. Convolutional Correlation Analysis for Enhancing the Performance of SSVEP-Based Brain-Computer Interface[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28(12): 2681-2690.
- [9] Ding W, Shan J, Fang B, et al. Filter Bank Convolutional Neural Network for Short Time-Window Steady-State Visual Evoked Potential Classification[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021, 29: 2615-2624.
- [10] Chen J, Zhang Y, Pan Y, et al. A transformer-based deep neural network model for SSVEP classification[J]. Neural Networks, 2023, 164: 521-534.
- [11] Pan Y, Chen J, Zhang Y, et al. An efficient CNN-LSTM network with spectral normalization and label smoothing technologies for SSVEP frequency recognition[J]. Journal of Neural Engineering, 2022, 19(5): 056014.
- [12] Muresan D D, Parks T W. Orthogonal, exactly periodic subspace decomposition[J]. IEEE Transactions on Signal Processing, 2003, 51(9): 2270-2279. [DOI: 10.1109/TSP.2003.815381](10.1109/TSP.2003.815381)
- [13] Wang Y, Chen X, Gao X, et al. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10): 1746-1752.[DOI: 10.1109/TNSRE.2016.2627556](10.1109/TNSRE.2016.2627556)


