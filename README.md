# Anonymous_EPCA

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

- **Benchmark dataset (Dataset I)**
  [1] Wang Y, Chen X, Gao X, et al. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10): 1746-1752.
  
  https://bci.med.tsinghua.edu.cn/download.html

- **BETA dataset (Dataset II)**
  [2] Liu B, Huang X, Wang Y, et al. BETA: A Large Benchmark Database Toward SSVEP-BCI Application[J]. Frontiers in Neuroscience, 2020, 14.
  
  https://bci.med.tsinghua.edu.cn/download.html

- **eldBETA dataset (Dataset III)**
  [3] Liu B, Wang Y, Gao X, et al. eldBETA: A Large Eldercare-oriented Benchmark Database of SSVEP-BCI for the Aging Population[J]. Scientific Data, 2022, 9(1): 252.
  
  https://doi.org/10.6084/m9.figshare.18032669

- **GuHF dataset (Dataset IV)**
  [4] Gu M, Pei W, Gao X, et al. An open dataset for human SSVEPs in the frequency range of 1-60 Hz[J]. Scientific Data, 2024, 11(1): 196.
  
  https://doi.org/10.6084/m9.figshare.23641092

- **OPMMEG dataset (Dataset V)**  
  The OPMMEG dataset will be made available by the corresponding author upon reasonable request after publication.

---

## 4. Notes on Implementations
This repository includes Python implementations of several SSVER decoding methods:

- **./my_code/algorithms/cca.py**  
  Implementation of FBCCA (Filter Bank Canonical Correlation Analysis).

- **./my_code/algorithms/trca.py**  
  Implementation of TRCA (Task-Related Component Analysis), ms-TRCA, eTRCA, and ms-eTRCA.

- **./my_code/algorithms/ress.py**  
  Implementation of RESS (Rhythmic Entrainment Source Separation) and eRESS.

- **./my_code/algorithms/tdca.py**  
  Implementation of TDCA (Task Discriminant Component Analysis) and ePRCA.

- **./my_code/algorithms/epca.py**  
  Implementation of EPCA (Exactly Periodic Component Analysis) and eEPCA.

---

## 5. Usage Examples

### Standard Methods (EPCA / RESS / PRCA / TRCA)
```python
from epca import EPCA
from ress import RESS
from prca import PRCA
from trca import TRCA

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

model = PRCA(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank
)

model = TRCA(weights_filterbank=weights_filterbank)

# X_train: list of trials
# y_train: list of labels
model.fit(X_train, y_train)

# X_test: list of trials
Y_pred = model.predict(X_test)
```

### Enhanced Methods (eEPCA / eRESS / ePRCA / eTRCA)
```python
from epca import EEPCA
from ress import ERESS
from prca import EPRCA
from trca import ETRCA

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

model = EPRCA(
    stim_freqs=stim_freqs,
    srate=srate,
    weights_filterbank=weights_filterbank
)

model = ETRCA(weights_filterbank=weights_filterbank)

model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
```
