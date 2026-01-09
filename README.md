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

### Cite
If you use any part of the code, please cite the following publication:

- [1] Wang F, Cao F, Yang J, et al. Enhancing Individual Calibration Classification in SSVER Based BCI with Exactly Periodic Component Analysis[J]. IEEE Transactions on Industrial Informatics, 2026.

### Acknowledgements
- [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2018, 65(1): 104-112.[DOI: 10.1109/TBME.2017.2694818](10.1109/TBME.2017.2694818)
- [2] Wong C M, Wan F, Wang B, et al. Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs[J]. Journal of Neural Engineering, 2020, 17(1): 016026.[DOI: 10.1088/1741-2552/ab2373](10.1088/1741-2552/ab2373)
- [3] Xu W, Ke Y, Ming D. Improving the Performance of Individually Calibrated SSVEP Classification by Rhythmic Entrainment Source Separation[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024: 1-1.[DOI: 10.1109/TNSRE.2024.3503772](10.1109/TNSRE.2024.3503772)
- [4] Cohen M X, Gulbinaite R. Rhythmic entrainment source separation: Optimizing analyses of neural responses to rhythmic sensory stimulation[J]. NeuroImage, 2017, 147: 43-56.[DOI: 10.1016/j.neuroimage.2016.11.036](10.1016/j.neuroimage.2016.11.036)
- [5] Ke Y, Liu S, Ming D. Enhancing SSVEP Identification With Less Individual Calibration Data Using Periodically Repeated Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2024, 71(4): 1319-1331.[DOI: 10.1109/TBME.2023.3333435](10.1109/TBME.2023.3333435)
- [6] Muresan D D, Parks T W. Orthogonal, exactly periodic subspace decomposition[J]. IEEE Transactions on Signal Processing, 2003, 51(9): 2270-2279. [DOI: 10.1109/TSP.2003.815381](10.1109/TSP.2003.815381)
- [7] Wang Y, Chen X, Gao X, et al. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10): 1746-1752.[DOI: 10.1109/TNSRE.2016.2627556](10.1109/TNSRE.2016.2627556)


