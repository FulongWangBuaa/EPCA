# Overview
`EPCA (Exactly Periodic Component Analysis)` is used for enhancing individual calibration classification in SSVER-based BCI.

Implementation code for the `EPCA` method is provided here.


# Requirement
This project depends on the SSVEPAnalysisToolbox Python package, which can be installed following the instructions available at: [https://ssvep-analysis-toolbox.readthedocs.io/en/latest/Installation\_pages/Installation.html](https://ssvep-analysis-toolbox.readthedocs.io/en/latest/Installation_pages/Installation.html)

# Dataset
The `Benchmark` dataset can be accessed at: [https://bci.med.tsinghua.edu.cn/download.html](https://bci.med.tsinghua.edu.cn/download.html)



# Note
- The file `trca.py` is a Python implementation of the `TRCA (Task-Related Component Analysis)` [1] , `ms-TRCA`[2] ,`eTRCA` and `ms-eTRCA` method.
- The file `ress.py` is a Python implementation of the `RESS (Rhythmic Entrainment Source Separation)` [3] and `eRESS` method.
- The file `prca.py` is a Python implementation of the `PRCA (Periodically Repeated Component Analysis)` [5] and `ePRCA` method.
- The file `epca.py` is a Python implementation of the `EPCA (Exactly Periodic Component Analysis)` and `eEPCA` method. (The code will be released upon acceptance of the paper.)

# Usage Example
```python
# EPCA, RESS, PRCA, TRCA
from epca import EPCA
from ress import RESS
from prca import PRCA
from trca import TRCA
model = EPCA(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank)
model = RESS(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank,
             ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3})
model = PRCA(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank)
model = TRCA(weights_filterbank=weights_filterbank)

# X_train: List[]   y_train:List[]
model.fit(X_train,y_train)

# X_test: List[]
Y_pred = model.predict(X_test)
```

```python
# eEPCA, eRESS, ePRCA, eTRCA
from epca import EEPCA
from ress import ERESS
from prca import EPRCA
from trca import ETRCA
model = EEPCA(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank)
model = ERESS(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank,
             ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3})
model = EPRCA(stim_freqs=stim_freqs,srate=srate,weights_filterbank=weights_filterbank)
model = ETRCA(weights_filterbank=weights_filterbank)

# X_train: List[]   y_train:List[]
model.fit(X_train,y_train)

# X_test: List[]
Y_pred = model.predict(X_test)
```

# Cite
If you use any part of the code, please cite the following publication:

- ...

# Acknowledgements
- [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2018, 65(1): 104-112.[DOI: 10.1109/TBME.2017.2694818](10.1109/TBME.2017.2694818)
- [2] Wong C M, Wan F, Wang B, et al. Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs[J]. Journal of Neural Engineering, 2020, 17(1): 016026.[DOI: 10.1088/1741-2552/ab2373](10.1088/1741-2552/ab2373)
- [3] Xu W, Ke Y, Ming D. Improving the Performance of Individually Calibrated SSVEP Classification by Rhythmic Entrainment Source Separation[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024: 1-1.[DOI: 10.1109/TNSRE.2024.3503772](10.1109/TNSRE.2024.3503772)
- [4] Cohen M X, Gulbinaite R. Rhythmic entrainment source separation: Optimizing analyses of neural responses to rhythmic sensory stimulation[J]. NeuroImage, 2017, 147: 43-56.[DOI: 10.1016/j.neuroimage.2016.11.036](10.1016/j.neuroimage.2016.11.036)
- [5] Ke Y, Liu S, Ming D. Enhancing SSVEP Identification With Less Individual Calibration Data Using Periodically Repeated Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2024, 71(4): 1319-1331.[DOI: 10.1109/TBME.2023.3333435](10.1109/TBME.2023.3333435)
- [6] Muresan D D, Parks T W. Orthogonal, exactly periodic subspace decomposition[J]. IEEE Transactions on Signal Processing, 2003, 51(9): 2270-2279. [DOI: 10.1109/TSP.2003.815381](10.1109/TSP.2003.815381)
- [7] Wang Y, Chen X, Gao X, et al. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10): 1746-1752.[DOI: 10.1109/TNSRE.2016.2627556](10.1109/TNSRE.2016.2627556)
