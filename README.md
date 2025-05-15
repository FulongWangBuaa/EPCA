# Overview
`EPCA (Exactly Periodic Component Analysis)` is used for enhancing individual calibration classification in SSVER-based BCI.

Implementation code for the `EPCA` method is provided here.


# Requirement
This project depends on the SSVEPAnalysisToolbox Python package, which can be installed following the instructions available at: [https://ssvep-analysis-toolbox.readthedocs.io/en/latest/Installation\_pages/Installation.html](https://ssvep-analysis-toolbox.readthedocs.io/en/latest/Installation_pages/Installation.html)

# Dataset
The `Benchmark` dataset can be accessed at: [https://bci.med.tsinghua.edu.cn/download.html](https://bci.med.tsinghua.edu.cn/download.html)



## Note
- The file `trca.py` is a Python implementation of the `TRCA` [1] , `ms-TRCA`[2] ,`eTRCA` and `ms-eTRCA` method.
- The file `ress.py` is a Python implementation of the `RESS` [3] and `eRESS` method.
- The file `epca.py` is a Python implementation of the `EPCA` and `eEPCA` method. (The code will be released upon acceptance of the paper.)

# Cite
If you use any part of the code, please cite the following publication:

- ...

# Acknowledgements
- [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2018, 65(1): 104-112.[DOI: 10.1109/TBME.2017.2694818](10.1109/TBME.2017.2694818)
- [2] Wong C M, Wan F, Wang B, et al. Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs[J]. Journal of Neural Engineering, 2020, 17(1): 016026.[DOI: 10.1088/1741-2552/ab2373](10.1088/1741-2552/ab2373)
- [3] Xu W, Ke Y, Ming D. Improving the Performance of Individually Calibrated SSVEP Classification by Rhythmic Entrainment Source Separation[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024: 1-1.[DOI: 10.1109/TNSRE.2024.3503772](10.1109/TNSRE.2024.3503772)
- [4] Muresan D D, Parks T W. Orthogonal, exactly periodic subspace decomposition[J]. IEEE Transactions on Signal Processing, 2003, 51(9): 2270-2279. [DOI: 10.1109/TSP.2003.815381](10.1109/TSP.2003.815381)



