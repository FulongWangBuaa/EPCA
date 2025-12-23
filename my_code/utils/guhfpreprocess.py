# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

# ch_names_guhf= [
#         'FP1', 'FPz', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
#         'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
#         'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5',
#         'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3',
#         'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
#         'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
ch_names_guhf = ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PZ']

def suggested_weights_filterbank(num_subbands: Optional[int] = 5) -> List[float]:
    """
    Provide suggested weights of filterbank for guhf dataset

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def guhf_suggested_ch(ch_num) -> List[int]:
    ch_suggest_order = ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PZ']
    ch_used = ch_suggest_order[:ch_num]

    pick_ch_guhf_idx = [ch_names_guhf.index(pick_ch) for pick_ch in ch_used]
    return pick_ch_guhf_idx


def preprocess(dataself,
               X: ndarray) -> ndarray:
    """
    Suggested preprocessing function for guhf dataset
    
    notch filter at 50 Hz
    """
    srate = dataself.srate

    # notch filter at 50 Hz
    f0 = 50
    Q = 35
    notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    preprocess_X = signal.filtfilt(notchB, notchA, X, axis = 1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))
    
    return preprocess_X
    
def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for guhf dataset
    """
    srate = dataself.srate
    
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    
    for k in range(1, num_subbands+1, 1):
        Wp = [(8*k)/(srate/2), 90/(srate/2)]
        Ws = [(8*k-2)/(srate/2), 100/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                break
            except:
                gstop -= 1
        if gstop<20:
            raise ValueError("""
Filterbank cannot be processed. You may try longer signal lengths.
Filterbank order: {:n}
gstop: {:n}
bpB: {:s}
bpA: {:s}
Required signal length: {:n}
Signal length: {:n}""".format(k,
                                gstop,
                                str(bpB),
                                str(bpA),
                                3*(max(len(bpB),len(bpA))-1),
                                X.shape[1]))
        
        
    return filterbank_X