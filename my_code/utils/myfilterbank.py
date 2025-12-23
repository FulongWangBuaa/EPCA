import numpy as np
import warnings
from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple, cast
from scipy import signal

def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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


def filterbank_1(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 1) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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


def filterbank_2(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 2) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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


def filterbank_3(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 3) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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


def filterbank_4(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 4) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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



def filterbank_5(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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



def filterbank_6(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 6) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
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