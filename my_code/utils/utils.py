from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, linspace, pi, sin, cos, expand_dims, concatenate
import numpy as np
from scipy import signal

def nextpow2(n):
    '''
    Retrun the first P such that 2 ** P >= abs(n).
    '''
    return np.ceil(np.log2(np.abs(n)))

def fft(X : ndarray,
        fs : float,
        detrend_flag : bool = True,
        NFFT : Optional[int] = None):
    """
    Calculate FFT

    Parameters
    -----------
    X : ndarray
        Input signal. The shape is (1*N) where N is the sampling number.
    fs : float
        Sampling freqeuncy.
    detrend_flag : bool
        Whether detrend. If True, X will be detrended first. Default is True.
    NFFT : Optional[int]
        Number of FFT. If None, NFFT is equal to 2^nextpow2(X.shape[1]). Default is None.

    Returns
    -------------
    freqs : ndarray
        Corresponding frequencies
    fft_res : ndarray
        FFT result
    """
    X_raw, X_col = X.shape
    if X_raw!=1:
        raise ValueError('The row number of the input signal for the FFT must be 1.')
    if X_col==1:
        raise ValueError('The column number of the input signal for the FFT cannot be 1.')
    if NFFT is None:
        NFFT = 2 ** nextpow2(X_col)
    if type(NFFT) is not int:
        NFFT = int(NFFT)
    
    if detrend_flag:
        X = signal.detrend(X, axis = 1)

    fft_res = np.fft.fft(X, NFFT, axis = 1)
    freqs = np.fft.fftfreq(NFFT, 1/fs)
    freqs = np.expand_dims(freqs,0)
    if NFFT & 0x1:
        fft_res = fft_res[:,:int((NFFT+1)/2)]
        freqs = freqs[:,:int((NFFT+1)/2)]
    else:
        fft_res = fft_res[:,:int(NFFT/2)]
        freqs = freqs[:,:int(NFFT/2)]
    # fft_res = fft_res/X_col
    
    return freqs, fft_res

def freqs_snr(X : ndarray,
              target_fre : float,
              srate : float,
              Nh : int,
              detrend_flag : bool = True,
              NFFT : Optional[int] = None):
    """
    Calculate FFT and then calculate SNR
    """
    freq, fft_res = fft(X, srate, detrend_flag = detrend_flag, NFFT = NFFT)
    abs_fft_res = np.abs(fft_res)

    stim_amp = 0
    for n in range(Nh):
        freLoc = np.argmin(np.abs(freq - (target_fre*(n+1))))
        stim_amp += abs_fft_res[0,freLoc]
    snr = 10*np.log10(stim_amp/(np.sum(abs_fft_res)-stim_amp))
    return snr

def amplitude_spectrum(X,sfreq=1000):
    sfreq = sfreq
    if X.ndim == 1:
        n = len(X)
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量  
        Y = np.fft.fft(X)  # 计算FFT  
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[:n//2]
    else:
        n = X.shape[-1]
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量
        Y = np.fft.fft(X,axis=-1)
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[...,:n//2]
    
def find_nearest_index(array, values):
    array = np.asarray(array)
    if not isinstance(values, list):
        index=(np.abs(array - values)).argmin()
        return index
    else:
        index = []
        for value in values:
            index.append((np.abs(array - value)).argmin())
        return index
    
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    """
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd)

    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

def print_significance(p_value):
        if p_value < 0.0001:
            return "****"
        elif p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns" 
        

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def half_violin(ax, data, pos, side='right', width=0.3, **kwargs):
    import scipy.stats
    # 计算核密度估计
    kde = scipy.stats.gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    y = kde(x)

    # 归一化y值
    y = y / y.max() * width

    if side == 'right':
        ax.fill_betweenx(x, pos, pos + y, **kwargs)
    else:
        ax.fill_betweenx(x, pos - y, pos, **kwargs)


def bandpass_filter(dataself, X, lowcut, highcut, order=5):
    nyq = dataself.srate / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if X.ndim == 1:
        Y = signal.filtfilt(b, a, X)
    else:
        Y = signal.filtfilt(b, a, X,axis=X.ndim-1)
    return Y