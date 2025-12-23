import numpy as np
import warnings
from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple, cast
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eig, pinv, qr
from scipy import signal
from functools import partial
from joblib import Parallel, delayed
from copy import deepcopy
from functools import reduce
from scipy.stats import pearsonr

from SSVEPAnalysisToolbox.algorithms.basemodel import BaseModel
from SSVEPAnalysisToolbox.algorithms.utils import gen_template

def get_primes(max: int = 1000000) -> list[int]:
    """
    Generate all prime numbers up to a given maximum.

    Parameters
    ----------
    max : int, optional
        The maximum number up to which to generate primes, by default 1000000.

    Returns
    -------
    array_like
        An array of all prime numbers up to the given maximum.
    """
    primes = np.arange(3, max + 1, 2)
    isprime = np.ones((max - 1) // 2, dtype=bool)
    for factor in primes[: int(np.sqrt(max))]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2 :: factor] = 0
    return np.insert(primes[isprime], 0, 2)

def get_factors(n: int, remove_1: bool = False, remove_n: bool = False) -> set:
    """
    Get all factors of a given number.

    Parameters
    ----------
    n : int
        The number to factor.
    remove_1 : bool, optional
        If True, remove 1 from the factors, by default False.
    remove_n : bool, optional
        If True, remove `n` from the factors, by default False.

    Returns
    -------
    set
        A set of all factors of the given number.
    """
    facs = set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )
    # get rid of 1 and the number itself
    if remove_1:
        facs.remove(1)
    if remove_n and n != 1:
        facs.remove(n)
    return facs  # retuned as a set


def project(
        data: list[float],
        p: int = 2,
        trunc_to_integer_multiple: bool = False,
        orthogonalize: bool = False,
        return_single_period: bool = False,
    ) -> list[float]:
        """
        Projects the data onto a lower-dimensional space.

        Parameters
        ----------
            data : list[float]
                The data to be projected.
            p : int, optional
                The period (default is 2).
            trunc_to_integer_multiple : bool, optional
                Whether or not to truncate the window to fit an integer multiple of the period (default is False).
            orthogonalize : bool, optional
                Whether or not to orthogonalize the projections (default is False).
            return_single_period : bool, optional
                Whether or not to return a single period (default is False).

        Returns
        -------
            list[float]
                The projected data.
        """
        PRIMES = set(get_primes(10000)) 
        cp = data.copy()
        samples_short = int(
            np.ceil(len(cp) / p) * p - len(cp)
        )  # calc how many samples short for rectangle
        cp = np.pad(cp, (0, samples_short))  # pad it
        cp = cp.reshape(int(len(cp) / p), p)  # reshape it to a rectangle

        if trunc_to_integer_multiple:
            if samples_short == 0:
                single_period = np.mean(cp, 0)  # don't need to omit the last row
            else:
                single_period = np.mean(
                    cp[:-1], 0
                )  # just take the mean of the truncated version and output a single period
        else:
            ## this is equivalent to the method presented in Sethares but significantly faster.
            ## do the mean manually. get the divisors from the last row since the last samples_short values will be one less than the others
            divs = np.zeros(cp.shape[1])
            for i in range(cp.shape[1]):
                if i < (cp.shape[1] - samples_short):
                    divs[i] = cp.shape[0]
                else:
                    divs[i] = cp.shape[0] - 1
            single_period = np.sum(cp, 0) / divs  # get the mean manually

        projection = np.tile(single_period, int(data.size / p) + 1)[
            : len(data)
        ]  # extend the period and take the good part

        # a faster, cleaner way to orthogonalize that is equivalent to the method
        # presented in "Orthogonal, exactly periodic subspace decomposition" (D.D.
        # Muresan, T.W. Parks), 2003. Setting trunc_to_integer_multiple gives a result
        # that is almost exactly identical (within a rounding error; i.e. 1e-6).
        # For the outputs of each to be identical, the input MUST be the same length
        # with DC removed since the algorithm in Muresan truncates internally and
        # here we allow the output to assume the dimensions of the input. See above
        # line of code.
        if orthogonalize:
            for f in get_factors(p, remove_1=True, remove_n=True):
                if f in PRIMES:
                    # remove the projection at p/prime_factor, taking care not to remove things twice.
                    projection = projection - project(
                        projection, int(p / f), trunc_to_integer_multiple, False
                    )

        if return_single_period:
            return projection[0:p]  # just a single period
        else:
            return projection  # the whole thing



import numpy as np
import warnings
from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple, cast
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eig, pinv, qr
from scipy import signal
from functools import partial
from joblib import Parallel, delayed
from copy import deepcopy

from SSVEPAnalysisToolbox.algorithms.basemodel import BaseModel
from SSVEPAnalysisToolbox.algorithms.utils import (
    gen_template, sort, canoncorr, separate_trainSig, qr_list, blkrep, eigvec, cholesky,
    inv, repmat
)

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


def _epca_U(X: list, stim_freq: float, srate: float):
    """
    Calculate spatial filters of epca

    Parameters
    ------------
    X : list
        List of EEG data
        Length: (trial_num,)
        Shape of EEG: (channel_num, signal_len)

    Returns
    -----------
    U : ndarray
        Spatial filter
        shape: (channel_num * n_component)
    """
    n_trials = len(X)
    n_channels, n_samples = X[0].shape

    # Ln = int(np.round(srate/stim_freq))
    # Pn = int(np.floor(n_samples/Ln))
    # tmp_data_prca = np.zeros((n_channels,Ln,n_trials,Pn))
    # for idx_trial in range(n_trials):
    #     data_trial = X[idx_trial]
    #     for idx_pn in range(Pn):
    #         tmp = data_trial[:,idx_pn*Ln:(idx_pn+1)*Ln]
    #         tmp_data_prca[:,:,idx_trial,idx_pn] = tmp - np.mean(tmp,axis=1,keepdims=True)
    # PRCA
    # 所有周期重复成分按时间拼接
    # X1 = tmp_data_prca.reshape((n_channels, -1)) # (n_channels, Ln*n_trials*Pn)
    X1 = np.concatenate(X,axis=1)

    X_mean = np.mean(X, axis=0)
    X3 = np.zeros(X_mean.shape)
    periodic = int(round(srate/stim_freq))
    for ch_idx in range(n_channels):
        X3[ch_idx,:] = project(X_mean[ch_idx,:], periodic, True, True)

    Q = np.dot(X1,X1.T)/X1.shape[1]
    S = np.dot(X3,X3.T)/X3.shape[1] - Q
    eig_val,eig_vec = eig(S,Q)

    sort_idx = np.argsort(eig_val)[::-1]
    U = eig_vec[:,sort_idx]
    return U

def _epca_coor2(A,B):
    '''
    CORR2 2-D correlation coefficient

    Parameters
    ----------
    A: (n_samples, n_stimulus)
    B: (n_samples, n_stimulus)

    Returns
    -------
    R: float
    '''
    if any([A.shape[0] != B.shape[0], A.shape[1] != B.shape[1]]):
        raise ValueError('A has shape {:s}, B has shape {:s}', str(A.shape), str(B.shape))
    A = np.double(A)
    B = np.double(B)

    A = A - np.mean(A)
    B = B - np.mean(B)

    R = np.sum(np.sum(A*B,axis=0,keepdims=True))/np.sqrt(np.sum(np.sum(A*A,axis=0,keepdims=True))*np.sum(np.sum(B*B,axis=0,keepdims=True)))
    return R

def _r_cca_canoncorr(X: ndarray,
                     Y: List[ndarray],
                     U: ndarray,
                     V: ndarray,
                     stim_freqs: List[float], 
                     srate: float) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
        List shape: (stimulus_num,)
        Reference shape: (filterbank_num, channel_num, signal_len)
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    """
    n_banks, n_channels, n_samples = X.shape
    n_stims = len(Y)

    R = np.zeros((n_banks, n_stims))

    for idx_fb in range(n_banks):
        X_tmp = X[idx_fb,:,:]
        for idx_class in range(n_stims):
            stim_freq = stim_freqs[idx_class]

            Y_tmp = Y[idx_class][idx_fb,:,:]

            w1 = U[idx_fb,idx_class,:,:]
            w2 = V[idx_fb,idx_class,:,:]

            A = X_tmp.T @ w1
            B = Y_tmp.T @ w2

            r = _epca_coor2(A,B)
            R[idx_fb,idx_class] = r
    return R

class EPCA(BaseModel):
    """
    EPCA method
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: float = 1000,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        super().__init__(ID = 'EPCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.srate = srate
        self.stim_freqs = stim_freqs

    def __copy__(self):
        copy_model = EPCA(stim_freqs = self.stim_freqs,
                          srate = self.srate,
                          n_component = self.n_component,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model
    
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('TRCA requires training label')
        if X is None:
            raise ValueError('TRCA requires training data')

        # List of shape: (stimulus_num,)
        # Template shape: (filterbank_num, channel_num, signal_len)
        template_sig = gen_template(X, Y)
                                          
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        n_component = self.n_component
        U_trca = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(_epca_U)(X=X_single_class, srate=self.srate, stim_freq=stim_freq) for X_single_class,stim_freq in zip(X_train, self.stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _epca_U(X = X_single_class,srate=self.srate,stim_freq=stim_freq)
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
        self.model['U'] = U_trca

    def predict(self,X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")

        template_sig = self.model['template_sig']
        U = self.model['U']

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, Y=template_sig, U=U, V=U, stim_freqs=self.stim_freqs, srate=self.srate))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(_r_cca_canoncorr(X=a, Y=template_sig, U=U, V=U, stim_freqs=self.stim_freqs, srate=self.srate))

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]

        return Y_pred, r


class EEPCA(BaseModel):
    """
    eEPCA method
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: float = 1000,
                 n_component: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        if n_component is not None:
            warnings.warn("Although 'n_component' is provided, it will not considered in eEPCA")
        n_component = 1
        super().__init__(ID = 'eEPCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.srate = srate
        self.stim_freqs = stim_freqs

    def __copy__(self):
        copy_model = EEPCA(stim_freqs = self.stim_freqs,
                          srate = self.srate,
                          n_component = None,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model
    
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('eTRCA requires training label')
        if X is None:
            raise ValueError('eTRCA requires training data')

        template_sig = gen_template(X, Y)  # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        # n_component = 1
        U_epca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(_epca_U)(X=X_single_class, srate=self.srate, stim_freq=stim_freq) for X_single_class,stim_freq in zip(X_train, self.stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _epca_U(X = X_single_class,srate=self.srate,stim_freq=stim_freq)
                    )

            for stim_idx, u in enumerate(U):
                U_epca[filterbank_idx, 0, :, stim_idx] = u[:channel_num,0]
        U_epca = np.repeat(U_epca, repeats = stimulus_num, axis = 1)

        self.model['U'] = U_epca


    def predict(self,X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")

        template_sig = self.model['template_sig']
        U = self.model['U']

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, Y=template_sig, U=U, V=U, stim_freqs=self.stim_freqs, srate=self.srate))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(_r_cca_canoncorr(X=a, Y=template_sig, U=U, V=U, stim_freqs=self.stim_freqs, srate=self.srate))

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]

        return Y_pred, r


