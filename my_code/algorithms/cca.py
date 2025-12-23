from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy
import warnings

import numpy as np

from my_code.utils.ssveputils import (
    qr_remove_mean, qr_inverse, mldivide, canoncorr, qr_list, 
    gen_template, sort, separate_trainSig, blkrep, blkmat, eigvec,
    svd, repmat
)

def _msetcca_cal_template_U(X_single_stimulus : ndarray,
                            I : ndarray):
    """
    Calculate templates and trials' spatial filters in multi-set CCA
    """
    trial_num, filterbank_num, channel_num, signal_len = X_single_stimulus.shape
    n_component = 1
    # prepare center matrix
    # I = np.eye(signal_len)
    LL = repmat(I, trial_num, trial_num) - blkrep(I, trial_num)
    # calculate templates and spatial filters of each filterbank
    U_trial = []
    CCA_template = []
    for filterbank_idx in range(filterbank_num):
        X_single_stimulus_single_filterbank = X_single_stimulus[:,filterbank_idx,:,:]
        template = blkmat(X_single_stimulus_single_filterbank)
        # calculate spatial filters of trials
        Sb = template @ LL @ template.T
        Sw = template @ template.T
        eig_vec = eigvec(Sb, Sw)[:,:n_component]
        U_trial.append(np.expand_dims(eig_vec, axis = 0))
        # calculate template
        template = []
        for trial_idx in range(trial_num):
            template_temp = eig_vec[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component].T @ X_single_stimulus_single_filterbank[trial_idx,:,:]
            template.append(template_temp)
        template = np.concatenate(template, axis = 0)
        CCA_template.append(np.expand_dims(template, axis = 0))
    U_trial = np.concatenate(U_trial, axis = 0)
    CCA_template = np.concatenate(CCA_template, axis = 0)
    return U_trial, CCA_template

def _oacca_cal_u1_v1(filteredData : ndarray,
                     sinTemplate : ndarray,
                     old_Cxx : ndarray,
                     old_Cxy : ndarray):
    """
    Calculate online adaptive multi-stimulus spatial filter in OACCA

    Parameters
    --------------
    filteredData : ndarray
        input signal
    sinTemplate : ndarray
        reference signal
    old_Cxx : ndarray
        Covariance matrix of input signal in previous step
    old_Cxy : ndarray
        Covariance matrix of input signal and reference signal in previous step
    """
    # Calculate multi-stimulus 
    # filteredData = x_single_trial[k,:,:]
    # sinTemplate = Y[prototype_res][:,:signal_len]

    channel_num = filteredData.shape[0]
    harmonic_num = sinTemplate.shape[0]

    # self.model['Cxx'][:,:,k] = self.model['Cxx'][:,:,k] + filteredData @ filteredData.T
    new_Cxx = old_Cxx + filteredData @ filteredData.T
    # self.model['Cxy'][:,:,k] = self.model['Cxy'][:,:,k] + filteredData @ sinTemplate.T
    new_Cxy = old_Cxy + filteredData @ sinTemplate.T

    CCyy = np.eye(harmonic_num)
    CCyx = new_Cxy.T
    CCxx = new_Cxx
    CCxy = new_Cxy
    A1 = np.concatenate((np.zeros(CCxx.shape), CCxy), axis = 1)
    A2 = np.concatenate((CCyx, np.zeros(CCyy.shape)), axis = 1)
    A = np.concatenate((A1, A2), axis = 0)
    B1 = np.concatenate((CCxx, np.zeros(CCxy.shape)), axis = 1)
    B2 = np.concatenate((np.zeros(CCyx.shape), CCyy), axis = 1)
    B = np.concatenate((B1, B2), axis = 0)
    eig_vec = eigvec(A, B)
    u1 = eig_vec[:channel_num,:]
    v1 = eig_vec[channel_num:,:]
    if u1[0,0] == 1:
        # warnings.warn("Warning: updated U is not meaningful and thus adjusted.")
        u1 = np.zeros((channel_num,1))
        u1[-3:] = 1
    u1 = u1[:,0]
    v1 = v1[:,0]

    return u1, v1, new_Cxx, new_Cxy

def _oacca_cal_u0(sf1x : ndarray, 
                  old_covar_mat : ndarray):
    """
    Calculate updated prototype filter in OACCA

    Parameters
    -------------
    sf1x : ndarray
        Spatial filter obtained from CCA
    old_covar_mat : ndarray
        Covariance matrix of spatial filter in previous step
    """
    channel_num = old_covar_mat.shape[0]

    # sf1x = cca_sfx[k,cca_res,:,:] 
    sf1x = sf1x/np.linalg.norm(sf1x)
    # sf1y = cca_sfy[k,cca_res,:,:]
    # sf1y = sf1y/np.linalg.norm(sf1y)

    # self.model['covar_mat'][:,:,k] = self.model['covar_mat'][:,:,k] + sf1x @ sf1x.T
    new_covar_mat = old_covar_mat + sf1x @ sf1x.T
    eig_vec = eigvec(new_covar_mat)
    u0 = eig_vec[:channel_num,0]
    return u0, new_covar_mat
    
    # for class_i in range(stimulus_num):
    #     self.model['U0'][k,class_i,:,0] = u0 

def _r_cca_canoncorr_withUV(X: ndarray,
                            Y: List[ndarray],
                            U: ndarray,
                            V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
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
    filterbank_num, channel_num, signal_len = X.shape
    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            # r = stats.pearsonr(a, b)[0]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r
    return R

def _r_cca_qr_withUV(X: ndarray,
                  Y_Q: List[ndarray],
                  Y_R: List[ndarray],
                  Y_P: List[ndarray],
                  U: ndarray,
                  V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on qr decomposition for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y_Q : List[ndarray]
        Q of reference signals
    Y_R: List[ndarray]
        R of reference signals
    Y_P: List[ndarray]
        P of reference signals
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
    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y_R[0].shape[-1]
    stimulus_num = len(Y_Q)
    
    Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]) for i in range(len(Y_Q))]
    if len(Y[0].shape)==2: # reference
        Y = [Y_tmp.T for Y_tmp in Y]
    elif len(Y[0].shape)==3: # template
        Y = [np.transpose(Y_tmp, (0,2,1)) for Y_tmp in Y]
    else:
        raise ValueError('Unknown data type')
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
                
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            # r = stats.pearsonr(a, b)[0]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r
    return R
    
def _r_cca_canoncorr(X: ndarray,
                     Y: List[ndarray],
                     n_component: int,
                     force_output_UV: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
    """
    Calculate correlation of CCA based on canoncorr for single trial data 

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
    n_component : int
        Number of eigvectors for spatial filters.
    force_output_UV : Optional[bool]
        Whether return spatial filter 'U' and weights of harmonics 'V'

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)
    """
    filterbank_num, channel_num, signal_len = X.shape
    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)
    
    # R1 = np.zeros((filterbank_num,stimulus_num))
    # R2 = np.zeros((filterbank_num,stimulus_num))
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
                
            if n_component == 0 and force_output_UV is False:
                D = canoncorr(tmp.T, Y_tmp.T, False)
                r = D[0]
            else:
                A_r, B_r, D = canoncorr(tmp.T, Y_tmp.T, True)
                
                a = A_r[:channel_num, :n_component].T @ tmp
                b = B_r[:harmonic_num, :n_component].T @ Y_tmp
                a = np.reshape(a, (-1))
                b = np.reshape(b, (-1))
                
                # r = stats.pearsonr(a, b)[0]
                r = np.corrcoef(a, b)[0,1]
                U[k,i,:,:] = A_r[:channel_num, :n_component]
                V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R

def _r_cca_qr(X: ndarray,
           Y_Q: List[ndarray],
           Y_R: List[ndarray],
           Y_P: List[ndarray],
           n_component: int,
           force_output_UV: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
    """
    Calculate correlation of CCA based on QR decomposition for single trial data 

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y_Q : List[ndarray]
        Q of reference signals
    Y_R: List[ndarray]
        R of reference signals
    Y_P: List[ndarray]
        P of reference signals
    n_component : int
        Number of eigvectors for spatial filters.
    force_output_UV : Optional[bool]
        Whether return spatial filter 'U' and weights of harmonics 'V'

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)
    """
    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y_R[0].shape[-1]
    stimulus_num = len(Y_Q)
    
    Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]) for i in range(len(Y_Q))]
    if len(Y[0].shape)==2: # reference
        Y = [Y_tmp.T for Y_tmp in Y]
    elif len(Y[0].shape)==3: # template
        Y = [np.transpose(Y_tmp, (0,2,1)) for Y_tmp in Y]
    else:
        raise ValueError('Unknown data type')
    
    # R1 = np.zeros((filterbank_num,stimulus_num))
    # R2 = np.zeros((filterbank_num,stimulus_num))
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        X_Q, X_R, X_P = qr_remove_mean(tmp.T)
        for i in range(stimulus_num):
            if len(Y_Q[i].shape)==2: # reference
                Y_Q_tmp = Y_Q[i]
                Y_R_tmp = Y_R[i]
                Y_P_tmp = Y_P[i]
                Y_tmp = Y[i]
            elif len(Y_Q[i].shape)==3: # template
                Y_Q_tmp = Y_Q[i][k,:,:]
                Y_R_tmp = Y_R[i][k,:,:]
                Y_P_tmp = Y_P[i][k,:]
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            svd_X = X_Q.T @ Y_Q_tmp
            if svd_X.shape[0]>svd_X.shape[1]:
                full_matrices=False
            else:
                full_matrices=True
            
            if n_component == 0 and force_output_UV is False:
                D = svd(svd_X, full_matrices, False)
                r = D[0]
            else:
                L, D, M = svd(svd_X, full_matrices, True)
                M = M.T
                A = mldivide(X_R, L) * np.sqrt(signal_len - 1)
                B = mldivide(Y_R_tmp, M) * np.sqrt(signal_len - 1)
                A_r = np.zeros(A.shape)
                for n in range(A.shape[0]):
                    A_r[X_P[n],:] = A[n,:]
                B_r = np.zeros(B.shape)
                for n in range(B.shape[0]):
                    B_r[Y_P_tmp[n],:] = B[n,:]
                
                a = A_r[:channel_num, :n_component].T @ tmp
                b = B_r[:harmonic_num, :n_component].T @ Y_tmp
                a = np.reshape(a, (-1))
                b = np.reshape(b, (-1))
                
                # r2 = stats.pearsonr(a, b)[0]
                # r = stats.pearsonr(a, b)[0]
                r = np.corrcoef(a, b)[0,1]
                U[k,i,:,:] = A_r[:channel_num, :n_component]
                V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
            # R1[k,i] = r1
            # R2[k,i] = r2
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R
    

class FBCCA():
    """
    Filter bank standard CCA based on canoncorr
    
    Computational time - Long
    Required memory - Small
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: bool = False,
                 update_UV: bool = True):
        """
        Special Parameters
        ----------
        force_output_UV : Optional[bool] 
            Whether store U and V. Default is False
        update_UV: Optional[bool]
            Whether update U and V in next time of applying "predict" 
            If false, and U and V have not been stored, they will be stored
            Default is True
        """
        if n_component < 0:
            raise ValueError('n_component must be larger than 0')
        
        self.ID = 'sCCA (canoncorr)'
        self.n_component = n_component
        self.n_jobs = n_jobs
        
        self.model = {}
        self.model['weights_filterbank'] = weights_filterbank

        self.force_output_UV = force_output_UV
        self.update_UV = update_UV
        
        self.model['U'] = None # Spatial filter of EEG
        self.model['V'] = None # Weights of harmonics
        
    def __copy__(self):
        copy_model = FBCCA(n_component = self.n_component,
                           n_jobs = self.n_jobs,
                           weights_filterbank = self.model['weights_filterbank'],
                           force_output_UV = self.force_output_UV,
                           update_UV = self.update_UV)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def fit(self,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        ----------------------
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if ref_sig is None:
            raise ValueError('sCCA requires sine-cosine-based reference signal')
           
            
        self.model['ref_sig'] = ref_sig
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
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
        n_component = self.n_component
        Y = self.model['ref_sig']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                if self.n_jobs is not None:
                    r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, n_component=n_component, Y=Y, force_output_UV=True))(a) for a in X))
                else:
                    r = []
                    U = []
                    V = []
                    for a in X:
                        r_temp, U_temp, V_temp = _r_cca_canoncorr(a, n_component=n_component, Y=Y, force_output_UV=True)
                        r.append(r_temp)
                        U.append(U_temp)
                        V.append(V_temp)
                self.model['U'] = U
                self.model['V'] = V
            else:
                if self.n_jobs is not None:
                    r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, n_component=n_component, Y=Y, force_output_UV=False))(a) for a in X)
                else:
                    r = []
                    for a in X:
                        r.append(
                            _r_cca_canoncorr(a, n_component=n_component, Y=Y, force_output_UV=False)
                        )
        else:
            U = self.model['U']
            V = self.model['V']
            if self.n_jobs is not None:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=Y))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
            else:
                r = []
                for a, u, v in zip(X,U,V):
                    r.append(
                        _r_cca_canoncorr_withUV(X=a, U=u, V=v, Y=Y)
                    )
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred, r
    

class FBCCA_qr():
    """
    Standard CCA based on qr decomposition
    
    Computational time - Short
    Required memory - Large
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: bool = False,
                 update_UV: bool = True):
        """
        Special Parameters
        ----------
        force_output_UV : Optional[bool] 
            Whether store U and V. Default is False
        update_UV: Optional[bool]
            Whether update U and V in next time of applying "predict" 
            If false, and U and V have not been stored, they will be stored
            Default is True
        """
        if n_component < 0:
            raise ValueError('n_component must be larger than 0')
        
        self.ID = 'sCCA (qr)'
        self.n_component = n_component
        self.n_jobs = n_jobs
        
        self.model = {}
        self.model['weights_filterbank'] = weights_filterbank

        self.force_output_UV = force_output_UV
        self.update_UV = update_UV
        
        self.model['U'] = None # Spatial filter of EEG
        self.model['V'] = None # Weights of harmonics
        
    def __copy__(self):
        copy_model = FBCCA_qr(n_component = self.n_component,
                                    n_jobs = self.n_jobs,
                                    weights_filterbank = self.model['weights_filterbank'],
                                    force_output_UV = self.force_output_UV,
                                    update_UV = self.update_UV)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def fit(self,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        -----------
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if ref_sig is None:
            raise ValueError('sCCA requires sine-cosine-based reference signal')
            
        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)
            
        self.model['ref_sig_Q'] = ref_sig_Q
        self.model['ref_sig_R'] = ref_sig_R
        self.model['ref_sig_P'] = ref_sig_P
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
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
        n_component = self.n_component
        Y_Q = self.model['ref_sig_Q']
        Y_R = self.model['ref_sig_R']
        Y_P = self.model['ref_sig_P']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                if self.n_jobs is not None:
                    r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True))(a) for a in X))
                else:
                    r = []
                    U = []
                    V = []
                    for a in X:
                        r_temp, U_temp, V_temp = _r_cca_qr(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True)
                        r.append(r_temp)
                        U.append(U_temp)
                        V.append(V_temp)
                self.model['U'] = U
                self.model['V'] = V
            else:
                if self.n_jobs is not None:
                    r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False))(a) for a in X)
                else:
                    r = []
                    for a in X:
                        r.append(
                            _r_cca_qr(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False)
                        )
        else:
            U = self.model['U']
            V = self.model['V']
            if self.n_jobs is not None:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
            else:
                r = []
                for a, u, v in zip(X,U,V):
                    r.append(
                        _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P)
                    )
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred, r