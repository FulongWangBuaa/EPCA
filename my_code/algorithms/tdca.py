import numpy as np
from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple, cast
from scipy.linalg import eigh,solve,eig, pinv, qr
from scipy.stats import pearsonr
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin



def isPD(B: ndarray) -> bool:
    """Returns true when input matrix is positive-definite, via Cholesky decompositon method.

    Parameters
    ----------
    B : ndarray
        Any matrix, shape (N, N)

    Returns
    -------
    bool
        True if B is positve-definite.

    Notes
    -----
        Use numpy.linalg rather than scipy.linalg. In this case, scipy.linalg has unpredictable behaviors.
    """

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A: ndarray) -> ndarray:
    """Find the nearest positive-definite matrix to input.

    Parameters
    ----------
    A : ndarray
        Any square matrxi, shape (N, N)

    Returns
    -------
    A3 : ndarray
        positive-definite matrix to A

    Notes
    -----
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1]_, which
    origins at [2]_.

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988):
           https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    print("Replace current matrix with the nearest positive-definite matrix.")

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `numpy.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    eye = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += eye * (-mineig * k**2 + spacing)
        k += 1

    return A3


def robust_pattern(W : ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.
        Referring to the method mentioned in article [1],the constructed spatial filter only shows how to combine
        information from different channels to extract signals of interest from EEG signals, but if our goal is
        neurophysiological interpretation or visualization of weights, activation patterns need to be constructed
        from the obtained spatial filters.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_channels, n_filters).
    Cx : ndarray
        Covariance matrix of eeg data, shape (n_channels, n_channels).
    Cs : ndarray
        Covariance matrix of source data, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_channels, n_patterns), each column is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging.
           Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A

def xiang_dsp_kernel(
    X: ndarray, y: ndarray
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    DSP: Discriminal Spatial Patterns, only for two classes[1]_.
    Import train data to solve spatial filters with DSP,
    finds a projection matrix that maximize the between-class scatter matrix and
    minimize the within-class scatter matrix. Currently only support for two types of data.

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    X : ndarray
        EEG train data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels of EEG data, shape (n_trials, )

    Returns
    -------
    W : ndarray
        spatial filters, shape (n_channels, n_filters)
    D : ndarray
        eigenvalues in descending order
    M : ndarray
        mean value of all classes and trials, i.e. common mode signals, shape (n_channel, n_samples)
    A : ndarray
        spatial patterns, shape (n_channels, n_filters)

    Notes
    -----
    the implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(y == label) for label in labels])
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms, Ss = zip(
        *[
            (
                np.mean(X[y == label], axis=0),
                np.sum(
                    np.matmul(X[y == label], np.swapaxes(X[y == label], -1, -2)), axis=0
                ),
            )
            for label in labels
        ]
    )
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(
        Ss
        - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(
        n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )

    D, W = eigh(nearestPD(Sb), nearestPD(Sw))
    ix = np.argsort(D)[::-1]  # in descending order
    D, W = D[ix], W[:, ix]
    A = robust_pattern(W, Sb, W.T @ Sb @ W)

    return W, D, M, A


def xiang_dsp_feature(
    W: ndarray, M: ndarray, X: ndarray, n_component: int = 1
) -> ndarray:
    """
    Return DSP features in paper [1]_.

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    M : ndarray
        common template for all classes, shape (n_channel, n_samples)
    X : ndarray
        eeg test data, shape (n_trials, n_channels, n_samples)
    n_component : int, optional
        length of the spatial filters, first k components to use, by default 1

    Returns
    -------
    features: ndarray
        features, shape (n_trials, n_component, n_samples)

    Raises
    ------
    ValueError
        n_component should less than half of the number of channels

    Notes
    -----
    1. instead of meaning of filtered signals in paper [1]_., we directly return filtered signals.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    W, M, X = np.copy(W), np.copy(M), np.copy(X)
    max_components = W.shape[1]
    if n_component > max_components:
        raise ValueError("n_component should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_component].T, X - M)
    return features


def proj_ref(Yf: ndarray):
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    return P


def aug_2(X: ndarray, n_samples: int, padding_len: int, P: ndarray, training: bool = True):
    X = X.reshape((-1, *X.shape[-2:]))
    n_trials, n_channels, n_points = X.shape
    if n_points < padding_len + n_samples:
        raise ValueError("the length of X should be larger than l+n_samples.")
    aug_X = np.zeros((n_trials, (padding_len + 1) * n_channels, n_samples))
    if training:
        for i in range(padding_len + 1):
            aug_X[:, i * n_channels : (i + 1) * n_channels, :] = X[
                ..., i : i + n_samples
            ]
    else:
        for i in range(padding_len + 1):
            aug_X[:, i * n_channels : (i + 1) * n_channels, : n_samples - i] = X[
                ..., i:n_samples
            ]
    aug_Xp = aug_X @ P
    aug_X = np.concatenate([aug_X, aug_Xp], axis=-1)
    return aug_X


def tdca_feature(
    X: ndarray,
    templates: ndarray,
    W: ndarray,
    M: ndarray,
    Ps: List[ndarray],
    padding_len: int,
    n_component: int = 1,
    training=False,
):
    rhos = []
    for Xk, P in zip(templates, Ps):
        a = xiang_dsp_feature(
            W,
            M,
            aug_2(X, P.shape[0], padding_len, P, training=training),
            n_component=n_component,
        )
        b = Xk[:n_component, :]
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return rhos


class TDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
                 padding_len: int, 
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,):
        self.padding_len = padding_len
        self.n_component = n_component
        if n_component < 0:
            raise ValueError('n_component must be larger than 0')
        
        self.ID = 'TDCA'
        self.n_jobs = n_jobs
        
        self.model = {}
        # self.model['weights_filterbank'] = weights_filterbank
    def __copy__(self):
        copy_model = TDCA(padding_len = self.padding_len,
                          n_component = self.n_component,
                          n_jobs = self.n_jobs)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self, 
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        X = np.array(X)[:,0,:,:]
        y = np.array(Y)
        Yf = np.array(ref_sig)

        

        X -= np.mean(X, axis=-1, keepdims=True)
        
        self.classes_ = np.unique(y)
        self.Ps_ = [proj_ref(Yf[i]) for i in range(len(self.classes_))]
        # raise ValueError(X.shape, y.shape, Yf.shape)
        aug_X_list, aug_Y_list = [], []
        for i, label in enumerate(self.classes_):
            aug_X_list.append(
                aug_2(
                    X[y == label],
                    self.Ps_[i].shape[0],
                    self.padding_len,
                    self.Ps_[i],
                    training=True,
                )
            )
            aug_Y_list.append(y[y == label])

        aug_X = np.concatenate(aug_X_list, axis=0)
        aug_Y = np.concatenate(aug_Y_list, axis=0)
        self.W_, _, self.M_, _ = xiang_dsp_kernel(aug_X, aug_Y)

        self.templates_ = np.stack(
            [
                np.mean(
                    xiang_dsp_feature(
                        self.W_,
                        self.M_,
                        aug_X[aug_Y == label],
                        n_component=self.W_.shape[1],
                    ),
                    axis=0,
                )
                for label in self.classes_
            ]
        )
        # return self

    def predict(self, X: List[ndarray]):
        X = np.array(X)[:,0,:,:]
        n_component = self.n_component
        X -= np.mean(X, axis=-1, keepdims=True)
        X = X.reshape((-1, *X.shape[-2:]))
        rhos = [
            tdca_feature(
                tmp,
                self.templates_,
                self.W_,
                self.M_,
                self.Ps_,
                self.padding_len,
                n_component=n_component,
            )
            for tmp in X
        ]
        feat = np.stack(rhos)

        labels = self.classes_[np.argmax(feat, axis=-1)]
        labels = [labels[i] for i in range(len(labels))]
        return labels,np.max(feat, axis=-1)