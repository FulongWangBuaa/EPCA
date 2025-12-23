from typing import Union, Optional, Dict, List, Tuple, cast
from numpy import ndarray, transpose
from copy import deepcopy
import os
from itertools import combinations
import numpy as np
import py7zr

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose


from SSVEPAnalysisToolbox.utils.download import download_single_file
from SSVEPAnalysisToolbox.datasets import BaseDataset,BenchmarkDataset
from SSVEPAnalysisToolbox.utils.io import loadmat
from SSVEPAnalysisToolbox.datasets.subjectinfo import SubInfo

from my_code.evaluator.MyBaseEvaluator import TrialInfo

from mne.filter import resample

class MyBenchmarkDataset(BenchmarkDataset):

    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,35+1,1)]
    def __init__(self, path, resample_freq: Optional[float] = None):
        super().__init__(path)
        
        if resample_freq is not None:
            if resample_freq != self.srate:
                self.srate = resample_freq
                self.resample_freq = resample_freq
            else:
                self.resample_freq = None
        else:
            self.resample_freq = None


    def leave_block_out(self,
                        block_idx: List[int]) -> Tuple[List[int], List[int]]:
        """
        Generate testing and training blocks for specific block based on leave-out rule

        Parameters
        ----------
        block_idx : int
            Specific block index

        Returns
        -------
        test_block: List[int]
            Testing block
        train_block: List[int]
            Training block
        """
        if any(idx < 0 for idx in block_idx):
            raise ValueError('Block index cannot be negative')
        if any(idx > self.block_num-1 for idx in block_idx):
            raise ValueError('Block index should be smaller than {:d}'.format(self.block_num-1))
            
        train_block = deepcopy(block_idx)
        all_block = [i for i in range(self.block_num)]
        test_block = [i for i in all_block if i not in train_block]

        return train_block,test_block

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))
        
        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, sub_info.ID + '.mat')
        
        mat_data = loadmat(file_path)
        data = mat_data['data']
        # block_num * stimulus_num * ch_num * whole_trial_samples
        data = transpose(data, (3,2,0,1))

        if self.resample_freq is not None:
            data = resample(data, self.resample_freq, 250)

        return data


    def gen_trials_leave_out(self,
                            tw_seq: List[float],
                            trains: List[int],
                            harmonic_num: int,
                            ch_used: List[int],
                            stims: Optional[List[int]] = None,
                            subjects: Optional[List[int]] = None,
                            t_latency: Optional[float] = None,
                            shuffle: bool = False) -> list:
        '''
        Generate evaluation trials for one dataset
        Evaluations will be carried out on each subject and each signal length
        Training and testing datasets are separated based on the leave-block-out rule

        Parameters
        ----------
        dataset_idx : int
            dataset index of dataset_container
        tw_seq : List[float]
            List of signal length
        dataset_container : list
            List of datasets
        harmonic_num : int
            Number of harmonics
        trials: List[int]
            List of trial index
        ch_used : List[int]
            List of channels
        subjects : Optional[List[int]]
            List of subject indices
            If None, all subjects will be included
        t_latency : Optional[float]
            Latency time
            If None, default latency time of dataset will be used
        shuffle : bool
            Whether shuffle

        Returns
        -------
        trial_container : list
            List of trial information
        '''
        if subjects is None:
            sub_num = len(self.subjects)
            subjects = list(range(sub_num))
        if stims is None:
            stim_num = self.stim_info['stim_num']
            stims = list(range(stim_num))
        if t_latency is None:
            t_latency = self.default_t_latency
        block_num = self.block_num
        trial_container = []
        for tw in tw_seq:
            for sub_idx in subjects:
                for train_idx,num_train in enumerate(trains):
                    cv_labels = list(combinations(range(block_num), num_train))
                    for idx_cv, cv_label in enumerate(cv_labels):
                        train_block, test_block = self.leave_block_out(list(cv_label))
                        train_trial = TrialInfo().add_dataset(dataset_idx = 0,
                                                            sub_idx = sub_idx,
                                                            block_idx = train_block,
                                                            trial_idx = stims,
                                                            ch_idx = ch_used,
                                                            harmonic_num = harmonic_num,
                                                            tw = tw,
                                                            t_latency = t_latency,
                                                            shuffle = shuffle)
                        test_trial = TrialInfo().add_dataset(dataset_idx = 0,
                                                            sub_idx = sub_idx,
                                                            block_idx = test_block,
                                                            trial_idx = stims,
                                                            ch_idx = ch_used,
                                                            harmonic_num = harmonic_num,
                                                            tw = tw,
                                                            t_latency = t_latency,
                                                            shuffle = shuffle)
                        trial_container.append([train_trial, test_trial])
        return trial_container