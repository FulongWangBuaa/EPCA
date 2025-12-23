import os
import numpy as np
import py7zr
import mne
import scipy.io as sio
from copy import deepcopy

from itertools import combinations
import mat73

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose

from SSVEPAnalysisToolbox.datasets.basedataset import BaseDataset
from SSVEPAnalysisToolbox.datasets.subjectinfo import SubInfo
from SSVEPAnalysisToolbox.utils.download import download_single_file
from SSVEPAnalysisToolbox.utils.io import loadmat
from SSVEPAnalysisToolbox.utils.download import download_single_file
from SSVEPAnalysisToolbox.evaluator.baseevaluator import TrialInfo


class MyGuHFDataset64(BaseDataset):
    """An open dataset for human SSVEPs in the frequency
      range of 1-60 Hz.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    subject_info : SubInfo
        Subject information.
    """

    _CHANNELS = [
        'FP1', 'FPz', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3',
        'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
        'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
    # _CHANNELS = ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PZ']

    # 50Hz排除在外
    # _FREQS = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    #     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    #     41, 42, 43, 44, 45, 46, 47, 48, 49,
    #     51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    _FREQS = [10,11,12,13,14,15,
              20,21,22,23,24,25,
              40,41,42,43,44,45,
              55,56,57,58,59,60]
    _PHASES = [0] * 24

    _SUBJECTS = [SubInfo(ID = 's{:d}'.format(sub_idx)) for sub_idx in range(1,30+1,1)]

    def __init__(self,
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'GuHF',
                         url = '',
                         paths = path,
                         channels = self._CHANNELS,
                         srate = 1000,
                         block_num = 6,
                         trial_num = len(self._FREQS),
                         trial_len = 5,
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = [],
                         path_support_file = path_support_file,
                         t_prestim = 0,
                         t_break = 0.14,
                         default_t_latency = 0.14)
        self.trial_label_check_list = {}
        for trial_i in range(self.trial_num):
            self.trial_label_check_list[trial_i] = trial_i


    def download_single_subject(self,
                                subject: SubInfo):
        source_url = self.url + subject.ID + '.mat.7z'
        desertation = os.path.join(subject.path, subject.ID + '.mat.7z')
        
        data_file = os.path.join(subject.path, 'data_' + subject.ID + '_64_2.mat')

        download_flag = True
        
        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation)
            
                with py7zr.SevenZipFile(desertation,'r') as archive:
                    archive.extractall(subject.path)
                    
                os.remove(desertation)
            except:
                download_flag = False
        
        return download_flag, source_url, desertation
    

    def download_file(self,
                      file_name: str):
        source_url = self.url + file_name
        desertation = os.path.join(self.path_support_file, file_name)

        download_flag = True
        
        if not os.path.isfile(desertation):
            try:
                download_single_file(source_url, desertation)
            except:
                download_flag = False

        return download_flag, source_url, desertation

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))
        
        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, 'data_' + sub_info.ID + '_64_3.mat')
        
        # mat_data = loadmat(file_path)
        # ch_num * samples * stimulus_num * block_num
        # 64 * 5140 * 24 * 12
        data = loadmat(file_path)['datas']
        # ch_num * samples * stimulus_num * block_num

        # block_num * stimulus_num * ch_num * sample_num
        data = transpose(data, (3,2,0,1))[0:6]

        return data
    

    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return self.trial_label_check_list[trial_idx]
    

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