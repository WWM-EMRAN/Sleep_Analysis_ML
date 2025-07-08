"""
File Name: HumachLab_Utility.py
Author: Emran Ali
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 13/05/2021 5:54 am
"""


import os
import time
import datetime
import shutil
import numbers



class HumachLab_Tester:

    def __init__(self):
        self.pkl_path = None

        self.raw_data_directory = None
        self.raw_metadata_path = None
        self.result_root_directory = None
        self.result_subdirectory = None
        self.experiment_subdirectory = None
        self.feature_subdirectory = None
        self.channel_feature_subdirectories = None
        self.log_path = None
        self.feat_file_name = None
        self.seg_file_name = None

        self.experiment_type = None
        self.subexperiment_type = None
        self.experiment_no = None

        self.all_channels = None
        self.removable_channel_list = None

        self.class_name = 'seizureState'
        self.selected_channel = None
        self.selected_channel_list = None

        #Extra data for temporarily transfer
        self.exp_special_name = None
        self.dataset_name = None
        self.is_multi_channel_analysis = False
        self.seg_len = 5
        self.overlap = 0
        self.all_pats = None
        self.force_to_file_name = False
        self.from_file = None
        self.to_file = None

        self.test_split_ratio = (30 / 100)  # (1/28)
        self.test_splitting_nature = 0  # ### 0= leave-one out, >0= random splitting times, <0= leave p out & p is based on splitting ratio
        self.validation_split_ratio = (30 / 100)
        self.validation_splitting_nature = 1
        self.cv_rounds = 2
        self.num_fold = 5
        self.is_train_will_all_data = False
        self.patients_or_folds_done = 0

        self.preprocessing_type = 1
        self.classification_type = 1

        return