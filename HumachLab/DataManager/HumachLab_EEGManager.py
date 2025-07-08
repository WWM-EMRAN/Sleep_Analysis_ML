# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%
import os
from itertools import islice
import itertools
import copy
import numpy as np
import pandas as pd
import mne
from mne.io.edf.edf import RawEDF
from datetime import datetime

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.Utility.HumachLab_Enums import *
    from HumachLab.DataManager.HumachLab_Alfred_EEGManager import *
    from HumachLab.DataManager.HumachLab_CHBMIT_EEGManager import *
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###



############################### This data manager is for AMD dataset ############################################


class HumachLab_EEGManager:
    ### EEG directory
    data_directory = './'

    ### Maximum number of patients and records in dataset
    max_patients = 25 #Initialized in constructor again
    max_records = 100 #Initialized in constructor again

    ### Record index array: contain all indices of the record having data
    ### [patient_no, record_no, has_seizure_file:True/False]
    record_indices = []
    patient_list = []

    ### Dummy channel list to remove
    _removable_dummy_channels = []

    ### Removing duplicate channels
    _is_remove_duplicate_channels = False
    _duplicate_channels = {}



    ###
    ### Initializing parameters
    ###
    def __init__(self, logger, dataset_name=All_Datasets.chbmit, data_directory='./', metadata_path='./', removable_channel_list=[], is_remove_duplicate_channels=False):
        self.logger = logger
        self.dataset_wise_manager = None

        if dataset_name in [All_Datasets.alfred.value, All_Datasets.amd.value]:
            self.dataset_wise_manager = HumachLab_Alfred_EEGManager(logger, data_directory=data_directory, metadata_path=metadata_path, removable_channel_list=removable_channel_list, is_remove_duplicate_channels=is_remove_duplicate_channels)
        elif dataset_name==All_Datasets.chbmit.value:
            self.dataset_wise_manager = HumachLab_CHBMIT_EEGManager(logger, data_directory=data_directory, metadata_path=metadata_path, removable_channel_list=removable_channel_list, is_remove_duplicate_channels=is_remove_duplicate_channels)
        elif dataset_name==All_Datasets.siena.value:
            pass
        else:
            return
        return





    ##################################################################
    ###
    ### External Callable functions
    ###
    # Get list of all patients
    def get_record_info_for_all_patient(self):
        result = self.dataset_wise_manager.get_record_info_for_patient([], [])
        return result



    #  patient_no- list of patients' numbers & record_no- list of lists of records for that patient
    def get_records_info_for_patients(self, patient_no:list, record_no:list=[]):
        result = None
        if str(patient_no.__class__.__name__) in ['list'] and str(record_no.__class__.__name__) in ['list']:
            if len(patient_no)>0 and len(record_no)>0 and len(patient_no)==len(record_no):
                result = self.dataset_wise_manager.get_record_info_for_patient(patient_no, record_no)
            else:
                self.logger.info('Please provide items in the list for patients and same number of lists inside the list of records.')
        else:
            self.logger.info('Please provide both patients and records in the form of list.')

        for p in result:
            for r in p.records:
                # self.logger.info(f'{r.__dict__}')
                pass
        return result



    # Get list of all patients with seizure only
    def get_record_info_with_seizure_for_all_patient(self, patient_list:list):
        patient_list_with_seizures = []
        self.patient_list = []

        for p in patient_list:
            # self.logger.info(f'{p.num_records}, {p.record_numbers}')
            re = []
            re_no = []
            for r in p.records:
                if r.num_seizures>0:
                    # self.logger.info(f'{r.record_no}')
                    re.append(r)
                    re_no.append(r.record_no)
                    # p.records.remove(r)
                    # p.num_records -= 1
                    # p.record_numbers.remove(r.record_no)
            # self.logger.info(f'##### {p.num_records}, {p.record_numbers}, {re_no}')

            if len(re_no)>0:
                p_copy = copy.deepcopy(p)
                p_copy.records = re
                p_copy.num_records = len(re)
                p_copy.record_numbers = re_no
                patient_list_with_seizures.append(p_copy)

        self.patient_list = patient_list_with_seizures

        return patient_list_with_seizures



    # # Get list of all patients with seizure only
    # def get_record_info_with_seizure_for_all_patient(self, patient_list:list):
    #     patient_list_with_seizures = []
    #     self.patient_list = []
    #
    #     for p in patient_list:
    #         # self.logger.info(f'{p.patient_no}  {len(p.records)}')
    #         ind = []
    #         rno = []
    #         chn = []
    #         for r in p.records:
    #             # self.logger.info(f'{r.record_no}, no s: {r.num_seizures}')
    #             if r.num_seizures>0:
    #                 ind.append(p.records.index(r))
    #                 rno.append(r.record_no)
    #                 chn.append(r.channel_id)
    #
    #         p.records = [i for j, i in enumerate(p.records) if j in ind]
    #         p.num_records = len(p.records)
    #         p.record_numbers = rno
    #         p.channels = [i for j, i in enumerate(p.channels) if j in chn]
    #         if p.num_records>0:
    #             patient_list_with_seizures.append(p)
    #
    #     self.patient_list = patient_list_with_seizures
    #     tmp_ind = [l for i, j in enumerate(self.record_indices) for k, l in enumerate(j) if l[2] > 0]
    #     self.record_indices = [list(item[1]) for item in itertools.groupby(sorted(tmp_ind), key=lambda x: x[0])]
    #
    #     return patient_list_with_seizures


    # Get list of all patients with specific channels
    def get_record_info_for_specific_channels(self, patient_list:list, channels:list=[]):
        patient_list_for_specific_channels = []
        self.patient_list = []

        if len(channels)==0:
            self.patient_list = patient_list
            return patient_list

        for p in patient_list:
            # self.logger.info(f'@@@@ {channels}, {len(p.channels)}, {p.channels}, \n {len(p.removable_channels_)}, {p.removable_channels_})
            i=0
            for chn in p.channels:
                # self.logger.info(f'#### {len(chn.values())}, {chn.values()}')
                for key, val in chn.items():
                    # self.logger.info(f'{val}, {channels}, {not (val in channels)}')
                    if not (val in channels):
                        # self.logger.info(f'Rem: {i}, {key}')
                        p.removable_channels_[i][key] = val
                        # p.removable_channels_[p.channels.index(chn)][key] = val
                i += 1
            # self.logger.info(f'$$$$ {len(p.channels)}, {p.channels}, \n {len(p.removable_channels_)}, {p.removable_channels_}')
            # self.logger.info(f'========\n {len(p.channels[3])}, {p.channels[3]}, \n {len(p.removable_channels_[3])}, {p.removable_channels_[3]}')

        patient_list_for_specific_channels = patient_list
        self.patient_list = patient_list_for_specific_channels

        return patient_list_for_specific_channels










