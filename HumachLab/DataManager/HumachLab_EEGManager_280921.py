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

import HumachLab
from HumachLab_EEGDataStucture import Patient, Record
from HumachLab_Enums import *

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
        self.dataset_name = dataset_name
        self.data_directory = os.path.abspath(data_directory)
        if not (os.path.exists(self.data_directory)):
            self.logger.info(f'Raw directory does not exist!')
            self.data_directory = os.path.abspath('./')

        self.metadata_path = './'
        if self.dataset_name==All_Datasets.alfred:
            self.metadata_path = os.path.abspath(metadata_path)
            if not (os.path.exists(self.metadata_path)):
                self.logger.info(f'Raw metadata path does not exist!')
                self.metadata_path = os.path.abspath('./')

        self.data_directory = self.data_directory.replace('\\', '/')
        self.metadata_path = self.metadata_path.replace('\\', '/')

        #self._all_files = []
        self.max_patients = 25
        self.max_records = 100
        self.record_indices = []
        self.patient_list = []
        self._removable_dummy_channels = removable_channel_list
        self._is_remove_duplicate_channels = is_remove_duplicate_channels
        self._duplicate_channels = {}
        return



    #################################################################

    ### Remove duplicate channels and change channel names
    def _get_removable_channels(self, chans:dict):
        rem_chans = {}
        ind = 0
        for key, chan in chans.items():
            # self.logger.info(f'Check: , {chan}, {chan in self._removable_dummy_channels}')
            sub_chans = (list(chans.values()))[:ind]

            if len(self._removable_dummy_channels)>0 and (chan in self._removable_dummy_channels):
                rem_chans[key] = chan
            else:
                if (chan in sub_chans):
                    i = sub_chans.index(chan)
                    key2 = (list(chans.keys()))[i]
                    chans[key2] = '{}-{}'.format(chan, 0)
                    chans[key] = '{}-{}'.format(chan, 1)
                    if self._is_remove_duplicate_channels:
                        rem_chans[key] = '{}-{}'.format(chan, 1)
                else:
                    found_list = [ch for ch in sub_chans if (ch.count('-')==2) and ch.find(chan)!=-1]
                    if len(found_list)>0:
                        last_found_ch_ind = int((found_list[-1].split('-'))[-1])
                        chans[key] = '{}-{}'.format(chan, (last_found_ch_ind+1))
                        if self._is_remove_duplicate_channels:
                            rem_chans[key] = '{}-{}'.format(chan, (last_found_ch_ind+1))

            ind += 1

        return rem_chans


    ###
    ### Convert summary data from specific summary file
    ###
    def _convert_summary_from_specific_summary_file(self, summary_file):
        sample_rate = 0
        channels = []
        removable_chn_ = []
        data_for_channels = []
        data = []

        tmp_chn = {}
        # tmp_rem_chn_ = {}
        tmp_dat = {}

        rat_ind = -1
        chn_ind = -1
        dat_ind = -1
        ind_cnt = -1
        dat_cnt = -1

        self.logger.info(f'---> Getting summary data...')

        with open(summary_file, 'r') as f:
            for line in f:
                # self.logger.info(fline)
                byte_array = bytes(line, 'utf-8')
                words = line.strip().split(':')

                if (line == '\n'):
                    # self.logger.info(f'Data block ends...')
                    ### Channels list
                    if (chn_ind > -1):
                        # self.logger.info(f'tmp_chn:{tmp_chn}')
                        tmp_rem_chn_ = self._get_removable_channels(tmp_chn)
                        channels.append(tmp_chn)
                        tmp_chn = {}
                        removable_chn_.append(tmp_rem_chn_)
                        # tmp_rem_chn_ = {}
                        chn_ind = -1

                    ### Data list
                    if (dat_ind > -1):
                        # self.logger.info(f'data finished')
                        data.append(tmp_dat)
                        tmp_dat = {}
                        dat_ind = -1

                elif (line.find('Channels') >= 0):
                    ind_cnt += 1
                    dat_cnt = -1

                elif (len(words) >= 2 and (words[1] != '')):
                    words[0] = words[0].strip()
                    words[1] = words[1].strip()

                    if (words[0].find('Sampling Rate') >= 0):
                        sample_rate = (words[1].split())[0]
                        rat_ind += 1
                        chn_ind = -1
                        dat_ind = -1

                    elif (words[0].find('Channel') >= 0):
                        chn_ind += 1
                        tmp_chn[words[0]] = words[1]

                        rat_ind = -1
                        dat_ind = -1

                    else:  # elif(words[0].find('File Name')>=0):
                        if (words[0].find('File Start Time') >= 0 or words[0].find('File End Time') >= 0):
                            words[1] += ':' + words[2] + ':' + words[3]

                        dat_ind += 1
                        tmp_dat[words[0]] = words[1]
                        rat_ind = -1
                        chn_ind = -1

                        if (words[0].find('File Name') >= 0):
                            dat_cnt += 1
                            file_nm = words[1]
                            # self.logger.info(f'ind_cnt:{ind_cnt}===dat_ind:{dat_cnt}\n{data_for_channels}')
                            if len(data_for_channels) == ind_cnt:
                                data_for_channels.append(dat_cnt + 1)
                            else:
                                data_for_channels[ind_cnt] = (dat_cnt + 1)
                        # self.logger.info(f'data_for_channels:{dat_ind+1}')

                else:
                    # self.logger.info(f'---> Something is going on here...')
                    pass

            # self.logger.info(f'HHHHHH', len(channels), '\n', len(removable_chn_))

            ### Last data doesn't automatically added
            data.append(tmp_dat)

        ### Split data input according to the number of channels
        data_iter = iter(data)
        data = [list(islice(data_iter, elem)) for elem in data_for_channels]
        # self.logger.info(f'sample_rate:{sample_rate} \nchannels:{channels} \ndata:{data}')

        self.logger.info('===> Summary data completed!!!')

        return (sample_rate, channels, removable_chn_, data)



    def _populate_patient_info_with_summary_data(self, patient_no, summary_file):
        samp_rate, chans, rem_chns, rec_info = self._convert_summary_from_specific_summary_file(summary_file)
        # self.logger.info(f'{samp_rate} \n {chans} \n {rec_info}')

        # Populating patient data
        patient = Patient(patient_no)
        patient.sam_rate = samp_rate
        patient.channels = chans
        patient.removable_channels_ = rem_chns
        # patient.num_records = len(rec_info)
        # self.logger.info(f'Num rec: {len(rec_info)}')

        # Populating record data
        ch_id = 0
        for rec_group in rec_info:
            for rec in rec_group:
                rec_no = int((((rec['File Name'].split('.'))[0]).split('_'))[1])
                patient.record_numbers.append(rec_no)
                record = Record(rec_no)
                record.patient_no = patient_no
                record.file_name = rec['File Name']
                # self.logger.info(f'Channel ID: {ch_id}')
                record.channel_id = ch_id
                file_st_time = self._calculate_time_in_seconds_from_string(rec['File Start Time'])
                file_end_time = self._calculate_time_in_seconds_from_string(rec['File End Time'])
                if file_end_time<file_st_time:
                    file_end_time += 24*3600
                record.record_times = (file_st_time, file_end_time)
                record.record_duration = file_end_time - file_st_time
                num_seiz = int(rec['Number of Seizures in File'])
                record.num_seizures = num_seiz
                # self.logger.info(f'Patient: {patient_no} Record: {rec_no}, NumSeiz: {num_seiz}')
                if num_seiz>0:
                    for sind in range(1, (num_seiz+1)):
                        seiz_st_key = 'Seizure {} Start Time'.format(sind)
                        seiz_end_key = 'Seizure {} End Time'.format(sind)

                        seiz_st = -1
                        seiz_end = -1

                        # seiz_st = int((rec[seiz_st_key].split(' '))[0])
                        # seiz_end = int((rec[seiz_end_key].split(' '))[0])

                        try:
                            seiz_st = int((rec[seiz_st_key].split(' '))[0])
                            seiz_end = int((rec[seiz_end_key].split(' '))[0])
                        except:
                            seiz_st_key = 'Seizure Start Time'
                            seiz_end_key = 'Seizure End Time'
                            seiz_st = int((rec[seiz_st_key].split(' '))[0])
                            seiz_end = int((rec[seiz_end_key].split(' '))[0])

                        record.seizure_times.append(tuple((seiz_st, seiz_end)))
                        record.seizure_durations.append((seiz_end-seiz_st))


                record.data = None

                patient.records.append(record)

                if num_seiz>0:
                    patient.seizure_record_numbers.append(rec_no)

            ch_id += 1

        # self.logger.info(f'+++ {len(patient.records)}  {patient.record_numbers}')
        nr = len(patient.records)
        patient.num_records = nr
        patient.num_seizure_records = len(patient.seizure_record_numbers)

        # for r in patient.records:
        #     self.logger.info(f'{r.__dict__}')
        # self.logger.info(f'{patient.__dict__}')
        if nr==0:
            patient = None

        return patient



    ### Convert time in string to an integer indicating time in seconds, considering 12:00 as 00 secs
    def _calculate_time_in_seconds_from_string(self, str_time):
        time_arr = str_time.split(':')
        time_in_seconds = int(time_arr[0])*3600 + int(time_arr[1])*60 + int(time_arr[2])
        return time_in_seconds



    ###################### Patient and Record Manager ###################
    ###
    ### Traverse all files from a directory including its nested locations
    ### Checks if files exists for a patient and record
    ###  info_type= 0-all, 1-edf, 2-seizure & 3-summary
    def _has_record_info_for_patient(self, patient_no, record_no, info_type):
        patient_no_str = '{:0>02}'.format(patient_no) if patient_no < 99 else '{}'.format(
            patient_no)  # .format(patient_no) | f'{patient_no:0>02}'
        record_no_str = '{:0>02}'.format(record_no) if record_no < 99 else '{}'.format(record_no)

        record_found = True
        record_path = self.data_directory + '/'
        #self.logger.info(f'came... {record_path}')

        if patient_no > 0 and record_no > 0 and info_type == 1:
            record_path += 'chb{0}/chb{0}_{1}.edf'.format(patient_no_str, record_no_str)
        elif patient_no > 0 and record_no > 0 and info_type == 2:
            record_path += 'chb{0}/chb{0}_{1}.edf.seizures'.format(patient_no_str, record_no_str)
        elif patient_no > 0 and record_no == 0 and info_type == 3:
            record_path += 'chb{0}/chb{0}-summary.txt'.format(patient_no_str)
        else:
            record_path = ''
            record_found = False

        if record_found:
            if not os.path.exists(record_path):
                record_path = ''
                record_found = False
            else:
                pass
                # self.logger.info(f'--> {patient_no}, {record_no}, {info_type}, {record_path}')

        # self.logger.info(f'--> {patient_no}, {record_no}, {info_type}, {record_path}')

        return record_found, record_path



    ###
    ### Record Management functions
    ###
    def _generate_single_record_for_single_patient(self, patient_no, record_no):
        edf_stat, edf_path = self._has_record_info_for_patient(patient_no, record_no, 1)
        seizure_stat, seizure_path = self._has_record_info_for_patient(patient_no, record_no, 2)
        tmp_ind = []
        edf_data = None # Data will be loaded at the time of access finally

        if edf_stat:
            tmp_ind.append(patient_no)
            tmp_ind.append(record_no)

            # edf_data = self._convert_specific_edf_from_file(edf_path)

            if seizure_stat:
                tmp_ind.append(1)
            else:
                tmp_ind.append(0)

        return tmp_ind, edf_path



    ##################################################################
    ###
    ### External Callable functions
    ###
    # Get list of all patients
    def get_record_info_for_all_patient(self):
        result = self._get_record_info_for_patient([], [])
        return result



    #  patient_no- list of patients' numbers & record_no- list of lists of records for that patient
    def get_records_info_for_patients(self, patient_no:list, record_no:list=[]):
        result = None
        if str(patient_no.__class__.__name__) in ['list'] and str(record_no.__class__.__name__) in ['list']:
            if len(patient_no)>0 and len(record_no)>0 and len(patient_no)==len(record_no):
                result = self._get_record_info_for_patient(patient_no, record_no)
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






###############################################################################################################

######################################          CHB MIT        ################################################

###############################################################################################################

    # def _get_record_info_for_patient(self, patient_no, record_no):
    #     list_of_patients = []
    #     all_patient_list = []
    #     self.patient_list = []
    #     self.record_indices = []
    #
    #     if len(patient_no) == 0 and len(record_no) == 0:
    #         list_of_patients = range(1, (self.max_patients))
    #     else:
    #         list_of_patients = patient_no
    #
    #     for p in list_of_patients:
    #         list_of_record = []
    #         if len(patient_no) > 0 and len(record_no) > 0:
    #             ind = list_of_patients.index(p)
    #             list_of_record = record_no[ind]
    #
    #         patient = self._generate_records_for_single_patient(p, list_of_record)
    #
    #         # self.logger.info(f'KKKKK {len(patient.channels)}, \n, {len(patient.removable_channels_)}')
    #
    #         if patient:
    #             all_patient_list.append(patient)
    #
    #     self.patient_list = all_patient_list
    #     self.record_indices = [list(item[1]) for item in itertools.groupby(sorted(self.record_indices), key=lambda x: x[0])]
    #
    #     return all_patient_list
    #
    #
    #
    # def _generate_records_for_single_patient(self, patient_no, r_no):
    #     # summary
    #     summary_stat, summary_path = self._has_record_info_for_patient(patient_no, 0, 3)
    #     patient = None
    #     if not summary_stat:
    #         self.logger.info('Summary for Patient {} is not found.'.format(patient_no))
    #     else:
    #         patient = self._populate_patient_info_with_summary_data(patient_no, summary_path)
    #         # self.logger.info(f'LLLLLL {len(patient.channels)}, \n {len(patient.removable_channels_)}')
    #
    #     if not patient:
    #         self.logger.info(f'No record found for patient {patient_no}')
    #     else:
    #         missing_rec_ind = []
    #         present_chn_ind = []
    #         rearrange_chn_ind = 0
    #         ind_cnt = 0
    #         for record_no in patient.record_numbers:
    #             tmp_ind = []
    #             edf_data = None
    #
    #             if len(r_no)==0 or (len(r_no)>0 and (record_no in r_no)):
    #                 tmp_ind, edf_path = self._generate_single_record_for_single_patient(patient_no, record_no)
    #
    #             if len(tmp_ind) > 0:
    #                 # self.logger.info(f'Rec {record_no} found for Patient {patient_no}: {tmp_ind}')
    #                 self.record_indices.append(tmp_ind)
    #                 # patient.data = edf_data
    #                 tmp_record: Record = patient.records[ind_cnt]
    #                 tmp_record.file_path = edf_path
    #                 tmp_record.data = edf_data
    #                 # self.logger.info(f'{tmp_ind} --> {tmp_record.channel_id}')
    #
    #                 # if self._is_remove_duplicate_channels:
    #                 #     dropable_chns = (patient.removable_channels_[tmp_record.channel_id])
    #                 #     dropable_chns = list(dropable_chns.values())
    #                 #     edf_data.drop_channels(dropable_chns)
    #
    #                 if len(r_no) > 0:
    #                     present_chn_ind.append(tmp_record.channel_id)
    #                     # tmp_record.channel_id = rearrange_chn_ind
    #                     # self.logger.info(f'{tmp_ind} ====> {tmp_record.channel_id}')
    #                 # self.logger.info(f'+++ {patient.channels}, {tmp_record.channel_id}')
    #             else:
    #                 indx = patient.record_numbers.index(record_no)
    #                 missing_rec_ind.append(indx)
    #
    #             ind_cnt += 1
    #             rearrange_chn_ind += 1
    #
    #         # Deal with missing records
    #         # self.logger.info(f'LLLLLL222 {len(patient.channels)}, \n {len(patient.removable_channels_)}')
    #         patient.record_numbers = [i for j, i in enumerate(patient.record_numbers) if j not in missing_rec_ind]
    #         patient.records = [i for j, i in enumerate(patient.records) if j not in missing_rec_ind]
    #         # if len(r_no) > 0:
    #         #     patient.channels = [i for j, i in enumerate(patient.channels) if j in present_chn_ind]
    #         #     patient.removable_channels_ = [i for j, i in enumerate(patient.removable_channels_) if j in present_chn_ind]
    #         patient.num_records -= len(missing_rec_ind)
    #         # self.logger.info(f'LLLLLL333 {len(patient.channels)}, \n {len(patient.removable_channels_)}')
    #
    #     return patient






###############################################################################################################

######################################          Alfred         ################################################

###############################################################################################################
    def _get_record_info_for_patient(self, patient_no, record_no):
        self.logger.info(f'---> Getting summary data Alfred...')
        all_patient_list = []
        self.patient_list = []
        self.record_indices = []

        # Reading excel metadata
        num_active_cols = 11
        xl_data = pd.ExcelFile(self.metadata_path)
        sheets = xl_data.sheet_names
        demographic_df = (pd.read_excel(xl_data, sheets[0])).iloc[:, :num_active_cols]
        demographic_df = demographic_df.dropna()
        # print(demographic_df)
        # print(demographic_df.dtypes)
        cols = demographic_df.columns
        convert_dict = {}
        for i, c in enumerate(cols):
            if i in [0, 2]:
                convert_dict[c] = int
            elif i == 1:
                convert_dict[c] = str
            #     else:
            #         convert_dict[c] = datetime64[ns]
            elif i in [3, 5, 8, 9]: #[8, 9]:  # [3, 5, 8, 9]:
                demographic_df[c] = pd.to_datetime(demographic_df[c], format='%H:%M:%S')
                # try:
                #     demographic_df[c] = (pd.to_datetime(demographic_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
                # except:
                #     demographic_df[c] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
                # try:
                #     demographic_df[f'{c} 2'] = (pd.to_datetime(demographic_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
                # except:
                #     demographic_df[f'{c} 2'] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
            elif i in [4, 6]:
                demographic_df[c] = pd.to_datetime(demographic_df[c], format='%d/%m/%Y')
                # demographic_df[c] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d')
                # demographic_df[f'{c} 2'] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d %H:%M:%S')
        demographic_df = demographic_df.astype(convert_dict)

        # print(demographic_df)
        # print(demographic_df.dtypes)
        # Forming patient info
        list_of_patients = []
        list_of_all_records = []
        pats = [1]
        recs = list(demographic_df['Number'])
        # self.logger.info(f'=========== {pats}, {recs}, {record_no}')

        if len(patient_no) == 0:
            list_of_patients = pats
        else:
            list_of_patients = [i for i in pats if i in patient_no]
            if len(list_of_patients) != len(patient_no):
                self.logger.info(f'Not all the patients asked for exists!')

        for ind, p in enumerate(list_of_patients):
            list_of_record = []
            if len(record_no) == 0:
                list_of_record = recs
            else:
                list_of_record = [i for i in recs if i in record_no]
                if len(list_of_record) != len(record_no):
                    self.logger.info(f'Not all the patients asked for exists!')
            list_of_all_records.append(list_of_record)

        for ind, p in enumerate(list_of_patients):
            list_of_record = list_of_all_records[ind]
            # self.logger.info(f'-------------- {p}, {list_of_record}, {list_of_all_records}')
            patient = self._generate_records_for_single_patient(p, list_of_record, xl_data, demographic_df)

            if patient:
                all_patient_list.append(patient)

        self.patient_list = all_patient_list
        self.record_indices = [list(item[1]) for item in itertools.groupby(sorted(self.record_indices), key=lambda x: x[0])]

        self.logger.info(f'---> Finished retrieving summary data Alfred...')
        return all_patient_list


    def _generate_records_for_single_patient(self, patient_no, rec_list, xl_data, demographic_df):
        seizure_indicators = ['A', 'B'] #['Sz', 'IIC'] ['A', 'B', 'IIC']
        rec_nums = []
        records = []
        sz_times = []
        sz_durations = []

        # Patient metadata
        patient = Patient(patient_no)
        pat_info_path = f'{self.data_directory}/cEEG-001/cEEG-001.edf'
        edf_data = mne.io.read_raw_edf(pat_info_path)
        edf_dict = edf_data.__dict__
        edf_dict_info = edf_dict['info']
        patient.sam_rate = edf_dict_info['sfreq']
        pat_chns = self._create_channel_dictionary_from_list(edf_dict_info['ch_names'])
        patient.channels.append( pat_chns )
        patient.removable_channels_.append( self._get_removable_channels(pat_chns) )
        patient.record_numbers = rec_list
        patient.num_records = len(rec_list)

        # Record metadata
        # record_sheet = f'cEEG-{i:>03} Annotations'
        # record_sheet = f'Events'
        sheets = xl_data.sheet_names
        num_active_cols = 15
        events_df = (pd.read_excel(xl_data, sheets[1])).iloc[:, :num_active_cols]
        # print(events_df)
        # print(events_df.dtypes)
        cols = (events_df.columns)[:3]
        part_df = events_df[cols]
        part_df.ffill(inplace=True)
        events_df[cols] = part_df[cols]
        events_df = events_df[events_df['Number'].isin(rec_list)]
        # convert_dict = {'Number': int, 'Event No.':int, 'Event Duration': str, 'Event Start Time': str, 'Event End Time': str}
        convert_dict = {'Number': int, 'Event Duration': str, 'Event Start Time': str, 'Event End Time': str}
        events_df = events_df.astype(convert_dict)
        events_df.dropna(subset=['Event Duration'])
        events_df = events_df[events_df['Event Duration'] != 'nan']
        events_df = events_df[events_df['Seizure Definition'].isin(seizure_indicators)]

        cols = events_df.columns
        convert_dict = {}
        for i, c in enumerate(cols):
            if i in [0, 3]:
                convert_dict[c] = int
            elif i == 1:
                convert_dict[c] = str
            elif i in [8, 9]:  # [3, 5, 8, 9]:
                events_df[c] = pd.to_datetime(events_df[c], format='%H:%M:%S')
                # events_df[c] = pd.to_datetime(events_df[c], format='%d/%m/%Y')
                # events_df[c] = pd.to_datetime(events_df[c], format='%Y-%m-%d %H:%M:%S')
                # try:
                #     events_df[c] = (pd.to_datetime(events_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
                # except:
                #     events_df[c] = pd.to_datetime(events_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
                # try:
                #     events_df[f'{c} 2'] = (pd.to_datetime(events_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
                # except:
                #     events_df[f'{c} 2'] = pd.to_datetime(events_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
        events_df = events_df.astype(convert_dict)

        # print(events_df)
        # print(events_df.dtypes)
        # self.logger.info(f'&&&&&&&&&&&&&\n, {events_df}')
        sz_rec_lst = events_df['Number'].unique()
        patient.seizure_record_numbers = sz_rec_lst
        patient.num_seizure_records = len(sz_rec_lst)

        # all_recs = self._generate_info_for_single_record(patient_no, sz_rec_lst, events_df, demographic_df)
        all_recs = self._generate_info_for_single_record(patient_no, rec_list, events_df, demographic_df)
        patient.records = all_recs

        return patient


    def _generate_info_for_single_record(self, pat, recs, record_df, demographic_df):
        seizure_indicators = ['A', 'B'] #['Sz', 'IIC'] ['A', 'B', 'IIC']
        record_list = []

        # Reading record info from excel
        if (record_df.shape)[0]>0:
            for rec in recs:
                rec_path = f'{self.data_directory}/cEEG-{rec:>03}/'
                record = Record(rec)
                record.patient_no = pat
                record.channel_id = 0
                ddf = demographic_df[demographic_df['Number']==rec]
                # ddf['Segment Start Time 2'] = (pd.to_datetime(ddf['Segment Start Time'], format='%H:%M:%S')).dt.strftime( '%H:%M:%S')
                # ddf['Segment End Time 2'] = (pd.to_datetime(ddf['Segment End Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')

                val = ddf[ddf['Number']==rec]['Segment Start Time'].values[0]
                seg_st = self._calculate_time_in_seconds_from_string2(val)
                val = ddf[ddf['Number']==rec]['Segment End Time'].values[0]
                seg_en = self._calculate_time_in_seconds_from_string2(val)
                # val = ddf[ddf['Number']==rec]['Segment Start Time 2'].values[0]
                # seg_st = self._calculate_time_in_seconds_from_string2(val)
                # val = ddf[ddf['Number']==rec]['Segment End Time 2'].values[0]
                # seg_en = self._calculate_time_in_seconds_from_string2(val)
                if seg_en < seg_st:
                    seg_en += 24 * 3600
                record.record_times = (seg_st, seg_en)
                record.record_duration = seg_en-seg_st

                # Reading seizure info from excel
                rdf = record_df[record_df['Number']==rec]
                ev_numbers = rdf['Event No.']#.sort()
                for ev_no in ev_numbers:
                    # rdf['Event Start Time 2'] = (pd.to_datetime(rdf['Event Start Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
                    # rdf['Event End Time 2'] = (pd.to_datetime(rdf['Event End Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')

                    val = rdf[rdf['Event No.']==ev_no]['Event Start Time'].values[0]
                    ev_st = self._calculate_time_in_seconds_from_string2(val)
                    val = rdf[rdf['Event No.']==ev_no]['Event End Time'].values[0]
                    ev_en = self._calculate_time_in_seconds_from_string2(val)
                    # val = rdf[rdf['Event No.']==ev_no]['Event Start Time 2'].values[0]
                    # ev_st = self._calculate_time_in_seconds_from_string2(val)
                    # val = rdf[rdf['Event No.']==ev_no]['Event End Time 2'].values[0]
                    # ev_en = self._calculate_time_in_seconds_from_string2(val)
                    if ev_en < ev_st:
                        ev_en += 24 * 3600
                    ev_st = ev_st-seg_st
                    ev_en = ev_en-seg_st
                    record.seizure_times.append( tuple((ev_st, ev_en)) )
                    record.seizure_durations.append( (ev_en-ev_st) )
                record.num_seizures = len(record.seizure_durations)

                path = os.walk(rec_path)
                for root, directories, files in path:
                    for file in files:
                        if file.find('.edf')>0:
                            record.file_name = file
                            record.file_path = f'{rec_path}/{file}'
                            break

                record.data: RawEDF = None
                record_list.append(record)

        return record_list


    def _create_channel_dictionary_from_list(self, chns):
        chn_dict = {}
        for i, c in enumerate(chns):
            chn_dict[f'Channel {i+1}'] = c
        return chn_dict

    def _calculate_time_in_seconds_from_string2(self, str_time):
        str_time = str(np.datetime_as_string(str_time))
        # datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
        time_arr = (str_time.split('T')[1]).split('.')[0]
        time_arr = time_arr.split(':')
        time_in_seconds = int(time_arr[0])*3600 + int(time_arr[1])*60 + int(time_arr[2])
        return time_in_seconds




###############################################################################################################

######################################        Alfred OLD       ################################################

###############################################################################################################
    # def _get_record_info_for_patient(self, patient_no, record_no):
    #     self.logger.info(f'---> Getting summary data...')
    #     all_patient_list = []
    #     self.patient_list = []
    #     self.record_indices = []
    #
    #     # Reading excel metadata
    #     num_active_cols = 11
    #     xl_data = pd.ExcelFile(self.metadata_path)
    #     sheets = xl_data.sheet_names
    #     demographic_df = (pd.read_excel(xl_data, sheets[0])).iloc[:, :num_active_cols]
    #     demographic_df = demographic_df.dropna()
    #     cols = demographic_df.columns
    #     convert_dict = {}
    #     for i, c in enumerate(cols):
    #         if i in [0, 2]:
    #             convert_dict[c] = int
    #         elif i == 1:
    #             convert_dict[c] = str
    #         #     else:
    #         #         convert_dict[c] = datetime64[ns]
    #         elif i in [8, 9]:  # [3, 5, 8, 9]:
    #             try:
    #                 demographic_df[f'{c} 2'] = (pd.to_datetime(demographic_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
    #             except:
    #                 demographic_df[f'{c} 2'] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
    #         elif i in [4, 6]:
    #             demographic_df[f'{c} 2'] = pd.to_datetime(demographic_df[c], format='%Y-%m-%d %H:%M:%S')
    #     demographic_df = demographic_df.astype(convert_dict)
    #
    #     # Forming patient info
    #     list_of_patients = []
    #     list_of_all_records = []
    #     pats = [1]
    #     recs = list(demographic_df['Number'])
    #     # self.logger.info(f'=========== {pats}, {recs}, {record_no}')
    #
    #     if len(patient_no) == 0:
    #         list_of_patients = pats
    #     else:
    #         list_of_patients = [i for i in pats if i in patient_no]
    #         if len(list_of_patients) != len(patient_no):
    #             self.logger.info(f'Not all the patients asked for exists!')
    #
    #     for ind, p in enumerate(list_of_patients):
    #         list_of_record = []
    #         if len(record_no) == 0:
    #             list_of_record = recs
    #         else:
    #             list_of_record = [i for i in recs if i in record_no]
    #             if len(list_of_record) != len(record_no):
    #                 self.logger.info(f'Not all the patients asked for exists!')
    #         list_of_all_records.append(list_of_record)
    #
    #     for ind, p in enumerate(list_of_patients):
    #         list_of_record = list_of_all_records[ind]
    #         # self.logger.info(f'-------------- {p}, {list_of_record}, {list_of_all_records}')
    #         patient = self._generate_records_for_single_patient(p, list_of_record, xl_data, demographic_df)
    #
    #         if patient:
    #             all_patient_list.append(patient)
    #
    #     self.patient_list = all_patient_list
    #     self.record_indices = [list(item[1]) for item in itertools.groupby(sorted(self.record_indices), key=lambda x: x[0])]
    #
    #     self.logger.info(f'---> Finished retrieving summary data...')
    #     return all_patient_list
    #
    #
    # def _generate_records_for_single_patient(self, patient_no, rec_list, xl_data, demographic_df):
    #     seizure_indicators = ['A', 'B'] #['Sz', 'IIC'] ['A', 'B', 'IIC']
    #     rec_nums = []
    #     records = []
    #     sz_times = []
    #     sz_durations = []
    #
    #     # Patient metadata
    #     patient = Patient(patient_no)
    #     file_name_keyword = f'Rework'
    #     pat_info_path = f'{self.data_directory}/cEEG-001/cEEG-001 (Seizure Example) {file_name_keyword}.edf'
    #     edf_data = mne.io.read_raw_edf(pat_info_path)
    #     edf_dict = edf_data.__dict__
    #     edf_dict_info = edf_dict['info']
    #     patient.sam_rate = edf_dict_info['sfreq']
    #     pat_chns = self._create_channel_dictionary_from_list(edf_dict_info['ch_names'])
    #     patient.channels.append( pat_chns )
    #     patient.removable_channels_.append( self._get_removable_channels(pat_chns) )
    #     patient.record_numbers = rec_list
    #     patient.num_records = len(rec_list)
    #
    #     # Record metadata
    #     # record_sheet = f'cEEG-{i:>03} Annotations'
    #     # record_sheet = f'Events'
    #     sheets = xl_data.sheet_names
    #     num_active_cols = 15
    #     events_df = (pd.read_excel(xl_data, sheets[1])).iloc[:, :num_active_cols]
    #     cols = (events_df.columns)[:3]
    #     part_df = events_df[cols]
    #     part_df.ffill(inplace=True)
    #     events_df[cols] = part_df[cols]
    #     events_df = events_df[events_df['Number'].isin(rec_list)]
    #     # convert_dict = {'Number': int, 'Event No.':int, 'Event Duration': str, 'Event Start Time': str, 'Event End Time': str}
    #     convert_dict = {'Number': int, 'Event Duration': str, 'Event Start Time': str, 'Event End Time': str}
    #     events_df = events_df.astype(convert_dict)
    #     events_df.dropna(subset=['Event Duration'])
    #     events_df = events_df[events_df['Event Duration'] != 'nan']
    #     events_df = events_df[events_df['Seizure Definition'].isin(seizure_indicators)]
    #
    #     cols = events_df.columns
    #     convert_dict = {}
    #     for i, c in enumerate(cols):
    #         if i in [0, 3]:
    #             convert_dict[c] = int
    #         elif i == 1:
    #             convert_dict[c] = str
    #         elif i in [8, 9]:  # [3, 5, 8, 9]:
    #             try:
    #                 events_df[f'{c} 2'] = (pd.to_datetime(events_df[c], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
    #             except:
    #                 events_df[f'{c} 2'] = pd.to_datetime(events_df[c], format='%Y-%m-%d %H:%M:%S')  # format='%H:%M:%S')
    #     events_df = events_df.astype(convert_dict)
    #
    #     # self.logger.info(f'&&&&&&&&&&&&&\n, {events_df}')
    #     sz_rec_lst = events_df['Number'].unique()
    #     patient.seizure_record_numbers = sz_rec_lst
    #     patient.num_seizure_records = len(sz_rec_lst)
    #
    #     # all_recs = self._generate_info_for_single_record(patient_no, sz_rec_lst, events_df, demographic_df)
    #     all_recs = self._generate_info_for_single_record(patient_no, rec_list, events_df, demographic_df)
    #     patient.records = all_recs
    #
    #     return patient
    #
    #
    # def _generate_info_for_single_record(self, pat, recs, record_df, demographic_df):
    #     seizure_indicators = ['A', 'B'] #['Sz', 'IIC'] ['A', 'B', 'IIC']
    #     record_list = []
    #
    #     # Reading record info from excel
    #     if (record_df.shape)[0]>0:
    #         for rec in recs:
    #             rec_path = f'{self.data_directory}/cEEG-{rec:>03}/'
    #             record = Record(rec)
    #             record.patient_no = pat
    #             record.channel_id = 0
    #             ddf = demographic_df[demographic_df['Number']==rec]
    #             # ddf['Segment Start Time 2'] = (pd.to_datetime(ddf['Segment Start Time'], format='%H:%M:%S')).dt.strftime( '%H:%M:%S')
    #             # ddf['Segment End Time 2'] = (pd.to_datetime(ddf['Segment End Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
    #
    #             val = ddf[ddf['Number']==rec]['Segment Start Time 2'].values[0]
    #             seg_st = self._calculate_time_in_seconds_from_string2(val)
    #             val = ddf[ddf['Number']==rec]['Segment End Time 2'].values[0]
    #             seg_en = self._calculate_time_in_seconds_from_string2(val)
    #             if seg_en < seg_st:
    #                 seg_en += 24 * 3600
    #             record.record_times = (seg_st, seg_en)
    #             record.record_duration = seg_en-seg_st
    #
    #             # Reading seizure info from excel
    #             rdf = record_df[record_df['Number']==rec]
    #             ev_numbers = rdf['Event No.']#.sort()
    #             for ev_no in ev_numbers:
    #                 # rdf['Event Start Time 2'] = (pd.to_datetime(rdf['Event Start Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
    #                 # rdf['Event End Time 2'] = (pd.to_datetime(rdf['Event End Time'], format='%H:%M:%S')).dt.strftime('%H:%M:%S')
    #
    #                 val = rdf[rdf['Event No.']==ev_no]['Event Start Time 2'].values[0]
    #                 ev_st = self._calculate_time_in_seconds_from_string2(val)
    #                 val = rdf[rdf['Event No.']==ev_no]['Event End Time 2'].values[0]
    #                 ev_en = self._calculate_time_in_seconds_from_string2(val)
    #                 if ev_en < ev_st:
    #                     ev_en += 24 * 3600
    #                 ev_st = ev_st-seg_st
    #                 ev_en = ev_en-seg_st
    #                 record.seizure_times.append( tuple((ev_st, ev_en)) )
    #                 record.seizure_durations.append( (ev_en-ev_st) )
    #             record.num_seizures = len(record.seizure_durations)
    #
    #             path = os.walk(rec_path)
    #             file_name_keyword = f'Rework'
    #             for root, directories, files in path:
    #                 for file in files:
    #                     if file.find(file_name_keyword)>0:
    #                         record.file_name = file
    #                         record.file_path = f'{rec_path}/{file}'
    #                         break
    #
    #             record.data: RawEDF = None
    #             record_list.append(record)
    #
    #     return record_list
    #
    #
    # def _create_channel_dictionary_from_list(self, chns):
    #     chn_dict = {}
    #     for i, c in enumerate(chns):
    #         chn_dict[f'Channel {i+1}'] = c
    #     return chn_dict
    #
    # def _calculate_time_in_seconds_from_string2(self, str_time):
    #     datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
    #     time_arr = str_time.split(':')
    #     time_in_seconds = int(time_arr[0])*3600 + int(time_arr[1])*60 + int(time_arr[2])
    #     return time_in_seconds


