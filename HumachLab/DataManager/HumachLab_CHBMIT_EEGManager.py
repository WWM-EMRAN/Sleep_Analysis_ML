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
    from HumachLab.DataManager.HumachLab_EEGDataStucture import Patient, Record
    from HumachLab.Utility.HumachLab_StaticMethods import *
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###



class HumachLab_CHBMIT_EEGManager:

    def __init__(self, logger, data_directory='./', metadata_path='./', removable_channel_list=[], is_remove_duplicate_channels=False):
        self.logger = logger
        self.logger.info(f'Processing ALFRED data...')

        self.data_directory = os.path.abspath(data_directory)
        if not (os.path.exists(self.data_directory)):
            self.logger.info(f'Raw directory does not exist!')
            self.data_directory = os.path.abspath('./')

        # self.metadata_path = os.path.abspath(metadata_path)
        # if not (os.path.exists(self.metadata_path)):
        #     self.logger.info(f'Raw metadata path does not exist!')
        #     self.metadata_path = os.path.abspath('./')

        self.data_directory = self.data_directory.replace('\\', '/')
        # self.metadata_path = self.metadata_path.replace('\\', '/')

        # self._all_files = []
        self.max_patients = 25
        self.max_records = 100
        self.record_indices = []
        self.patient_list = []
        self._removable_dummy_channels = removable_channel_list
        self._is_remove_duplicate_channels = is_remove_duplicate_channels
        self._duplicate_channels = {}
        return

    ###############################################################################################################

    ######################################          CHB MIT        ################################################

    ###############################################################################################################

    def get_record_info_for_patient(self, patient_no, record_no):
        list_of_patients = []
        all_patient_list = []
        self.patient_list = []
        self.record_indices = []

        if len(patient_no) == 0 and len(record_no) == 0:
            list_of_patients = range(1, (self.max_patients))
        else:
            list_of_patients = patient_no

        for p in list_of_patients:
            list_of_record = []
            if len(patient_no) > 0 and len(record_no) > 0:
                ind = list_of_patients.index(p)
                list_of_record = record_no[ind]

            patient = self._generate_records_for_single_patient(p, list_of_record)

            # self.logger.info(f'KKKKK {len(patient.channels)}, \n, {len(patient.removable_channels_)}')

            if patient:
                all_patient_list.append(patient)

        self.patient_list = all_patient_list
        self.record_indices = [list(item[1]) for item in
                               itertools.groupby(sorted(self.record_indices), key=lambda x: x[0])]

        return all_patient_list

    def _generate_records_for_single_patient(self, patient_no, r_no):
        # summary
        summary_stat, summary_path = self._has_record_info_for_patient(patient_no, 0, 3)
        patient = None
        if not summary_stat:
            self.logger.info('Summary for Patient {} is not found.'.format(patient_no))
        else:
            patient = self._populate_patient_info_with_summary_data(patient_no, summary_path)
            # self.logger.info(f'LLLLLL {len(patient.channels)}, \n {len(patient.removable_channels_)}')

        if not patient:
            self.logger.info(f'No record found for patient {patient_no}')
        else:
            missing_rec_ind = []
            present_chn_ind = []
            rearrange_chn_ind = 0
            ind_cnt = 0
            for record_no in patient.record_numbers:
                tmp_ind = []
                edf_path = ''
                edf_data = None

                if len(r_no) == 0 or (len(r_no) > 0 and (record_no in r_no)):
                    tmp_ind, edf_path = self._generate_single_record_for_single_patient(patient_no, record_no)

                if len(tmp_ind) > 0:
                    # self.logger.info(f'Rec {record_no} found for Patient {patient_no}: {tmp_ind}')
                    self.record_indices.append(tmp_ind)
                    # patient.data = edf_data
                    tmp_record: Record = patient.records[ind_cnt]
                    tmp_record.file_path = edf_path
                    tmp_record.data = edf_data
                    # self.logger.info(f'{tmp_ind} --> {tmp_record.channel_id}')

                    # if self._is_remove_duplicate_channels:
                    #     dropable_chns = (patient.removable_channels_[tmp_record.channel_id])
                    #     dropable_chns = list(dropable_chns.values())
                    #     edf_data.drop_channels(dropable_chns)

                    if len(r_no) > 0:
                        present_chn_ind.append(tmp_record.channel_id)
                        # tmp_record.channel_id = rearrange_chn_ind
                        # self.logger.info(f'{tmp_ind} ====> {tmp_record.channel_id}')
                    # self.logger.info(f'+++ {patient.channels}, {tmp_record.channel_id}')
                else:
                    indx = patient.record_numbers.index(record_no)
                    missing_rec_ind.append(indx)

                ind_cnt += 1
                rearrange_chn_ind += 1

            # Deal with missing records
            # self.logger.info(f'LLLLLL222 {len(patient.channels)}, \n {len(patient.removable_channels_)}')
            patient.record_numbers = [i for j, i in enumerate(patient.record_numbers) if j not in missing_rec_ind]
            patient.records = [i for j, i in enumerate(patient.records) if j not in missing_rec_ind]
            # if len(r_no) > 0:
            #     patient.channels = [i for j, i in enumerate(patient.channels) if j in present_chn_ind]
            #     patient.removable_channels_ = [i for j, i in enumerate(patient.removable_channels_) if j in present_chn_ind]
            patient.num_records -= len(missing_rec_ind)
            # self.logger.info(f'LLLLLL333 {len(patient.channels)}, \n {len(patient.removable_channels_)}')

        return patient


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
                        tmp_rem_chn_ = HumachLab_StaticMethods.get_removable_channels(tmp_chn, self._removable_dummy_channels, self._is_remove_duplicate_channels)
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
