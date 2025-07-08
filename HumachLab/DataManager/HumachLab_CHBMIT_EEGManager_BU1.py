# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from array import *
import mne
from itertools import islice
from datetime import datetime
from mne.io.edf.edf import RawEDF
import matplotlib.pyplot as plt

from HumachLab_CHBMIT_EEG_DataStucture import Patient, Record



class HumachLab_CHBMIT_EEGManager_BU1:
    ### EEG directory
    data_directory = './'

    ### File locations: all + data, seizure and summary only
    #_all_files = []
    data_files = []

    ### Maximum number of patients and records in dataset
    max_patients = 25 #Initialized in constructor again
    max_records = 100 #Initialized in constructor again

    ### Record index array: contain all indices of the record having data
    ### [patient_no, record_no, has_seizure_file:True/False]
    record_indices = []
    patient_list = []

    ### Dummy channel list to remove
    _removable_dummy_channels = []



    ###
    ### Initializing parameters
    ###
    def __init__(self, data_directory='./', removable_channel_list=[]):
        self.data_directory = os.path.abspath(data_directory)
        if not (os.path.exists(self.data_directory)):
            self.data_directory = os.path.abspath('./')
        self.data_directory = self.data_directory.replace('\\', '/')
        # if not self.data_directory[-1]=='/':
        #     self.data_directory += '/'

        #self._all_files = []
        self.data_files = []
        self.max_patients = 25
        self.max_records = 100
        self.record_indices = []
        self.patient_list = []
        self._removable_dummy_channels = removable_channel_list
        return

    ###################### Directory Manager ###################
    ###
    ### Traverse all files from a directory including its nested locations
    ###
    def list_all_directories_and_files(self, directory):
        locations = [all_files for all_files in os.walk(directory) if all_files[0].find('/.') == -1]

        list_of_folders = []
        list_of_subfolders = []
        list_of_files = []

        for item in locations:
            list_of_folders.append(item[0])
            list_of_subfolders.append([x for x in item[1] if not x.startswith('.')])
            list_of_files.append([x for x in item[2] if not x.startswith('.')])

        return (list_of_folders, list_of_subfolders, list_of_files)


    ###
    ### List all files to manage
    ###
    def list_all_files_to_manage(self):
        directory = self.data_directory
        list_of_folders, list_of_subfolders, list_of_files = self.list_all_directories_and_files(directory)
        list_of_locations = []

        for loc in list_of_folders:
            files = list_of_files[list_of_folders.index(loc)]
            if len(files) > 0:
                loc = loc.rstrip('/')
                list_of_locations.extend([(loc + '/' + file) for file in files])

        list_of_locations = [file.replace('\\', '/') for file in list_of_locations]
        list_of_locations = [file for file in list_of_locations if file.find('/.') < 0]

        #self._all_files += list_of_locations
        self.data_files += [file for file in list_of_locations if
                            (file.endswith('.edf') or file.endswith('.seizures') or file.endswith('summary.txt'))]

        return (list_of_locations)


    #################################################################
    ### Raw data management

    ###
    ### Convert specific edf file to csv file
    ###
    def _convert_specific_edf_from_file(self, data_file):
        # with tqdm(total=100, desc='Converting EDF file..') as pbar:
        print('---> Getting EDF data...')
        edf_data = None
        edf_data = mne.io.read_raw_edf(data_file, exclude=self._removable_dummy_channels)
        # pbar.update(100)
        # pbar.refresh(nolock=False, lock_args=None)
        # edf_data.plot()

        # Post processing of addition/removal of any data column goes here...
        print('===> EDF data completed!!!')
        return (edf_data)


    ###
    ### Convert summary data from specific summary file
    ###
    def _convert_summary_from_specific_summary_file(self, summary_file):
        sample_rate = 0
        channels = []
        data_for_channels = []
        data = []

        tmp_chn = {}
        tmp_dat = {}

        rat_ind = -1
        chn_ind = -1
        dat_ind = -1
        ind_cnt = -1
        dat_cnt = -1

        print('---> Getting summary data...')

        with open(summary_file, 'r') as f:
            for line in f:
                # print(line)
                byte_array = bytes(line, 'utf-8')
                words = line.strip().split(':')

                if (line == '\n'):
                    # print('Data block ends...')
                    if (chn_ind > -1):
                        # print('tmp_chn:{}'.format(tmp_chn))
                        channels.append(tmp_chn)
                        tmp_chn = {}
                        chn_ind = -1

                    if (dat_ind > -1):
                        # print('data finished')
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
                        should_remove_dummy = True if len(self._removable_dummy_channels) > 0 else False
                        if not (should_remove_dummy and (words[1] in self._removable_dummy_channels)):
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
                            # print('ind_cnt:{}===dat_ind:{}\n{}'.format(ind_cnt, dat_cnt, data_for_channels))
                            if len(data_for_channels) == ind_cnt:
                                data_for_channels.append(dat_cnt + 1)
                            else:
                                data_for_channels[ind_cnt] = (dat_cnt + 1)
                        # print('data_for_channels:{}'.format(dat_ind+1))

                else:
                    # print('---> Something is going on here...')
                    pass

            ### Last data doesn't automatically added
            data.append(tmp_dat)

        ### Split data input according to the number of channels
        data_iter = iter(data)
        data = [list(islice(data_iter, elem)) for elem in data_for_channels]
        # print('sample_rate:{} \nchannels:{} \ndata:{}'.format(sample_rate, channels, data))

        print('===> Summary data completed!!!')

        return (sample_rate, channels, data)


    def _populate_patient_info_with_summary_data(self, patient_no, summary_file):
        samp_rate, chans, rec_info = self._convert_summary_from_specific_summary_file(summary_file)
        # print(samp_rate, '\n', chans, '\n', rec_info)

        # Populating patient data
        patient = Patient(patient_no)
        patient.sam_rate = samp_rate
        patient.channels = chans
        # patient.num_records = len(rec_info)
        # print('Num rec: {}'.format(len(rec_info)))

        # Populating record data
        ch_id = 0
        for rec_group in rec_info:
            for rec in rec_group:
                rec_no = int((((rec['File Name'].split('.'))[0]).split('_'))[1])
                patient.record_numbers.append(rec_no)
                record = Record(rec_no)
                record.patient_no = patient_no
                record.file_name = rec['File Name']
                record.channel_id = ch_id
                file_st_time = self._calculate_time_in_seconds_from_string(rec['File Start Time'])
                file_end_time = self._calculate_time_in_seconds_from_string(rec['File End Time'])
                record.record_times = (file_st_time, file_end_time)
                record.record_duration = file_end_time - file_st_time
                num_seiz = int(rec['Number of Seizures in File'])
                record.num_seizures = num_seiz
                print('Patient: {} Record: {}, NumSeiz: {}'.format(patient_no, rec_no, num_seiz))
                if num_seiz>0:
                    for sind in range(1, (num_seiz+1)):
                        seiz_st_key = 'Seizure {} Start Time'.format(sind)
                        seiz_end_key = 'Seizure {} End Time'.format(sind)

                        seiz_st = -1
                        seiz_end = -1

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

            ch_id += 1

        # print('+++ ', len(patient.records), patient.record_numbers)
        nr = len(patient.records)
        patient.num_records = nr
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
        #print('came...', record_path)

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
                print('--> ', patient_no, record_no, info_type, record_path)

        # print('--> ', patient_no, record_no, info_type, record_path)

        return record_found, record_path


    ###
    ### Record Management functions
    ###
    def _generate_records_for_all_patient(self):
        self.patient_list = []
        for p in range(1, (self.max_patients)):
            patient = self._generate_records_for_single_patient(p, 0)
            if patient:
                self.patient_list.append(patient)

        return self.patient_list


    def _generate_records_for_single_patient(self, patient_no, r_no):
        self.record_indices = []
        # summary
        summary_stat, summary_path = self._has_record_info_for_patient(patient_no, 0, 3)
        patient = None
        if not summary_stat:
            print('Summary for Patient {} is not found.'.format(patient_no))
        else:
            patient = self._populate_patient_info_with_summary_data(patient_no, summary_path)

        if not patient:
            print('No record found for patient {}'.format(patient_no))
        else:
            missing_rec_ind = []
            ind_cnt = 0
            for record_no in patient.record_numbers:
                tmp_ind = []
                edf_data = None
                if r_no==0 or (r_no>0 and record_no==r_no):
                    tmp_ind, edf_data = self._generate_single_record_for_single_patient(patient_no, record_no)

                if len(tmp_ind) > 0:
                    # print('Rec {} found for Patient {}: {}'.format(record_no, patient_no, tmp_ind))
                    self.record_indices.append(tmp_ind)
                    # patient.data = edf_data
                    tmp_record: Record = patient.records[ind_cnt]
                    tmp_record.data = edf_data
                    ### for single record remove all other channels in the channel list
                    if r_no>0:
                        patient.channels = [chans for chans in patient.channels if ((patient.channels.index(chans)) == tmp_record.channel_id)]
                else:
                    indx = patient.record_numbers.index(record_no)
                    missing_rec_ind.append(indx)

                ind_cnt += 1

            ### Deal with missing records
            patient.record_numbers = [i for j, i in enumerate(patient.record_numbers) if j not in missing_rec_ind]
            patient.records = [i for j, i in enumerate(patient.records) if j not in missing_rec_ind]
            patient.num_records -= len(missing_rec_ind)


        return patient


    def _generate_single_record_for_single_patient(self, patient_no, record_no):
        edf_stat, edf_path = self._has_record_info_for_patient(patient_no, record_no, 1)
        seizure_stat, seizure_path = self._has_record_info_for_patient(patient_no, record_no, 2)
        tmp_ind = []
        edf_data = None

        if edf_stat:
            tmp_ind.append(patient_no)
            tmp_ind.append(record_no)

            edf_data = self._convert_specific_edf_from_file(edf_path)

            if seizure_stat:
                tmp_ind.append(1)
            else:
                tmp_ind.append(0)

        return tmp_ind, edf_data



    def _get_record_info_for_patient(self, patient_no, record_no):
        data = None
        if patient_no==0 and record_no==0:
            data = self._generate_records_for_all_patient()
        elif patient_no>0 and record_no==0:
            data = self._generate_records_for_single_patient(patient_no, record_no)
        elif patient_no>0 and record_no>0:
            data = self._generate_records_for_single_patient(patient_no, record_no)
        return data


    def get_record_info_for_all_patient(self):
        result = self._get_record_info_for_patient(0, 0)
        return result


    def get_record_info_for_single_patient(self, patient_no, record_no=0):
        result = self._get_record_info_for_patient(patient_no, record_no)
        return result



########################################
# Testing...............
# directory = os.path.abspath('./../data/')
# # directory = "C:\Users\aliem\OneDrive\ -\ Deakin University\Deakin\ Uni\ AU\MS\My\ Research\Physionet_CHB-MIT_EEG_Epilepcy\CHB_MIT_EEG_Dataset\1.0.0"
# # directory = './../CHB_MIT_EEG_Dataset/1.0.0/'
#
# manager = HumachLab_CHBMIT_EEGManager(directory, removable_channel_list=['-'])
#
# ### All Patient info
# p_list = manager.get_record_info_for_all_patient()
# print(p_list)
# for p in p_list:
#     r = p.records[0]
#     print(p.patient_no, p.record_numbers, p.records[0].num_seizures, '\n-', manager.record_indices, '\n-', r.data)
#
# ### Single Patient info
# # p = manager.get_record_info_for_single_patient(12)
# # r = p.records[0]
# # print(p.patient_no, p.record_numbers, p.records[0].num_seizures, '\n-', manager.record_indices, '\n-', r.data)
# # r.data.plot()



