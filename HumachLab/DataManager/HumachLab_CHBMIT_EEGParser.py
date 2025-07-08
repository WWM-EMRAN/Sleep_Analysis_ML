# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from array import *
import mne
from numpy import genfromtxt
from itertools import islice
from ast import literal_eval
from io import StringIO


# %matplotlib qt


class HumachLab_CHBMIT_EEGParser:
    ### Parsing directory
    parsing_directory = './'

    ### List of all files
    _all_files = []

    ### List of edf, seizure and summary files
    _data_files = []

    ### List of edf, seizure and summary files
    _parsed_data_files = []

    ### Control parsing specific file type
    _is_parse_edf_files = True
    _is_parse_seizure_files = True
    _is_parse_summary_files = True

    ### Parsed file lists
    _parsed_edf_files = []
    _parsed_seizure_files = []
    _parsed_summary_files = []

    ### Parsed data lists
    _parsed_edf_data = {}
    _parsed_seizure_data = {}
    _parsed_summary_data = {}
    _removable_dummy_channels = []

    ###
    ### Initializing parameters
    ###
    def __init__(self, parsing_directory='./', removable_dummy_channels=[], is_parse_edf_files=True, is_parse_seizure_files=True,
                 is_parse_summary_files=True):
        self.parsing_directory = os.path.abspath(parsing_directory)
        if not(os.path.exists(self.parsing_directory)):
            self.parsing_directory = os.path.abspath('./')
        self.parsing_directory = self.parsing_directory.replace('\\', '/')

        self._data_files = []
        self._parsed_data_files = []
        self._removable_dummy_channels = removable_dummy_channels

        self._is_parse_edf_files = is_parse_edf_files
        self._is_parse_seizure_files = is_parse_seizure_files
        self._is_parse_summary_files = is_parse_summary_files

        self._parsed_edf_files = []
        self._parsed_seizure_files = []
        self._parsed_summary_files = []

        self._parsed_edf_data = {}
        self._parsed_seizure_data = {}
        self._parsed_summary_data = {}

        return

    ###
    ### Convert specific edf file to csv file
    ###
    def _convert_specific_edf_to_csv(self, data_file):
        # res = data_file.rfind('.', 0, len(data_file))
        # data_file_csv = data_file[:res]
        data_file_csv = data_file + ".csv"
        self._parsed_data_files.append(data_file_csv)
        self._parsed_edf_files.append(data_file_csv)

        if os.path.exists(data_file_csv):
            print('Retrieving EDF file...')
            my_data = genfromtxt(data_file_csv, delimiter=',', dtype=str)
            print('EDF file already converted!')
            return (my_data)

        with tqdm(total=100, desc='Converting EDF file..') as pbar:
            print('--> Conversion of EDF file started!')
            edf = mne.io.read_raw_edf(data_file, exclude=self._removable_dummy_channels)
            pbar.update(40)
            # pbar.refresh(nolock=False, lock_args=None)
            header = ','.join(edf.ch_names)

            # print(header)

            # Post preprocessing of addition/removal of any data column goes here...

            # print((edf.get_data().T)[0])

            np_data = edf.get_data().T
            pbar.update(20)
            # pbar.refresh(nolock=False, lock_args=None)
            np.savetxt(data_file_csv, np_data, delimiter=',', header=header)
            pbar.update(40)
            # pbar.refresh(nolock=False, lock_args=None)
            print('==> Conversion of EDF file completed!')
            return (np_data)

    ###
    ### Convert data from specific seizures file
    ###
    def _convert_data_from_specific_seizures_file(self, annotation_file):

        # res = annotation_file.rfind('.', 0, len(annotation_file))
        # annotation_file_csv = annotation_file[:res]
        # annotation_file_csv = annotation_file_csv+".csv"
        annotation_file_csv = annotation_file + ".csv"
        self._parsed_data_files.append(annotation_file_csv)
        self._parsed_seizure_files.append(annotation_file_csv)

        if os.path.exists(annotation_file_csv):
            print('Retrieving Seizure file...')
            my_data = genfromtxt(annotation_file_csv, delimiter=',', dtype=str)
            print('Seizure file already converted!')
            return (my_data)

        with tqdm(total=100, desc='Converting seizures file..') as pbar2:
            print('--> Conversion of Seizure file started!')
            data_bytes: bytearray
            with open(annotation_file, 'rb') as fh:
                data_bytes = bytearray(fh.read())
            pbar2.update(40)
            # pbar2.refresh(nolock=False, lock_args=None)
            data_array = []  # array('i', [])
            # print(data_array)
            for byte in data_bytes:
                # i = int.from_bytes(byte, byteorder='little')
                # print(i)
                # print(byte)
                data_array.append(byte)

            # np_annotation = np.array(data_array)
            np_annotation = np.fromiter(data_array, dtype=int)
            pbar2.update(20)
            # pbar2.refresh(nolock=False, lock_args=None)
            np.savetxt(annotation_file_csv, np_annotation, delimiter=',')
            pbar2.update(40)
            # pbar2.refresh(nolock=False, lock_args=None)
            print('==> Conversion of Seizure file completed!')
            return (np_annotation)

    ###
    ### Convert summary data from specific summary file
    ###
    def _convert_summary_from_specific_summary_file(self, summary_file):

        file_size = os.stat(summary_file).st_size

        summary_file_csv = summary_file + "-data.csv"

        self._parsed_data_files.append(summary_file + "-sam_rate.csv")
        self._parsed_data_files.append(summary_file + "-channels.csv")
        self._parsed_data_files.append(summary_file + "-data_for_channels.csv")
        self._parsed_data_files.append(summary_file_csv)

        self._parsed_summary_files.append(summary_file + "-sam_rate.csv")
        self._parsed_summary_files.append(summary_file + "-channels.csv")
        self._parsed_summary_files.append(summary_file + "-data_for_channels.csv")
        self._parsed_summary_files.append(summary_file_csv)

        if os.path.exists(summary_file_csv):
            print('Retrieving Summary file...')
            my_data1 = np.genfromtxt(summary_file + "-sam_rate.csv", delimiter=',', dtype=str)
            my_data2 = np.genfromtxt(summary_file + "-channels.csv", delimiter=',', dtype=str)
            my_data3 = np.genfromtxt(summary_file + "-data_for_channels.csv", delimiter=',', dtype=str)
            #my_data3
            #my_data4 = np.genfromtxt(summary_file_csv, dtype=object)
            ###########
            dtf = pd.read_csv(summary_file_csv, dtype=object)
            #dtf = dtf.dropna(how='any')

            #dtf.dropna(how='any', axis=1)
            # for i in dtf.iloc(0):
            #     print('dd- ', i)

            # dtf = dtf.apply(literal_eval)

            my_data4_test = dtf.values.tolist()
            my_data4 = []
            for dat in my_data4_test:
                dat = [x for x in dat if str(x)!='nan']
                dat = [eval(x) for x in dat]
                my_data4.append(dat)

            #print('\nDDD-', dtf,'\n===', my_data4)

            #print(dtf.apply(lambda x: type(x)))
            ###########
            np_samRate = np.array(my_data1)
            np_channels = np.array(my_data2)
            np_data_for_channels = np.array(my_data3)
            np_data = np.array(my_data4)
            print('Summary file already converted!')
            return (np_samRate, np_channels, np_data_for_channels, np_data)

        sample_rate = []
        channels = []
        data = []
        data_for_channels = []

        szr_tmp_arr = []
        tmp_szr = {}

        tmp_rat = {}
        tmp_chn = {}
        tmp_dat = {}

        rat_ind = -1
        chn_ind = -1
        dat_ind = -1
        szr_ind = -1
        ind_cnt = -1
        dat_cnt = -1

        print('--> Conversion of Summary file started!')

        with tqdm(total=100, desc='Converting summary file..') as pbar3:
            with open(summary_file, 'r') as f:
                for line in f:
                    # print(line)
                    byte_array = bytes(line, 'utf-8')
                    byte_len = len(byte_array)
                    pbar3.update((100 / file_size) * byte_len)

                    words = line.strip().split(':')

                    if (line == '\n'):
                        # print('Data block ends...')

                        if (rat_ind > -1):
                            # print('tmp_rat:{}'.format(tmp_rat))
                            sample_rate.append(tmp_rat)
                            tmp_rat = {}
                            rat_ind = -1

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
                        # print('Data block...')
                        # continue
                        # print(line)

                        words[0] = words[0].strip()
                        words[1] = words[1].strip()
                        # print(words)

                        if (words[0].find('Sampling Rate') >= 0):
                            rat_ind += 1
                            tmp_rat[words[0]] = words[1]
                            chn_ind = -1
                            dat_ind = -1

                        elif (words[0].find('Channel') >= 0):
                            chn_ind += 1
                            should_remove_dummy = True if len(self._removable_dummy_channels)>0 else False
                            if not(should_remove_dummy and (words[1] in self._removable_dummy_channels)):
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
                tmp_dat = {}
                dat_ind = -1
        # print('sample_rate:{} \nchannels:{} \ndata_for_channels:{} \ndata:{}'.format(sample_rate, channels, data_for_channels, data))

        ### Split data input according to the number of channels
        data_iter = iter(data)
        data = [list(islice(data_iter, elem)) for elem in data_for_channels]
        #print('DATA_X: ', data)

        np_samRate = np.array(sample_rate)
        np_channels = np.array(channels)
        np_data_for_channels = np.array(data_for_channels)
        np_data = np.array(data)

        # print(np_data[0].keys())

        df = pd.DataFrame(sample_rate)
        df.to_csv(summary_file + '-sam_rate.csv', index=False)
        df = pd.DataFrame(channels)
        df.to_csv(summary_file + '-channels.csv', index=False)
        df = pd.DataFrame(data_for_channels)
        df.to_csv(summary_file + '-data_for_channels.csv', index=False)
        df = pd.DataFrame(data)
        df.to_csv(summary_file + '-data.csv', index=False)

        print('==> Summary file completed!')

        return (np_samRate, np_channels, np_data_for_channels, np_data)

    ###
    ### Get start time and duration of seizure from specific seizures file
    ###
    def _get_seizure_start_time_and_duration(self, annotation_file):
        data = self._convert_data_from_specific_seizures_file(annotation_file)
        print(data[1])
        # print(data[38], " --- ", data[41], " --- ", data[49])
        st_time = int((str(bin(data[38]).replace('0b', '')) + str(bin(data[41]).replace('0b', ''))), 2)
        dur_time = data[49]

        print(str(st_time) + " --- " + str(dur_time))
        return (st_time, dur_time)


    ###################### Directory Manager ###################
    ###
    ### Traverse all files from a directory including its nested locations
    ###
    def _list_all_directories_and_files(self, directory):
        locations = [all_files for all_files in os.walk(directory) if all_files[0].find('/.') == -1]

        list_of_folders = []
        list_of_subfolders = []
        list_of_files = []

        for item in locations:
            list_of_folders.append(item[0])
            list_of_subfolders.append([x for x in item[1] if not x.startswith('.')])
            list_of_files.append([x for x in item[2] if not x.startswith('.')])

        # print(list_of_folders,'--',list_of_subfolders,'==',list_of_files)
        return (list_of_folders, list_of_subfolders, list_of_files)

        ###

    ### List all files to parse
    ###
    def _list_all_files_to_parse(self, root_directory):
        list_of_folders, list_of_subfolders, list_of_files = self._list_all_directories_and_files(root_directory)
        list_of_locations = []

        for loc in list_of_folders:
            files = list_of_files[list_of_folders.index(loc)]
            if len(files) > 0:
                # print(loc,'---',files)
                loc = loc.rstrip('/')
                list_of_locations.extend([(loc + '/' + file) for file in files])

        list_of_locations = [file.replace('\\', '/') for file in list_of_locations]

        self._all_files += list_of_locations
        self._data_files += [file for file in list_of_locations if
                            (file.endswith('.edf') or file.endswith('.seizures') or file.endswith('summary.txt'))]
        # self.parsed_data_files += [file for file in list_of_locations if (not file.endswith('.edf') and not file.endswith('.seizures') and not file.endswith('summary.txt'))]

        return (list_of_locations)

    ###
    ### Parse files accordingly
    ###
    def parse_CHBMIT_EEGData(self):
        root_directory = self.parsing_directory
        files = self._list_all_files_to_parse(root_directory)

        for file in files:
            #print(files.index(file),'-->: ', file)

            patient = ''
            record = ''
            file_name = (file.split('/'))[-1]

            ### For testing - to minimize the volume of data
            # if file.find('12')==-1:
            #     continue

            if file.endswith('.edf') and self._is_parse_edf_files:
                print('edf file: ', file)
                fileData = self._convert_specific_edf_to_csv(file)
                patient = file_name[3:5]
                record = file_name[6:8]

                if patient not in self._parsed_edf_data:
                    self._parsed_edf_data[patient] = []
                self._parsed_edf_data[patient].append(fileData)

            elif file.endswith('.seizures') and self._is_parse_seizure_files:
                print('seizures file: ', file)
                seizure_data = self._convert_data_from_specific_seizures_file(file)
                patient = file_name[3:5]
                record = file_name[6:8]

                if patient not in self._parsed_seizure_data:
                    self._parsed_seizure_data[patient] = []
                self._parsed_seizure_data[patient].append(seizure_data)

            elif file.endswith('summary.txt') and self._is_parse_summary_files:
                print('summary file: ', file)
                sample_rate, channels, data_for_channels, data = self._convert_summary_from_specific_summary_file(file)
                summary = [sample_rate, channels, data]
                patient = file_name[3:5]

                if patient not in self._parsed_summary_data:
                    self._parsed_summary_data[patient] = []
                self._parsed_summary_data[patient].append(summary)

            else:
                print('Other file: ', file)
                pass

        ##########################################################################################
        ### ??? It can't specify which data, seizure and summary is for which patient and record
        ### This is still left to do

        return (self._parsed_edf_data, self._parsed_seizure_data, self._parsed_summary_data)


#%%
########################################
# Testing...............
# directory = './../data/'
# parser = HumachLab_CHBMIT_EEGParser(directory, removable_dummy_channels=['-'])
# all_edf_data, all_seizure_data, all_summary_data = parser.parse_CHBMIT_EEGData()
# print('\nData: {} \nSeizure: {} \nSummary: {}\n'.format(all_edf_data, all_seizure_data, all_summary_data))

