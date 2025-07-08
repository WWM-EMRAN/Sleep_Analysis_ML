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



class HumachLab_Alfred_EEGManager:

    def __init__(self, logger, data_directory='./', metadata_path='./', removable_channel_list=[], is_remove_duplicate_channels=False):
        self.logger = logger
        self.logger.info(f'Processing ALFRED data...')

        self.data_directory = os.path.abspath(data_directory)
        if not (os.path.exists(self.data_directory)):
            self.logger.info(f'Raw directory does not exist!')
            self.data_directory = os.path.abspath('./')

        self.metadata_path = os.path.abspath(metadata_path)
        if not (os.path.exists(self.metadata_path)):
            self.logger.info(f'Raw metadata path does not exist!')
            self.metadata_path = os.path.abspath('./')

        self.data_directory = self.data_directory.replace('\\', '/')
        self.metadata_path = self.metadata_path.replace('\\', '/')

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

######################################          Alfred         ################################################

###############################################################################################################
    def get_record_info_for_patient(self, patient_no, record_no):
        self.logger.info(f'---> Getting ALFRED summary data Alfred...')
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
        recs = demographic_df['Number'].values.tolist()
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

        self.logger.info(f'---> Finished retrieving ALFRED summary data...')
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
        patient.removable_channels_.append( HumachLab_StaticMethods.get_removable_channels(pat_chns, self._removable_dummy_channels, self._is_remove_duplicate_channels) )
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

                val = ddf.loc[(ddf['Number']==rec), 'Segment Start Time'].values[0]
                seg_st = self._calculate_time_in_seconds_from_string2(val)
                val = ddf.loc[(ddf['Number']==rec), 'Segment End Time'].values[0]
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

                    val = rdf.loc[(rdf['Event No.']==ev_no), 'Event Start Time'].values[0]
                    ev_st = self._calculate_time_in_seconds_from_string2(val)
                    val = rdf.loc[(rdf['Event No.']==ev_no), 'Event End Time'].values[0]
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



