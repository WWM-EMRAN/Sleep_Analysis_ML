"""
File Name: HumachLab_DataManager.py
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 1:11 pm
"""

import os, sys
import pandas as pd
import numpy as np

from sklearn.utils import resample
import copy
import pickle
import json

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.DataManager.HumachLab_EEGManager import *
    from HumachLab.Utility.HumachLab_StaticMethods import *
elif sys_ind==2:
    from HumachLab import *
    # To register the DataStructure module in the system to be used by the pickle module
    import HumachLab.DataManager.HumachLab_EEGDataStucture as dm
    sys.modules['HumachLab_EEGDataStucture'] = dm
else:
    pass
### END: My modules ###



class HumachLab_DataManager:

    def __init__(self, logger):
        self.logger = logger

        self.serial_col = 'dp_serial'
        self.pat_id_col = 'pat_id'
        self.rec_id_col = 'rec_id'
        self.seg_id_col = 'seg_id'
        self.channel_cols = ['ch_id', 'ch_name']
        self.extra_cols =  [self.serial_col, self.pat_id_col, self.rec_id_col, self.seg_id_col]  + self.channel_cols
        self.class_name = 'seizureState'
        self.pred_col = 'prediction'
        return


    # ######################################
    # ### Retrieve metadata from dataset and post-process metadata
    # ######################################
    def get_metadata_of_the_dataset(self, dataset_name, raw_data_directory, raw_metadata_path, removable_channel_list, list_of_patients=[], list_of_lists_of_records=[[]], is_remove_duplicate_channels=True):
        self.manager_obj = HumachLab_EEGManager(self.logger, dataset_name, raw_data_directory, raw_metadata_path,
                                                removable_channel_list=removable_channel_list, is_remove_duplicate_channels=is_remove_duplicate_channels)
        self.logger.info(f'Metadata about the dataset is retrieved from: {self.manager_obj.data_directory}')

        if len(list_of_patients)==0:
            metadata = self.manager_obj.get_record_info_for_all_patient()
        else:
            metadata = self.manager_obj.get_records_info_for_patients(list_of_patients, list_of_lists_of_records)
        return metadata

    def filter_metadata_for_seizure_only_records(self, metadata):
        seizure_metadata = self.manager_obj.get_record_info_with_seizure_for_all_patient(metadata)
        return seizure_metadata

    def filter_metadata_for_specific_channels(self, metadata, channels):
        channels_metadata = self.manager_obj.get_record_info_for_specific_channels(metadata, channels)
        return channels_metadata

    def show_metadata_details(self, metadata):
        for p in metadata:
            self.logger.info(f'\n========================================\n Patient: {p.patient_no} \n========================================\n')
            self.logger.info(p.__dict__)
            for r in p.records:
                self.logger.info(f'\n----------------------------------------\n Record: {r.record_no} \n----------------------------------------\n')
                self.logger.info(f'{r.__dict__}')
        return


    # ######################################
    # ### Retrieve and save metadata from and to file
    # ######################################
    def save_metadata_info_to_file(self, metadata, pkl_path):
        self.save_dictionary_to_file(metadata, pkl_path, 'Dataset Metadata', dump_level=2)
        return


    def get_metadata_info_from_file(self, pkl_path, patient_numbers=[], seizure_only=False):
        PIK = pkl_path
        metadata = []
        with open(PIK, "rb") as f:
            for _ in range(pickle.load(f)):
                p = (pickle.load(f))

                if len(patient_numbers) > 0 and (p.patient_no not in patient_numbers):
                    continue
                p_cp = copy.deepcopy(p)

                p.num_records = 0
                p.record_numbers = []
                p.records = []

                for r in p_cp.records:
                    if (not seizure_only) or (seizure_only and r.record_no in p.seizure_record_numbers):
                        p.record_numbers.append(r.record_no)
                        p.records.append(r)

                p.num_records = len(p.record_numbers)

                metadata.append(p)
            self.logger.info(f'Metadata is retrieved from the file: {PIK}')
        return metadata


    def get_metadata_info_for_specific_channels(self, metadata, ch_list=[]):
        for p in metadata:
            for i in range(len(p.channels)):
                chn = (p.channels)[i]
                for key, val in chn.items():
                    if not (val in ch_list):
                        p.removable_channels_[i][key] = val
        return metadata


    def get_patient_and_records_for_specific_channels(self, metadata, ch_list=[]):
        for p in metadata:
            for i in range(len(p.channels)):
                chn = (p.channels)[i]
                for key, val in chn.items():
                    if not (val in ch_list):
                        p.removable_channels_[i][key] = val
        return metadata


    def get_metadata_info_from_file_all_channels(self, pkl_path, patient_numbers=[], seizure_only=False, channels_list=[], match_all_channel_group=False):
        PIK = pkl_path
        metadata = []
        with open(PIK, "rb") as f:
            for _ in range(pickle.load(f)):
                p = (pickle.load(f))

                if len(patient_numbers) > 0 and (p.patient_no not in patient_numbers):
                    continue
                p_cp = copy.deepcopy(p)

                p.num_records = 0
                p.record_numbers = []
                p.records = []

                # self.logger.info(f'=== {p.patient_no}')

                for r in p_cp.records:
                    if (not seizure_only) or (seizure_only and r.record_no in p.seizure_record_numbers):

                        if len(channels_list) > 0:
                            # self.logger.info(f'{r.record_no}')
                            chnss = []
                            chn_dic = p_cp.channels[r.channel_id]
                            rem_chn_dic = p_cp.removable_channels_[r.channel_id]
                            for key in chn_dic.keys():
                                if key not in rem_chn_dic.keys():
                                    chnss.append(chn_dic[key])

                            test_list1 = copy.deepcopy(channels_list)
                            test_list2 = copy.deepcopy(chnss)
                            # self.logger.info(f'-- {test_list1} \n== {test_list2}')
                            test_list1.sort()
                            test_list2.sort()
                            # tmplst = [item for item in test_list1 if (item in test_list2)]

                            # self.logger.info(f'{test_list1 == test_list2}\n-- {test_list1} \n== {test_list2}')
                            if match_all_channel_group:
                                if not (test_list1 == test_list2):
                                    continue
                            else:
                                # check = all((item in test_list2) for item in test_list1)

                                cnt = 0
                                mat = len(test_list1)
                                for c in test_list1:
                                    if c in test_list2:
                                        cnt += 1
                                check = (mat==cnt)

                                if not check:
                                    continue

                        p.record_numbers.append(r.record_no)
                        p.records.append(r)

                p.num_records = len(p.record_numbers)

                metadata.append(p)
            self.logger.info('Metadata is retrieved from the file...')
        return metadata


    def get_and_show_patients_and_corresponding_records(self, metadata, pat_list = [], show_patient_and_record_details=False):
        pats = []
        recs = []
        self.logger.info(f'#################################################################')
        if len(pat_list)>0:
            self.logger.info(f'Selected patients: {pat_list}')

        # show_patient_and_record_details = True

        for p in metadata:
            if show_patient_and_record_details:
                self.logger.info(f'{p.__dict__}')
            if len(pat_list)==0 or (len(pat_list)>0 and (p.patient_no in pat_list)):
                self.logger.info(f'{p.patient_no} === {p.record_numbers}')
                pats.append(p.patient_no)
                recs.append(p.record_numbers)

                #Extra record view
                if show_patient_and_record_details:
                    for r in p.records:
                        self.logger.info(f'{r.__dict__}')
                    self.logger.info(f'\n')
        return pats, recs


    # ######################################
    # ### Load and manage data/dataframe from file
    # ######################################
    def load_external_data(self, dat_file_name, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise):
    # def load_external_data(self, dat_file_name, channels=[], drop_nan=False, all_channels_columnwise=False):
        self.logger.info(f'Start loading external data file...')
        self.logger.info(f'External file:{dat_file_name}')
        print(all_channels_columnwise)
        df = pd.DataFrame()

        # './AllData/All_Data_Left_Hemisphere.csv' 'All_Data.csv'
        # all_data_csv = f'{self.all_data_directory}{dat_file_name}'
        all_data_csv = dat_file_name
        colwise_file_found = False

        if all_channels_columnwise:
            tmp_ind = dat_file_name.rfind('.')
            tmp_path = f'{dat_file_name[:tmp_ind]}_colwise{dat_file_name[tmp_ind:]}'
            print(tmp_path)
            if os.path.isfile(tmp_path):
                self.logger.info(f'New external file: {tmp_path}')
                all_data_csv = tmp_path
                colwise_file_found = True
            else:
                self.logger.info(f'Column-wise data file not found, loading the row-wise one...')

        if os.path.isfile(all_data_csv):
            df = pd.read_csv(all_data_csv)
            # df = pd.read_csv(all_data_csv, usecols=self.channels)
            self.logger.info(f'{df.shape}')
        else:
            self.logger.info(f'Data file not found..')

        df = df[df != np.inf]

        self.logger.info(f'Finish loading external data file...')
        return df, colwise_file_found


    def load_data_from_feature_directory(self, path, drop_nan=False, channels=[], patients=[], records=[]):
        self.logger.info(f'{path} {channels}')
        df = pd.DataFrame()
        path2 = os.walk(path)
        class_name = self.class_name

        for root, directories, files in path2:
            # for directory in directories:
            #     self.logger.info(f'{directory}')
            for file in files:
                #             self.logger.info(f'{file}')
                items = file.split('_')
                file_ext = (file.split('.'))[-1]
                items = file.split('_')

                if file_ext!='csv' or len(items)<5:
                    # self.logger.info(f'This is not the data file: {file}')
                    continue

                pat = int(items[1])
                rec = int(items[2])
                ch = (items[3].split('.'))[0]
                ch_name = (items[4].split('.'))[0]

                if len(patients) == 0 or (len(patients) > 0 and (pat in patients)):
                    iind = patients.index(pat)
                    recs = records[iind]
                    if len(records) == 0 or (len(records) > 0 and (rec in recs)):
                        if len(channels) == 0 or (len(channels) > 0 and (ch_name in channels)):
                            #                         self.logger.info(f'{file}')
                            #                         continue
                            file = path + f'/{file}'
                            extr_feat = pd.read_csv(file)

                            pArr = [pat for i in range(len(extr_feat.index))]
                            rArr = [rec for i in range(len(extr_feat.index))]
                            segArr = [i+1 for i in range(len(extr_feat.index))]
                            cArr = [ch for i in range(len(extr_feat.index))]
                            cnArr = [ch_name for i in range(len(extr_feat.index))]
                            dat_info = pd.DataFrame({self.pat_id_col: pArr, self.rec_id_col: rArr, self.seg_id_col: segArr, self.channel_cols[0]: cArr, self.channel_cols[1]: cnArr})

                            extr_feat = dat_info.join([extr_feat])

                            df = df.append(extr_feat, ignore_index=True)

        df = df[df != np.inf]
        return df


    def load_all_data_from_feature_directory(self, path, feature_subdirectories, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise):
        self.logger.info(f'Start loading data from feature files...')
        df = pd.DataFrame()
        path2 = os.walk(path)
        class_name = self.class_name
        ex_cols = self.extra_cols + [class_name]
        join_cols = [self.pat_id_col, self.rec_id_col, class_name]
        drop_cols = self.channel_cols

        for root, directories, files in path2:
            # self.logger.info(f'1== {directories}')
            selected_dirs = [sdir for sdir in directories if (feature_subdirectories) in sdir]
            # self.logger.info(f'2== {selected_dirs}')
            # self.logger.info(f'2== {selected_dirs} {channels}')
            # dir_com
            selected_dirs = [sdir for sdir in selected_dirs if (len(sdir.split('_'))>1) and ((sdir.split('_')[-1]) in channels)]
            # self.logger.info(f'3=={selected_dirs}')
            # Sorting directory based on channel
            selected_dirs = [d for c in channels for d in selected_dirs if c in d]
            directories = selected_dirs

            for directory in directories:
                self.logger.info(f'== {directory}')
                chn = directory.split('_')
                if len(chn)<2 or (not directory.startswith(feature_subdirectories)):
                    continue
                elif chn[-1] not in channels:
                    continue

                chn = chn[-1]
                directory = path + directory + '/'
                # self.logger.info(f'== {directory}')
                #             continue

                tdf = self.load_data_from_feature_directory(directory, drop_nan=drop_nan, channels=channels, patients=patients, records=records)
                df = pd.concat([df, tdf])
                self.logger.info(f'{tdf.shape} {df.shape}')

        # indx = [j for j in range(df.shape[0])]
        # srl_dat = pd.DataFrame(indx)
        # df.insert(0, self.serial_col, srl_dat)

        # if df.shape[0]>0:
        #     if replace_transitions:
        #         df = self.get_data_after_replacing_transitions_targets(df, replace_transitions)
        #     if all_channels_columnwise:
        #         df = self.create_columnwise_features_from_dataset(df, channels)
        self.logger.info(f'Finish loading data from feature files...')
        return df



    def filter_and_replace_data_in_dataframe(self, cdf, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise):
        # ###Data filling & sorting and row-wise or column-wise data rearrangement
        df = cdf.copy()
        # print('XXXXXX', df.shape, df.columns.values)
        if df.shape[0] > 0:
            # Filtering based on channels and patients
            # print('XXXXXX1111', df.shape, df.columns.values)
            if len(patients) > 0:
                df = self.get_data_for_specific_patients_and_records(df, patients, records)
            if len(channels) > 0:
                df = self.get_data_for_specific_channels(df, channels)
            # print('XXXXXX2222', df.shape, df.columns.values)

            # Do post-processing for row-wise or column-wise data rearrangement
            if all_channels_columnwise:
                df = self.create_columnwise_features_from_dataset(df, channels)
            # Data filling and sorting
            # df = self.fill_or_remove_nan_value_for_all_channels(df, patients, channels, drop_nan)
            df = self.fill_or_remove_nan_value_for_all_channels(df, drop_nan=drop_nan)
            # Replace if any positive value is found otherwise remove transition segments
            df = self.get_data_after_replacing_transitions_targets(df, replace_transitions)
            # if replace_transitions:
            #     self.dataset = self.dataManager.get_data_after_replacing_transitions_targets(self.dataset, replace_transitions)

            if self.serial_col not in df.columns.values.tolist():
                indx = [j for j in range(df.shape[0])]
                srl_dat = pd.DataFrame(indx)
                df.reset_index(drop=True, inplace=True)
                df.insert(0, self.serial_col, srl_dat)
                self.logger.info(f'Total: {len(indx)} serial found')
        else:
            self.logger.info(f'No data to post-process...')
        return df



    def get_data_after_replacing_transitions_targets(self, df, replace_transitions):
        sel_df = df.copy()
        print('YYYYYY', sel_df.shape, sel_df.columns.values)
        class_name = self.class_name
        if replace_transitions>=0:
            sel_df[class_name] = sel_df[class_name].replace([2, 3], replace_transitions)
        else:
            sel_df = sel_df[~sel_df[class_name].isin([2, 3])]
        return sel_df


    def fill_or_remove_nan_value_for_all_channels(self, cdf, patients=[], channels=[], drop_nan=False):
        self.logger.info(f'Dealing with NaN data...')
        df = cdf.copy()

        # Remove unnecessary patients and channels
        if len(patients)>0:
            df = df[df[self.pat_id_col].isin(patients)]
        if len(channels)>0:
            df = df[df[self.channel_cols[1]].isin(channels)]

        # Data filling and sorting
        df = df[df != np.inf]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        r, _ = np.where(df.isna())
        rows_with_nan = list(r)
        self.logger.info(f'{len(rows_with_nan)} rows with NaN at indices: {rows_with_nan}')

        # fill or drop
        if not drop_nan:
            df = df.fillna(0)
        else:
            # df = df.dropna()
            data_with_nan = df.iloc[rows_with_nan]
            rsrl = list(data_with_nan[self.serial_col].values)
            rpid = list(data_with_nan[self.pat_id_col].values)
            rrid = list(data_with_nan[self.rec_id_col].values)
            rsid = list(data_with_nan[self.seg_id_col].values)

            rem_indx = []
            for srl, p, r, s in zip(rsrl, rpid, rrid, rsid):
                ii = list(df[((df[self.serial_col] == srl) & (df[self.pat_id_col] == p) & (df[self.rec_id_col] == r) & (df[self.seg_id_col] == s))].index)
                rem_indx += ii

            self.logger.info(f'Row indices to remove: {rem_indx}')
            df = df.drop(rem_indx)
        df.sort_index()

        # Convert data type of the column
        all_cols = df.columns
        d_keys = [self.serial_col, self.pat_id_col, self.rec_id_col, self.seg_id_col, self.class_name]
        convert_dict = dict()
        for k in d_keys:
            # convert_dict[k] = int
            if k in all_cols:
                convert_dict[k] = int

        if self.channel_cols[0] in all_cols:
            convert_dict[self.channel_cols[0]] = int
            convert_dict[self.channel_cols[1]] = str

        df = df.astype(convert_dict)

        return df


    def create_columnwise_features_from_dataset(self, cdf, channels=[]):
        self.logger.info(f'Converting data column wise...')
        df = cdf.copy()
        tmpdf = pd.DataFrame()
        if len(channels)==0:
            # channels = df.columns.values.tolist()
            channels = df[self.channel_cols[1]].unique().tolist()
        extra_col_start = 0 if (self.serial_col in df.columns.values.tolist()) else 1
        ex_cols_with_target = self.extra_cols[extra_col_start:] + [self.class_name]
        # ex_cols_with_target = self.extra_cols + [self.class_name]
        join_cols = [self.pat_id_col, self.rec_id_col, self.class_name]
        drop_cols = self.channel_cols

        for ii, chn in enumerate(channels):
            self.logger.info(f'Columnizing channel: {ii} {chn}')
            tdf = df[df[self.channel_cols[1]] == chn]

            tcols = [c for c in tdf.columns if (c not in ex_cols_with_target)]
            tmp_cols = [(f'{c}_{chn}') for c in tcols]
            tmp_cols = ex_cols_with_target + tmp_cols
            # tmp_cols
            # print(f'HHHHH: {tdf.columns.values.tolist()}, {tmp_cols}')
            tdf.columns = tmp_cols
            tdf = tdf.reset_index(drop=True)

            self.logger.info(f'{tmpdf.shape} {tdf.shape}')
            if tmpdf.empty:
                tdf.drop(drop_cols, axis=1, inplace=True)
                tmpdf = pd.concat([tmpdf, tdf])
            else:
                tdf.drop(ex_cols_with_target, axis=1, inplace=True)
                tmpdf = pd.merge(tmpdf, tdf, left_index=True, right_index=True)
                # tmpdf = pd.merge(tmpdf, tdf, left_index=True, right_index=True)
                # tmpdf = pd.merge(tmpdf, tdf, how='inner', left_on=join_cols, right_on=join_cols)
                # tmpdf.drop(ex_cols, axis=1, inplace=True)
            self.logger.info(f'{tmpdf.shape} {tdf.shape}')

        df = tmpdf
        return df


    # ######################################
    # ### Other dataframe operations
    # ######################################

    def show_class_distribution(self, df):
        class_distribution = df[self.class_name].value_counts().to_dict()
        self.logger.info(f'Class distribution: {class_distribution}')

        ns = class_distribution[0]
        s = class_distribution[1]

        nsp = round(((ns/(ns+s))*100), 2)
        sp = round(((s/(ns+s))*100), 2)

        self.logger.info(f'Total non-sizure segments: {ns} and seizure segments: {s}')
        self.logger.info(f'Non-sizure and seizure ratio: {nsp}:{sp}')
        return class_distribution

    def seperate_seizure_and_nonseizure_data(self, df):
        class_name = self.class_name
        sdf = pd.DataFrame()
        nsdf = pd.DataFrame()

        sdf = df[df[class_name]>0]
        nsdf = df[df[class_name]==0]

        return sdf, nsdf

    def up_or_down_sample_data(self, df, sampling_crieteria=0): #sampling_crieteria: 0-no sampling, 1-down sampling, 2-up sampling
        cdf = df.copy()

        if sampling_crieteria>0:
            class_name = self.class_name
            df_nsz = df[df[class_name] == 0]
            df_sz = df[df[class_name] == 1]

            df_majority = df_nsz
            df_minority = df_sz

            if df_nsz.shape[0]<df_sz.shape[0]:
                df_majority = df_sz
                df_minority = df_nsz

            number_of_datapoints = (len(df_minority.index) if sampling_crieteria==1 else len(df_majority.index))

            df_downsampled = resample((df_majority if sampling_crieteria==1 else df_minority),
                                               replace=False,  # sample without replacement
                                               n_samples=number_of_datapoints,  # to match minority class
                                               random_state=123)  # reproducible results

            # Combine minority class with downsampled majority class
            cdf = pd.concat([df_downsampled, (df_minority.index if sampling_crieteria==1 else df_majority.index)])

        return cdf



    # ######################################
    # ### Other dataframe management process
    # ######################################

    def get_file_naming_detail_for_dataset(self, fdf):
        file_naming_detail_for_dataset = f''
        cprf = []

        c = fdf[self.channel_cols[1]].unique()
        cprf.append(len(c))
        p = fdf[self.pat_id_col].unique()
        cprf.append(len(p))
        r = 0  # fdf.loc[fdf['pat_id'],self.rec_id_col]#fdf[self.rec_id_col].unique()
        for i in p:
            r += len(fdf.loc[fdf['pat_id'] == i, 'rec_id'].unique())
        cprf.append(r)
        cl = fdf.columns.values.tolist()
        cli = cl.index(self.class_name)
        f = cl[cli+1:]
        cprf.append(len(f))

        file_naming_detail_for_dataset = f'ch{cprf[0]}_pat{cprf[1]}_rec{cprf[2]}_ft{cprf[3]}'

        return file_naming_detail_for_dataset, cprf


    def save_dataframe_to_file(self, sdf, pkl_path, data_desc, force_save=False):
        self.logger.info(f'Start saving dataframe to file...')
        PIK = pkl_path
        msg_str = ''

        if os.path.exists(PIK) and (not force_save):
            msg = f'{data_desc} file already exist: {PIK}. Do you want to overwrite (y/n)? '
            ans = input(msg)
            msg += f'{ans}\n'
            msg_str += msg
            if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                msg = f'This will delete all the contents of the {data_desc} file, Are you sure (y/n)? '
                ans = input(msg)
                msg += f'{ans}\n'
                msg_str += msg
                if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                    try:
                        os.remove(PIK)
                        msg_str += f'{data_desc} file removed successfully: {PIK}\n'
                    except:
                        msg_str += f'Can not remove {data_desc} file: {PIK}\n'
                else:
                    msg_str += f'Working with existing {data_desc} file: {PIK}\n'
            else:
                msg_str += f'Working with existing {data_desc} file: {PIK}\n'

        try:
            sdf.to_csv(PIK, index=False)
        except:
            msg_str += f'Problem creating {data_desc} file: {PIK}\n'

        self.logger.info(msg_str)
        self.logger.info(f'Finish saving dataframe to file...')
        return


    def save_dictionary_to_file(self, dictdata, pkl_path, data_desc, dump_level=0, force_save=False):
        self.logger.info(f'Start saving dictionary to file...')
        PIK = pkl_path
        msg_str = ''

        if os.path.exists(PIK) and (not force_save):
            msg = f'{data_desc} file already exist: {PIK}. Do you want to overwrite (y/n)? '
            ans = input(msg)
            msg += f'{ans}\n'
            msg_str += msg
            if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                msg = f'This will delete all the contents of the {data_desc} file, Are you sure (y/n)? '
                ans = input(msg)
                msg += f'{ans}\n'
                msg_str += msg
                if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                    try:
                        os.remove(PIK)
                        msg_str += f'{data_desc} file removed successfully: {PIK}\n'
                    except:
                        msg_str += f'Can not remove {data_desc} file: {PIK}\n'
            else:
                msg_str += f'Working with existing {data_desc} file: {PIK}\n'

        try:
            fo_mode = 'wb' if dump_level>0 else 'w'
            with open(PIK, fo_mode) as f:
                if dump_level==0:
                    f.write(json.dumps(dictdata))
                elif dump_level==1:
                    pickle.dump(dictdata, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(len(dictdata), f)
                    for d in dictdata:
                        pickle.dump(d, f)
                msg_str += f'{data_desc} is written to the file: {PIK}\n'
        except:
            msg_str += f'Problem creating {data_desc} file: {PIK}\n'

        self.logger.info(msg_str)
        self.logger.info(f'Finish saving dictionary to file...')
        return


    def load_previously_saved_results(self, log_load_paths, exp_srl_no):
        path = os.walk(log_load_paths)
        file_name_list = ['all_models', 'bst_models', 'tr_score', 'ts_score', 'bst_score', 'metadata', 'bst_prediction_data', 'splitter']
        models, bst_models, tr_scores, ts_scores, bst_scores, metadata, prdiction_data, splitter = [], [], [], [], [], [], [], []

        for root, directories, files in path:
            for file in files:
                self.logger.info(file)
                if not (file.startswith(tuple(file_name_list))):
                    print('skipping....')
                    continue

                ff = file.split('.')
                load_list_ind = ff[0]
                file_type = ff[-1]
                ff = load_list_ind.split('_')
                load_list_ind = ff[-1]
                load_list_ind = int(load_list_ind)
                # print(load_list_ind, load_list, file_type)

                if (load_list_ind == exp_srl_no) and (file.startswith(tuple(file_name_list))):
                    self.logger.info(f'>> {file}')
                    #                 self.logger.info(f'\n\n------------------------------------------------------------------\n{file}\n------------------------------------------------------------------\n')
                    file2 = log_load_paths + file
                    #                 self.logger.info(file_type)

                    if file_type == 'dat':
                        # % Retrieving Patient object previously stored using pickle
                        retrieved_data = []
                        with open(file2, "rb") as f:
                            for _ in range(pickle.load(f)):
                                retrieved_data.append(pickle.load(f))

                        if file.startswith('all_models'):
                            models = retrieved_data
                            self.logger.info(f'Models are retrieved from the file...')
                        elif file.startswith('bst_models'):
                            bst_models = retrieved_data
                            self.logger.info(f'Best models are retrieved from the file...')

                    elif file_type == 'sdat':
                        # % Retrieving Patient object previously stored using pickle
                        splttr = []
                        with open(file2, "rb") as f:
                            splttr.append(pickle.load(f))

                        splitter = splttr
                        self.logger.info(f'Data splitter is retrieved from the file...')

                    elif file_type == 'txt':
                        # ### Retrieving dictionary using pickle
                        with open(file2, 'rb') as f:
                            dict_data = pickle.load(f)

                            if file.startswith('metadata'):
                                metadata = dict_data
                                self.logger.info(f'Metadata is read from the file...')
                            elif file.startswith('tr_score'):
                                tr_scores = dict_data
                                self.logger.info(f'Training scores are retrieved from the file...')
                            elif file.startswith('ts_score'):
                                ts_scores = dict_data
                                self.logger.info(f'Test scores are retrieved from the file...')
                            elif file.startswith('bst_score'):
                                bst_scores = dict_data
                                self.logger.info(f'Best models test scores are retrieved from the file...')

                    elif file_type == 'csv':
                        df = pd.read_csv(file2)

                        if file.startswith('bst_prediction_data'):
                            prdiction_data = df
                            self.logger.info(f'Prediction data is read from the file...')

        return models, bst_models, tr_scores, ts_scores, bst_scores, metadata, prdiction_data, splitter


    def load_all_previously_saved_results(self, log_load_paths, exp_list):
        models, bst_models, tr_scores, ts_scores, bst_scores, metadata, prdiction_data, splitter = [], [], [], [], [], [], [], []
        print(log_load_paths, exp_list)
        for i in range(len(exp_list)):
            exp = exp_list[i]
            self.logger.info(
                '\n==================================================================\n' + 'DATA FOR EXPERIMENT: ' + str(
                    exp) + '\n==================================================================')

            log_path = f'{log_load_paths}Experiment_Classifier_{exp:02}/'

            mod, bst_mod, tr_scr, ts_scr, bst_scr, met, prd, splt = self.load_previously_saved_results(log_path, exp)

            models.append(mod)
            bst_models.append(bst_mod)
            tr_scores.append(tr_scr)
            ts_scores.append(ts_scr)
            bst_scores.append(bst_scr)
            metadata.append(met)
            prdiction_data.append(prd)
            splitter.append(splt)

        return models, bst_models, tr_scores, ts_scores, bst_scores, metadata, prdiction_data, splitter


    # ### Other file management process
    # ######################################
    def save_portion_of_data_from_a_file_to_another_file(self, from_file, to_file, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise):
        df = self.load_external_data(from_file, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise)
        df = self.filter_and_replace_data_in_dataframe(df, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise)
        self.save_dataframe_to_file(df, to_file, 'Partial data')
        return


    # ### Other dataframe management process
    # ######################################

    def get_shuffled_data(self, df):
        clf_df = df.sample(frac=1)
        return clf_df

    def get_randomized_data(self, df):
        return self.get_shuffled_data(df)


    def get_data_for_specific_channels(self, df, sel_channels):
        sel_df = df.copy()
        chn_name_col = self.channel_cols[1]
        sel_cols = sel_df.columns.values.tolist()
        print(chn_name_col, chn_name_col in sel_cols)

        if chn_name_col in sel_cols:
            print('Chan in the row')
            sel_df = df[df[chn_name_col].isin(sel_channels)]
        else:
            feat_cols = sel_cols[(sel_cols.index(self.class_name)+1):]
            print('Chan in the column', len(feat_cols))
            cols_to_drop = []
            for ccol in feat_cols:
                # print(ccol)
                if ccol.split('_')[-1] not in sel_channels:
                    cols_to_drop.append(ccol)

            print('Columns to drop', len(cols_to_drop))
            sel_df.drop(cols_to_drop, axis=1, inplace=True)
        return sel_df


    def get_data_for_specific_patients_and_records(self, df, sel_patients, sel_records):
        sel_df = pd.DataFrame()
        # print('TTTTTTT ', len(sel_patients))
        if len(sel_patients)>0:
            for pat, recs in zip(sel_patients, sel_records):
                # print('TTTTTTT222 ', pat, len(recs))
                tmp_df = None
                if len(recs)>0:
                    tmp_df = df[(df[self.pat_id_col]==pat) & (df[self.rec_id_col].isin(recs))]
                else:
                    tmp_df = df[(df[self.pat_id_col] == pat)]
                sel_df = pd.concat([sel_df, tmp_df])
        else:
            sel_df = df
        return sel_df


    def get_data_for_specific_channels_and_features_set(self, df, chns_with_cols, make_random_data=False):
        # Copy dataset and set new serial number for tracking
        # chns_with_cols = [('A1', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('A2', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('C4', ('numberOfZeroCrossing',)), ('O2', ('numberOfZeroCrossing',)),
        #                   ('Cz', ('numberOfZeroCrossing',))]

        dat_cols = df.columns.values.tolist()
        ch_name_col = self.channel_cols[1]
        is_data_colwise = False
        if ch_name_col not in dat_cols:
            is_data_colwise = True

        clf_df = pd.DataFrame()
        acm_df = df.copy()
        sel_cols = []

        for i, (ch, feats) in enumerate(chns_with_cols):
            tmp_df = pd.DataFrame()
            if not is_data_colwise:
                tmp_df = acm_df[acm_df[ch_name_col]==ch]
                sel_cols.extend(list(feats))
            else:
                if i==0:
                    clf_df = acm_df[self.extra_cols]

                for ft in feats:
                    col_to_add = f'{ch}_{ft}'
                    self.logger.info(col_to_add)
                    tmp_df = acm_df[f'{ft}_{ch}']

            clf_df = pd.concat([clf_df, tmp_df], axis=(1 if is_data_colwise else 0))

        if not is_data_colwise:
            sel_cols = list(set(sel_cols))
            ex_cols_with_target = self.extra_cols + [self.class_name]
            sel_cols = ex_cols_with_target + sel_cols
            clf_df = clf_df[sel_cols]

        self.logger.info(f'{clf_df.shape}, {clf_df.columns}')

        clf_df = clf_df.reset_index(drop=True)

        indx = [j for j in range(clf_df.shape[0])]
        srl_dat = pd.DataFrame(indx)
        clf_df.insert(0, self.serial_col, srl_dat)

        # Shuffle data randomly
        if make_random_data:
            clf_df = self.get_shuffled_data(clf_df)
        return clf_df



