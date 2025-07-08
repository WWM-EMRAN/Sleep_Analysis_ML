"""
File Name: HumachLab_DataManager.py
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 1:11 pm
"""

import os
import pandas as pd
import numpy as np

from sklearn.utils import resample
import copy
import pickle
import json
from HumachLab import *
from HumachLab_EEGDataStucture import Patient, Record
from HumachLab_EEGManager import *


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
        return


    # ######################################
    # ### Retrieve metadata from dataset and post-process metadata
    # ######################################
    def get_metadata_of_the_dataset(self, raw_data_directory, raw_metadata_path, removable_channel_list, list_of_patients=[], list_of_lists_of_records=[[]], is_remove_duplicate_channels=True):
        self.manager_obj = HumachLab_EEGManager(self.logger, raw_data_directory, raw_metadata_path,
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
        self.save_dictionary_to_file(self, metadata, pkl_path, 'Dataset Metadata', dump_level=2)
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
    def load_external_data(self, dat_file_name, channels=[], drop_nan=False, all_channels_columnwise=False):
        df = pd.DataFrame()

        # './AllData/All_Data_Left_Hemisphere.csv' 'All_Data.csv'
        # all_data_csv = f'{self.all_data_directory}{dat_file_name}'
        all_data_csv = dat_file_name

        if os.path.isfile(all_data_csv):
            df = pd.read_csv(all_data_csv)
            # df = pd.read_csv(all_data_csv, usecols=self.channels)
            self.logger.info(f'{df.shape}')
        else:
            self.logger.info(f'Data file not found..')

        if len(channels)>0:
            df = df[channels]
        return df


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

        indx = [j for j in range(df.shape[0])]
        srl_dat = pd.DataFrame(indx)
        df.insert(0, self.serial_col, srl_dat)
        return df


    def fill_or_remove_nan_value_for_all_channels(self, cdf, patients, channels, drop_nan):
        self.logger.info(f'Dealing with NaN data...')
        df = cdf.copy()

        # Remove unnecessary patients and channels
        if len(patients)>0:
            df = df[df[self.pat_id_col].isin(patients)]
        if len(channels)>0:
            df = df[df[self.channel_cols[1]].isin(channels)]

        # Data filling and sorting
        df = df[df != np.inf]
        # fill or drop
        if not drop_nan:
            df = df.fillna(0)
        else:
            # df = df.dropna()
            r, _ = np.where(df.isna())
            rows_with_nan = list(r)
            self.logger.info(f'Rows with NaN: {rows_with_nan}')

            data_with_nan = df.iloc[rows_with_nan]

            rpid = list(data_with_nan[self.pat_id_col].values)
            rrid = list(data_with_nan[self.rec_id_col].values)
            rsid = list(data_with_nan[self.seg_id_col].values)

            rem_indx = []
            for p, r, s in zip(rpid, rrid, rsid):
                ii = list(
                    df[((df[self.pat_id_col] == p) & (df[self.rec_id_col] == r) & (df[self.seg_id_col] == s))].index)
                rem_indx += ii

            self.logger.info(f'Row indices to remove: {rem_indx}')
            df = df.drop(rem_indx)
        df.sort_index()

        # Convert data type of the column
        all_cols = df.columns
        convert_dict = {self.pat_id_col: int,
                        self.rec_id_col: int,
                        self.seg_id_col: int,
                        self.class_name: int
                        }
        if self.channel_cols[0] in all_cols:
            convert_dict[self.channel_cols[0]] = int
            convert_dict[self.channel_cols[1]] = str

        df = df.astype(convert_dict)

        return df


    def create_columnwise_features_from_dataset(self, cdf, channels):
        self.logger.info(f'Converting data column wise...')
        df = cdf.copy()
        tmpdf = pd.DataFrame()
        ex_cols = self.extra_cols + [self.class_name]
        join_cols = [self.pat_id_col, self.rec_id_col, self.class_name]
        drop_cols = self.channel_cols

        for chn in channels:
            self.logger.info(f'Columnizing channel: {chn}')
            tdf = df[df[self.channel_cols[1]] == chn]

            tcols = [c for c in tdf.columns if (c not in ex_cols)]
            tmp_cols = [(f'{c}_{chn}') for c in tcols]
            tmp_cols = ex_cols + tmp_cols
            tdf.columns = tmp_cols
            tdf = tdf.reset_index(drop=True)

            self.logger.info(f'{tmpdf.shape} {tdf.shape}')
            if tmpdf.empty:
                tdf.drop(drop_cols, axis=1, inplace=True)
                tmpdf = pd.concat([tmpdf, tdf])
            else:
                tdf.drop(ex_cols, axis=1, inplace=True)
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
        print(class_distribution)

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

    def shuffle_dataset(self, df):
        # Shuffling the dataframe
        df = df.sample(frac=1)
        return df

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
        # print(fdf.shape, fdf.columns.values)
        file_naming_detail_for_dataset = f''
        cprf = []

        c = fdf[self.channel_cols[1]].unique()
        cprf.append(len(c))
        p = fdf[self.pat_id_col].unique()
        cprf.append(len(p))
        r = fdf[self.rec_id_col].unique()
        cprf.append(len(r))
        cl = fdf.columns.values.tolist()
        cli = cl.index(self.class_name)
        f = cl[cli+1:]
        cprf.append(len(f))

        file_naming_detail_for_dataset = f'ch{cprf[0]}_pat{cprf[1]}_rec{cprf[2]}_ft{cprf[3]}'

        return file_naming_detail_for_dataset, cprf


    def save_dataframe_to_file(self, sdf, pkl_path):
        PIK = pkl_path
        msg_str = ''

        if os.path.exists(PIK):
            msg = f'Dataframe file already exist: {PIK}. Do you want to overwrite (y/n)? '
            ans = input(msg)
            msg += f'{ans}\n'
            msg_str += msg
            if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                msg = f'This will delete all the contents of the dataframe file, Are you sure (y/n)? '
                ans = input(msg)
                msg += f'{ans}\n'
                msg_str += msg
                if ans == 'Yes' or ans == 'yes' or ans == 'Y' or ans == 'y':
                    try:
                        os.remove(PIK)
                        msg_str += f'Dataframe file removed successfully: {PIK}\n'
                    except:
                        msg_str += f'Can not remove dataframe file: {PIK}\n'
            else:
                msg_str += f'Working with existing dataframe file: {PIK}\n'

        try:
            sdf.to_csv(PIK, index=False)
        except:
            msg_str += f'Problem creating metadata file: {PIK}\n'

        self.logger.info(msg_str)
        return


    def save_dictionary_to_file(self, dictdata, pkl_path, data_desc, dump_level=0):
        PIK = pkl_path
        msg_str = ''

        if os.path.exists(PIK):
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
        return


    # ### Other dataframe management process
    # ######################################

    def get_shuffled_data(self, df):
        clf_df = df.sample(frac=1)
        return clf_df

    def get_data_for_specific_channels_and_features_set(self, df, chns_with_cols, make_random_data=False):
        # Copy dataset and set new serial number for tracking
        # chns_with_cols = [('A1', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('A2', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('C4', ('numberOfZeroCrossing',)), ('O2', ('numberOfZeroCrossing',)),
        #                   ('Cz', ('numberOfZeroCrossing',))]

        dat_cols = df.columns.values.tolist()
        is_data_colwise = False
        if self.channel_cols[1] not in dat_cols:
            is_data_colwise = True

        clf_df = pd.DataFrame()
        acm_df = df.copy()
        sel_cols = []

        for i, (ch, feats) in enumerate(chns_with_cols):
            if not is_data_colwise:
                acm_df = (acm_df[acm_df[self.channel_cols[1]==ch]])
                sel_cols.extend(list(feats))
                continue

            for ft in feats:
                self.logger.info(f'{ch}_{ft}')
                if clf_df.shape[0] == 0:
                    tmp_df = (acm_df[(acm_df[self.channel_cols[1]] == ch)])[self.extra_cols + [ft]]
                else:
                    tmp_df = (acm_df[(acm_df[self.channel_cols[1]] == ch)])[[ft]]
                # tmp_df.columns = [f'{ch}_{ft}', ]
                tmp_df.rename(columns={ft: f'{ch}_{ft}'}, inplace=True)
                tmp_df = tmp_df.reset_index(drop=True)
                clf_df = pd.concat([clf_df, tmp_df], axis=1)

        if not is_data_colwise:
            sel_cols = list(set(sel_cols))
            sel_cols = self.extra_cols + sel_cols
            acm_df = acm_df[sel_cols]
        self.logger.info(f'{clf_df.shape}, {clf_df.columns}')

        clf_df = clf_df.reset_index(drop=True)

        indx = [j for j in range(clf_df.shape[0])]
        srl_dat = pd.DataFrame(indx)
        clf_df.insert(0, self.serial_col, srl_dat)

        # Shuffle data randomly
        if make_random_data:
            clf_df = self.get_shuffled_data(clf_df)
        return clf_df

    def get_data_for_specific_channels_and_features(self, df, chns_with_cols, make_random_data=False):
        # Copy dataset and set new serial number for tracking
        # chns_with_cols = [('A1', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('A2', ('variance', 'meanAbsoluteValue', 'fd_maximum', 'fd_mean', 'fd_variance')),
        #                   ('C4', ('numberOfZeroCrossing',)), ('O2', ('numberOfZeroCrossing',)),
        #                   ('Cz', ('numberOfZeroCrossing',))]

        clf_df = pd.DataFrame()
        acm_df = df.copy()
        for (ch, feats) in chns_with_cols:
            for ft in feats:
                print(f'{ch}_{ft}')
                if clf_df.shape[0] == 0:
                    tmp_df = (acm_df[(acm_df[self.channel_cols[1]] == ch)])[self.extra_cols + [ft]]
                else:
                    tmp_df = (acm_df[(acm_df[self.channel_cols[1]] == ch)])[[ft]]
                # tmp_df.columns = [f'{ch}_{ft}', ]
                tmp_df.rename(columns={ft: f'{ch}_{ft}'}, inplace=True)
                tmp_df = tmp_df.reset_index(drop=True)
                clf_df = pd.concat([clf_df, tmp_df], axis=1)

        self.logger.info(f'{clf_df.shape}, {clf_df.columns}')

        clf_df = clf_df.reset_index(drop=True)

        indx = [i for i in range(clf_df.shape[0])]
        srl_dat = pd.DataFrame(indx)
        clf_df.insert(0, self.serial_col, srl_dat)

        # Shuffle data randomly
        if make_random_data:
            clf_df = self.get_shuffled_data(clf_df)
        return clf_df


    def get_random_data_for_channel(self, df, chns, cols2, col_wise=False, ):
        #Copy dataset and set new serial number for tracking
        clf_df = df.copy()
        final_cols2 = []
        if not col_wise:
            if len(chns)>0:
                clf_df = clf_df[clf_df[self.channel_cols[1]].isin(chns)]
            final_cols2 = cols2.copy()
        else:
            final_cols2 = df.columns.values.tolist()
            # new_cols2 = df.columns.values.tolist()
            # for c2 in cols2:
            #     final_cols2.extend( [c for c in new_cols2 if c2 in c] )

        clf_df = clf_df[final_cols2]
        clf_df = clf_df.reset_index(drop=True)

        indx = [i for i in range(clf_df.shape[0])]
        srl_dat = pd.DataFrame(indx)
        clf_df.insert (0, self.serial_col, srl_dat)

        #Shuffle data randomly
        clf_df = clf_df.sample(frac = 1)
        return clf_df


