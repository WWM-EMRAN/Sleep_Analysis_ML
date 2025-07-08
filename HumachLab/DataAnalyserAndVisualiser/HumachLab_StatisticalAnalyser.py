"""
File Name: HumachLab_StatisticalAnalyser.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 12:53 pm
"""

import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib

# # matplotlib.rc('figure', figsize=(16, 12))
# # matplotlib.rc('font', size=28)
# matplotlib.rc('figure', figsize=(20, 15))
# matplotlib.rc('font', size=32)
# # matplotlib.rc('axes', titlesize=28)
# matplotlib.rc('axes', facecolor='white')
# matplotlib.rc('axes.spines', top=False, right=False)
# plt.style.use('seaborn-whitegrid')


# if using a Jupyter notebook, include:
# %matplotlib inline
from sklearn.utils import resample
import pickle
import json
from HumachLab import *
# from HumachLab.DataManager.HumachLab_EEGDataStucture import Patient, Record
import copy
#     from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, roc_auc_score

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.DataManager.HumachLab_DataManager import *
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###


class HumachLab_StatisticalAnalyser:

    def __init__(self, logger):
        self.logger = logger

        self.pat_id_col = 'pat_id'
        self.rec_id_col = 'rec_id'
        self.seg_id_col = 'seg_id'
        self.channel_cols = ['ch_id', 'ch_name']
        self.extra_cols =  [self.pat_id_col, self.rec_id_col, self.seg_id_col]  + self.channel_cols
        self.class_name = 'seizureState'

        self.dataManager = HumachLab_DataManager(logger)
        return


    # ######################################
    # ### P value and AUC calculations
    # ######################################

    #%% General p and auc value calculation
    # ## p-value calculatoin
    def get_pValues_for_dataframe(self, df, sort_values=False, replace_data=1):
        class_name = self.class_name
        #     print('COLS: ', df.columns)
        y = df[class_name]
        X = df.drop(columns=[class_name])
        columns = X.columns.tolist()
        #     scores, pval_lst = chi2(X, y)
        scores, pval_lst = f_classif(X, y)
        pval_lst = list(pval_lst)
        pvalues = pd.DataFrame({'features': columns, 'pvalue': pval_lst})

        if replace_data == 1:
            pvalues['pvalue'] = pvalues['pvalue'].fillna(1)

        if sort_values:
            pvalues.sort_values(by=['pvalue'], ascending=True, inplace=True)

        return pvalues

    # ## auc-value calculation
    def get_aucValues_for_dataframe(self, df, sort_values=False, replace_data=1):
        class_name = self.class_name
        y = df[class_name].values
        X = df.drop(columns=[class_name])
        columns = X.columns.tolist()
        auc_lst = []
        auc_rel_lst = []

        for c in columns:
            xx = X[c].values
            val = roc_auc_score(y, xx)
            auc_lst.append(val)
            val = 0.5 + abs(val - 0.5)
            auc_rel_lst.append(val)

        aucValues = pd.DataFrame({'features': columns, 'AUC': auc_lst, 'relativeAUC': auc_rel_lst})

        if replace_data == 1:
            aucValues['AUC'] = aucValues['AUC'].fillna(0.50)
            aucValues['relativeAUC'] = aucValues['relativeAUC'].fillna(0.50)

        if sort_values:
            aucValues.sort_values(by=['relativeAUC'], ascending=False, inplace=True)

        return aucValues

    # ## p and auc-value calculation
    def get_pValue_and_aucValues_for_dataframe(self, df, sort_values=False, replace_data=1):
        df_p_and_auc_Vals = pd.DataFrame()
        
        df_pVals = self.get_pValues_for_dataframe(df)
        df_aucVals = self.get_pValues_for_dataframe(df)

        # df_p_and_auc_Vals = pd.concat([df_pVals, df_aucVals], axis=1)
        df_p_and_auc_Vals = pd.merge(df_pVals, df_aucVals, on="features")

        return df_p_and_auc_Vals


    #%% Special p and auc value calculation:

    # ##patient and feature wise p and auc values on overlall data regardless of channels
    def get_all_patient_and_feature_wise_pvalue_and_auc(self, df, patients, cols, class_n=''):
        class_name = self.class_name if len(class_n)==0 else class_n
        res_data = pd.DataFrame()
        patientWise_pAuc_overallDat = pd.DataFrame()

        #     for i in range(1, len(patients)+1):
        for pat in patients:
            cdf = df[df[self.pat_id_col] == pat]
            scdf = cdf[cols[(cols.index(class_name)):]]  # .reindex(sc1df.index)
            arr = np.array(scdf)
            arr2 = arr  # min_max_normalizer(arr)
            result = pd.DataFrame(arr2, index=(scdf.index), columns=cols[(cols.index(class_name)):])
            res_data = pd.concat([res_data, result], axis=0)

            p = 1
            auc = 0.5
            if result.shape[0] > 0:
                p = self.get_pValues_for_dataframe(result)
                auc = self.get_pValues_for_dataframe(result)
            else:
                continue
            pAuc = pd.merge(p, auc, on="features")

            cc = [pat for j in range(pAuc.shape[0])]
            dd = {'patient': cc}
            pat_no = pd.DataFrame(dd)
            com_df = pd.concat([pat_no, pAuc], axis=1)
            patientWise_pAuc_overallDat = pd.concat([patientWise_pAuc_overallDat, com_df])

        ppval = self.get_pValues_for_dataframe(res_data)
        pauc = self.get_pValues_for_dataframe(res_data)

        patientWise_pAuc_overallDat.reset_index(drop=True, inplace=True)
        # Feature-wise p_auc for overall data
        featWise_pAuc_overallDat = pd.merge(ppval, pauc, on="features")
        # ppval_pauc = pd.concat([com_vals, com_df])
        return featWise_pAuc_overallDat, patientWise_pAuc_overallDat


    # ##patient and feature wise p and auc values for individual channels
    def get_all_patient_and_feature_wise_pvalue_and_auc_for_all_channels(self, df, patients, channels, cols, class_n):
        class_name = self.class_name if len(class_n)==0 else class_n
        featWise_pAuc_chnDat = pd.DataFrame()
        patientWise_pAuc_chnDat = pd.DataFrame()

        for i,chn in enumerate(channels):
            new_df = df[df['ch_name'] == chn]
            feat_pAuc, pat_pAuc = self.get_all_patient_and_feature_wise_pvalue_and_auc(new_df, patients, cols, class_name)

            cc = [chn for j in range(feat_pAuc.shape[0])]
            dd = {'channel': cc}
            chnn = pd.DataFrame(dd)

            com_df = pd.concat([chnn, feat_pAuc], axis=1)
            featWise_pAuc_chnDat = pd.concat([featWise_pAuc_chnDat, com_df])

            cc2 = [chn for j in range(pat_pAuc.shape[0])]
            dd2 = {'channel': cc2}
            chnn2 = pd.DataFrame(dd2)
            # print(chnn2.shape, pat_pAuc.shape)
            pat_pAuc.reset_index(drop=True, inplace=True)
            # print(chnn2, pat_pAuc)

            com_df2 = pd.concat([chnn2, pat_pAuc], axis=1)
            patientWise_pAuc_chnDat = pd.concat([patientWise_pAuc_chnDat, com_df2])

        featWise_pAuc_chnDat.reset_index(drop=True, inplace=True)
        patientWise_pAuc_chnDat.reset_index(drop=True, inplace=True)
        return featWise_pAuc_chnDat, patientWise_pAuc_chnDat


    # ## Process the p-value and AUC-value for each channels
    def get_channelwise_pvalue_and_auc(self, df, class_name, channels, cols):
        class_name = self.class_name
        com_vals = pd.DataFrame()

        for chn in channels:
            self.logger.info(f'### p-Valu and AUC-Value calculation started for channel: {chn}\n')
            chdf = df[df[self.channel_cols[1]] == chn]
            dschdf = chdf[cols[(cols.index(class_name)):]]

            p = self.get_pValues_for_dataframe(dschdf)
            # self.logger.info(f'{p}\n\n')
            auc = self.get_pValues_for_dataframe(dschdf)
            # self.logger.info(f'{auc}\n\n')

            com_df = pd.merge(p, auc, on="features")
            # self.logger.info(com_df)

            cc = [chn for j in range(p.shape[0])]
            dd = {'channels': cc}
            chnn = pd.DataFrame(dd)
            com_df = pd.concat([chnn, com_df], axis=1)
            com_vals = pd.concat([com_vals, com_df])

            self.logger.info(f'*** p-Valu and AUC-Value calculation finished for channel: {chn}\n')

        return com_vals


    # ######################################
    # ### Mean and STD calculations
    # ######################################

    # ## mean, std of all features overall
    def get_feature_mean_std(self, df, cols, patients=[], channels=[]):
        if len(patients)>0:
            df = df[df[self.pat_id_col].isin(patients)]
        if len(channels)>0:
            df = df[df[self.channel_cols[1]].isin(channels)]

        class_name = self.class_name
        ft_mean_std = pd.DataFrame()
        # fts = cols2[5:]
        feats = cols[(cols.index(class_name) + 1):]
        for ft in feats:
            nsdf = df[df[class_name] == 0][ft]
            sdf = df[df[class_name] == 1][ft]

            nsmean = nsdf.mean()
            nsstd = nsdf.std()
            smean = sdf.mean()
            sstd = sdf.std()

            ms_dict = {'features': [ft], 'non_siez_mean': [nsmean], 'non_siez_std': [nsstd], 'siez_mean': [smean], 'siez_std': [sstd]}
            msdf = pd.DataFrame(ms_dict)

            ft_mean_std = pd.concat([ft_mean_std, msdf])
        return ft_mean_std


    # ## mean, std of all features overall for all channels
    def get_feature_mean_std_for_all_channels(self, df, cols, channels, patients=[]):
        if len(patients)>0:
            df = df[df[self.pat_id_col].isin(patients)]

        class_name = self.class_name
        ch_mean_std = pd.DataFrame()
        chns = channels
        #     fts = cols2[5:]
        fts = cols[(cols.index(class_name) + 1):]
        for ch in chns:
            df2 = df[df['ch_name'] == ch]

            msdf = self.get_feature_mean_std(df2, cols)
            msdf.insert(0, 'channel', [ch]*len(fts))
            ch_mean_std = pd.concat([ch_mean_std, msdf])
        return ch_mean_std




    # ######################################
    # ### Saving all p-value, AUC and mean-std values
    # ######################################

    # # ## mean, std of all features overall
    # def save_all_and_channelwise_pValue_AUC_Mean_STD_data(self, save_dir, seiz_only, ana_meta_data, featWise_pAuc_overallDat,
    #                   patientWise_pAuc_overallDat, featWise_pAuc_chnDat, patientWise_pAuc_chnDat, feat_mean_std,
    #                   chan_feat_mean_std, file_naming_detail_for_dataset, save_file_name_extra=''):
    #     # Save metadata
    #     save_file_name = f'{save_dir}analysis-metadata_{file_naming_detail_for_dataset}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.txt'
    #     self.dataManager.save_dictionary_to_file(ana_meta_data, save_file_name, 'Analysis Metadata', dump_level=0)
    #
    #     # Save feature-wise p & auc for all data
    #     save_file_name = f'{save_dir}featWise-pAuc-allDat_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     featWise_pAuc_overallDat.to_csv(save_file_name, index=False)
    #
    #     # Save patient-wise p & auc for all data
    #     save_file_name = f'{save_dir}patWise-pAuc-allDat_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     patientWise_pAuc_overallDat.to_csv(save_file_name, index=False)
    #
    #     # Save feature-wise p & auc for channel-wise data
    #     save_file_name = f'{save_dir}featWise-pAuc-chnDat_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     featWise_pAuc_chnDat.to_csv(save_file_name, index=False)
    #
    #     # Save patient-wise p & auc for channel-wise data
    #     save_file_name = f'{save_dir}patWise-pAuc-chnDat_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     patientWise_pAuc_chnDat.to_csv(save_file_name, index=False)
    #
    #     # Save feat mean-std
    #     save_file_name = f'{save_dir}feat-meanstd_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     feat_mean_std.to_csv(save_file_name, index=False)
    #
    #     # Save channel-wise feat mean-std
    #     save_file_name = f'{save_dir}chn-feat-meanstd_ch-{tot_chns}_pat-{tot_pats}_rec-{tot_recs}_feats-{tot_feats}'
    #     if len(save_file_name_extra) > 0:
    #         save_file_name += f'_{save_file_name_extra}'
    #     save_file_name += '.csv'
    #     chan_feat_mean_std.to_csv(save_file_name, index=False)
    #
    #     return


