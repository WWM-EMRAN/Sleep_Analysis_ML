# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%

import decimal
import random
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from HumachLab import *
# from HumachLab_DataLoader import *
# from HumachLab_Utility import Humachlab_Utility
# from HumachLab.DataManager.HumachLab_DataManager import HumachLab_DataManager
# from HumachLab.Utility.HumachLab_Enums import ML_Classifiers, ML_Performace_Metrics



#%%


class HumachLab_Preprocessor:

    def __init__(self, logger):
        self.logger = logger
        return



    def get_selected_feature_groups_from_dataframe(self, df, class_name, ch_col_name, prep_feature_type=[]):
        self.logger.info(f'Domain based feature selection started...')
        all_cols = df.columns.values.tolist()
        selected_feats = HumachLab_StaticMethods.get_dataset_columns_containing_featuregroup_names(all_cols, class_name, ch_col_name, prep_feature_type)

        sel_df = df[selected_feats]
        del df
        return sel_df


    def get_selected_features_from_dataframe(self, df, class_name, ch_col_name, prep_feature_name=[]):
        self.logger.info(f'Individual feature selection started...')
        all_cols = df.columns.values.tolist()
        selected_feats = HumachLab_StaticMethods.get_dataset_columns_containing_feature_names(all_cols, class_name, ch_col_name, prep_feature_name)

        sel_df = df[selected_feats]
        del df
        return sel_df


    def get_PCA_from_dataframe(self, df, class_name, num_comp=0, allowed_variance=2):
        self.logger.info(f'PCA analysis started...')
        all_cols = df.columns.values.tolist()
        label_indx = all_cols.index(class_name)
        common_cols_data, feats_data = df.iloc[:, :label_indx+1], df.iloc[:, label_indx+1:]
        feat_cols = feats_data.columns.values.tolist()
        del df
        num_components = feats_data.shape[0]

        # sc = StandardScaler()
        # feats_data_sc = sc.fit_transform(feats_data)
        # pca = PCA()
        # X_reduced = pca.fit_transform(feats_data_sc)

        sc = StandardScaler()
        feats_data = sc.fit_transform(feats_data)
        pca = PCA()
        feats_data = pca.fit_transform(feats_data)

        explained_variance = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(explained_variance)

        explained_variance = explained_variance.tolist()
        cum_sum_eigenvalues = cum_sum_eigenvalues.tolist()

        cols = [f'PC{i + 1}' for i in range(feats_data.shape[1])]

        #     principal_df = pd.DataFrame(X_reduced.n_components, columns = cols)
        principal_df = pd.DataFrame(feats_data, columns=cols)
        del feats_data
        # feats_data = None

        if (num_comp > 0) and (num_comp <= num_components):
            principal_df = principal_df.iloc[:, :num_comp]

        if allowed_variance!=2:
            absolute_difference_function = lambda list_value: abs(list_value - allowed_variance)
            closest_value = min(cum_sum_eigenvalues, key=absolute_difference_function)
            indx_closest_value = cum_sum_eigenvalues.index(closest_value)
            print(f'{allowed_variance}, {closest_value}, {indx_closest_value}, {len(cum_sum_eigenvalues)}, {cum_sum_eigenvalues}')
            principal_df = principal_df.iloc[:, :indx_closest_value]

        common_cols_data.reset_index(inplace=True)
        principal_df.reset_index(inplace=True)

        self.logger.info(f'Dataset columns: {len(feat_cols)} \nPincipal componants: {len(principal_df.columns.values.tolist())}')
        principal_df = pd.concat([common_cols_data, principal_df], axis=1)
        self.logger.info(f'New dataset columns (Pincipal componants): {len(principal_df.columns.values.tolist())}')

        return principal_df, explained_variance, cum_sum_eigenvalues


    def get_resamplled_data(self, df, class_name, sort_col, random_sampling=True, up_or_down_sampling=0, min_scale=1.0, max_scale=3.0): ## 0-no, 1-down, 2-up, 3-bound
        self.logger.info(f'Data resampling started...')

        sz_dat = df[df[class_name]==1]
        nsz_dat = df[df[class_name]==0]
        sz_cnt = sz_dat.shape[0]
        nsz_cnt = nsz_dat.shape[0]
        new_df = pd.DataFrame()

        self.logger.info(f'Imbalanced data size: Seizue= {sz_cnt}, Non-seizure= {nsz_cnt}')

        if up_or_down_sampling==0:
            new_df = df
            self.logger.info(f'No resampling...')
        elif up_or_down_sampling==1:
            downsampled_dat = resample(nsz_dat, replace=True, n_samples=sz_cnt, random_state=42) if random_sampling else resample(nsz_dat, replace=True, n_samples=sz_cnt)
            new_df = pd.concat([sz_dat, downsampled_dat])
            self.logger.info(f'Downsampling finished...')
        elif up_or_down_sampling==2:
            upsampled_dat = resample(sz_dat, replace=True, n_samples=nsz_cnt, random_state=42) if random_sampling else resample(sz_dat, replace=True, n_samples=nsz_cnt)
            new_df = pd.concat([nsz_dat, upsampled_dat])
            self.logger.info(f'Upsampling finished...')
        elif up_or_down_sampling==3:
            mn_cnt = int(sz_cnt*min_scale)
            mx_cnt = int(mn_cnt*max_scale)
            upsampled_dat = resample(sz_dat, replace=True, n_samples=mn_cnt, random_state=42) if random_sampling else resample(sz_dat, replace=True, n_samples=mn_cnt)
            downsampled_dat = resample(nsz_dat, replace=True, n_samples=mx_cnt, random_state=42) if random_sampling else resample(nsz_dat, replace=True, n_samples=mx_cnt)
            new_df = pd.concat([downsampled_dat, upsampled_dat])
            self.logger.info(f'Bound-sampling finished...')
        else:
            new_df = df
            self.logger.info(f'No resampling...')

        new_df = new_df.sort_values(sort_col)
        self.logger.info(f'Data size after resampling: Seizue= {new_df[new_df[class_name]==1].shape[0]}, Non-seizure= {new_df[new_df[class_name]==0].shape[0]}')

        return new_df






