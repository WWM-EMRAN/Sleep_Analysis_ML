# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sb
import os
import pickle
import decimal, math
from datetime import datetime
import json
from scipy import stats
import random


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, ParameterGrid
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from HumachLab_Utility import Humachlab_Utility



# should_balanced_data
#%%


class HumachLab_CHBMIT_Classifier:

    dataset = None
    should_use_params = False
    cross_validation_rounds = 10
    log_serial_no = 0
    log_path = ''
    should_balanced_data = False
    exp_name = 'test'
    test_split_ratio = 0.30
    classifier_methods = ['rf', 'dt', 'knn', 'nb', 'svm']



######################################
# ### Necessary parameter setup for data management and training-testing

    def __init__(self, logger, log_path='./', test_no='00', should_use_params=False, parameter_optimization=3, early_stop=False, cross_validation_rounds=5, exp_name='test', test_split_ratio=0.30, class_name='class', classifier_methods=[], random_training=0, should_balanced_data = False, channels=[], patients=[], records=[]):

        self.classifier_methods = classifier_methods

        self.dataset = pd.DataFrame()
        self.should_use_params = should_use_params
        self.early_stop = early_stop
        self.parameter_optimization = parameter_optimization
        self.should_balanced_data = should_balanced_data
        self.cross_validation_rounds = cross_validation_rounds
        self.log_json_data = {'models':[]}
        self.exp_name = exp_name
        self.test_split_ratio = test_split_ratio
        self.is_validate_models = True
        self.log_path = log_path
        self.test_no = test_no
        self.logger = logger
        self.class_name = class_name
        self.random_training = random_training

        self.channels = channels
        self.patients = patients
        self.records = records

        self.pat_id_col = 'pat_id'
        self.channel_cols = ['ch_id', 'ch_name']
        self.extra_cols =  ['pat_id', 'rec_id'] + self.channel_cols

        self.validation_patients = []
        self.training_patients = []
        return


######################################
# ### Loading or populating data for training

    def get_external_data(self, dat_file_name='All_Data_Left_Hemisphere.csv', replace_transitions=False, all_channels_columnwise=False):
        patients = self.patients
        channels = self.channels

        df = pd.DataFrame()

        # './AllData/All_Data_Left_Hemisphere.csv' 'All_Data.csv'
        # all_data_csv = f'{self.all_data_directory}{dat_file_name}'
        all_data_csv = dat_file_name

        if os.path.isfile(all_data_csv):
            df = pd.read_csv(all_data_csv)
            self.logger.info(f'{df.shape}')

            self.logger.info(f'Shape before: {df.shape}')
            df = df[df != np.inf]
            df = df.fillna(0)

            if replace_transitions:
                df[self.class_name] = df[self.class_name].replace([2, 3], 0)
            else:
                df = df[~df[self.class_name].isin([2, 3])]

            self.logger.info(f'Shape after: {df.shape}')
            if len(channels)>0:
                df = df[df['ch_name'].isin(channels)]
            if len(patients)>0:
                df = df[df['pat_id'].isin(patients)]

            if all_channels_columnwise:

                tmpdf = pd.DataFrame()
                ex_cols = ['pat_id', 'rec_id', 'ch_id', 'ch_name', self.class_name]
                join_cols = ['pat_id', 'rec_id', self.class_name]
                drop_cols = ['ch_id', 'ch_name']

                for chn in channels:
                    self.logger.info(f'{chn}')
                    tdf = df[df['ch_name']==chn]

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
            self.logger.info(f'Shape after2: {df.shape}')
        return df


    def load_data(self, path, class_name, channels=[], patients=[], records=[]):
        # patient = ':02d'.format(patient)
        self.logger.info(f'{path} {channels}')
        df = pd.DataFrame()
        path2 = os.walk(path)

        for root, directories, files in path2:
            # for directory in directories:
            #     self.logger.info(f'{directory}')
            for file in files:
                #             self.logger.info(f'{file}')
                items = file.split('_')
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
                            cArr = [ch for i in range(len(extr_feat.index))]
                            cnArr = [ch_name for i in range(len(extr_feat.index))]
                            dat_info = pd.DataFrame({'pat_id': pArr, 'rec_id': rArr, 'ch_id': cArr, 'ch_name': cnArr})

                            extr_feat = dat_info.join([extr_feat])

                            df = df.append(extr_feat, ignore_index=True)

        #     self.logger.info(f'{df}')
        # if df.shape[0] > 0:
        #     df[class_name] = df[class_name].replace([2, 3], 0)
        #         df.seizureState.replace({2: 0, 3: 0}, inplace=True)

        df = df[df != np.inf]
        df = df.fillna(0)

        return df


    # def load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', all_channels_columnwise=False, dataframe=None):
    def set_or_load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', replace_transitions=False, all_channels_columnwise=False, dataframe=None, dat_file_name=''):

        if ((dataframe is None) and (path is None) and (dat_file_name is None)):
            self.logger.worning(f'No data is provided or instructed to retrieve...')
            return
        elif (dat_file_name is not None):
            df = self.get_external_data(dat_file_name=dat_file_name, replace_transitions=replace_transitions, all_channels_columnwise=all_channels_columnwise)
            self.dataset = df
        elif (dataframe is not None):
            self.dataset = dataframe
            self.logger.info(f'External data is set...')
        else:
            class_name = self.class_name
            channels = self.channels
            patients = self.patients
            records = self.records

            # patient = ':02d'.format(patient)
            df = pd.DataFrame()
            path2 = os.walk(path)
            ex_cols = ['pat_id', 'rec_id', 'ch_id', 'ch_name', class_name]
            join_cols = ['pat_id', 'rec_id', class_name]
            drop_cols = ['ch_id', 'ch_name']

            for root, directories, files in path2:
                selected_dirs = [sdir for sdir in directories if (feature_subdirectories) in sdir]
                # self.logger.info(f'{selected_dirs} {channels}')
                selected_dirs = [sdir for sdir in selected_dirs if (sdir.split('_')[1]) in channels]
                # self.logger.info(f'{selected_dirs}')
                # Sorting directory based on channel
                selected_dirs = [d for c in channels for d in selected_dirs if c in d]
                directories = selected_dirs

                for directory in directories:
                    #             self.logger.info(f'== {directory}')
                    chn = directory.split('_')[1]
                    directory = path + directory + '/'
                    self.logger.info(f'== {directory}')
                    #             continue

                    tdf = self.load_data(directory, class_name, channels=channels, patients=patients, records=records)
                    self.logger.info(f'{df.shape} {tdf.shape}')
                    #             continue

                    # Do post-processing for row-wise or column-wise data rearrengement
                    if all_channels_columnwise:
                        tcols = [c for c in tdf.columns if (c not in ex_cols)]
                        tmp_cols = [(f'{ft}_{chn}') for ft in tcols]
                        tmp_cols = ex_cols + tmp_cols

                        tdf.columns = tmp_cols

                        if df.empty:
                            tdf.drop(drop_cols, axis=1, inplace=True)
                            df = pd.concat([df, tdf])
                        else:
                            tdf.drop(ex_cols, axis=1, inplace=True)
                            df = pd.merge(df, tdf, left_index=True, right_index=True)
                        self.logger.info(f'{df.shape} {tdf.shape}')

                    # Adding data to existing raw
                    else:
                        df = pd.concat([df, tdf])

            self.logger.info(f'Shape before: {df.shape}')
            if replace_transitions:
                df[class_name] = df[class_name].replace([2, 3], 0)
            else:
                df = df[~df[class_name].isin([2, 3])]
            self.logger.info(f'Shape after: {df.shape}')

            self.dataset = df

        self.logger.info(f'Dataset set with shape: {self.dataset.shape}')
        self.logger.info(f'{list(self.dataset.columns)}')

        return



######################################
# ### Patient specific data splitting and feature-target seperator

    def _patient_specific_data_splitter(self, dat, p_id, test_split_ratio=0.0):
        tr_dat, ts_dat = None, None

        if test_split_ratio == 0.0:
            self.logger.info(f'All..')
            ts_dat = dat[dat[self.pat_id_col] == p_id]
            tr_dat = dat[dat[self.pat_id_col] != p_id]
        else:
            unique_pat_ids = list(dat[self.pat_id_col].unique())
            self.logger.info(f'{unique_pat_ids}')
            no_of_patients = len(unique_pat_ids)
            # no_of_ts_patients = int(round(test_split_ratio * no_of_patients))
            no_of_ts_patients = int(math.floor(test_split_ratio * no_of_patients))
            if no_of_ts_patients==0:
                no_of_ts_patients = 1
            no_of_tr_patients = no_of_patients - no_of_ts_patients
            self.logger.info(f'{no_of_ts_patients} {no_of_tr_patients}')
            tr_pat_ids, ts_pat_ids = None, None
            tot_ind = (unique_pat_ids.index(p_id) + no_of_ts_patients)
            if tot_ind > (no_of_patients):
                tr_pat_ids = unique_pat_ids[(tot_ind - no_of_patients): unique_pat_ids.index(p_id)]
                ts_pat_ids = list(set(unique_pat_ids) - set(tr_pat_ids))
            else:
                ts_pat_ids = unique_pat_ids[
                             unique_pat_ids.index(p_id): tot_ind]
                tr_pat_ids = list(set(unique_pat_ids) - set(ts_pat_ids))

            self.logger.info(f'{ts_pat_ids} {tr_pat_ids}')
            ts_dat = dat[dat[self.pat_id_col].isin(ts_pat_ids)]
            tr_dat = dat[dat[self.pat_id_col].isin(tr_pat_ids)]

        return tr_dat, ts_dat


    def _patient_specific_feature_target_seperator(self, dat):
        #pat_id	rec_id	ch_id	ch_name	|seizureState	|features...
        target = dat[self.class_name]
        features = dat.drop(self.extra_cols, axis=1)

        return features, target



######################################
# ### Patient specific training and testing

    def patient_specific_testing(self, classifier_method='rf'):

        # self.log_path = self.create_log_path(self.log_path_orig)
        self.logger.info(f'\n##########********** Classification with: {classifier_method} **********##########\n#################################################################\n')

        data = self.dataset
        if data is None:
            self.logger.info(f'No data is provided, please try again after setting proper data...')
            return

        all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores = [], [], [], [], []
        unique_pat_ids = list(data[self.pat_id_col].unique())
        all_y_tests = []
        all_y_preds = []

        for pat_id in unique_pat_ids:
            self.logger.info(f'\n### Testing STARTS for patient {pat_id}')

            training_data, test_data = self._patient_specific_data_splitter(data, pat_id)

            best_model_index = 0
            all_trained_models, all_trained_models_scores, best_model_index = self.patient_specific_training(training_data, classifier_method)
            model_scores, test_patient = [], {'exp_name': self.exp_name, 'test_patient': pat_id}

            # continue

            for i in range(len(all_trained_models)):
                model = all_trained_models[i]
                model_scrs = test_patient
                X_test, y_test = self._patient_specific_feature_target_seperator(test_data)
                y_pred = model.predict(X_test)

                ind = all_trained_models.index(model)
                tr_mod_scr = all_trained_models_scores[ind]
                # self.logger.info(f'$$$-> {ind} {tr_mod_scr['validation_patient']}')
                model_scrs = {**model_scrs, **{'validation_patient': tr_mod_scr['validation_patient'], 'training_patient': tr_mod_scr['training_patient']}}

                mod_scr = self.calculate_model_scores(model, y_pred, y_test)
                model_scrs = {**model_scrs, **mod_scr}

                model_scores.append(model_scrs)
                self.logger.info(f'== {model_scores}')

                # # For best model
                if i==best_model_index:
                    # self.logger.info(f'bst mod ind =========================> {pat_id} {best_model_index}')
                    all_best_model.append(model)
                    all_best_test_scores.append(model_scrs)

                    all_y_tests.append(y_test)
                    all_y_preds.append(y_pred)

            all_models.append(all_trained_models)
            all_models_training_scores.append(all_trained_models_scores)
            all_models_test_scores.append(model_scores)

            self.logger.info(f'### Testing ENDS for patient {pat_id}')

        self.save_and_show_model_scores(all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores)
        self.save_metadata_and_prediction_targets(classifier_method, unique_pat_ids, all_y_tests, all_y_preds)

        return


    def patient_specific_training(self, data, classifier_method='rf'):
        all_trained_models, all_trained_models_scores = [], []
        unique_pat_ids = list(data[self.pat_id_col].unique())

        # Random or leave-1-out patients list
        loop_iter = len(unique_pat_ids)
        index_list = [i for i in range(0, loop_iter)]
        if self.random_training>0:
            index_list = []
            while len(index_list) < self.random_training:
                rr = random.randint(0, loop_iter-1)
                if rr not in index_list:
                    index_list.append(rr)
            index_list.sort()

        val_pat = []
        tr_pat = []

        # # Best model and best model scores are recorded
        bst_model_ind = 0
        bst_model_f1_score = -1 ## Search for best model based on higher F1 score

        for i in index_list:
            pat_id = unique_pat_ids[i]
            self.logger.info(f'{i}- Training and validation STARTS for patient {pat_id}')
            training_data, validation_data = None, None

            if self.is_validate_models:
                training_data, validation_data = self._patient_specific_data_splitter(data, pat_id, test_split_ratio=self.test_split_ratio)
            else:
                training_data, validation_data = data, None

            # continue
            val_pat = list(validation_data[self.pat_id_col].unique())
            tr_pat = list(training_data[self.pat_id_col].unique())

            model, model_scores, model_scr = None, {'exp_name': self.exp_name, 'validation_patient': val_pat, 'training_patient': tr_pat}, None

            if classifier_method=='rf':
                model, model_scr = self.random_forest(training_data, validation_data)
            elif classifier_method == 'knn':
                model, model_scr = self.k_neares_neighbours(training_data, validation_data)

            # # Best model and best model scores are recorded
            f1scr = model_scr['f1_score']
            # self.logger.info(f'********** F1 {f1scr}')
            if f1scr>bst_model_f1_score:
                bst_model_ind = i
                bst_model_f1_score = f1scr

            # # If two models having same F1 score, then, the model with maximum number of data points can be selected
            # ### TO DO--->

            model_scores.update(model_scr)
            # model_scores = pd.DataFrame(model_scores)

            all_trained_models.append(model)
            all_trained_models_scores.append(model_scores)

            self.logger.info(f'Training and validation ENDS for patient {pat_id}')

            # Stop training model if one already gets highest F1 score, ie. 100%
            if self.early_stop and f1scr==100:
                break

        self.validation_patients.append(val_pat)
        self.training_patients.append(tr_pat)

        return all_trained_models, all_trained_models_scores, bst_model_ind


######################################
# ### Apply Grid-search and calculate scores

    def run_model_gridSearch(self, mods, params, training_data, test_data):
        self.logger.info(f'\n------------------------------------------------------------------\n GridSearch: {mods.__class__.__name__} : \n')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        # scoring = ['accuracy', 'precision', 'recall']
        # scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        scoring = ['precision', 'f1']
        refit = 'precision'

        model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)

        # if self.should_use_params:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2)
        # else:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2) #Normal model

        X_train, y_train = self._patient_specific_feature_target_seperator(training_data)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test = self._patient_specific_feature_target_seperator(test_data)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores


######################################
# ### Apply Randomized-search and calculate scores

    def run_model_randomizedSearch(self, mods, params, training_data, test_data):
        self.logger.info(f'\n------------------------------------------------------------------\n RandomizedSearch: {mods.__class__.__name__} : \n')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        # scoring = ['accuracy', 'precision', 'recall']
        # scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        scoring = ['precision', 'f1']
        refit = 'precision'

        model = RandomizedSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, iid=False, n_jobs=-1, verbose=2)

        # if self.should_use_params:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2)
        # else:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2) #Normal model

        X_train, y_train = self._patient_specific_feature_target_seperator(training_data)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test = self._patient_specific_feature_target_seperator(test_data)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores


######################################
# ### Apply Baysian-search and calculate scores

    def run_model_baysianSearch(self, mods, params, training_data, test_data):
        self.logger.info(f'\n------------------------------------------------------------------\n BaysianSearch: {mods.__class__.__name__} : \n')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        # scoring = ['accuracy', 'precision', 'recall']
        # scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        scoring = ['precision', 'f1']
        refit = 'precision'

        model = BayesSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_iter=(self.cross_validation_rounds*2), iid=False, n_jobs=-1, verbose=2)

        # if self.should_use_params:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2)
        # else:
        #     model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, verbose=2) #Normal model

        X_train, y_train = self._patient_specific_feature_target_seperator(training_data)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test = self._patient_specific_feature_target_seperator(test_data)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores


######################################
# ### Apply Custom grid-search and calculate scores

    def run_model_customGridSearch(self, mods, params, training_data, test_data):
        self.logger.info(f'\n------------------------------------------------------------------\n CustomGridSearch: {mods.__class__.__name__} : \n')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        # scoring = ['accuracy', 'precision', 'recall']
        # scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        scoring = ['precision', 'f1']
        refit = 'precision'

        # ###Documentation
        # https: // scikit - learn.org / stable / modules / cross_validation.html
        # model = BayesSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)
        param_grid = ParameterGrid(parameters)
        for pars in param_grid:
            pass

        X_train, y_train = self._patient_specific_feature_target_seperator(training_data)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test = self._patient_specific_feature_target_seperator(test_data)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores


#############################


    def calculate_model_scores(self, mods, y_pred, y_test):
        output_string = ''

        # output_string += f'\n------------------------------------------------------------------\n{mods.__class__.__name__} \n'
        confMat = confusion_matrix(y_test, y_pred).tolist()
        acc = round(accuracy_score(y_test, y_pred)*100, 2)
        prec = round(precision_score(y_test, y_pred)*100, 2)
        reca_sens = round(recall_score(y_test, y_pred)*100, 2)
        f1sc = round(f1_score(y_test, y_pred)*100, 2)
        spec = round((confMat[0][0] / (confMat[0][0]+confMat[1][0]))*100, 2)

        # scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc, 'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec, 'f1_score': f1sc}
        scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc,
                    'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec,
                    'f1_score': f1sc}

        self.logger.info(f'{scr_dict}')

        return scr_dict


    def save_and_show_model_scores(self, all_models, bst_models, all_training_scrs, all_test_scrs, bst_test_srcs):

        # ### Save model scores and details
        test_no = self.test_no

        PIK = f'{self.log_path}/all_models_{test_no}.dat'
        with open(PIK, 'wb') as f:
            pickle.dump(len(all_models), f)
            for i in range(len(all_models)):
                p = all_models[i]
                pickle.dump(p, f)
            self.logger.info(f'All models are written to the file...')

        PIK = f'{self.log_path}/bst_models_{test_no}.dat'
        with open(PIK, 'wb') as f:
            pickle.dump(len(bst_models), f)
            for i in range(len(bst_models)):
                p = bst_models[i]
                pickle.dump(p, f)
            self.logger.info(f'Best models are written to the file...')

        PIK = f'{self.log_path}/tr_score_{test_no}.dat'
        with open(PIK, 'wb') as f:
            pickle.dump(len(all_training_scrs), f)
            for i in range(len(all_training_scrs)):
                p = all_training_scrs[i]
                pickle.dump(p, f)
            self.logger.info(f'Training score is written to the file...')

        PIK = f'{self.log_path}/ts_score_{test_no}.dat'
        with open(PIK, 'wb') as f:
            pickle.dump(len(all_test_scrs), f)
            for i in range(len(all_test_scrs)):
                p = all_test_scrs[i]
                pickle.dump(p, f)
            self.logger.info(f'Test score is written to the file...')

        PIK = f'{self.log_path}/bst_score_{test_no}.dat'
        with open(PIK, "wb") as f:
            pickle.dump(len(bst_test_srcs), f)
            for i in range(len(bst_test_srcs)):
                p = bst_test_srcs[i]
                pickle.dump(p, f)
            self.logger.info(f'Best score is written to the file...')

        # ### Show model scores and details
        self.logger.info(f'{len(bst_models)} {len(all_test_scrs)} {len(all_training_scrs)} {len(bst_test_srcs)}')

        for bst in bst_test_srcs:
            ind = bst_test_srcs.index(bst)

            self.logger.info(f'\nRESULT {ind+1}\n')
            self.logger.info(f'##################################################################\n')
            self.logger.info(f'BEST TEST SCORES\n')
            self.logger.info(f'------------------------------------------------------------------\n')
            self.logger.info(f'{bst}')
            self.logger.info(f'==================================================================\n')
            self.logger.info(f'ALL TEST SCORES\n')
            self.logger.info(f'------------------------------------------------------------------\n')
            # for tss in all_test_scrs:
            #     self.logger.info(f'{tss}\n')
            self.logger.info(f'{all_test_scrs[ind]}')
            self.logger.info(f'==================================================================\n')
            self.logger.info(f'ALL TRAINING SCORES\n')
            self.logger.info(f'------------------------------------------------------------------\n')
            # for trs in all_training_scrs:
            #     self.logger.info(f'{trs}\n')
            self.logger.info(f'{all_training_scrs[ind]}')
            self.logger.info(f'==================================================================\n')

        return


    def save_metadata_and_prediction_targets(self, classifier_method, pat_ids, y_tests, y_preds):
        test_no = self.test_no

        # ### Saving metadata for the classifier setup
        PIK = f'{self.log_path}/metadata_{test_no}.txt'
        metadata_dict = {}
        metadata_dict['exp_no'] = test_no
        metadata_dict['exp_name'] = self.exp_name
        metadata_dict['patients'] = self.patients
        metadata_dict['records'] = self.records
        metadata_dict['channels'] = self.channels
        metadata_dict['dataset_columns'] = list(self.dataset.columns)
        metadata_dict['class_name'] = self.class_name
        metadata_dict['dataset_shape'] = self.dataset.shape
        metadata_dict['balancing_data'] = self.should_balanced_data
        metadata_dict['test_split_ratio'] = self.test_split_ratio
        metadata_dict['classifier_method'] = classifier_method
        metadata_dict['validation_randomeness'] = 'leave-one-out' if self.random_training==0 else f'random-{self.random_training}'
        metadata_dict['parameters_used'] = self.should_use_params
        metadata_dict['cross_validation_rounds'] = self.cross_validation_rounds
        metadata_dict['validation_patients'] = self.validation_patients
        metadata_dict['training_patients'] = self.training_patients

        #should_use_params=True, cross_validation_rounds=2, exp_name=f'Test {test_no}: {test_details}', test_no=f'{test_no}', test_split_ratio=0.3, class_name=

        # ### Saving dictionary using pickle
        with open(PIK, 'wb') as f:
            pickle.dump(metadata_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f'Metadata is written to the file...')

        # # ### Retrieving dictionary using pickle
        # with open(PIK, 'rb') as f:
        #     dict_data = pickle.load(f)
        #     self.logger.info(f'Metadata is read from the file...')


        # ### Saving prediction target data
        save_file = f'{self.log_path}/bst_prediction_data_{test_no}.csv'

        all_pat_id, all_y_tst, all_y_prd = [], [], []

        for i in range(len(y_tests)):
            y_tst = y_tests[i]
            y_prd = y_preds[i]
            pat_id = [pat_ids[i] for cc in range(len(y_tst))]
            # self.logger.info(f'{list(y_tst)} \n\n {list(y_prd)}')

            all_pat_id += pat_id
            all_y_tst += list(y_tst)
            all_y_prd += list(y_prd)

        df = pd.DataFrame(list(zip(all_pat_id, all_y_tst, all_y_prd)), columns=['pat_id', f'{classifier_method}_y_test', f'{classifier_method}_y_pred'])
        df.to_csv(save_file, index=False)
        self.logger.info(f'Prediction result is written to the file...')

        return



######################################
# ### All models' implementation

    def call_all_model_optimization(self, classifier, parameters, training_data, test_data):
        if self.parameter_optimization == 1:
            model, model_scores = self.run_model_gridSearch(classifier, parameters, training_data, test_data)
        elif self.parameter_optimization == 2:
            model, model_scores = self.run_model_randomizedSearch(classifier, parameters, training_data, test_data)
        elif self.parameter_optimization == 3:
            model, model_scores = self.run_model_baysianSearch(classifier, parameters, training_data, test_data)
        elif self.parameter_optimization == 4:
            model, model_scores = self.run_model_customGridSearch(classifier, parameters, training_data, test_data)
        return model, model_scores

    def random_forest(self, training_data, test_data):
        classifier = RandomForestClassifier()
        n_estimators = list(range(1, 500, 5))
        max_depth = list(range(1, 100))
        max_features = list(range(2, 20, 1))
        splitter = ['best', 'random']
        min_samples_split = list(range(1, 10))
        min_samples_leaf = list(range(1, 10))
        max_leaf_nodes = list(range(1, 100))

        # parameters = {'n_estimators': list(range(100, 1000, 50))}
        # parameters = {'n_estimators': list(range(100, 500, 50))}
        parameters = {'n_estimators': [30, 50, 75, 100, 200, 500, 750, 1000]}
        # parameters = {'n_estimators': list(range(1, 10, 1)), 'max_depth': list(range(1, 10)), 'max_leaf_nodes': list(range(1, 10))}
        # parameters = {'n_estimators': n_estimator, 'max_depth': max_depth, 'max_features': max_features, 'splitter': splitter, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_leaf_nodes': max_leaf_nodes}

        model, model_scores = self.call_all_model_optimization(classifier, parameters, training_data, test_data)

        return model, model_scores


    def k_neares_neighbours(self, training_data, test_data):
        classifier = KNeighborsClassifier()
        n_neighbors = list(range(1, 100))
        metric = ['manhattan', 'minkowski','euclidean']
        n_splits = list(range(2, 10))

        # parameters = {'n_neighbors': list(range(100, 1000, 50))}
        # parameters = {'n_neighbors': list(range(2, 11, 1))}
        parameters = {'n_neighbors': [2, 3, 5, 9, 13, 19, 29]}
        # parameters = {'n_neighbors': n_neighbors, 'metric': metric, 'n_splits': n_splits}

        model, model_scores = self.call_all_model_optimization(classifier, parameters, training_data, test_data)

        return model, model_scores


    # #### Models...
    # def naive_bias(self):
    #     parameters = {}
    #
    #     self.run_model_gridSearch(GaussianNB(), parameters)
    #     return
    #
    #
    # def support_vector_machine(self):
    #     c_range = list(self.float_range('0.000001', '1', '10'))
    #     gamma_range = list(self.float_range('0.000001', '1', '10'))
    #     kernal_range = ['linear', 'rbf', 'poly', 'sigmoid']
    #     degree = list(range(1, 10))
    #
    #     parameters = {'C': c_range, 'gamma': gamma_range, 'kernel': kernal_range, 'degree': degree}
    #
    #     self.run_model_gridSearch(SVC(random_state=42), parameters)
    #     return
    #
    #
    # def neares_neighbours(self):
    #     k_range = list(range(1, 100))
    #     metric = ['manhattan', 'minkowski','euclidean']
    #
    #     parameters = {'n_neighbors':k_range, 'metric': metric}
    #
    #     self.run_model_gridSearch(KNeighborsClassifier(), parameters)
    #     return
    #
    #
    # def decision_tree(self):
    #     depth_range = list(range(1, 100))
    #     criterion = ['gini', 'entropy']
    #     splitter = ['best', 'random']
    #     min_samples_split = list(range(1, 10))
    #     min_samples_leaf = list(range(1, 10))
    #     max_leaf_nodes = list(range(1, 100))
    #     parameters = {'max_depth': depth_range, 'criterion': criterion, 'splitter': splitter, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_leaf_nodes':max_leaf_nodes}
    #
    #     self.run_model_gridSearch(DecisionTreeClassifier(random_state = 42), parameters)
    #     return







