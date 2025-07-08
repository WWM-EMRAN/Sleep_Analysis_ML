# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%
import sys
import time

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
import copy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit, LeavePOut, ParameterGrid
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
from HumachLab_CHBMIT_Enums import ML_Classifiers, ML_Performace_Metrics



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
    validation_split_ratio = 0.30
    test_split_ratio = 0.30



######################################
# ### Necessary parameter setup for data management and training-testing

    def __init__(self, logger, log_path='./', test_no='00', should_use_params=False, parameter_optimization=3, early_stop=False, cross_validation_rounds=5, exp_name='test', class_name='class', test_split_ratio=0.30, test_splitting_nature=0, validation_split_ratio=0.30, validation_splitting_nature=0, should_balanced_data = False, best_model_scoring_metrics=[ML_Performace_Metrics.PREC, ML_Performace_Metrics.F1SCR], channels=[], patients=[], records=[]):

        self.classifier_methods = ['svm', 'knn', 'nb', 'dt', 'rf']

        self.dataset = pd.DataFrame()
        self.should_use_params = should_use_params
        self.early_stop = early_stop
        self.parameter_optimization = parameter_optimization
        self.should_balanced_data = should_balanced_data
        self.cross_validation_rounds = cross_validation_rounds
        self.log_json_data = {'models':[]}
        self.exp_name = exp_name
        self.test_split_ratio = test_split_ratio
        self.test_splitting_nature = test_splitting_nature
        self.validation_split_ratio = validation_split_ratio
        self.validation_splitting_nature = validation_splitting_nature
        self.is_validate_models = True
        self.log_path = log_path
        self.test_no = test_no
        self.logger = logger
        self.class_name = class_name
        self.best_model_scoring_metrics = best_model_scoring_metrics

        self.channels = channels
        self.patients = patients
        self.records = records

        self.pat_id_col = 'pat_id'
        self.rec_id_col = 'rec_id'
        self.seg_id_col = 'seg_id'
        self.channel_cols = ['ch_id', 'ch_name']
        self.extra_cols =  [self.pat_id_col, self.rec_id_col, self.seg_id_col]  + self.channel_cols

        self.validation_patients = []
        self.training_patients = []
        return


######################################
# ### Loading or populating data for training

    def get_external_data(self, dat_file_name='All_Data_Left_Hemisphere.csv', drop_nan=False, all_channels_columnwise=False):
        channels = self.channels

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
        return df


    def load_data(self, path, class_name, drop_nan=False, channels=[], patients=[], records=[]):
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


    def load_all_data(self, path, feature_subdirectories, replace_transitions, drop_nan, all_channels_columnwise):
        self.logger.info(f'{self.log_path}, {path}, {feature_subdirectories}')
        class_name = self.class_name
        channels = self.channels
        patients = self.patients
        records = self.records

        # patient = ':02d'.format(patient)
        df = pd.DataFrame()
        path2 = os.walk(path)
        ex_cols = self.extra_cols + [self.class_name]
        join_cols = [self.pat_id_col, self.rec_id_col, self.class_name]
        drop_cols = self.channel_cols

        for root, directories, files in path2:
            # self.logger.info(f'1== {directories}')
            selected_dirs = [sdir for sdir in directories if (feature_subdirectories) in sdir]
            # self.logger.info(f'2== {selected_dirs}')
            # self.logger.info(f'{selected_dirs} {channels}')
            selected_dirs = [sdir for sdir in selected_dirs if (len(sdir.split('_'))>1) and ((sdir.split('_')[1]) in channels)]
            # self.logger.info(f'3=={selected_dirs}')
            # Sorting directory based on channel
            selected_dirs = [d for c in channels for d in selected_dirs if c in d]
            directories = selected_dirs

            for directory in directories:
                # self.logger.info(f'== {directory}')
                chn = directory.split('_')
                if len(chn)<2 or (not directory.startswith(feature_subdirectories)):
                    continue
                elif chn[1] not in channels:
                    continue

                chn = chn[1]
                directory = path + directory + '/'
                # self.logger.info(f'== {directory}')
                #             continue

                tdf = self.load_data(directory, class_name, drop_nan=drop_nan, channels=channels, patients=patients, records=records)
                df = pd.concat([df, tdf])
                self.logger.info(f'{tdf.shape} {df.shape}')
        return df


    def _fill_or_remove_nan_value_for_all_channels(self, cdf, drop_nan):
        self.logger.info(f'Dealing with NaN data...')
        df = cdf.copy()
        patients = self.patients
        channels = self.channels

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


    def _create_columnwise_features_from_dataset(self, cdf, channels):
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


    # def load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', all_channels_columnwise=False, dataframe=None):
    def set_or_load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', replace_transitions=0, drop_nan=False, all_channels_columnwise=False, dataframe=None, dat_file_name=''):

        class_name = self.class_name
        channels = self.channels
        patients = self.patients
        records = self.records

        if ((dataframe is None) and (path is None) and (dat_file_name is None)):
            self.logger.worning(f'No data is provided or instructed to retrieve...')
            return
        elif (dat_file_name is not None):
            df = self.get_external_data(dat_file_name=dat_file_name, drop_nan=drop_nan, all_channels_columnwise=all_channels_columnwise)
            self.dataset = df
            self.logger.info(f'External data is set from a file...')
        elif (dataframe is not None):
            self.dataset = dataframe
            self.logger.info(f'External data is set... Please check the column name compatibility...')
        else:
            df = self.load_all_data(path, feature_subdirectories, replace_transitions, drop_nan, all_channels_columnwise)
            self.dataset = df
            self.logger.info(f'Dataset set from feature files...')

        # Initial data shape
        self.logger.info(f'Initial DataShape: {self.dataset.shape}')

        # ###Data filling & sorting and row-wise or column-wise data rearrangement
        if self.dataset.shape[0]>0:
            # Data filling and sorting
            self.dataset = self._fill_or_remove_nan_value_for_all_channels(self.dataset, drop_nan)
            # Do post-processing for row-wise or column-wise data rearrangement
            if all_channels_columnwise:
                self.dataset = self._create_columnwise_features_from_dataset(self.dataset, channels)

        # Replace if any positive value is found otherwise remove transition segments
        if replace_transitions>=0:
            self.dataset[class_name] = self.dataset[class_name].replace([2, 3], replace_transitions)
        else:
            self.dataset = self.dataset[~self.dataset[class_name].isin([2, 3])]

        self.logger.info(f'DataShape: {self.dataset.shape} '
                         f'\nDataColumns:{list(self.dataset.columns)} '
                         f'\nDataChannels:{self.channels if all_channels_columnwise else self.dataset[self.channel_cols[1]].unique()}')

        return


    ######################################
    # ### Patient specific indices splitting and feature-target data separator
    def _get_patients_training_test_indices_using_cross_validation(self, patients_list, tr_indices, ts_indices):
        pats = np.array(patients_list)
        tr_ids = list(pats[tr_indices])
        ts_ids = list(pats[ts_indices])
        tr_ids.sort()
        ts_ids.sort()
        return tr_ids, ts_ids


    def _get_data_from_patient_indices(self, indices):
        data = copy.deepcopy(self.dataset)
        data = data[data[self.pat_id_col].isin(indices)]
        target = data[self.class_name]
        extra_cols = data[self.extra_cols]
        features = data.drop(self.extra_cols, axis=1)
        data = None
        return features, target, extra_cols



######################################
# ### Patient specific training and testing

    def patient_specific_testing(self, classifier_method=ML_Classifiers.RF):

        # self.log_path = self.create_log_path(self.log_path_orig)
        self.logger.info(f'\n#################################################################\n##########********** Classification with: {ML_Classifiers.get_short_form(str(classifier_method.value))} **********##########\n#################################################################')

        if self.dataset is None:
            self.logger.info(f'No data is provided, please try again after setting proper data...')
            return

        all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores = [], [], [], [], []
        unique_pat_ids = list(self.dataset[self.pat_id_col].unique())
        all_extra = []
        all_y_tests = []
        all_y_preds = []

        splitter = None
        if self.test_splitting_nature > 0:
            # Random splitting
            splitter = ShuffleSplit(n_splits=self.test_splitting_nature, test_size=self.test_split_ratio, random_state=random.randint(1, 1000)) #rs
        elif self.test_splitting_nature == 0:
            # Leave-one-out
            splitter = LeavePOut(p=1) #loo
        else:
            # Leave-p-out
            splitter = LeavePOut(p=int(len(self.patients)*self.test_split_ratio)) #lpo

        util = Humachlab_Utility()
        ii = 0

        for train_ids, test_ids in splitter.split(unique_pat_ids):
            train_ids, test_ids = self._get_patients_training_test_indices_using_cross_validation(unique_pat_ids, train_ids, test_ids)
            # pat_id = test_ids[0]

            util.start_timer()
            self.logger.info(f'#################################################################\n@## Testing STARTS for patient {test_ids}')
            self.logger.info(f'Total testing:{len(test_ids)}, Total training & validation:{len(train_ids)}')
            self.logger.info(f'Testing:{test_ids}, Training & validation:{train_ids}')
            self.logger.info(f'#################################################################')

            best_model_index = 0
            all_trained_models, all_trained_models_scores, best_model_index = self.patient_specific_training(train_ids, classifier_method)
            model_scores, test_patient = [], {'exp_name': self.exp_name, 'test_patient': test_ids}

            #Uncomment if only want to see patient splitting
            # continue

            util2 = Humachlab_Utility()
            util2.start_timer()
            for i in range(len(all_trained_models)):
                model = all_trained_models[i]
                model_scrs = test_patient
                X_test, y_test, extra = self._get_data_from_patient_indices(test_ids)
                y_pred = model.predict(X_test)

                # Add test serial number with extra
                arr = [ii for ttt in range((extra.shape)[0])]
                tst_srl = pd.DataFrame(arr, columns=['test_number'])
                extra = pd.concat([tst_srl, extra], axis=1)

                ind = all_trained_models.index(model)
                tr_mod_scr = all_trained_models_scores[ind]
                # self.logger.info(f'$$$-> {ind} {tr_mod_scr['validation_patient']}')
                model_scrs = {**model_scrs, **{'validation_patient': tr_mod_scr['validation_patient'], 'training_patient': tr_mod_scr['training_patient']}}

                mod_scr = self.calculate_model_scores(model, y_pred, y_test)
                model_scrs = {**model_scrs, **mod_scr}

                model_scores.append(model_scrs)
                self.logger.info(f'*Training{i+1:>2} Score= {tr_mod_scr}')
                self.logger.info(f'#Testing {i+1:>2} Score= {model_scores}')

                # # For best model
                if i==best_model_index:
                    # self.logger.info(f'bst mod ind =========================> {test_ids} {best_model_index}')
                    all_best_model.append(model)
                    all_best_test_scores.append(model_scrs)

                    all_extra.append(extra)
                    all_y_tests.append(y_test)
                    all_y_preds.append(y_pred)

                X_test, y_test = None, None

            util2.end_timer()
            time_str = util2.time_calculator()
            self.logger.info(f'@@@ ## Each round of testing time for patients {test_ids}: {time_str}\n')

            all_models.append(all_trained_models)
            all_models_training_scores.append(all_trained_models_scores)
            all_models_test_scores.append(model_scores)

            self.logger.info(f'## Best Test Score= {all_best_test_scores[ii]}')
            self.logger.info(f'### Testing ENDS for patient {test_ids}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ # Each round of training and testing time for patients {test_ids}: {time_str}\n')
            self.logger.info(f'#################################################################')
            ii += 1

        self.save_and_show_model_scores(all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores)
        self.save_metadata_and_prediction_targets(classifier_method, unique_pat_ids, all_extra, all_y_tests, all_y_preds)

        return


    def patient_specific_training(self, training_ids, classifier_method=ML_Classifiers.RF):
        all_trained_models, all_trained_models_scores = [], []

        splitter = None
        if self.is_validate_models:
            if self.validation_splitting_nature > 0:
                # Random splitting
                splitter = ShuffleSplit(n_splits=self.validation_splitting_nature, test_size=self.validation_split_ratio, random_state=random.randint(1, 1000)) #rs
            elif self.validation_splitting_nature == 0:
                # Leave-one-out
                splitter = LeavePOut(p=1) #loo
            else:
                # Leave-p-out
                splitter = LeavePOut(p=int(len(self.patients)*self.validation_split_ratio)) #lpo
        else:
            self.logger.worning(f'Model is not going to be validated, this might cause problem and program crashes...')
            pass

        # # Best model and best model scores are recorded
        bst_model_ind = -1
        bst_model_score = None ## Search for best model based on higher F1 score
        i = 0
        scoring = self.best_model_scoring_metrics
        util = Humachlab_Utility()

        for train_ids, val_ids in splitter.split(training_ids):
            train_ids, val_ids = self._get_patients_training_test_indices_using_cross_validation(training_ids, train_ids, val_ids)

            util.start_timer()
            self.logger.info(f'@** Training and validation STARTS for patient {val_ids}')
            self.logger.info(f'Total validation:{len(val_ids)}, Total training:{len(train_ids)}')
            self.logger.info(f'Validation:{val_ids}, Training:{train_ids}')
            self.logger.info(f'------------------------------------------------------------------')

            if not self.is_validate_models:
                train_ids, val_ids = training_ids, None

            model, model_scores, model_scr = None, {'exp_name': self.exp_name, 'validation_patient': val_ids, 'training_patient': train_ids}, None

            #Uncomment if only want to see patient splitting
            # continue

            # Apply ML model
            model, model_scr = self.apply_ml_model(classifier_method, train_ids, val_ids)

            # # Best model and best model scores are recorded
            # # Choose best model based on a number of parameters
            new_best = True
            if i!=0:
                new_best = False
                for ss in scoring:
                    ss = ss.value
                    if model_scr[ss] > bst_model_score[ss]:
                        new_best = True
                        break
                    elif model_scr[ss] == bst_model_score[ss]:
                        continue
                    else:
                        break

            if new_best:
                bst_model_ind = i
                bst_model_score = model_scr

            # # If two models having same F1 score, then, the model with maximum number of data points can be selected
            # ### TO DO--->

            model_scores.update(model_scr)
            # model_scores = pd.DataFrame(model_scores)

            all_trained_models.append(model)
            all_trained_models_scores.append(model_scores)

            self.logger.info(f'*** Training and validation ENDS for patient {val_ids}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ ** Each round of training and validation time for patients {val_ids}: {time_str}\n')
            self.logger.info(f'------------------------------------------------------------------')
            # #Increamenting model index
            i += 1

            self.validation_patients.append(val_ids)
            self.training_patients.append(train_ids)

            # Stop training model if one already gets highest score, ie. 100%
            if self.early_stop:
                cnt = 0
                for ss in scoring:
                    if model_scr[ss]==100.00:
                        cnt += 1
                if cnt == len(scoring):
                    break

        return all_trained_models, all_trained_models_scores, bst_model_ind


######################################
# ### Apply Custom grid-search and calculate scores

    # def _do_custom_grideSearch(self, classifier_method, parameters, scoring, cv, refit, return_train_score, n_jobs=-1, verbose=2):
    def _do_custom_grideSearch(self, classifier_method, parameters, train_ids, val_ids, scoring, cv, split_ratio):
        # mods = self.get_ml_model_instances(classifier_method, parameters)
        all_info = []
        all_scr = []
        best_scr = None
        all_mdl = []
        best_mdl = None

        param_grid = ParameterGrid(parameters)
        self.logger.info(f'Fitting {cv} folds for each of {len(param_grid)} candidates, totalling {cv*len(param_grid)} fits:')
        # random_splitter = ShuffleSplit(n_splits=self.validation_splitting_nature, test_size=self.validation_split_ratio, random_state=random.randint(1, 1000))  # rs
        random_splitter = ShuffleSplit(n_splits=cv, test_size=split_ratio, random_state=random.randint(1, 1000))  # rs
        i = 0
        for tr_ids, ts_ids in random_splitter.split(train_ids):
            tr_ids, ts_ids = self._get_patients_training_test_indices_using_cross_validation(tr_ids, ts_ids)
            for pars in param_grid:
                if len(parameters.keys())==0:
                    parameters = None
                mods = self.get_ml_model_instances(classifier_method, parameters)
                X, y, _ = self._get_data_from_patient_indices(tr_ids)
                Xt, yt, _ = self._get_data_from_patient_indices(tr_ids)
                st = time.time()
                mods.fit(X, y.values.ravel())
                yp = mods.predict(Xt)
                # scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc,
                #             'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec,
                #             'false_positive_rate':fpr, 'f1_score': f1sc}
                scrs = self.calculate_model_scores(mods, yp, yt)

                all_mdl.append(mods)
                all_scr.append(scrs)

                # ### Score calculator
                if i==0:
                    best_mdl = mods
                    best_scr = scrs

                new_best = False
                for ss in scoring:
                    if scrs[ss]>best_scr[ss]:
                        new_best = True
                        break
                    elif scrs[ss]==best_scr[ss]:
                        continue
                    else:
                        break
                if new_best:
                    best_mdl = mods
                    best_scr = scrs

                # ### Time calculator
                et = time.time()
                # Time calculation
                millis = int(et-st)
                seconds = (millis / 1000) % 60
                seconds = int(seconds)
                minutes = (millis / (1000 * 60)) % 60
                minutes = int(minutes)
                hours = (millis / (1000 * 60 * 60)) % 24
                final_time = f''
                if hours > 0:
                    final_time = f'{round(hours, 2):>6} hr'
                elif minutes > 0:
                    final_time = f'{round((minutes + (seconds / 60)), 2):?6} min'
                else:
                    final_time = f'{round(seconds, 2):>6} sec'

                self.logger.info(f'[CV-{i:02}] END {pars:.>50s}; total time={final_time}')
            i += 1

        all_info.append(best_mdl, best_scr, all_mdl, all_scr)

        return best_mdl, all_info

    def run_model_customGridSearch(self, classifier_method, params, train_ids, val_ids):
        mods = self.get_ml_model_instances(classifier_method)
        # mods = classifier_method
        self.logger.info(f'------------------------------------------------------------------\nCustomGridSearch: {ML_Classifiers.get_short_form(str(classifier_method.value))} - {params} ')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        scoring, refit = self.get_ml_scoring_metrices(self.best_model_scoring_metrics, self.best_model_scoring_metrics[0])

        # ### Documentation
        # https: // scikit - learn.org / stable / modules / cross_validation.html
        # model = self._do_custom_grideSearch(classifier_method, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)
        model, all_models_and_scores = self._do_custom_grideSearch(classifier_method, parameters, train_ids, val_ids, scoring=scoring, cv=self.cross_validation_rounds, split_ratio=self.validation_split_ratio)

        self.logger.info(f'ML model for training: {model}')

        X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores


######################################
# ### Apply Grid-search and calculate scores

    def custom_precision_func(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        with open(self.log_path+'ddd.txt', 'a') as f:
            f.write(str(cm)+'\n')
            f.write(str(y_true)+'\n')
            f.write(str(y_pred)+'\n')
            f.write('\n\n')

        prec = 0 if ((cm[1][1]+cm[0][1]))==0 else cm[1][1]/(cm[1][1]+cm[0][1])
        return prec

    def run_model_gridSearch(self, classifier_method, params, train_ids, val_ids):
        mods = self.get_ml_model_instances(classifier_method)
        self.logger.info(f'------------------------------------------------------------------\nGridSearch: {ML_Classifiers.get_short_form(str(classifier_method.value))} - {params} ')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        scoring, refit = self.get_ml_scoring_metrices(self.best_model_scoring_metrics, self.best_model_scoring_metrics[0])

        model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)

        # ### Scoring from custom method
        # score = make_scorer(self.custom_precision_func, greater_is_better=False)
        # # scoring = {'precision': score, 'f1':make_scorer(f1_score)}
        # model = GridSearchCV(mods, parameters, scoring=score, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)

        self.logger.info(f'ML model for training: {model}')

        X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
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

    def run_model_randomizedSearch(self, classifier_method, params, train_ids, val_ids):
        mods = self.get_ml_model_instances(classifier_method)
        self.logger.info(f'------------------------------------------------------------------\nRandomizedSearch: {ML_Classifiers.get_short_form(str(classifier_method.value))} - {params} ')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        scoring, refit = self.get_ml_scoring_metrices(self.best_model_scoring_metrics, self.best_model_scoring_metrics[0])

        model = RandomizedSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, iid=False, n_jobs=-1, verbose=2)

        self.logger.info(f'ML model for training: {model}')

        X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
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

    def run_model_baysianSearch(self, classifier_method, params, train_ids, val_ids):
        mods = self.get_ml_model_instances(classifier_method)
        self.logger.info(f'------------------------------------------------------------------\nBaysianSearch: {ML_Classifiers.get_short_form(str(classifier_method.value))} - {params} ')
        parameters = {}
        model = mods
        model_scores = None
        if self.should_use_params:
            parameters = params

        scoring, refit = self.get_ml_scoring_metrices(self.best_model_scoring_metrics, self.best_model_scoring_metrics[0])

        model = BayesSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_iter=(self.cross_validation_rounds*2), iid=False, n_jobs=-1, verbose=2)

        self.logger.info(f'ML model for training: {model}')

        X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        model = model.fit(X_train, y_train.values.ravel())

        if self.is_validate_models:
            X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            y_pred = model.predict(X_test)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
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
        # output_string = ''

        # output_string += f'------------------------------------------------------------------\n{mods.__class__.__name__} '
        confMat = confusion_matrix(y_test, y_pred).tolist()
        tn = confMat[0][0]
        fp = confMat[0][1]
        fn = confMat[1][0]
        tp = confMat[1][1]

        acc = round(accuracy_score(y_test, y_pred)*100, 2)
        prec = round(precision_score(y_test, y_pred)*100, 2) #precision or positive predictive value (PPV)
        reca_sens = round(recall_score(y_test, y_pred)*100, 2) #sensitivity, recall, hit rate, or true positive rate (TPR)
        f1sc = round(f1_score(y_test, y_pred)*100, 2)
        spec = round((tn / (tn+fp))*100, 2) #specificity, selectivity or true negative rate (TNR)
        fpr = round((fp / (fp+tn))*100, 2) #fall-out or false positive rate (FPR)
        fnr = round((fn / (fn+tp))*100, 2) #miss rate or false negative rate (FNR)

        # scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc, 'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec, 'f1_score': f1sc}
        scr_dict = {'model_name': str(mods.__class__.__name__), 'method_name': str(mods.estimator), 'method_parameters': str(mods.best_params_), 'method_scores': str(round(mods.best_score_*100,2)),
                    ML_Performace_Metrics.CONF_MAT.value: confMat, ML_Performace_Metrics.ACC.value: acc, ML_Performace_Metrics.PREC.value: prec,
                    ML_Performace_Metrics.RECL.value: reca_sens, ML_Performace_Metrics.SEN.value: reca_sens, ML_Performace_Metrics.SPEC.value: spec,
                    ML_Performace_Metrics.FPR.value: fpr, ML_Performace_Metrics.FNR.value: fnr, ML_Performace_Metrics.F1SCR.value: f1sc}

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

        # ### Writing scores in text file
        PIK = f'{self.log_path}/tr_score_{test_no}.txt'
        with open(PIK, 'wb') as f:
            pickle.dump(all_training_scrs, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(len(all_training_scrs), f)
            # for i in range(len(all_training_scrs)):
            #     p = all_training_scrs[i]
            #     pickle.dump(p, f)
            self.logger.info(f'Training score is written to the file...')

        PIK = f'{self.log_path}/ts_score_{test_no}.txt'
        with open(PIK, 'wb') as f:
            pickle.dump(all_test_scrs, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(len(all_test_scrs), f)
            # for i in range(len(all_test_scrs)):
            #     p = all_test_scrs[i]
            #     pickle.dump(p, f)
            self.logger.info(f'Test score is written to the file...')

        PIK = f'{self.log_path}/bst_score_{test_no}.txt'
        with open(PIK, "wb") as f:
            pickle.dump(bst_test_srcs, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(len(bst_test_srcs), f)
            # for i in range(len(bst_test_srcs)):
            #     p = bst_test_srcs[i]
            #     pickle.dump(p, f)
            self.logger.info(f'Best score is written to the file...')

        # ### Show model scores and details
        self.logger.info(f'{len(bst_models)} {len(all_test_scrs)} {len(all_training_scrs)} {len(bst_test_srcs)}')

        for bst in bst_test_srcs:
            ind = bst_test_srcs.index(bst)

            self.logger.info(f'\nRESULT {ind+1}')
            self.logger.info(f'##################################################################')
            self.logger.info(f'BEST TEST SCORES\n')
            self.logger.info(f'------------------------------------------------------------------')
            self.logger.info(f'{bst}')
            self.logger.info(f'==================================================================')
            self.logger.info(f'ALL TEST SCORES\n')
            self.logger.info(f'------------------------------------------------------------------')
            # for tss in all_test_scrs:
            #     self.logger.info(f'{tss}\n')
            self.logger.info(f'{all_test_scrs[ind]}')
            self.logger.info(f'==================================================================')
            self.logger.info(f'ALL TRAINING SCORES\n')
            self.logger.info(f'------------------------------------------------------------------')
            # for trs in all_training_scrs:
            #     self.logger.info(f'{trs}\n')
            self.logger.info(f'{all_training_scrs[ind]}')
            self.logger.info(f'==================================================================')

        return


    def save_metadata_and_prediction_targets(self, classifier_method, pat_ids, all_extra, y_tests, y_preds):
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
        # sz = (self.dataset[self.dataset[self.class_name]!=0].shape)[0]
        sz = (self.dataset[self.dataset[self.class_name]==1].shape)[0]
        nsz = (self.dataset[self.dataset[self.class_name]==0].shape)[0]
        metadata_dict['seizure_nonseizure_amount'] = (sz, nsz)
        metadata_dict['balancing_data'] = self.should_balanced_data
        metadata_dict['test_split_ratio'] = self.test_split_ratio
        val_nature = 'leave-one-out' if self.test_splitting_nature == 0 else (f'random-{self.test_splitting_nature}' if self.test_splitting_nature > 0 else f'leave-{int(len(self.patients) * self.test_split_ratio)}-out')
        tst_nature = 'leave-one-out' if self.validation_splitting_nature==0 else (f'random-{self.validation_splitting_nature}' if self.validation_splitting_nature>0 else f'leave-{int(len(self.patients)*self.validation_split_ratio)}-out')
        metadata_dict['validation_splitting_nature'] = val_nature
        metadata_dict['validation_split_ratio'] = self.validation_split_ratio
        metadata_dict['validation_splitting_nature'] = tst_nature
        metadata_dict['classifier_method'] = classifier_method
        metadata_dict['parameters_used'] = self.should_use_params
        metadata_dict['cross_validation_rounds'] = self.cross_validation_rounds
        metadata_dict['validation_patients'] = self.validation_patients
        metadata_dict['training_patients'] = self.training_patients

        #should_use_params=True, cross_validation_rounds=2, exp_name=f'Test {test_no}: {test_details}', test_no=f'{test_no}', validation_split_ratio=0.3, class_name=

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

        all_extra_vals, all_pat_id, all_y_tst, all_y_prd = pd.DataFrame(), [], [], []

        for i in range(len(y_tests)):
            extra = all_extra[i]
            y_tst = y_tests[i]
            y_prd = y_preds[i]

            # pat_id = [pat_ids[i] for cc in range(len(y_tst))]
            # self.logger.info(f'{list(y_tst)} \n {list(y_prd)}')
            # all_pat_id += pat_id

            all_extra_vals = pd.concat([all_extra_vals, extra])
            all_y_tst += list(y_tst)
            all_y_prd += list(y_prd)
            # print(len(y_tst), len(y_prd))

        # df = pd.DataFrame(list(zip(all_pat_id, all_y_tst, all_y_prd)), columns=[self.pat_id_col, f'{classifier_method}_y_test', f'{classifier_method}_y_pred'])
        df = all_extra_vals.copy()
        clf_str = ML_Classifiers.get_short_form(str(classifier_method.value))
        tmp_df = pd.DataFrame(list(zip(all_y_tst, all_y_prd)), columns=[f'{clf_str}_y_test', f'{clf_str}_y_pred'])
        df.reset_index(drop=True, inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, tmp_df], axis=1)
        df.to_csv(save_file, index=False)
        self.logger.info(f'Prediction result is written to the file...')

        return



######################################
# ### All models' implementation

    def apply_ml_model(self, classifier_method, train_ids, val_ids):
        parameters = self.get_parameters_for_ml_models(classifier_method)
        model, model_scores = self.call_all_model_optimization(classifier_method, parameters, train_ids, val_ids)
        return model, model_scores


    def call_all_model_optimization(self, classifier_method, parameters, train_ids, val_ids):
        model, model_scores = None, None
        if self.parameter_optimization == 1:
            model, model_scores = self.run_model_gridSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 2:
            model, model_scores = self.run_model_randomizedSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 3:
            model, model_scores = self.run_model_baysianSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 4:
            model, model_scores = self.run_model_customGridSearch(classifier_method, parameters, train_ids, val_ids)
        return model, model_scores


    def get_ml_model_instances(self, classifier_method, parameters=None):
        classifier = None
        # ####### rf #######
        # rf - random_forest classifier
        if classifier_method == ML_Classifiers.RF:
            classifier = RandomForestClassifier() if (parameters is None) else RandomForestClassifier(parameters)
        # ####### knn #######
        # knn - k_neares_neighbours classifier
        elif classifier_method == ML_Classifiers.kNN:
            classifier = KNeighborsClassifier() if (parameters is None) else KNeighborsClassifier(parameters)
        # ####### nb #######
        # knn - naieve bias classifier
        elif classifier_method == ML_Classifiers.NB:
            classifier = GaussianNB() if (parameters is None) else GaussianNB(parameters)
        # ####### svm/svc #######
        # knn - support vector classifier
        elif classifier_method == ML_Classifiers.SVC:
            classifier = SVC() if (parameters is None) else SVC(parameters)
        # ####### knn #######
        # knn - k_neares_neighbours classifier
        elif classifier_method == ML_Classifiers.DT:
            classifier = DecisionTreeClassifier() if (parameters is None) else DecisionTreeClassifier(parameters)

        # ####### ####### #######
        return classifier


    def get_ml_scoring_metrices(self, scr, reft=None):
        model_scoring_mets = [ML_Performace_Metrics.ACC, ML_Performace_Metrics.PREC, ML_Performace_Metrics.RECL,
                              ML_Performace_Metrics.SEN, ML_Performace_Metrics.SPEC, ML_Performace_Metrics.FPR,
                              ML_Performace_Metrics.FNR, ML_Performace_Metrics.F1, ML_Performace_Metrics.ROC_AUC]

        scoring = [ML_Performace_Metrics.ACC.value]
        bst_mod_mets_1 = None
        i = 0
        for met in self.best_model_scoring_metrics:
            if i==0:
                scoring.clear()
                if (reft is not None):
                    if reft == ML_Performace_Metrics.F1SCR:
                        reft = ML_Performace_Metrics.F1
                    if (reft not in model_scoring_mets):
                        reft = None

            if met == ML_Performace_Metrics.F1SCR:
                met = ML_Performace_Metrics.F1

            if met in model_scoring_mets:
                scoring.append(met.value)
            i += 1

        refit = (scoring[0]) if reft is None else reft.value

        return scoring, refit


    ############################################################################
    def get_parameters_for_ml_models(self, classifier_method):
        parameters = {}
        # Parameter generation method name
        method_name = f'{str(classifier_method.value)}_parameters'

        try:
            method = getattr(self, method_name)
            # Call method for parameter generation
            self.logger.info(f'Calling method: {method_name}')
            parameters = method()
        except AttributeError:
            self.logger.worning(f'No such method exists with the name: {method_name}')
            raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, method_name))

        # ####### ####### #######
        return parameters




############################################################################
    def generate_parameter_dictionary(self, par_names, par_vals, par_ind):
        final_par_names = []
        par_dict = {}

        for i in par_ind:
            pn = par_names[i]
            pv = par_vals[i]
            exec(f'{pn}={pv}')
            final_par_names.append(pn)

        for par in final_par_names:
            par_dict[par] = eval(par)

        return par_dict


    def float_range(self, start, stop, step):
        start = decimal.Decimal(start)
        stop = decimal.Decimal(stop)
        while start < stop:
            yield float(start)
            start *= decimal.Decimal(step)


    # ### ML Classifier Method Parameters
    def random_forest_parameters(self):

        # ### Parameter generation using function
        par_names = ['n_estimators', 'max_depth', 'max_features', 'splitter', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
        par_vals = [list(range(1, 500, 5)),
                    list(range(1, 100)),
                    list(range(2, 20, 1)),
                    ['best', 'random'],
                    list(range(1, 10)),
                    list(range(1, 10)),
                    list(range(1, 100))]

        par_vals = [[30, 50, 75, 100, 200, 500, 750, 1000], [2, 3, 5, 7], [5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000]]
        par_vals = [[5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000]]
        # par_vals = [[2, 3, 5, 7]]
        par_ind = [0]
        parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

        return parameters


    def k_nearest_neighbors_parameters(self):

        # ### Parameter generation using function
        par_names = ['n_neighbors', 'p', 'metric', 'n_splits']
        par_vals = [list(range(2, 100)),
                    list(range(2, 100)),
                    ['manhattan', 'minkowski', 'euclidean'],
                    list(range(2, 10))]

        par_vals = [list(range(100, 1000, 50)), list(range(2, 11, 1)), [2, 3, 5, 9, 13, 19, 29]]
        par_vals = [[2, 3, 5, 9, 13, 19, 29]]
        par_ind = [0]
        parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

        return parameters


    def naive_bias_parameters(self):

        # ### Parameter generation using function
        par_names = ['var_smoothing']
        par_vals = [np.logspace(0,-9, num=100)]

        # par_vals = []
        # par_vals = []
        par_ind = [0]
        parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

        return parameters


    def support_vector_classifier_parameters(self):

        # ### Parameter generation using function
        par_names = ['C', 'gamma', 'kernel', 'degree']
        par_vals = [list(self.float_range('0.000001', '1', '10')),
                    list(self.float_range('0.000001', '1', '10')),
                    ['linear', 'rbf', 'poly', 'sigmoid'],
                    list(range(1, 10))]

        par_vals = [list(self.float_range('0.000001', '1', '10')), list(self.float_range('0.00001', '1', '10')), list(self.float_range('0.0001', '1', '10'))]
        par_vals = [list(self.float_range('0.000001', '1', '10'))]
        par_ind = [0]
        parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

        return parameters


    def decision_tree_parameters(self):

        # ### Parameter generation using function
        par_names = ['max_depth', 'criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
        par_vals = [list(range(1, 100)),
                    ['gini', 'entropy'],
                    ['best', 'random'],
                    list(range(1, 10)),
                    list(range(1, 10)),
                    list(range(1, 100))]

        par_vals = [list(range(1, 100)), list(range(1, 100, 2)), list(range(1, 100, 3))]
        par_vals = [list(range(1, 100))]
        par_ind = [0]
        parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

        return parameters







