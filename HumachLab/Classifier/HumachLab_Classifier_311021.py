# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

#%%

import decimal
import random

from sklearn.model_selection import ShuffleSplit, LeavePOut, ParameterGrid
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from HumachLab import *
# from HumachLab_DataLoader import *
# from HumachLab_Utility import Humachlab_Utility
from HumachLab_DataManager import HumachLab_DataManager
from HumachLab_Preprocessor import HumachLab_Preprocessor
from HumachLab_Enums import ML_Classifiers, ML_Performace_Metrics



# should_balanced_data
#%%


class HumachLab_Classifier:

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

        # self.classifier_methods = ['svm', 'knn', 'nb', 'dt', 'rf']
        self.is_patient_spesific_analysis = True
        self.exp_type = f'patient-wise training and testing'

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
        self.num_fold = validation_splitting_nature
        self.best_model_indices = []
        self.log_path = log_path
        self.test_no = test_no
        self.logger = logger
        # self.class_name = class_name
        self.best_model_scoring_metrics = best_model_scoring_metrics

        self.channels = channels
        self.patients = patients
        self.records = records

        # self.pat_id_col = 'pat_id'
        # self.rec_id_col = 'rec_id'
        # self.seg_id_col = 'seg_id'
        # self.channel_cols = ['ch_id', 'ch_name']
        # self.extra_cols =  [self.pat_id_col, self.rec_id_col, self.seg_id_col]  + self.channel_cols

        self.validation_patients = []
        self.training_patients = []

        self.dataManager = HumachLab_DataManager(logger)
        self.serial_col = self.dataManager.serial_col
        self.pat_id_col = self.dataManager.pat_id_col
        self.rec_id_col = self.dataManager.rec_id_col
        self.seg_id_col = self.dataManager.seg_id_col
        self.channel_cols = self.dataManager.channel_cols
        self.extra_cols =  self.dataManager.extra_cols
        self.class_name = self.dataManager.class_name
        self.pred_col = self.dataManager.pred_col

        self.preprocessor = HumachLab_Preprocessor(logger)
        return


    # #########################################################################
    # Data loading ana management
    # #########################################################################

    # def load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', all_channels_columnwise=False, dataframe=None):
    def set_or_load_all_data(self, path='./', feature_subdirectories='single_channel_feafures', replace_transitions=0, drop_nan=False, all_channels_columnwise=False, dataframe=None, dat_file_name=''):

        class_name = self.class_name
        channels = self.channels
        patients = self.patients
        records = self.records
        df = pd.DataFrame()

        if ((dataframe is None) and (path is None) and (dat_file_name is None)):
            self.logger.worning(f'No data is provided or instructed to retrieve...')
            return
        elif (dataframe is not None):
            df = dataframe
            self.logger.info(f'External data is set... Please check the column name compatibility...')
        elif (dat_file_name is not None):
            # df = self.dataManager.load_external_data(dat_file_name=dat_file_name, drop_nan=drop_nan, all_channels_columnwise=all_channels_columnwise)
            df = self.dataManager.load_external_data(dat_file_name, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise)
            self.logger.info(f'External data is set from a file...')
        else:
            df = self.dataManager.load_all_data_from_feature_directory(path, feature_subdirectories, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise)
            self.logger.info(f'Dataset set from feature files...')

        # Initial data shape
        self.logger.info(f'Initial DataShape: {df.shape}')

        self.dataset = self.dataManager.filter_and_replace_data_in_dataframe(df, patients, records, channels, replace_transitions, drop_nan, all_channels_columnwise)
        df = None
        # self.dataset = self.dataset.fillna(0)
        # self.logger.info(f'AfterProcess DataShape: {self.dataset.shape}')

        self.logger.info(f'DataShape: {self.dataset.shape} '
                         f'\nDataColumns:{list(self.dataset.columns)} '
                         f'\nDataChannels:{self.channels if all_channels_columnwise else self.dataset[self.channel_cols[1]].unique()}')

        return


    # #########################################################################
    # Do Preprocessing
    # #########################################################################

    def do_preprocessing(self, preprocessing_type):
        #Preprocessing
        # new_dataset = None
        if preprocessing_type==1:
            self.dataset, _, _ = self.preprocessor.get_PCA_from_dataframe(self.dataset, self.class_name, num_comp=0, allowed_variance=0.99)
        else:
            self.logger.info(f'No preprocessing type is selected.')
        # self.dataset = new_dataset.copy()
        return


    # #########################################################################
    # Do classification
    # #########################################################################

    def do_classification(self, classification_type, clf_meth, num_fold):
        #Classification
        if classification_type == 1:
            self.patient_specific_testing(classifier_method=clf_meth)
        elif classification_type == 2:
            self.cross_patient_training_and_patient_specific_testing(classifier_method=clf_meth)
        elif classification_type == 3:
            self.cross_patient_training_and_testing(classifier_method=clf_meth, num_fold=num_fold)
        else:
            self.logger.info(f'No classification type is selected.')
        return


    ######################################
    # ### Patient specific indices splitting and feature-target data separator
    # def _get_patientwise_training_test_indices_using_cross_validation(self, patients_list, tr_indices, ts_indices):
    #     tr_ids, ts_ids = None, None
    #     if self.is_patient_spesific_analysis:
    #         pats = np.array(patients_list)
    #         tr_ids = list(pats[tr_indices])
    #         ts_ids = list(pats[ts_indices])
    #         tr_ids.sort()
    #         ts_ids.sort()
    #     return tr_ids, ts_ids


    # def _get_data_from_patient_indices(self, indices):
    #     data = copy.deepcopy(self.dataset)
    #     data = data[data[self.pat_id_col].isin(indices)]
    #     target = data[self.class_name]
    #     extra_cols = data[self.extra_cols]
    #     features = data.drop(self.extra_cols, axis=1)
    #     data = None
    #     return features, target, extra_cols


    #Get fold indices for all data regardless of patient specific analysis
    # def get_fold_indices_for_all_data(self, dat_size, fold=5):
    #     fold_size = dat_size / fold
    #     fold_range = []
    #     for i in range(1, (fold+1)):
    #         if i == fold:
    #             fold_range.append(dat_size)
    #         else:
    #             fold_range.append(int(i * fold_size))
    #     return fold_range


    def _get_training_test_indices_using_cross_validation(self, patients_or_serial_list, tr_indices, ts_indices, force_patient_spesific=False):
        dat_list = np.array(patients_or_serial_list)
        tr_ids = tr_indices
        ts_ids = ts_indices
        if self.is_patient_spesific_analysis or force_patient_spesific:
            tr_ids = list(dat_list[tr_indices])
            ts_ids = list(dat_list[ts_indices])
        tr_ids.sort()
        ts_ids.sort()
        return tr_ids, ts_ids


    def _get_data_from_patient_or_serial_indices(self, indices, force_patient_spesific=False):
        data = copy.deepcopy(self.dataset)
        # print('TTTTTTTTT', data.columns.values.tolist())
        all_cols = data.columns.values.tolist()
        if self.is_patient_spesific_analysis or force_patient_spesific:
            data = data[data[self.pat_id_col].isin(indices)]
        else:
            data = data[data[self.serial_col].isin(indices)]
        target = data[self.class_name].values.tolist()
        extra_cols = []
        ex_col = []
        if self.channel_cols[1] in all_cols:
            ex_col = self.extra_cols
        else:
            ex_col = self.extra_cols[:4]
        extra_cols = data[ex_col]
        features = data.drop(ex_col, axis=1)
        data = None
        return features, target, extra_cols


    def get_fold_serial_indices_for_all_data(self, fold=0, randomness_of_samples=False):
        if fold==0:
            fold = self.cross_validation_rounds
        dat_size = self.dataset.shape[0]
        fold_indices = []
        data_serial = self.dataset[self.serial_col].values.tolist()
        print(len(data_serial))
        if randomness_of_samples:
            data_serial = HumachLab_StaticMethods.shuffle_list(data_serial)
        fold_size = math.ceil(dat_size / fold)
        for i in range(fold):
            tmp_list = []
            si = (i*fold_size) if i > 0 else 0
            ei = dat_size if (i>=(fold-1)) else (si+fold_size)
            self.logger.info(f'{si}, {ei}')
            fold_indices.append(data_serial[si:ei])
        return fold_indices




    # #########################################################################
    # Patient specific analysis
    # #########################################################################

    def patient_specific_testing(self, classifier_method=ML_Classifiers.RF):
        self.is_patient_spesific_analysis = True
        self.exp_type = f'patient-wise training and testing'

        # self.log_path = self.create_log_path(self.log_path_orig)
        self.logger.info(f'\n#################################################################\n##########********** Patient-specific Classification with: {ML_Classifiers.get_short_form(str(classifier_method.value))} **********##########\n#################################################################')

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
        self.best_model_indices = []

        for train_ids, test_ids in splitter.split(unique_pat_ids):
            # train_ids, test_ids = self._get_patientwise_training_test_indices_using_cross_validation(unique_pat_ids, train_ids, test_ids)
            train_ids, test_ids = self._get_training_test_indices_using_cross_validation(unique_pat_ids, train_ids, test_ids)

            # pat_id = test_ids[0]
            # if ii == 5:
            #     break

            util.start_timer()
            self.logger.info(f'#################################################################\n@## {ii} Testing STARTS for patient {test_ids}')
            self.logger.info(f'Total testing:{len(test_ids)}, Total training & validation:{len(train_ids)}')
            self.logger.info(f'Testing:{test_ids}, Training & validation:{train_ids}')
            self.logger.info(f'#################################################################')

            best_model_index = 0
            all_trained_models, all_trained_models_scores, best_model_index = self.patient_specific_training(ii, train_ids, classifier_method)
            model_scores, test_patient = [], {'exp_name': self.exp_name, 'test_patient': test_ids}
            self.best_model_indices.append(best_model_index)

            #Uncomment if only want to see patient splitting
            # continue

            util2 = Humachlab_Utility()
            util2.start_timer()
            for i in range(len(all_trained_models)):
                model = all_trained_models[i]
                model_scrs = test_patient
                # X_test, y_test, extra = self._get_data_from_patient_indices(test_ids)
                X_test, y_test, extra = self._get_data_from_patient_or_serial_indices(test_ids)

                y_pred = model.predict(X_test)

                # Save individual prediction result
                self.save_individual_model_prediction_targets(classifier_method, f'test-{ii}_model-{i}', extra, y_test, y_pred)

                # Add test serial number with extra
                arr = [ii for ttt in range((extra.shape)[0])]
                tst_srl = pd.DataFrame(arr, columns=['test_number'])
                extra.reset_index(drop=True, inplace=True)
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
            self.logger.info(f'### {ii} Testing ENDS for patient {test_ids}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ # Each round of training and testing time for patients {test_ids}: {time_str}\n')
            self.logger.info(f'#################################################################')
            ii += 1

        self.save_and_show_model_scores(all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores)
        self.save_metadata_and_prediction_targets(classifier_method, unique_pat_ids, all_extra, all_y_tests, all_y_preds)

        return


    def patient_specific_training(self, test_iter, training_ids, classifier_method=ML_Classifiers.RF):
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
            # train_ids, val_ids = self._get_patientwise_training_test_indices_using_cross_validation(training_ids, train_ids, val_ids)
            train_ids, val_ids = self._get_training_test_indices_using_cross_validation(training_ids, train_ids, val_ids)


            util.start_timer()
            self.logger.info(f'@** {test_iter},{i} Training and validation STARTS for patient {val_ids}')
            self.logger.info(f'Total validation:{len(val_ids)}, Total training:{len(train_ids)}')
            self.logger.info(f'Validation:{val_ids}, Training:{train_ids}')
            self.logger.info(f'------------------------------------------------------------------')

            if not self.is_validate_models:
                train_ids, val_ids = training_ids, None

            model, model_scores, model_scr = None, {'exp_name': self.exp_name, 'validation_patient': val_ids, 'training_patient': train_ids}, None

            #Uncomment if only want to see patient splitting
            # continue

            # Apply ML model
            model, model_scr, target_and_prediction = self.apply_ml_model(classifier_method, train_ids, val_ids)

            if self.is_validate_models:
                # Save individual prediction result
                self.save_individual_model_prediction_targets(classifier_method, f'test-{test_iter}_validation-{i}', target_and_prediction[0], target_and_prediction[1], target_and_prediction[2])

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

            self.logger.info(f'*** {test_iter},{i} Training and validation ENDS for patient {val_ids}')
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




    # #########################################################################
    # Cross patient analysis
    # #########################################################################

    def cross_patient_training_and_testing(self, classifier_method=ML_Classifiers.RF, num_fold=5, randomness_of_samples=True):
        self.is_patient_spesific_analysis = False
        self.exp_type = f'cross patient training and testing'
        self.num_fold = num_fold

        # self.log_path = self.create_log_path(self.log_path_orig)
        self.logger.info(f'\n#################################################################\n##########********** Cross-patient Classification with: {ML_Classifiers.get_short_form(str(classifier_method.value))} **********##########\n#################################################################')

        if self.dataset is None:
            self.logger.info(f'No data is provided, please try again after setting proper data...')
            return

        # print('total* data: ', self.dataset.shape[0])

        serial_indices_for_folds = self.get_fold_serial_indices_for_all_data(fold=num_fold, randomness_of_samples=randomness_of_samples)
        # print(len(serial_indices_for_folds))
        all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores = [], [], [], [], []
        unique_pat_ids = list(self.dataset[self.pat_id_col].unique())
        all_extra = []
        all_y_tests = []
        all_y_preds = []

        util = Humachlab_Utility()
        ii = 0
        self.best_model_indices = []

        for i, test_ids in enumerate(serial_indices_for_folds):
            # X_data, y_data, extra_cols = self._get_data_from_patient_or_serial_indices(indices)
            train_ids = HumachLab_StaticMethods.get_list_without_itemlist_in_another_list(serial_indices_for_folds, test_ids)
            train_ids = HumachLab_StaticMethods.flattening_list(train_ids)

            # pat_id = test_ids[0]

            util.start_timer()
            self.logger.info(f'#################################################################\n@## {ii} Testing STARTS for data {test_ids[:3]+["..."]}')
            self.logger.info(f'Total testing:{len(test_ids)}, Total training & validation:{len(train_ids)}')
            self.logger.info(f'Testing:{test_ids[:3]+["..."]}, Training & validation:{train_ids[:3]+["..."]}')
            self.logger.info(f'#################################################################')

            best_model_index = 0
            dat_serials = (self.dataset[self.dataset[self.serial_col].isin(train_ids)])[self.serial_col].values.tolist()
            # dat_serials = (self.dataset[self.dataset[self.pat_id_col].isin(train_ids)])[self.serial_col].values.tolist()
            # print(dat_serials, len(self.dataset[self.serial_col]))
            all_trained_models, all_trained_models_scores, best_model_index = self.cross_patient_training(ii, dat_serials, classifier_method)
            model_scores, test_patient = [], {'exp_name': self.exp_name, 'test_patient': test_ids}
            self.best_model_indices.append(best_model_index)

            #Uncomment if only want to see patient splitting
            # continue

            util2 = Humachlab_Utility()
            util2.start_timer()
            for i in range(len(all_trained_models)):
                model = all_trained_models[i]
                model_scrs = test_patient
                # X_test, y_test, extra = self._get_data_from_patient_indices(test_ids)
                X_test, y_test, extra = self._get_data_from_patient_or_serial_indices(test_ids)

                y_pred = model.predict(X_test)

                # Save individual prediction result
                self.save_individual_model_prediction_targets(classifier_method, f'test-{ii}_model-{i}', extra, y_test, y_pred)

                # Add test serial number with extra
                arr = [ii for ttt in range((extra.shape)[0])]
                tst_srl = pd.DataFrame(arr, columns=['test_number'])
                extra.reset_index(drop=True, inplace=True)
                extra = pd.concat([tst_srl, extra], axis=1)
                # print('test##* ', extra)
                # ### Problem: not adding columnwise

                ind = all_trained_models.index(model)
                tr_mod_scr = all_trained_models_scores[ind]
                # self.logger.info(f'$$$-> {ind} {tr_mod_scr['validation_patient']}')
                model_scrs = {**model_scrs, **{'validation_patient': tr_mod_scr['validation_patient'], 'training_patient': tr_mod_scr['training_patient'], 'validation_serial': tr_mod_scr['validation_serial'], 'training_serial': tr_mod_scr['training_serial']}}

                # print('total** test: ', len(y_test))
                # print('total** pred: ', len(y_pred))
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
            self.logger.info(f'@@@ ## Each round of testing time for data {test_ids[:3]+["..."]}: {time_str}\n')

            all_models.append(all_trained_models)
            all_models_training_scores.append(all_trained_models_scores)
            all_models_test_scores.append(model_scores)

            self.logger.info(f'## Best Test Score= {all_best_test_scores[ii]}')
            self.logger.info(f'### {ii} Testing ENDS for patient {test_ids[:3]+["..."]}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ # Each round of training and testing time for data {test_ids[:3]+["..."]}: {time_str}\n')
            self.logger.info(f'#################################################################')
            ii += 1

        # print('total* test: ', len(all_y_tests))
        # print('total* pred: ', len(all_y_preds))
        # print('total*# extras: ', len(all_extra), all_extra)
        self.save_and_show_model_scores(all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores)
        self.save_metadata_and_prediction_targets(classifier_method, unique_pat_ids, all_extra, all_y_tests, all_y_preds)
        return



    def cross_patient_training_and_patient_specific_testing(self, classifier_method=ML_Classifiers.RF):#, num_fold=5, randomness_of_samples=True):
        self.is_patient_spesific_analysis = False
        self.exp_type = f'cross patient training and patient-wise testing'

        # self.log_path = self.create_log_path(self.log_path_orig)
        self.logger.info(f'\n#################################################################\n##########********** Cross-patient Classification with: {ML_Classifiers.get_short_form(str(classifier_method.value))} **********##########\n#################################################################')

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
        self.best_model_indices = []

        for train_ids, test_ids in splitter.split(unique_pat_ids):
            # train_ids, test_ids = self._get_patientwise_training_test_indices_using_cross_validation(unique_pat_ids, train_ids, test_ids)
            train_ids, test_ids = self._get_training_test_indices_using_cross_validation(unique_pat_ids, train_ids, test_ids, force_patient_spesific=True)

            # pat_id = test_ids[0]

            util.start_timer()
            self.logger.info(f'#################################################################\n@## {ii} Testing STARTS for patient {test_ids}')
            self.logger.info(f'Total testing:{len(test_ids)}, Total training & validation:{len(train_ids)}')
            self.logger.info(f'Testing:{test_ids}, Training & validation:{train_ids}')
            self.logger.info(f'#################################################################')

            best_model_index = 0
            dat_serials = (self.dataset[self.dataset[self.pat_id_col].isin(train_ids)])[self.serial_col].values.tolist()
            # dat_serials = (self.dataset[self.dataset[self.pat_id_col].isin(train_ids)])[self.serial_col].values.tolist()
            # serial_indices_for_folds = self.get_fold_serial_indices_for_all_data(fold=num_fold, randomness_of_samples=randomness_of_samples)
            # print(len(serial_indices_for_folds))
            all_trained_models, all_trained_models_scores, best_model_index = self.cross_patient_training(ii, dat_serials, classifier_method)
            model_scores, test_patient = [], {'exp_name': self.exp_name, 'test_patient': test_ids}
            self.best_model_indices.append(best_model_index)

            #Uncomment if only want to see patient splitting
            # continue

            util2 = Humachlab_Utility()
            util2.start_timer()
            for i in range(len(all_trained_models)):
                model = all_trained_models[i]
                model_scrs = test_patient
                # X_test, y_test, extra = self._get_data_from_patient_indices(test_ids)
                X_test, y_test, extra = self._get_data_from_patient_or_serial_indices(test_ids, force_patient_spesific=True)

                y_pred = model.predict(X_test)

                # Save individual prediction result
                self.save_individual_model_prediction_targets(classifier_method, f'test-{ii}_model-{i}', extra, y_test, y_pred)

                # Add test serial number with extra
                arr = [ii for ttt in range((extra.shape)[0])]
                tst_srl = pd.DataFrame(arr, columns=['test_number'])
                extra.reset_index(drop=True, inplace=True)
                extra = pd.concat([tst_srl, extra], axis=1)

                ind = all_trained_models.index(model)
                tr_mod_scr = all_trained_models_scores[ind]
                # self.logger.info(f'$$$-> {ind} {tr_mod_scr['validation_patient']}')
                model_scrs = {**model_scrs, **{'validation_patient': tr_mod_scr['validation_patient'], 'training_patient': tr_mod_scr['training_patient'], 'validation_serial': tr_mod_scr['validation_serial'], 'training_serial': tr_mod_scr['training_serial']}}

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
            self.logger.info(f'### {ii} Testing ENDS for patient {test_ids}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ # Each round of training and testing time for patients {test_ids}: {time_str}\n')
            self.logger.info(f'#################################################################')
            ii += 1

        self.save_and_show_model_scores(all_models, all_best_model, all_models_training_scores, all_models_test_scores, all_best_test_scores)
        self.save_metadata_and_prediction_targets(classifier_method, unique_pat_ids, all_extra, all_y_tests, all_y_preds)
        return


    def cross_patient_training(self, test_iter, dat_serials, classifier_method=ML_Classifiers.RF):
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
                splitter = LeavePOut(p=int(len(dat_serials)*self.validation_split_ratio)) #lpo
        else:
            self.logger.worning(f'Model is not going to be validated, this might cause problem and program crashes...')
            pass

        # print(splitter, len(dat_serials), dat_serials)
        # # Best model and best model scores are recorded
        bst_model_ind = -1
        bst_model_score = None ## Search for best model based on higher F1 score
        i = 0
        scoring = self.best_model_scoring_metrics
        util = Humachlab_Utility()

        for train_ids, val_ids in splitter.split(dat_serials):
            # train_ids, val_ids = self._get_patientwise_training_test_indices_using_cross_validation(training_ids, train_ids, val_ids)
            train_ids, val_ids = HumachLab_StaticMethods.get_list_from_a_list_of_indices(dat_serials, train_ids), HumachLab_StaticMethods.get_list_from_a_list_of_indices(dat_serials, val_ids)

            util.start_timer()
            self.logger.info(f'@** {test_iter},{i} Training and validation STARTS for data {val_ids[:3]+["..."]}')
            self.logger.info(f'Total validation:{len(val_ids)}, Total training:{len(train_ids)}')
            self.logger.info(f'Validation:{val_ids[:3]+["..."]}, Training:{train_ids[:3]+["..."]}')
            self.logger.info(f'------------------------------------------------------------------')

            if not self.is_validate_models:
                train_ids, val_ids = dat_serials, None

            model, model_scores, model_scr = None, {'exp_name': self.exp_name, 'validation_patient': [], 'training_patient': [], 'validation_serial': val_ids, 'training_serial': train_ids}, None

            #Uncomment if only want to see patient splitting
            # continue

            # Apply ML model
            model, model_scr, target_and_prediction = self.apply_ml_model(classifier_method, train_ids, val_ids)

            if self.is_validate_models:
                # Save individual prediction result
                self.save_individual_model_prediction_targets(classifier_method, f'test-{test_iter}_validation-{i}', target_and_prediction[0], target_and_prediction[1], target_and_prediction[2])


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

            self.logger.info(f'*** {test_iter},{i} Training and validation ENDS for data {val_ids[:3]+["..."]}')
            util.end_timer()
            time_str = util.time_calculator()
            self.logger.info(f'@@@ ** Each round of training and validation time for data {val_ids[:3]+["..."]}: {time_str}\n')
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



    # #########################################################################
    # Apply classifier validation methods: Grid, Random, binary search
    # #########################################################################

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
            # tr_ids, ts_ids = self._get_patientwise_training_test_indices_using_cross_validation(tr_ids, ts_ids)
            tr_ids, ts_ids = self._get_training_test_indices_using_cross_validation(tr_ids, ts_ids)

            for pars in param_grid:
                if len(parameters.keys())==0:
                    parameters = None
                mods = self.get_ml_model_instances(classifier_method, parameters)
                # X, y, _ = self._get_data_from_patient_indices(tr_ids)
                # Xt, yt, _ = self._get_data_from_patient_indices(tr_ids)
                X, y, _ = self._get_data_from_patient_or_serial_indices(tr_ids)
                Xt, yt, _ = self._get_data_from_patient_or_serial_indices(tr_ids)

                st = time.time()
                # mods.fit(X, y.values.ravel())
                mods.fit(X, y)
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

        # X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        X_train, y_train, _ = self._get_data_from_patient_or_serial_indices(train_ids)

        # model = model.fit(X_train, y_train.values.ravel())
        model = model.fit(X_train, y_train)

        target_and_prediction = []
        if self.is_validate_models:
            # X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            X_test, y_test, xtra = self._get_data_from_patient_or_serial_indices(val_ids)

            y_pred = model.predict(X_test)

            target_and_prediction.append(xtra)
            target_and_prediction.append(y_test)
            target_and_prediction.append(y_pred)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores, target_and_prediction


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
        # model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=None, verbose=2)
        # model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=50, verbose=2)

        # ### Scoring from custom method
        # score = make_scorer(self.custom_precision_func, greater_is_better=False)
        # # scoring = {'precision': score, 'f1':make_scorer(f1_score)}
        # model = GridSearchCV(mods, parameters, scoring=score, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)

        self.logger.info(f'ML model for training: {model}')

        # X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        X_train, y_train, _ = self._get_data_from_patient_or_serial_indices(train_ids)

        # model = model.fit(X_train, y_train.values.ravel())
        # model = model.fit(X_train, y_train.ravel())
        model = model.fit(X_train, y_train)

        target_and_prediction = []
        if self.is_validate_models:
            # X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            X_test, y_test, xtra = self._get_data_from_patient_or_serial_indices(val_ids)

            y_pred = model.predict(X_test)

            target_and_prediction.append(xtra)
            target_and_prediction.append(y_test)
            target_and_prediction.append(y_pred)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores, target_and_prediction


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

        # X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        X_train, y_train, _ = self._get_data_from_patient_or_serial_indices(train_ids)

        # model = model.fit(X_train, y_train.values.ravel())
        model = model.fit(X_train, y_train)

        target_and_prediction = []
        if self.is_validate_models:
            # X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            X_test, y_test, xtra = self._get_data_from_patient_or_serial_indices(val_ids)

            y_pred = model.predict(X_test)

            target_and_prediction.append(xtra)
            target_and_prediction.append(y_test)
            target_and_prediction.append(y_pred)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores, target_and_prediction


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

        # X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
        X_train, y_train, _ = self._get_data_from_patient_or_serial_indices(train_ids)

        # model = model.fit(X_train, y_train.values.ravel())
        model = model.fit(X_train, y_train)

        target_and_prediction = []
        if self.is_validate_models:
            # X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
            X_test, y_test, xtra = self._get_data_from_patient_or_serial_indices(val_ids)

            y_pred = model.predict(X_test)

            target_and_prediction.append(xtra)
            target_and_prediction.append(y_test)
            target_and_prediction.append(y_pred)

            model_scores = self.calculate_model_scores(model, y_pred, y_test)

        self.logger.info(f'Best model: {model}')
        self.logger.info(f'Best estimator of the model: {model.best_estimator_}')
        self.logger.info(f'Best parameters of the model: {model.best_params_}')

        # if self.should_use_params:
        #     bst_parameters = model.best_params_
        #     self.logger.info(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
        #     model.set_params(**bst_parameters)
        #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        return model, model_scores, target_and_prediction



    # #########################################################################
    # Generalised classifier detail and settings
    # #########################################################################
######################################
# ### All models' implementation

    def apply_ml_model(self, classifier_method, train_ids, val_ids):
        parameters = self.get_parameters_for_ml_models(classifier_method)
        model, model_scores, target_and_prediction = self.call_all_model_optimization(classifier_method, parameters, train_ids, val_ids)
        return model, model_scores, target_and_prediction


    def call_all_model_optimization(self, classifier_method, parameters, train_ids, val_ids):
        model, model_scores, target_and_prediction = None, None, None
        if self.parameter_optimization == 1:
            model, model_scores, target_and_prediction = self.run_model_gridSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 2:
            model, model_scores, target_and_prediction = self.run_model_randomizedSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 3:
            model, model_scores, target_and_prediction = self.run_model_baysianSearch(classifier_method, parameters, train_ids, val_ids)
        elif self.parameter_optimization == 4:
            model, model_scores, target_and_prediction = self.run_model_customGridSearch(classifier_method, parameters, train_ids, val_ids)
        return model, model_scores, target_and_prediction


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


    # def float_range(self, start, stop, step):
    #     start = decimal.Decimal(start)
    #     stop = decimal.Decimal(stop)
    #     while start < stop:
    #         yield float(start)
    #         start *= decimal.Decimal(step)


    # #########################################################################
    # Model parameter settings
    # #########################################################################
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
        par_vals = [list(HumachLab_StaticMethods.float_range('0.000001', '1', '10')),
                    list(HumachLab_StaticMethods.float_range('0.000001', '1', '10')),
                    ['linear', 'rbf', 'poly', 'sigmoid'],
                    list(range(1, 10))]

        par_vals = [list(HumachLab_StaticMethods.float_range('0.000001', '1', '10')), list(HumachLab_StaticMethods.float_range('0.00001', '1', '10')), list(HumachLab_StaticMethods.float_range('0.0001', '1', '10'))]
        par_vals = [list(HumachLab_StaticMethods.float_range('0.000001', '1', '10'))]
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



    # #########################################################################
    # Calculate and save classification details and model scores
    # #########################################################################
#############################

    def calculate_model_scores(self, mods, y_pred, y_test):
        # output_string = ''

        # output_string += f'------------------------------------------------------------------\n{mods.__class__.__name__} '
        perf_scores = HumachLab_StaticMethods.get_performance_scores(y_test, y_pred)
        confMat = perf_scores[0]
        tn = confMat[0][0]
        fp = confMat[0][1]
        fn = confMat[1][0]
        tp = confMat[1][1]

        acc = perf_scores[1]
        prec = perf_scores[2] #precision or positive predictive value (PPV)
        reca_sens = perf_scores[3] #sensitivity, recall, hit rate, or true positive rate (TPR)
        spec = perf_scores[5] #specificity, selectivity or true negative rate (TNR)
        fpr = perf_scores[6] #fall-out or false positive rate (FPR)
        fnr = perf_scores[7] #miss rate or false negative rate (FNR)
        f1sc = perf_scores[8]

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
        self.dataManager.save_dictionary_to_file(all_models, PIK, 'All Models', dump_level=2)

        PIK = f'{self.log_path}/bst_models_{test_no}.dat'
        self.dataManager.save_dictionary_to_file(bst_models, PIK, 'Best Models', dump_level=2)

        # ### Writing scores in text file
        PIK = f'{self.log_path}/tr_score_{test_no}.txt'
        self.dataManager.save_dictionary_to_file(all_training_scrs, PIK, 'Training Scores', dump_level=1)

        PIK = f'{self.log_path}/ts_score_{test_no}.txt'
        self.dataManager.save_dictionary_to_file(all_test_scrs, PIK, 'Testing Scores', dump_level=1)

        PIK = f'{self.log_path}/bst_score_{test_no}.txt'
        self.dataManager.save_dictionary_to_file(bst_test_srcs, PIK, 'Best Testing Scores', dump_level=1)

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
        metadata_dict['is_patient_spesific_analysis'] = self.is_patient_spesific_analysis
        metadata_dict['exp_type'] = self.exp_type
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
        metadata_dict['num_fold'] = self.num_fold
        metadata_dict['classifier_method'] = classifier_method
        metadata_dict['parameters_used'] = self.should_use_params
        metadata_dict['cross_validation_rounds'] = self.cross_validation_rounds
        metadata_dict['validation_patients'] = self.validation_patients
        metadata_dict['training_patients'] = self.training_patients
        metadata_dict['best_model_index'] = self.best_model_indices

        #should_use_params=True, cross_validation_rounds=2, exp_name=f'Test {test_no}: {test_details}', test_no=f'{test_no}', validation_split_ratio=0.3, class_name=

        # ### Saving dictionary using pickle
        self.dataManager.save_dictionary_to_file(metadata_dict, PIK, 'Testing Metadata', dump_level=1)


        # ### Saving prediction target data
        save_file = f'{self.log_path}/bst_prediction_data_{test_no}.csv'

        all_extra_vals, all_pat_id, all_y_tst, all_y_prd = pd.DataFrame(), [], [], []

        for i in range(len(y_tests)):
            extra = all_extra[i]
            y_tst = y_tests[i]
            y_prd = y_preds[i]
            # print('total*** ', len(y_tst), len(y_prd), extra.columns, extra.shape)

            # pat_id = [pat_ids[i] for cc in range(len(y_tst))]
            # self.logger.info(f'{list(y_tst)} \n {list(y_prd)}')
            # all_pat_id += pat_id

            all_extra_vals = pd.concat([all_extra_vals, extra])
            all_y_tst += list(y_tst)
            all_y_prd += list(y_prd)
            # print(len(y_tst), len(y_prd))

        # print('total*#* ', len(all_y_tst), len(all_y_prd), all_extra_vals.columns, all_extra_vals.shape)
        # df = pd.DataFrame(list(zip(all_pat_id, all_y_tst, all_y_prd)), columns=[self.pat_id_col, f'{classifier_method}_y_test', f'{classifier_method}_y_pred'])
        df = all_extra_vals.copy()
        # print(df)
        clf_str = ML_Classifiers.get_short_form(str(classifier_method.value))
        # tmp_df = pd.DataFrame(list(zip(all_y_tst, all_y_prd)), columns=[f'{clf_str}_y_test', f'{clf_str}_y_pred'])
        tmp_df = pd.DataFrame(list(zip(all_y_tst, all_y_prd)), columns=[self.class_name, self.pred_col])
        df.reset_index(drop=True, inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)
        # print(tmp_df)
        df = pd.concat([df, tmp_df], axis=1)
        # print(df)
        # MODIFY CHANGE
        df.sort_values([self.pat_id_col, self.rec_id_col, self.seg_id_col], ascending=[True, True, True], inplace=True)
        # print(df)

        # self.dataManager.save_dictionary_to_file(df, save_file, 'Prediction Result')
        self.dataManager.save_dataframe_to_file(df, save_file, 'Prediction Result')

        return


    def save_individual_model_prediction_targets(self, classifier_method, test_or_val_detail, extra, y_test, y_pred):
        test_no = self.test_no
        save_file = f'{self.log_path}/prediction_data_{test_no}_{test_or_val_detail}.csv'
        tmp_df = pd.DataFrame(list(zip(y_test, y_pred)), columns=[self.class_name, self.pred_col])
        extra.reset_index(drop=True, inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)
        # print(tmp_df)
        extra = pd.concat([extra, tmp_df], axis=1)
        self.dataManager.save_dataframe_to_file(extra, save_file, 'Prediction Result')
        return






