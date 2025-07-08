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
import decimal, math
from datetime import datetime
import json
from scipy import stats


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier





#%%


class HumachLab_CHBMIT_Classifier_BU1:

    dataset = None
    should_use_params = False
    cross_validation_rounds = 10
    log_serial_no = 0
    log_path = ''
    should_balanced_data = False
    exp_name = 'test'
    training_ratio = 0.6


    def __init__(self, log_path='./', should_use_params=False, cross_validation_rounds=5, should_balanced_data=0, exp_name='test'):
        self.dataset = pd.DataFrame()
        self.should_use_params = should_use_params
        self.cross_validation_rounds = cross_validation_rounds
        self.log_json_data = {'models':[]}
        self.log_path = log_path
        self.exp_name = exp_name
        self.training_ratio = 0.6
        '''
        should_balanced_data 
        0: Nothing to do
        1: Downsample dataset
        2: Upsample dataset
        3: Balance training dataset only
        9: Balance training dataset only with n fold random selection 
        '''
        self.should_balanced_data = should_balanced_data

        path2 = os.walk(self.log_path)
        self.log_serial_no = 0
        for root, directories, files in path2:
            max_fs = 0
            for file in files:
                # print(file)
                items = file.split('_')
                val2 = items[-1].split('.')
                val = int(val2[0])
                if max_fs < val:
                    max_fs = val

            self.log_serial_no = max_fs+1

        return


    def load_data(self, path, patient=0):
        # patient = ':02d'.format(patient)
        path2 = os.walk(path)

        for root, directories, files in path2:
            # for directory in directories:
            #     print(directory)
            for file in files:
                # print(file)
                items = file.split('_')
                pat = int(items[1])

                if patient==0 or (patient!=0 and patient==pat) :
                    file = path+f'/{file}'
                    extr_feat = pd.read_csv(file)
                    self.dataset = self.dataset.append(extr_feat, ignore_index=True)

        # print('======>', self.dataset.head())

        # nulls_df = self.dataset.isnull().sum().to_frame(name='null_count')
        # print(nulls_df)
        # self.dataset.fillna(0, inplace=True)
        # self.dataset.fillna(self.dataset.median(), inplace=True)
        # self.dataset = np.nan_to_num(self.dataset)
        self.dataset = self.dataset[self.dataset.seizureState < 2]

        # self.dataset = self.dataset.iloc[:550000, :]
        # self.dataset = self.dataset.iloc[550000:, :]
        # self.dataset = self.dataset.iloc[550000:600000, :]
        # self.dataset = self.dataset.iloc[600000:, :]
        # self.dataset = self.dataset.iloc[550000:560000, :]
        # self.dataset.to_csv('problematic_data2.csv', index=False)

        # Feature selection
        # self.select_important_features()
        # exit(1)

        # Data splitting, resampling, and balancing
        X, y = self.preprocess_data()

        self.log_json_data['experiment_name'] = self.exp_name
        self.log_json_data['all_data_dimension'] = self.dataset.shape
        self.log_json_data['all_feature_dimension'] = X.shape
        self.log_json_data['all_target_dimension'] = y.shape

        self.log_json_data['train_feature_dimension'] = self.X_train.shape
        self.log_json_data['train_target_dimension'] = self.y_train.shape
        self.log_json_data['test_feature_dimension'] = self.X_test.shape
        self.log_json_data['test_target_dimension'] = self.y_test.shape

        print(self.log_json_data, self.y_train.columns.tolist(), self.y_test.columns.tolist())

        self.log_json_data['all_target_frequency'] = self.dataset.seizureState.value_counts().to_dict()
        self.log_json_data['train_target_frequency'] = self.y_train.seizureState.value_counts().to_dict()
        self.log_json_data['test_target_frequency'] = self.y_test.seizureState.value_counts().to_dict()

        # testVals = self.y_test.seizureState.value_counts()
        # print(testVals)

        return


    def select_important_features(self):
        print('Here comes...')

        pearsoncorr = self.dataset.corr(method='pearson')
        sb.heatmap(pearsoncorr,
                   xticklabels=pearsoncorr.columns,
                   yticklabels=pearsoncorr.columns,
                   cmap='Spectral_r',
                   annot=True,
                   linewidth=0.5)
        plt.title('Correlation matrix')

        # Convert precip_type from string to int so it can be Categorical ['rain' 'snow'] = [0 1]
        self.dataset.seizureState = pd.factorize(self.dataset.seizureState)[1]
        print('Pearson correlation coefficient')
        for x in range(len(self.dataset.columns)):
            if x != 1:
                print(self.dataset.columns[1], 'and', self.dataset.columns[x]
                      , stats.pearsonr(self.dataset.iloc[:, 1], self.dataset.iloc[:, x])[0])

        return


    def preprocess_data(self):
        # ## All preprocessing goes here
        # ## Downsampling the data to solve class imbalance
        df_majority = self.dataset[self.dataset.seizureState == 0]
        df_minority = self.dataset[self.dataset.seizureState == 1]
        number_of_datapoints = len(df_minority.index)

        X = y = 0

        if self.should_balanced_data==1:
            # Downsample majority class
            df_majority_downsampled = resample(df_majority,
                                               replace=False,  # sample without replacement
                                               n_samples=number_of_datapoints,  # to match minority class
                                               random_state=123)  # reproducible results

            # Combine minority class with downsampled majority class
            self.dataset = pd.concat([df_majority_downsampled, df_minority])
            X = self.dataset.iloc[:, 1:]
            y = self.dataset.iloc[:, :1]
            X = X.astype('float64')
            y = y.astype('int')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1.0-self.training_ratio), random_state=0)

        elif self.should_balanced_data==2:
            # Upsample minority class
            number_of_datapoints = len(df_majority.index)
            df_minority_upsampled = resample(df_minority,
                                             replace=False,  # sample without replacement
                                             n_samples=number_of_datapoints,  # to match minority class
                                             random_state=123)  # reproducible results

            # Combine minority class with downsampled majority class
            self.dataset = pd.concat([df_minority_upsampled, df_majority])
            X = self.dataset.iloc[:, 1:]
            y = self.dataset.iloc[:, :1]
            X = X.astype('float64')
            y = y.astype('int')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1.0-self.training_ratio), random_state=0)

        # ## All preprocessing goes here - part 2
        # For training test splitting
        elif self.should_balanced_data==3:
            # Balanced training and imballence test
            # 70% of read seizure data will be used for training
            tmp_Train, tmp_Test = np.split(df_minority, [int(0.6*len(df_minority))])
            number_of_datapoints = len(tmp_Train.index)

            train_df_majority_downsampled = resample(df_majority,
                                                     replace=False,  # sample without replacement
                                                     n_samples=number_of_datapoints)  # to match minority class
                                                     # random_state=123)  # reproducible results
            # Combine minority class with downsampled majority class
            tmp_Train = pd.concat([train_df_majority_downsampled, tmp_Train])
            tst_data = df_majority.drop(train_df_majority_downsampled.index)
            tmp_Test = pd.concat([tst_data, tmp_Test])

            tst_data = pd.concat([tmp_Train, tmp_Test])
            X = tst_data.iloc[:, 1:]
            y = tst_data.iloc[:, :1]

            self.X_train = tmp_Train.iloc[:, 1:]
            self.y_train = tmp_Train.iloc[:, :1]
            self.X_test = tmp_Test.iloc[:, 1:]
            self.y_test = tmp_Test.iloc[:, :1]

        # n fold balanced training and inbalanced testing
        elif self.should_balanced_data==9:
            # 70% of read seizure data will be used for training
            tmp_Train, tmp_Test = np.split(df_minority, [int(self.training_ratio*len(df_minority))])
            number_of_datapoints = len(tmp_Train.index)

            train_df_majority_downsampled = resample(df_majority,
                                                     replace=False,  # sample without replacement
                                                     n_samples=self.cross_validation_rounds*number_of_datapoints)  # to match minority class
                                                     # random_state=123)  # reproducible results
            # Combine minority class with downsampled majority class
            tmp_Train = pd.concat([train_df_majority_downsampled, tmp_Train])
            # Shuffling the dataframe
            tmp_Train = tmp_Train.sample(frac=1)
            tst_data = df_majority.drop(train_df_majority_downsampled.index)
            tmp_Test = pd.concat([tst_data, tmp_Test])

            tst_data = pd.concat([tmp_Train, tmp_Test])
            X = tst_data.iloc[:, :-1]
            y = tst_data.iloc[:, -1:]

            self.X_train = tmp_Train.iloc[:, 1:]
            self.y_train = tmp_Train.iloc[:, :1]
            self.X_test = tmp_Test.iloc[:, 1:]
            self.y_test = tmp_Test.iloc[:, :1]

        else:
            X = self.dataset.iloc[:, 1:]
            y = self.dataset.iloc[:, :1]
            X = X.astype('float64')
            y = y.astype('int')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1.0-self.training_ratio), random_state=0)

        return X, y


    def load_and_show_model_scores(self, log_load_paths=''):

        if not log_load_paths:
            log_load_paths = self.log_path

        path2 = os.walk(log_load_paths)

        for root, directories, files in path2:
            for file in files:
                # print(file)
                if file.find('log')>=0 and file.find('.json')>=0:
                    print(f'\n\n------------------------------------------------------------------\n{file}\n------------------------------------------------------------------\n')
                    file = log_load_paths+file
                    with open(file, 'r') as fileObj:
                        data = fileObj.read()
                        # print(type(data), data)
                        data_dict = json.loads(data)
                        for key, val in data_dict.items():
                            if key=='models':
                                for mdl in val:
                                    print('\n------------------\n')
                                    for key2, val2 in mdl.items():
                                        if key2=='confusion_matrix':
                                            print(f'{key2}\n{val2[0]}\n{val2[1]}')
                                        else:
                                            print(key2, ': ', val2)
                                print('\n------------------\n')
                            else:
                                print(key, ': ', val)

        return


    def run_model_gridSearch(self, mods, params):
        print('\n------------------------------------------------------------------\n', mods.__class__.__name__, ': \n')
        parameters = {}
        if self.should_use_params:
            parameters = params

        # scoring = ['accuracy', 'precision', 'recall']
        # scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        scoring = 'recall'

        model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit='recall', return_train_score=True, verbose=2)
        # model = mods

        model.fit(self.X_train, self.y_train.values.ravel())

        if self.should_use_params:
            parameters = model.best_params_
            model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

        y_pred = model.predict(self.X_test)

        self.show_model_scores(mods, y_pred)

        return


    def show_model_scores(self, mods, y_pred):
        output_string = ''

        output_string += f'\n------------------------------------------------------------------\n{mods.__class__.__name__} \n'
        confMat = confusion_matrix(self.y_test, y_pred).tolist()
        acc = round(accuracy_score(y_pred, self.y_test)*100, 2)
        prec = round(precision_score(y_pred, self.y_test)*100, 2)
        reca_sens = round(recall_score(y_pred, self.y_test)*100, 2)
        f1sc = round(f1_score(y_pred, self.y_test)*100, 2)
        spec = round((confMat[0][0] / (confMat[0][0]+confMat[1][0]))*100, 2)

        print(type(confMat))

        scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc, 'precition': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec, 'f1_score': f1sc}
        self.log_json_data['models'].append(scr_dict)


        output_string += f'\nOriginal Total Values (seizureState): \n{self.dataset.seizureState.value_counts()}'
        output_string += f'\nOriginal Training Values (seizureState): \n{self.y_train.seizureState.value_counts()}'
        output_string += f'\nOriginal Test Values (seizureState): \n{self.y_test.seizureState.value_counts()}'
        output_string += f'\nConfusion Matrix:\n{confMat[0]} \n{confMat[1]}'
        output_string += f'\nAccuracy Score: {acc}'
        output_string += f'\nPrecision: {prec}'
        output_string += f'\nRecall or Sensitivity: {reca_sens}'
        output_string += f'\nSpecificity: {spec}'
        output_string += f'\nF1 Score: {f1sc}'
        output_string += f'\n------------------------------------------------------------------\n'

        print(output_string)

        return


    def write_model_scores(self):

        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)

        log_file_name = f'{self.log_path}logfile_{self.log_serial_no}.json'
        print(log_file_name, self.log_json_data)

        with open(log_file_name, 'a') as writer:
            writer.write(json.dumps(self.log_json_data))
            print('Data is written successfully to: ', log_file_name)

        # # You can read from json file like this
        # with open('file2.json', 'r') as file:
        #     d = file.read()
        #     print(type(d), d)
        #     data_dict = json.loads(d)
        #     print('\n\n', data_dict['model_scores'][1]['f1_score'])

        return


    #### Call models...
    def detect_seizure(self):

        round = 1

        if self.should_balanced_data==9:
            round = self.cross_validation_rounds
            self.log_serial_no -= 1

        new_train_X = np.array_split(self.X_train, round)
        new_train_y = np.array_split(self.y_train, round)

        for i in range(round):
            if self.should_balanced_data==9:
                self.log_json_data = {'models': []}

            self.X_train = new_train_X[i]
            self.y_train = new_train_y[i]

            #Different Models
            # self.naive_bias()
            # self.support_vector_machine()
            # self.neares_neighbours()
            # self.decision_tree()
            self.random_forest()

            if self.should_balanced_data==9:
                self.log_json_data['experiment_name'] = self.exp_name+f'-{i+1}'
                self.log_json_data['all_data_dimension'] = self.dataset.shape
                self.log_json_data['all_feature_dimension'] = self.X_train.shape+self.X_test.shape
                self.log_json_data['all_target_dimension'] = self.y_train.shape+self.y_test.shape

                self.log_json_data['train_feature_dimension'] = self.X_train.shape
                self.log_json_data['train_target_dimension'] = self.y_train.shape
                self.log_json_data['test_feature_dimension'] = self.X_test.shape
                self.log_json_data['test_target_dimension'] = self.y_test.shape

                self.log_json_data['all_target_frequency'] = self.dataset.seizureState.value_counts().to_dict()
                self.log_json_data['train_target_frequency'] = self.y_train.seizureState.value_counts().to_dict()
                self.log_json_data['test_target_frequency'] = self.y_test.seizureState.value_counts().to_dict()

                self.log_serial_no += 1

            self.write_model_scores()

        return


    def float_range(self, start, stop, step):
        start = decimal.Decimal(start)
        stop = decimal.Decimal(stop)
        while start < stop:
            yield float(start)
            start *= decimal.Decimal(step)


    #### Models...
    def naive_bias(self):
        parameters = {}

        self.run_model_gridSearch(GaussianNB(), parameters)
        return


    def support_vector_machine(self):
        c_range = list(self.float_range('0.000001', '1', '10'))
        gamma_range = list(self.float_range('0.000001', '1', '10'))
        kernal_range = ['linear', 'rbf', 'poly', 'sigmoid']
        degree = list(range(1, 10))

        parameters = {'C': c_range, 'gamma': gamma_range, 'kernel': kernal_range, 'degree': degree}

        self.run_model_gridSearch(SVC(random_state=42), parameters)
        return


    def neares_neighbours(self):
        k_range = list(range(1, 100))
        metric = ['manhattan', 'minkowski','euclidean']

        parameters = {'n_neighbors':k_range, 'metric': metric}

        self.run_model_gridSearch(KNeighborsClassifier(), parameters)
        return


    def decision_tree(self):
        depth_range = list(range(1, 100))
        criterion = ['gini', 'entropy']
        splitter = ['best', 'random']
        min_samples_split = list(range(1, 10))
        min_samples_leaf = list(range(1, 10))
        max_leaf_nodes = list(range(1, 100))
        parameters = {'max_depth': depth_range, 'criterion': criterion, 'splitter': splitter, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_leaf_nodes':max_leaf_nodes}

        self.run_model_gridSearch(DecisionTreeClassifier(random_state = 42), parameters)
        return


    def random_forest(self):
        n_estimator_range = list(range(1, 500, 5))
        depth_range = list(range(1, 100))
        feature_range = list(range(2, 20, 1))
        splitter = ['best', 'random']
        min_samples_split = list(range(1, 10))
        min_samples_leaf = list(range(1, 10))
        max_leaf_nodes = list(range(1, 100))

        parameters = {'n_estimators': n_estimator_range, 'max_depth': depth_range, 'max_features': feature_range, 'splitter': splitter, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_leaf_nodes':max_leaf_nodes}

        self.run_model_gridSearch(RandomForestClassifier(), parameters)
        return







