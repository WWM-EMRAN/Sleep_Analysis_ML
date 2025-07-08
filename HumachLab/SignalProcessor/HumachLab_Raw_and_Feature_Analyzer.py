# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""
#%%
import os
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
# %matplotlib inline
import statsmodels.api as sm
from sklearn.utils import resample
import mne
from matplotlib import interactive
interactive(True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
np.random.seed(123)
# %matplotlib



class HumachLab_Raw_and_Feature_Analyzer():

    def __init__(self):
        return


    #%% Fundamental methods: load_data, up/downsample_data, show_class_distribution


    def load_data(self, path, channels, patient=0):
        # patient = ':02d'.format(patient)
        df = pd.DataFrame()
        path2 = os.walk(path)

        for root, directories, files in path2:
            # for directory in directories:
            #     print(directory)
            for file in files:
                #             print(file)
                items = file.split('_')
                pat = int(items[1])
                ch_name = (items[4].split('.'))[0]
                #             print(ch_name)

                if patient == 0 or (patient != 0 and patient == pat):
                    if len(channels) == 0:
                        #                     print(file)
                        file = path + f'/{file}'
                        extr_feat = pd.read_csv(file)
                        #                     print(extr_feat)
                        df = df.append(extr_feat, ignore_index=True)

                    elif len(channels) > 0 and (ch_name in channels):
                        #                     print(file)
                        file = path + f'/{file}'
                        extr_feat = pd.read_csv(file)
                        #                     print(extr_feat)
                        df = df.append(extr_feat, ignore_index=True)

        #     print(df)
        df = df[df.seizureState < 2]

        return df


    def downsample_data(self, df, class_name):
        df_majority = df[df[class_name] == 0]
        df_minority = df[df[class_name] == 1]
        number_of_datapoints = len(df_minority.index)

        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample without replacement
                                           n_samples=number_of_datapoints,  # to match minority class
                                           random_state=123)  # reproducible results

        # Combine minority class with downsampled majority class
        df = pd.concat([df_majority_downsampled, df_minority])

        # Shuffling the dataframe
        df = df.sample(frac=1)

        return df


    def upsample_data(self, df, class_name):
        df_majority = df[df[class_name] == 0]
        df_minority = df[df[class_name] == 1]
        number_of_datapoints = len(df_majority.index)

        df_majority_downsampled = resample(df_minority,
                                           replace=False,  # sample without replacement
                                           n_samples=number_of_datapoints,  # to match minority class
                                           random_state=123)  # reproducible results

        # Combine minority class with downsampled majority class
        df = pd.concat([df_majority_downsampled, df_majority])

        # Shuffling the dataframe
        df = df.sample(frac=1)

        return df


    def show_class_distribution(self, df, class_name):
        class_distribution = df[class_name].value_counts().to_dict()
        print(class_distribution)

        return class_distribution


    #%% Feature selection methods


    def get_correlation_values(self, df, class_name, with_features=0, corr_threshold=0.50, varbose=0):
        # Using Pearson Correlation
        corr_mat = df.corr()

        if with_features == 0:
            # Correlation with output variable
            corr_target = corr_mat[class_name]

            feats = list(corr_target.index)
            corrs = list(corr_target.values)
            corrs_abs = list(abs(corr_target.values))

            data_dict = {'features': feats, 'correlation_with_target_abs': corrs_abs}
            if varbose == 1:
                data_dict = {'features': feats, 'correlation_with_target': corrs,
                             'correlation_with_target_abs': corrs_abs}
            corr_mat = pd.DataFrame(data_dict)
            corr_mat = corr_mat[corr_mat['features'] != class_name]
            corr_mat = corr_mat.reset_index()
            corr_mat = corr_mat.drop(columns=['index'])

            corr_mat = corr_mat.sort_values(by=['correlation_with_target_abs'])
            corr_mat = corr_mat[corr_mat['correlation_with_target_abs'] >= corr_threshold]

        return corr_mat


    def feature_selection_based_on_correlation_values(self, df, class_name, corr_threshold=0.90):
        corr_mat = self.show_correlation_values(df, class_name, with_features=0, corr_threshold=corr_threshold, varbose=0)
        relevant_features = corr_mat['features']

        return relevant_features


    def get_highly_correlated_features_to_remove(self, df, class_name, feature_similarity_threshold=0.90):
        #     Y = df[class_name]
        X = df.drop(columns=[class_name])
        #     columns = x.columns.tolist()

        # Create correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > feature_similarity_threshold)]

        return to_drop


    def feature_selection_with_backward_elimination_based_on_pvalue(self, df, class_name, pvalue_threshold=0.05, varbose=0):
        y = df[class_name]
        X = df.drop(columns=[class_name])

        # Backward Elimination
        cols = list(X.columns)
        pmax = 1
        while (len(cols) > 0):
            p = []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y, X_1).fit()
            p = pd.Series(model.pvalues.values[1:], index=cols)

            if varbose > 1:
                print(cols, '\n', p, '\n')

            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if (pmax > pvalue_threshold):

                if varbose > 0:
                    print('=====', feature_with_p_max, pmax)

                cols.remove(feature_with_p_max)
            else:
                break

        selected_features_BE = cols

        return selected_features_BE #(selected_features_BE, list(set(X.columns) - set(selected_features_BE)))


    def feature_selection_with_forward_selection_based_on_pvalue(self, df, class_name, pvalue_threshold=0.05, varbose=0):
        y = df[class_name]
        X = df.drop(columns=[class_name])
        selected_features_FS = []

        # Forward selection
        cols = list(X.columns)
        pmax = 0
        while (len(cols) > 0):
            p = []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y, X_1).fit()
            p = pd.Series(model.pvalues.values[1:], index=cols)

            if varbose > 1:
                print(cols, '\n', p, '\n')

            pmin = min(p)
            feature_with_p_min = p.idxmin()
            if (pmin < pvalue_threshold):

                if varbose > 0:
                    print('=====', feature_with_p_min, pmin)

                selected_features_FS.append(feature_with_p_min)
                cols.remove(feature_with_p_min)
            else:
                break

        return selected_features_FS #(selected_features_FS, list(set(X.columns) - set(selected_features_FS)))


    def feature_selection_with_performance_measure_based_on_roc_value(self, df, class_name, test_size=0.3,
                                                                      get_selected_feats=0):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.feature_selection import VarianceThreshold

        y = df[class_name]
        X = df.drop(columns=[class_name])
        columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # remove constant and quasi constant features
        constant_filter = VarianceThreshold(threshold=0.01)
        constant_filter.fit(X_train)
        X_train_filter = constant_filter.transform(X_train)
        X_test_filter = constant_filter.transform(X_test)

        #     #remove duplicate features
        #     X_train_T = X_train_filter.T
        #     X_test_T = X_test_filter.T
        #     X_train_T = pd.DataFrame(X_train_T)
        #     X_test_T = pd.DataFrame(X_test_T)

        # remove duplicate features
        X_train_T = X_train.T
        X_test_T = X_test.T
        X_train_T = pd.DataFrame(X_train_T)
        X_test_T = pd.DataFrame(X_test_T)

        duplicated_features = X_train_T.duplicated()

        features_to_keep = [not index for index in duplicated_features]
        X_train_unique = X_train_T[features_to_keep].T
        X_test_unique = X_test_T[features_to_keep].T

        roc_auc = []
        for feature in X_train_unique.columns:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X_train_unique[feature].to_frame(), y_train)
            y_pred = clf.predict(X_test_unique[feature].to_frame())
            roc_auc.append(roc_auc_score(y_test, y_pred))

        #     roc_values = pd.Series(roc_auc)
        #     roc_values.index = X_train_unique.columns
        #     roc_values.sort_values(ascending =False, inplace = True)

        indx = X_train_unique.columns
        relative_auc = [i if (i >= 0.50) else (1.0 - i) for i in roc_auc]

        data_dict = {'features': indx, 'AUC': roc_auc, 'relativeAUC': relative_auc}
        roc_values = pd.DataFrame(data_dict)
        roc_values.sort_values(by=['relativeAUC'], ascending=False, inplace=True)

        sel_feats = roc_values
        print(sel_feats)

        if get_selected_feats == 1:
            sel_feats = roc_values[roc_values['relativeAUC'] >= 0.51]

        #     X_train_roc = X_train_unique[sel_feats.index]
        #     X_test_roc = X_test_unique[sel_feats.index]
        #     Now best prediction can be achieved from X_train_roc, X_test_roc, y_train, y_test

        return sel_feats


    #%% Visualization methods:


    def show_correlation_heatmap(self, df):
        # Using Pearson Correlation
        plt.figure(figsize=(15, 15))
        cor = df.corr()
        sb.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

        return


    def show_distribution_plot(self, df, class_name):
        Y = df[class_name]
        x = df.drop(columns=[class_name])
        columns = x.columns.tolist()

        # plt.figure(find)

        fig = plt.figure(figsize=(20, 25))
        j = 0
        for i in columns:
            plt.subplot(6, 4, j + 1)
            j += 1
            sb.distplot(x[i][Y == 0], color='g', label='non-seizure')
            sb.distplot(x[i][Y == 1], color='r', label='seizure')
            plt.legend(loc='best')
        fig.suptitle('Seizure Data Analysis')
        fig.tight_layout()
        # fig.subplots_adjust(top=0.95)
        plt.show()

        return


    # def show_raw_or_feature_eeg_with_seizure(self):
    #     from matplotlib import interactive
    #     interactive(True)
    #
    #     x1 = list(x.flatten())
    #     y1 = list(y.flatten())
    #
    #     seg = list([i for i in range(len(y1))])
    #     colrs = ['red' if s == 1 else 'blue' for s in x1]
    #     dict1 = {'Segments': seg, 'Values': y1, 'Colors': colrs}
    #     ndf = pd.DataFrame(dict1)
    #
    #     plt.plot(y1)
    #     plt.plot([i for i in range(1, 20)], y1[1:20], 'r-')
    #     plt.show()
    #
    #     return


    def _get_the_list_of_seizures(self, x, seizure_states):
        seizures = []

        indices2 = [i for i, x in enumerate(seizure_states) if x == 2]
        indices3 = [i for i, x in enumerate(seizure_states) if x == 3]

        for i2 in indices2:
            i3 = indices3[indices2.index(i2)]
            seizures.append(x[i2:(i3 + 1)])

        return indices2, indices3, seizures


    def show_raw_or_feature_eeg_with_seizure(self, df, class_name, features=[]):

        y = df[class_name]
        X = df.drop(columns=[class_name])
        cols = X.columns.tolist()

        if len(features) > 0:
            cols = features

        #     print(cols)
        find = 1

        for col in cols:
            x1 = list(X[col].values.flatten())
            y1 = list(y.values.flatten())

            indices2, indices3, seizures = self._get_the_list_of_seizures(x1, y1)

            #         print(seizures)

            # plt.rcParams["figure.figsize"] = (36, 12)
            # fig = plt.figure(figsize=(36, 12))
            # plt.figure(find, figsize=(36, 12))
            plt.figure(figsize=(36, 12))
            plt.title(col)
            plt.plot(x1)

            for seiz in seizures:
                indx = seizures.index(seiz)
                #             plt.figure(indx)
                plt.plot([i for i in range(indices2[indx], (indices3[indx] + 1))], seiz, 'r-')

            plt.show()
            find += 1

        return












