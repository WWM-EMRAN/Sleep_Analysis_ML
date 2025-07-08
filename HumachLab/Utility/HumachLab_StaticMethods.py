"""
File Name: HumachLab_StaticMethods.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 1:57 pm
"""

import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import random, decimal

from HumachLab.SignalProcessor import HumachLab_FeatureDetails
from HumachLab.DataManager import HumachLab_DataManager


class HumachLab_StaticMethods:
    logger = None
    def __init__(self):
        return


    ###############################
    ######## EEG management
    @staticmethod
    def get_removable_channels(chans:dict, _removable_dummy_channels, _is_remove_duplicate_channels):
        rem_chans = {}
        ind = 0
        for key, chan in chans.items():
            # sprint(f'Check: , {chan}, {chan in _removable_dummy_channels}')
            sub_chans = (list(chans.values()))[:ind]

            if len(_removable_dummy_channels)>0 and (chan in _removable_dummy_channels):
                rem_chans[key] = chan
            else:
                if (chan in sub_chans):
                    i = sub_chans.index(chan)
                    key2 = (list(chans.keys()))[i]
                    chans[key2] = '{}-{}'.format(chan, 0)
                    chans[key] = '{}-{}'.format(chan, 1)
                    if _is_remove_duplicate_channels:
                        rem_chans[key] = '{}-{}'.format(chan, 1)
                else:
                    found_list = [ch for ch in sub_chans if (ch.count('-')==2) and ch.find(chan)!=-1]
                    if len(found_list)>0:
                        last_found_ch_ind = int((found_list[-1].split('-'))[-1])
                        chans[key] = '{}-{}'.format(chan, (last_found_ch_ind+1))
                        if _is_remove_duplicate_channels:
                            rem_chans[key] = '{}-{}'.format(chan, (last_found_ch_ind+1))
            ind += 1
        return rem_chans


    ###############################
    ######## Signal processing
    @staticmethod
    def get_dataset_columns_containing_feature_names(all_column_names, class_name, ch_col_name, feature_names):
        indx = all_column_names.index(class_name)
        all_extra = all_column_names[:indx + 1]
        all_feats = all_column_names[indx + 1:]
        feat_det_obj = HumachLab_FeatureDetails()
        sel_feats = feature_names

        already_colwise = False if (ch_col_name in all_extra) else True
        selected_feats = []
        print('XXXXXXX', class_name, ch_col_name, feature_names)
        for ff in all_feats:
            for tt in sel_feats:
                ff_lst = ff.split('_')
                if len(ff_lst)==1 and ff==tt:
                    selected_feats.append(ff)
                elif len(ff_lst)>1:
                    if not already_colwise and (tt==('_'.join(ff_lst))):
                        selected_feats.append(ff)
                    elif already_colwise and (tt==('_'.join(ff_lst[:-1]))):
                        selected_feats.append(ff)

        # [ff for ff in all_feats for ii in sel_feats if ff.startswith(f'{ii}_')]

        all_extra.extend(selected_feats)
        return all_extra

    @staticmethod
    def get_dataset_columns_containing_featuregroup_names(all_column_names, class_name, ch_col_name, featuregroup_names):
        indx = all_column_names.index(class_name)
        all_extra = all_column_names[:indx + 1]
        all_feats = all_column_names[indx + 1:]
        feat_det_obj = HumachLab_FeatureDetails()
        sel_feats = []
        for feat_name in featuregroup_names:
            ft = getattr(feat_det_obj, feat_name)
            sel_feats.extend(ft)

        already_colwise = False if (ch_col_name in all_extra) else True
        selected_feats = []
        for ff in all_feats:
            for tt in sel_feats:
                ff_lst = ff.split('_')
                if len(ff_lst) == 1 and ff == tt:
                    selected_feats.append(ff)
                elif len(ff_lst) > 1:
                    if not already_colwise and (tt == ('_'.join(ff_lst))):
                        selected_feats.append(ff)
                    elif already_colwise and (tt == ('_'.join(ff_lst[:-1]))):
                        selected_feats.append(ff)

        all_extra.extend(selected_feats)
        return all_extra

    @staticmethod
    def get_dataset_channel_names(df, logger):
        all_cols = df.columns.values.tolist()
        dataManager = HumachLab_DataManager(logger)
        class_name = dataManager.class_name
        channel_cols = dataManager.channel_cols[1]
        ch_names = []
        if (channel_cols in all_cols):
            ch_names = df[channel_cols].unique()
        else:
            indx = all_cols.index(class_name)
            # all_extra = all_cols[:indx + 1]
            all_feats = all_cols[indx + 1:]
            ch_names = list(set( [((ff.split('_'))[-1]) for ff in all_feats] ))
        del df
        return ch_names


    ###############################
    ######## Util
    @staticmethod
    def float_range(start, stop, step):
        start = decimal.Decimal(start)
        stop = decimal.Decimal(stop)
        while start <= stop:
            yield float(start)
            # start *= decimal.Decimal(step)
            start += decimal.Decimal(step)
        # return

    @staticmethod
    def shuffle_list(data_list, shuffle_times=1):
        shuffled_list = data_list.copy()
        for i in range(shuffle_times):
            random.shuffle(shuffled_list)
        return shuffled_list

    @staticmethod
    def flattening_list(data_list, remove_duplicates=False):
        flat_list = []
        for sublist in data_list:
            for item in sublist:
                if (remove_duplicates and (item not in flat_list)) or not remove_duplicates:
                    flat_list.append(item)
        return flat_list

    @staticmethod
    def is_all_items_of_a_list_in_another_list(data_list, in_list):
        result = all(elem in in_list for elem in data_list)
        return result

    @staticmethod
    def get_common_items_from_two_lists(data_list, other_list):
        result_list = [item for item in data_list if item in other_list]
        not_found_list = [item for item in data_list if item not in other_list]
        return result_list, not_found_list

    @staticmethod
    def get_unique_items_in_the_list(lst):
        unique_lst = []

        for i in lst:
            if i not in unique_lst:
                unique_lst.append(i)

        return unique_lst

    @staticmethod
    def get_list_without_itemlist_in_another_list(data_list, exclude_list):
        result_list = []
        for itemlist in data_list:
            if itemlist != exclude_list:
                result_list.append(itemlist)
        return result_list

    @staticmethod
    def get_list_from_a_list_of_indices(data_list, indices):
        result_list = []
        result_list = [data_list[i] for i in indices]
        return result_list

    @staticmethod
    def min_max_normalizer(sig):
        # sig-nparray
        # for i in range(1, len(sig[0])):
        sig = normalize(sig, norm='max', axis=0)
        return sig


    ###############################
    ######## Classifier performance
#     @staticmethod
#     def get_performance_scores(t, p, target_labels=[0, 1], print_data=True):
#         model_scores = []
#         import numpy as np
#         print(np.unique(np.array(t)), np.unique(np.array(p)))

#         conf_matrix = confusion_matrix(t, p, labels=target_labels).tolist()
#         # print(t, '\n', p, '\n', conf_matrix)
#         model_scores.append(conf_matrix)
#         tn = conf_matrix[0][0]
#         fp = conf_matrix[0][1]
#         fn = conf_matrix[1][0]
#         tp = conf_matrix[1][1]

#         acc = round(accuracy_score(t, p)*100, 2)
#         model_scores.append(acc)
#         prc = round(precision_score(t, p)*100, 2) #precision or positive predictive value (PPV)
#         model_scores.append(prc)
#         rec = round(recall_score(t, p)*100, 2) #sensitivity, recall, hit rate, or true positive rate (TPR)
#         model_scores.append(rec)
#         sns = rec
#         model_scores.append(sns)
#         spc = round((tn / (tn+fp))*100, 2) if (tn+fp)!=0 else round(0.0, 2) #specificity, selectivity or true negative rate (TNR)
#         model_scores.append(spc)
#         # fpr = round(100.0-(fp / (fp+tn))*100, 2) if (fp+tn)!=0 else round(0.0, 2) #fall-out or false positive rate (FPR)
#         fpr = round((fp / (fp+tn))*100, 2) if (fp+tn)!=0 else round(0.0, 2) #fall-out or false positive rate (FPR
#         model_scores.append(fpr)
#         # fnr = round(100.0-(fn / (fn+tp))*100, 2) if (fn+tp)!=0 else round(0.0, 2) #miss rate or false negative rate (FNR)
#         fnr = round((fn / (fn+tp))*100, 2) if (fn+tp)!=0 else round(0.0, 2) #miss rate or false negative rate (FNR)
#         model_scores.append(fnr)
#         f1s = round(f1_score(t, p)*100, 2)
#         model_scores.append(f1s)
        
#         a_fpr, a_tpr, a_thresholds = roc_curve(t, p, pos_label=1)
#         auc_s = round(auc(a_fpr, a_tpr)*100, 2)
#         model_scores.append(auc_s)

#         if print_data:
#             print(
#                 f'Confusion Matrix: \n{conf_matrix}\n'
#                 f'Accuracy (acc): {acc}\n'
#                 f'Precision (prc): {prc}\n'
#                 f'Recall (rec): {rec}\n'
#                 f'Sensitivity (sns): {sns}\n'
#                 f'Specificity (spc): {spc}\n'
#                 f'Inverse False Positive Rate (FPR): {fpr}\n'
#                 f'Inverse False Negative Rate (FNR): {fnr}\n'
#                 f'F1 Score (f1s): {f1s}\n'
#                 f'ROC AUC (AUC): {auc_s}'
#             )
#         return model_scores
    
    
    @staticmethod
    def get_performance_scores(t, p, target_labels=[0, 1], print_data=True):
        model_scores = []
        print(np.unique(np.array(t)), np.unique(np.array(p)))

        conf_matrix = confusion_matrix(t, p, labels=target_labels).tolist()
        # print(t, '\n', p, '\n', conf_matrix)
        model_scores.append(conf_matrix)
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        acc = round(accuracy_score(t, p)*100, 2)
        model_scores.append(acc)
        prc = round(precision_score(t, p)*100, 2) #precision or positive predictive value (PPV)
        model_scores.append(prc)
        rec = round(recall_score(t, p)*100, 2) #sensitivity, recall, hit rate, or true positive rate (TPR)
        model_scores.append(rec)
        sns = rec
        model_scores.append(sns)
        spc = round((tn / (tn+fp))*100, 2) if (tn+fp)!=0 else round(0.0, 2) #specificity, selectivity or true negative rate (TNR)
        model_scores.append(spc)
        # fpr = round(100.0-(fp / (fp+tn))*100, 2) if (fp+tn)!=0 else round(0.0, 2) #fall-out or false positive rate (FPR)
        fpr = round((fp / (fp+tn))*100, 2) if (fp+tn)!=0 else round(0.0, 2) #fall-out or false positive rate (FPR
        model_scores.append(fpr)
        # fnr = round(100.0-(fn / (fn+tp))*100, 2) if (fn+tp)!=0 else round(0.0, 2) #miss rate or false negative rate (FNR)
        fnr = round((fn / (fn+tp))*100, 2) if (fn+tp)!=0 else round(0.0, 2) #miss rate or false negative rate (FNR)
        model_scores.append(fnr)
        f1s = round(f1_score(t, p)*100, 2)
        model_scores.append(f1s)
        
        a_fpr, a_tpr, a_thresholds = roc_curve(t, p, pos_label=1)
        auc_s = round(auc(a_fpr, a_tpr)*100, 2)
        model_scores.append(auc_s)

        if print_data:
            print(
                f'Confusion Matrix: \n{conf_matrix}\n'
                f'Accuracy (acc): {acc}\n'
                f'Precision (prc): {prc}\n'
                f'Recall (rec): {rec}\n'
                f'Sensitivity (sns): {sns}\n'
                f'Specificity (spc): {spc}\n'
                f'Inverse False Positive Rate (FPR): {fpr}\n'
                f'Inverse False Negative Rate (FNR): {fnr}\n'
                f'F1 Score (f1s): {f1s}\n'
                f'ROC AUC (AUC): {auc_s}'
            )
        return model_scores


