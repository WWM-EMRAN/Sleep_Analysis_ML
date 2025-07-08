"""
File Name: HumachLab_CHBMIT_Enums.py 
Author: Emran Ali
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 21/05/2021 3:55 pm
"""

import enum


class HumachLab_CHBMIT_Enums(enum.Enum):
    pass


class ML_Classifiers(enum.Enum):
    SVC  = 'support_vector_classifier'
    NB   = 'naive_bias'
    kNN  = 'k_nearest_neighbors'
    DT   = 'decision_tree'
    RF   = 'random_forest'

    def get_short_form(name):
        short_name = ''
        name = name.split('_')
        name2 = [i[0] for i in name]
        short_name = short_name.join(name2).upper()
        return short_name

# scr_dict = {'model_name': str(mods.__class__.__name__), 'method_name': str(mods.estimator), 'method_parameters': str(mods.estimator_params), 'confusion_matrix': confMat,
# 'accuracy': acc,
#                     'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec,
                    # 'false_positive_rate': fpr, 'false_negative_rate': fnr, 'f1_score': f1sc}
class ML_Performace_Metrics(enum.Enum):
    CONF_MAT = 'confusion_matrix'
    ACC      = 'accuracy'
    PREC     = 'precision'             #precision or positive predictive value (PPV)
    RECL     = 'recall'                #sensitivity, recall, hit rate, or true positive rate (TPR)
    SEN      = 'sensitivity'           #sensitivity, recall, hit rate, or true positive rate (TPR)
    SPEC     = 'specificity'           #specificity, selectivity or true negative rate (TNR)
    FPR      = 'false_positive_rate'   #fall-out or false positive rate (FPR)
    FNR      = 'false_negative_rate'   #miss rate or false negative rate (FNR)
    F1SCR    = 'f1_score'

    F1       = 'f1'
    ROC_AUC  = 'roc_auc'




