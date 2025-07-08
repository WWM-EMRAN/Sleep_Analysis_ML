"""
File Name: HumachLab_Enums.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 27/07/2021 3:26 am
"""

import enum


class HumachLab_Enums(enum.Enum):
    pass


class All_Datasets(enum.Enum):
    amd      = 'AMD'
    alfred   = 'Alfred'
    chbmit   = 'CHBMIT'
    siena    = 'Siena'
    cap_sleep   = 'CAP_Sleep'
    sleep_edfx   = 'Sleep_EDFX' 
    sdrc   = 'SDRC'
    isruc   = 'ISRUC' 


class ML_Splitting_crieteria(enum.Enum):
    n_fold      = 'n-fold'
    f_fold      = 'five-fold'
    t_fold      = 'ten-fold'
    loso   = 'Leave-one-subject-out' 


class ML_Classifiers(enum.Enum):
    LogReg  = 'logistic_regression'
    SVC  = 'support_vector_classifier'
    SVCR  = 'support_vector_classifier_rbf'
    SVCP  = 'support_vector_classifier_poly'
    NB   = 'naive_bayes'
    GPC = 'gaussian_process_classifier' 
    kNN  = 'k_nearest_neighbors'
    DT   = 'decision_tree'
    RF   = 'random_forest'
    GBoost   = 'gradient_boosting'
    XGBoost   = 'xtreme_gradient_boosting'
    GPURF   = 'graphics_processing_unit_random_forest'
    MLP   = 'multi_layer_perceptron'
    STCK  = 'stacking_classifier'

    def get_short_form(name):
        short_name = ''
        name = name.split('_')
        name2 = [i[0] for i in name]
        short_name = short_name.join(name2).upper()
        return short_name

    def get_actual_name(name): 
        name_list = ['LogReg', 'SVC', 'SVCR', 'SVCP', 'NB', 'GPC', 'kNN', 'DT', 'RF', 'GBoost', 'XGBoost', 'MLP', 'STCK']
        actual_name_list = ['LogisticRegression', 'SVC', 'SVC_RBF', 'SVC_Poly', 'GaussianNB', 'GaussianProcessClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'MLPClassifier', 'StackingClassifier'] 
        actual_name = actual_name_list[name_list.index(name)]  
        return actual_name

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


    CONF_MAT_MACRO = 'confusion_matrix'
    ACC_MACRO      = 'accuracy'
    PREC_MACRO     = 'precision_marco'             #precision or positive predictive value (PPV)
    RECL_MACRO     = 'recall_marco'                #sensitivity, recall, hit rate, or true positive rate (TPR)
    SEN_MACRO      = 'sensitivity'           #sensitivity, recall, hit rate, or true positive rate (TPR)
    SPEC_MACRO     = 'specificity'           #specificity, selectivity or true negative rate (TNR)
    FPR_MACRO      = 'false_positive_rate'   #fall-out or false positive rate (FPR)
    FNR_MACRO      = 'false_negative_rate'   #miss rate or false negative rate (FNR)
    F1SCR_MACRO    = 'f1_score_marco'

    F1_MACRO       = 'f1_marco'
    ROC_AUC_MACRO  = 'roc_auc'




