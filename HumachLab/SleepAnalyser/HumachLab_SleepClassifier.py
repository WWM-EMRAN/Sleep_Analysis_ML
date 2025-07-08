
import random
import multiprocessing as mp

from sklearn.model_selection import ShuffleSplit, LeavePOut, KFold, ParameterGrid

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



###########################################################
# HML_ML_CLassifiers

# ### All models' implementation

class HumachLab_SleepClassifier:
    
    def __init__(self, dataset, class_column, split_column, splitting_crieteria, model_list, result_save_path): 
        self.dataset = dataset 
        self.class_column = class_column
        self.split_column = split_column 
        self.splitting_crieteria = splitting_crieteria #splitting_crieteria: =0 -loso, n>0 -n-fold
        self.model_list = model_list 
        self.result_save_path = result_save_path 
        return self 

#     def run_model_gridSearch(self, classifier_method, params, train_ids, val_ids):
#         mods = self.get_ml_model_instances(classifier_method)
#         print(f'------------------------------------------------------------------\nGridSearch: {ML_Classifiers.get_short_form(str(classifier_method.value))} - {params} ')
#         parameters = {}
#         model = mods
#         model_scores = None
#         if self.should_use_params:
#             parameters = params

#         scoring, refit = self.get_ml_scoring_metrices(self.best_model_scoring_metrics, self.best_model_scoring_metrics[0])
        
#         model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=50, verbose=2)
#         # model = GridSearchCV(mods, parameters, scoring=scoring, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=mp.cpu_count(), verbose=2)

#         # ### Scoring from custom method
#         # score = make_scorer(self.custom_precision_func, greater_is_better=False)
#         # # scoring = {'precision': score, 'f1':make_scorer(f1_score)}
#         # model = GridSearchCV(mods, parameters, scoring=score, cv=self.cross_validation_rounds, refit=refit, return_train_score=True, n_jobs=-1, verbose=2)

#         print(f'ML model for training: {model}')

#         # X_train, y_train, _ = self._get_data_from_patient_indices(train_ids)
#         X_train, y_train, _ = self._get_data_from_patient_or_serial_indices(train_ids, from_training=True)

#         # model = model.fit(X_train, y_train.values.ravel())
#         # model = model.fit(X_train, y_train.ravel())
#         X_train = np.nan_to_num(X_train)
#         model = model.fit(X_train, y_train)

#         target_and_prediction = []
#         if self.is_validate_models:
#             # X_test, y_test, _ = self._get_data_from_patient_indices(val_ids)
#             X_test, y_test, xtra = self._get_data_from_patient_or_serial_indices(val_ids)

#             y_pred = model.predict(X_test)

#             target_and_prediction.append(xtra)
#             target_and_prediction.append(y_test)
#             target_and_prediction.append(y_pred)

#             model_scores = self.calculate_model_scores(model, y_pred, y_test)

#         print(f'Best model: {model}')
#         print(f'Best estimator of the model: {model.best_estimator_}')
#         print(f'Best parameters of the model: {model.best_params_}')

#         # if self.should_use_params:
#         #     bst_parameters = model.best_params_
#         #     print(f'&&&&&&&&&&&&&&&&&&&&&& {bst_parameters}')
#         #     model.set_params(**bst_parameters)
#         #     # model = GridSearchCV(mods, parameters, scoring=scoring, verbose=2)

#         return model, model_scores, target_and_prediction

#     def apply_ml_model(self, classifier_method, train_ids, val_ids):
#         parameters = self.get_parameters_for_ml_models(classifier_method)
#         model, model_scores, target_and_prediction = self.call_all_model_optimization(classifier_method, parameters, train_ids, val_ids, parameter_optimization=1)
#         return model, model_scores, target_and_prediction


#     def call_all_model_optimization(self, classifier_method, parameters, train_ids, val_ids, parameter_optimization):
#         model, model_scores, target_and_prediction = None, None, None
#         if parameter_optimization == 1:
#             model, model_scores, target_and_prediction = self.run_model_gridSearch(classifier_method, parameters, train_ids, val_ids)
#         elif parameter_optimization == 2:
#             model, model_scores, target_and_prediction = self.run_model_randomizedSearch(classifier_method, parameters, train_ids, val_ids)
#         elif parameter_optimization == 3:
#             model, model_scores, target_and_prediction = self.run_model_baysianSearch(classifier_method, parameters, train_ids, val_ids)
#         elif parameter_optimization == 4:
#             model, model_scores, target_and_prediction = self.run_model_customGridSearch(classifier_method, parameters, train_ids, val_ids)
#         return model, model_scores, target_and_prediction


#     def get_ml_model_instances(self, classifier_method, parameters=None):
#         classifier = None

#         ### GPU code START
#         global GPUs
#         global HAS_GPU

#         # GPUs = GPUtil.getGPUs()
#         # tot_gpus = len(GPUs)
#         # HAS_GPU = True if len(GPUs) > 0 else False
#         # avl_GPUIDs = GPUtil.getAvailable(order = 'first', limit = tot_gpus, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
#         # tot_avl_gpus = len(avl_GPUIDs)
#         # print(f'For GPU based tasks. There are {tot_gpus} GPUs in the system and {tot_avl_gpus} are available. \nAvailable GPU IDs are: {avl_GPUIDs}')
#         allGPUs, bestGPU = HumachLab_Global.get_gpu_details(show_logs=False)
#         ### GPU code END

#         # ####### rf #######
#         # rf - random_forest classifier
#         if classifier_method == ML_Classifiers.RF:
#             classifier = RandomForestClassifier() if (parameters is None) else RandomForestClassifier(parameters)
#         # ####### knn #######
#         # knn - k_neares_neighbours classifier
#         elif classifier_method == ML_Classifiers.kNN:
#             classifier = KNeighborsClassifier() if (parameters is None) else KNeighborsClassifier(parameters)
#         # ####### nb #######
#         # knn - naieve bias classifier
#         elif classifier_method == ML_Classifiers.NB:
#             classifier = GaussianNB() if (parameters is None) else GaussianNB(parameters)
#         # ####### svm/svc #######
#         # knn - support vector classifier
#         elif classifier_method == ML_Classifiers.SVC:
#             classifier = SVC() if (parameters is None) else SVC(parameters)
#         # ####### knn #######
#         # knn - k_neares_neighbours classifier
#         elif classifier_method == ML_Classifiers.DT:
#             classifier = DecisionTreeClassifier() if (parameters is None) else DecisionTreeClassifier(parameters)

#         ### GPU code - Comment it if no gpu available or not linux system or no support for RapidsAI package
#         # ####### gpu-rf #######
#         # gpu-rf - gpu-random_forest classifier
#         # elif classifier_method == ML_Classifiers.GPURF and tot_avl_gpus>0:
#         #     classifier = gpuRandomForestClassifier() if (parameters is None) else gpuRandomForestClassifier(parameters)

#         # ####### None #######
#         # No classifier
#         else:
#             print(f'No classifier is selected...')

#         # ####### ####### #######
#         return classifier


#     def get_ml_scoring_metrices(self, scr, reft=None):
#         model_scoring_mets = [ML_Performace_Metrics.ACC, ML_Performace_Metrics.PREC, ML_Performace_Metrics.RECL,
#                               ML_Performace_Metrics.SEN, ML_Performace_Metrics.SPEC, ML_Performace_Metrics.FPR,
#                               ML_Performace_Metrics.FNR, ML_Performace_Metrics.F1, ML_Performace_Metrics.ROC_AUC]

#         scoring = [ML_Performace_Metrics.ACC.value]
#         bst_mod_mets_1 = None
#         i = 0
#         for met in self.best_model_scoring_metrics:
#             if i==0:
#                 scoring.clear()
#                 if (reft is not None):
#                     if reft == ML_Performace_Metrics.F1SCR:
#                         reft = ML_Performace_Metrics.F1
#                     if (reft not in model_scoring_mets):
#                         reft = None

#             if met == ML_Performace_Metrics.F1SCR:
#                 met = ML_Performace_Metrics.F1

#             if met in model_scoring_mets:
#                 scoring.append(met.value)
#             i += 1

#         refit = (scoring[0]) if reft is None else reft.value

#         return scoring, refit


#     ############################################################################
#     def get_parameters_for_ml_models(self, classifier_method):
#         parameters = {}
#         if not self.should_use_params:
#             return parameters

#         # Parameter generation method name
#         method_name = f'{str(classifier_method.value)}_parameters'

#         try:
#             method = getattr(self, method_name)
#             # Call method for parameter generation
#             print(f'Calling method: {method_name}')
#             parameters = method()
#         except AttributeError:
#             self.logger.worning(f'No such method exists with the name: {method_name}')
#             raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, method_name))

#         # ####### ####### #######
#         return parameters




#     ############################################################################
#     def generate_parameter_dictionary(self, par_names, par_vals, par_ind):
#         print(f'All parameters: {par_names}, {par_vals}, {par_ind}')
#         final_par_names = []
#         par_dict = {}

#         for i in par_ind:
#             pn = par_names[i]
#             pv = par_vals[i]
#             exec(f'{pn}={pv}')
#             final_par_names.append(pn)

#         for par in final_par_names:
#             par_dict[par] = eval(par)

#         return par_dict


#     # def float_range(self, start, stop, step):
#     #     start = decimal.Decimal(start)
#     #     stop = decimal.Decimal(stop)
#     #     while start < stop:
#     #         yield float(start)
#     #         start *= decimal.Decimal(step)


#     # #########################################################################
#     # Model parameter settings
#     # #########################################################################
#     # ### ML Classifier Method Parameters
#     def graphics_processing_unit_random_forest(self):

#         # ### Parameter generation using function
#         par_names = ['n_estimators', 'n_bins', 'n_streams', 'max_depth', 'max_features', 'splitter', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
#         par_vals = [list(range(1, 500, 5)),
#                     list(range(1, 100)),
#                     list(range(2, 20, 1)),
#                     ['best', 'random'],
#                     list(range(1, 10)),
#                     list(range(1, 10)),
#                     list(range(1, 100))]

#         par_vals = [[30, 50, 75, 100, 200, 500, 750, 1000], [2, 3, 5, 7], [5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000]]
#         par_vals = [[5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000], 15, 8]
#         # par_vals = [[2, 3, 5, 7]]
#         par_ind = [0, 1, 2]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters


#     def random_forest_parameters(self):

#         # ### Parameter generation using function
#         par_names = ['n_estimators', 'max_depth', 'max_features', 'splitter', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
#         par_vals = [list(range(1, 500, 5)),
#                     list(range(1, 100)),
#                     list(range(2, 20, 1)),
#                     ['best', 'random'],
#                     list(range(1, 10)),
#                     list(range(1, 10)),
#                     list(range(1, 100))]

#         par_vals = [[30, 50, 75, 100, 200, 500, 750, 1000], [2, 3, 5, 7], [5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000]]
#         par_vals = [[5, 7, 11, 15, 21, 30, 50, 75, 100, 200, 500, 750, 1000]]
#         par_vals = [[15, 21, 30, 50, 75, 100, 200, 500]]
#         par_vals = [[50, 75, 100]]
#         par_vals = [[15, 21, 30, 50, 75]]
#         # par_vals = [[2, 3, 5, 7]]
#         par_ind = [0]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters


#     def k_nearest_neighbors_parameters(self):

#         # ### Parameter generation using function
#         par_names = ['n_neighbors', 'p', 'metric', 'n_splits']
#         par_vals = [list(range(2, 100)),
#                     list(range(2, 100)),
#                     ['manhattan', 'minkowski', 'euclidean'],
#                     list(range(2, 10))]

#         par_vals = [list(range(100, 1000, 50)), list(range(2, 11, 1)), [2, 3, 5, 9, 13, 19, 29]]
#         par_vals = [[2, 3, 5, 9, 13, 19, 29]]
#         par_ind = [0]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters


#     def naive_bias_parameters(self):

#         # ### Parameter generation using function
#         par_names = ['var_smoothing']
#         par_vals = [np.logspace(0,-9, num=100)]
#         par_vals = [np.logspace(0, -9, num=100)]

#         # par_vals = []
#         # par_vals = []
#         par_ind = [0]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters


#     def support_vector_classifier_parameters(self):

#         # ### Parameter generation using function
#         par_names = ['C', 'kernel', 'gamma', 'degree', 'class_weightdict']
#         par_vals = [list(HumachLab_StaticMethods.float_range('0.001', '1', '0.01')),
#                     ['linear', 'rbf', 'poly', 'sigmoid'],
#                     list(HumachLab_StaticMethods.float_range('0.000001', '1', '10')),
#                     list(range(1, 10)),
#                     [None, 'balanced']]

#         # par_vals = [list(HumachLab_StaticMethods.float_range('0.000001', '1', '10')), list(HumachLab_StaticMethods.float_range('0.00001', '1', '10')), list(HumachLab_StaticMethods.float_range('0.0001', '1', '10'))]
#         par_vals = [list(HumachLab_StaticMethods.float_range('0.001', '1.', '0.1')), ['linear', 'rbf', 'poly']]
#         par_vals = [[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], ['linear', 'rbf', 'poly']]
#         par_ind = [0, 1]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters


#     def decision_tree_parameters(self):

#         # ### Parameter generation using function
#         par_names = ['max_depth', 'criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
#         par_vals = [list(range(1, 100)),
#                     ['gini', 'entropy'],
#                     ['best', 'random'],
#                     list(range(1, 10)),
#                     list(range(1, 10)),
#                     list(range(1, 100))]

#         par_vals = [list(range(1, 100)), list(range(1, 100, 2)), list(range(1, 100, 3))]
#         par_vals = [list(range(1, 100))]
#         par_ind = [0]
#         parameters = self.generate_parameter_dictionary(par_names, par_vals, par_ind)

#         return parameters



#     # #########################################################################
#     # Calculate and save classification details and model scores
#     # #########################################################################
#     #############################

#     def calculate_model_scores(self, mods, y_pred, y_test):
#         # output_string = ''

#         # output_string += f'------------------------------------------------------------------\n{mods.__class__.__name__} '
#         perf_scores = HumachLab_StaticMethods.get_performance_scores(y_test, y_pred)
#         confMat = perf_scores[0]
#         tn = confMat[0][0]
#         fp = confMat[0][1]
#         fn = confMat[1][0]
#         tp = confMat[1][1]

#         acc = perf_scores[1]
#         prec = perf_scores[2] #precision or positive predictive value (PPV)
#         reca_sens = perf_scores[3] #sensitivity, recall, hit rate, or true positive rate (TPR)
#         spec = perf_scores[5] #specificity, selectivity or true negative rate (TNR)
#         fpr = perf_scores[6] #fall-out or false positive rate (FPR)
#         fnr = perf_scores[7] #miss rate or false negative rate (FNR)
#         f1sc = perf_scores[8]

#         # scr_dict = {'model_name': str(mods.__class__.__name__), 'confusion_matrix': confMat, 'accuracy': acc, 'precision': prec, 'recall': reca_sens, 'sensitivity': reca_sens, 'specificity': spec, 'f1_score': f1sc}
#         scr_dict = {'model_name': str(mods.__class__.__name__), 'method_name': str(mods.estimator), 'method_parameters': str(mods.best_params_), 'method_scores': str(round(mods.best_score_*100,2)),
#                     ML_Performace_Metrics.CONF_MAT.value: confMat, ML_Performace_Metrics.ACC.value: acc, ML_Performace_Metrics.PREC.value: prec,
#                     ML_Performace_Metrics.RECL.value: reca_sens, ML_Performace_Metrics.SEN.value: reca_sens, ML_Performace_Metrics.SPEC.value: spec,
#                     ML_Performace_Metrics.FPR.value: fpr, ML_Performace_Metrics.FNR.value: fnr, ML_Performace_Metrics.F1SCR.value: f1sc}

#         print(f'{scr_dict}')

#         return scr_dict
    
    
    
    