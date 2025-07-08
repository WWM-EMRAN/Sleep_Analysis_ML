'''
######################################
Importing necessary modules
'''
import PSA_Imports 
import numpy as np 
import pandas as pd 
import itertools as it
from functools import reduce

from scipy.stats import ranksums, mannwhitneyu 
from scipy.stats import ttest_ind
from scipy.stats import f_oneway 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelBinarizer 
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

# import os, sys 
# import glob, re, copy 
# import numpy as np 
# import pandas as pd 
# # import matplotlib 
# # import matplotlib.pyplot as plt 




'''
######################################
Read and convert a transition matrix to one data sample     
'''
def convert_columnwise_data_to_rowwise(file): 
    #file = './Results/CAP_Sleep/Transition_Matrices/sub_tran_brux1_transition2_count.csv'
    
    tdf = pd.read_csv(file, index_col='From') 
    tdf
    
    indx = tdf.index.values.tolist()
    indx

    cols = tdf.columns.values.tolist()
    cols

    new_lst = list(it.product(indx, cols))
    new_lst

    new_cols = reduce(lambda lst, tpl: lst + [f"{tpl[0]}->{tpl[1]}"], new_lst, []) 
    new_cols
    
    arr = tdf.values
    arr

    arr = arr.reshape(1, arr.shape[0]*arr.shape[1]).tolist() 

    reshaped_tdf = pd.DataFrame(arr, columns=[new_cols]) 
    return reshaped_tdf




'''
######################################
Read and convert a transition matrix from dataframe to one data sample     
'''
def convert_columnwise_data_to_rowwise_from_dataframe(tdf): 
    all_col = tdf.columns.tolist() 
    if "From" in all_col:
        tdf = tdf.set_index("From") 
        
    indx = tdf.index.values.tolist()
    indx

    cols = tdf.columns.values.tolist()
    cols

    new_lst = list(it.product(indx, cols))
    new_lst 

    new_cols = reduce(lambda lst, tpl: lst + [f"{tpl[0]}->{tpl[1]}"], new_lst, []) 
    new_cols 
    # print('SMILE--->', new_cols) 
    
    arr = tdf.values
    arr

    arr = arr.reshape(1, arr.shape[0]*arr.shape[1]).tolist() 

    reshaped_tdf = pd.DataFrame(arr, columns=[new_cols]) 
    return reshaped_tdf



'''
######################################
Combine all samples to one dataset of similar features 
'''
def get_final_feature_data_from_transition_matrixes(all_file_names, fl_type): 
    all_tran_df = pd.DataFrame()

    for i, file in enumerate(all_file_names):
        components = file.split('/') 
        components = [components[2], components[-1]] 
        components = [components[0], components[-1].split('.')[0].split('_')] 
        print(i, file, components) 
        
        # result_type_subdirectory -=- ['All', 'Category', 'Subject', 'Subject_OneNight', 'Record']  
        # ---------------------------- ['Dataset', 'Category', 'Subject', 'Subject', 'Record']
        dataset = components[0]  
        info_type = components[1][:-3] 
        print('===>>>', fl_type, info_type)
        if fl_type=='all': 
            meta_data = {'Dataset':dataset}
        if fl_type=='cat': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1]}
        if fl_type=='subj': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2]}
        if fl_type=='rec': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2], 'Record':info_type[3]}
        if fl_type=='subj_': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2]}
            # fl_type = 'rec'
            if int(info_type[3])>1:
                continue 
        
        tdf = convert_columnwise_data_to_rowwise(file)  
        for i, (k,v) in enumerate(meta_data.items()):
            # print(k,[v]*tdf.shape[0]) 
            tdf.insert(i, k, [v]*tdf.shape[0]) 
            
        all_tran_df = pd.concat([all_tran_df, tdf], ignore_index=True) 

    return all_tran_df 



'''
######################################
Combine all samples to one dataset of similar features from dataset 
'''
def get_final_feature_data_from_transition_matrixes_from_dataset(all_mat_dict, fl_type): 
    all_tran_df = pd.DataFrame()

    for i, (key, mat_df) in enumerate(all_mat_dict.items()):        
        # result_type_subdirectory -=- ['All', 'Category', 'Subject', 'Subject_OneNight', 'Record']  
        # ---------------------------- ['Dataset', 'Category', 'Subject', 'Subject', 'Record']
        info_type = key.split(" ")
        dataset = info_type[0]  
        # print('===>>>', fl_type, info_type)
        if fl_type=='all': 
            meta_data = {'Dataset':dataset}
        if fl_type=='cat': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1]}
        if fl_type=='subj': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2]}
        if fl_type=='rec': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2], 'Record':info_type[3]}
        if fl_type=='subj_': 
            meta_data = {'Dataset':dataset, 'Category':info_type[1], 'Subject':info_type[2]}
            # fl_type = 'rec'
            if int(info_type[3])>1:
                continue 
        
        tdf = convert_columnwise_data_to_rowwise_from_dataframe(mat_df)  
        for i, (k,v) in enumerate(meta_data.items()):
            # print(k,[v]*tdf.shape[0]) 
            tdf.insert(i, k, [v]*tdf.shape[0]) 
            
        all_tran_df = pd.concat([all_tran_df, tdf], ignore_index=True) 

    return all_tran_df 



'''
######################################
Create class (label) from the category (sleep disorder type) 
'''
def map_category_to_class(dat_set, source_col='Category', class_name='Class', removable_cats=None, binary_class=True): 
    if class_name in dat_set.columns.tolist():
        dat_set = dat_set.drop(columns=[class_name])
    dat_set.insert(3, class_name, dat_set[source_col].values) 
    dat_set

    cat_val = dat_set[source_col].unique().tolist() 
    cat_val.remove('n')
    cat_val.insert(0, 'n')
    print(cat_val) 
    
    if removable_cats:
        cat_val = [c for c in cat_val if c not in removable_cats]
        dat_set = dat_set[dat_set[source_col].isin(cat_val)]
        dat_set.reset_index(drop=True, inplace=True)
    print(cat_val) 
        
    cls_map = dict(zip(cat_val, list(range(len(cat_val))))) 
    cls_map
    
    if binary_class:
        for k in cls_map.keys():
            if cls_map[k]>1:
                cls_map[k]=1

    dat_set.replace({class_name: cls_map}, inplace=True) 
    return cls_map, dat_set 



'''
######################################
Calculate P-value and AUC for a feature (one feature) 
'''
def calculate_p_and_auc_for_feature(feat_data, label_data, binary_class=True): 
    # Extract the independent variable and dependent variable
    X = feat_data.copy()  # Replace 'independent_variable' with your column name
    y = label_data.copy()  # Replace 'dependent_variable' with your column name
    # print(X, y) 
    # print("111 Binary classification?", binary_class)

    # Perform a one-way ANOVA and calculate the p-value
    p_value = 1.0
    if binary_class:
        _, p_valueX = ttest_ind(X[y==0], X[y==1])  # Assuming binary classification 
        # print("222 Binary classification?", binary_class)
    else: 
        groups = [X[y == label] for label in np.unique(y)] # For multiclass classification 
        _, p_valueX = f_oneway(*groups)
        # print("222 Not binary classification?", binary_class)
    p_value = p_valueX[0] if (p_valueX[0] is not None) else 1.0 

    # Display the p-value
    # print("P-value:", p_value)

    # Encode the target variable - For multiclass 
    if not binary_class: 
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        # print("333 Not binary classification?", binary_class)

    # Fit a logistic regression model and calculate the AUC
    # print("333.111 Binary classification?", binary_class)
    model = None 
    #if binary_class:
    #    model = LogisticRegression()
    #else:
    #    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # model = SVC(C=1.0, random_state=1, kernel='linear', probability=True)
    model = SVC(C=1.0, random_state=1, kernel='poly', gamma='auto', probability=True)
    # print("333.222 Binary classification?", model, X[:5], y[:5])
    model.fit(X, y)
    # print("333.333 Binary classification?", model)
    y_pred_proba = model.predict_proba(X)
    if binary_class: 
        y_pred_proba = y_pred_proba[:, 1]
        # print("444 Binary classification?", binary_class)

    # print(y_pred_proba) 
    auc = 0.0 
    if binary_class:
        aucX = roc_auc_score(y, y_pred_proba)
        # print("555 Binary classification?", binary_class)
    else:
        aucX = roc_auc_score(y, y_pred_proba, multi_class='ovr')
        # print("555 Not binary classification?", binary_class)
    auc = aucX if (aucX is not None) else 0.50 

    # Display the AUC
    # print("AUC:", auc)
    return p_value, auc 



'''
######################################
Calculate P-value and AUC for a dataset with many features  
'''
def calculate_p_and_auc_for_dataset(df, label_col, binary_class=True): 
    all_cols = df.columns.values.tolist() 
    
    feat_cols = [f for f in all_cols if f!=label_col]
#     print(feat_cols)
    
    all_p_list = [] 
    all_auc_list = [] 
    for ft in feat_cols:
#         print(ft)
        feat_data = df[[ft]] 
        label_data = df[label_col]
        p, auc = calculate_p_and_auc_for_feature(feat_data, label_data, binary_class=binary_class) 
        all_p_list.append(p) 
        all_auc_list.append(auc) 
        
#     all_p_and_auc_df = pd.DataFrame( {"Features": feat_cols, f"P_Value_{'bin' if binary_class else 'multi'}": all_p_list, f"AUC_{'bin' if binary_class else 'multi'}": all_auc_list} ) 
    all_p_and_auc_df = pd.DataFrame( {"Features": feat_cols, f"P_Value": all_p_list, f"AUC": all_auc_list} )  
    all_p_and_auc_df['AUC'] = all_p_and_auc_df['AUC'].where(all_p_and_auc_df['AUC']>=0.5, 1.0-all_p_and_auc_df['AUC'])
    return all_p_and_auc_df 



'''
######################################
Calculate statistical importance test: t-test, u-test 
'''
def get_statistical_significant_using_wilcoxon_and_mannwhitney_u_test(tmp_df, class_name, binary_class=True):  #for two/binary class only 
    unique_class = sorted(tmp_df[class_name].unique().tolist()) 
    feats = tmp_df.columns.values.tolist()[1:] 

    stat_significance_df = pd.DataFrame(columns=['Features', 'Wilcoxon_zscore', 'Wilcoxon_pvalue', 'MannWhitney_statistic', 'MannWhitney_pvalue']) 

    for f in feats:
        fd_0 = tmp_df[ (tmp_df[class_name]==0) ][f].values
        fd_1 = tmp_df[ (tmp_df[class_name]==1) ][f].values
    #     print("==>", f, fd_0.shape, fd_1.shape)
    #     print("==>", f, fd_0.shape, fd_1.shape, np.mean(fd_0), np.mean(fd_1))
        stat_value, p_value = ranksums(fd_0, fd_1)
    #     stat_value2, p_value2 = mannwhitneyu(fd_0, fd_1, use_continuity=True, alternative=None) 
    #     stat_value2, p_value2 = 0.0, 1.0 if ( (np.mean(fd_0)==0) and (np.mean(fd_0)==np.mean(fd_1))==True) else mannwhitneyu(fd_0, fd_1, use_continuity=True, alternative=None) 
        stat_value2, p_value2 = 0.0, 1.0 
        if not ( (np.mean(fd_0)==0) and (np.mean(fd_0)==np.mean(fd_1))==True):
#             print('NNNN--->', np.mean(fd_0), np.mean(fd_1)) 
            if np.mean(fd_0)!=np.mean(fd_1):                 
                res = mannwhitneyu(fd_0, fd_1, use_continuity=True, alternative=None) 
                stat_value2, p_value2 = res.statistic, res.pvalue 
    #     print(f, fd_0.shape, fd_1.shape, stat_value, p_value, stat_value2, p_value2) 
    #     print(f, fd_0.shape, fd_1.shape, stat_value, p_value, stat_value2, p_value2, p_value<0.05, p_value2<0.05, ((p_value<0.05)==(p_value2<0.05)) ) 
    #     print(f, p_value2<0.05, ((p_value<0.05)==(p_value2<0.05)) ) 
        new_row = {'Features':f, 'Wilcoxon_zscore':stat_value, 'Wilcoxon_pvalue':p_value, 'MannWhitney_statistic':stat_value2, 'MannWhitney_pvalue':p_value2}
        stat_significance_df = stat_significance_df.append(new_row, ignore_index=True)
    #     tdf = pd.DataFrame(new_row) 
    #     stat_significance_df = pd.concat([stat_significace_df, tdf]) 
    if binary_class==False:
        stat_significance_df[['Wilcoxon_zscore', 'Wilcoxon_pvalue', 'MannWhitney_statistic', 'MannWhitney_pvalue']] = np.nan 
    return stat_significance_df



'''
######################################
Calculate statistical measures: mean, std 
'''
def get_mean_SD_values(tmp_df, class_name, binary_class=True):   #for two/binary class only 
    unique_class = sorted(tmp_df[class_name].unique().tolist())
    feats = tmp_df.columns.values.tolist()[1:] 

    stat_df = pd.DataFrame(columns=['Features', 'Sum', 'Mean', 'STD', 'Healthy_Sum', 'Healthy_Mean', 'Healthy_STD', 'Disorder_Sum', 'Disorder_Mean', 'Disorder_STD']) 

    for f in feats:
        val = tmp_df.loc[:, [class_name,f]]
        fd_0 = val[ (val[class_name]==0) ][f].values
        fd_1 = val[ (val[class_name]==1) ][f].values
        
        new_row = {'Features':f, 'Sum':np.sum(val[f].values), 'Mean':np.mean(val[f].values), 'STD':np.std(val[f].values), 
                   'Healthy_Sum':np.sum(fd_0), 'Healthy_Mean':np.mean(fd_0), 'Healthy_STD':np.std(fd_0), 
                   'Disorder_Sum':np.sum(fd_1), 'Disorder_Mean':np.mean(fd_1), 'Disorder_STD':np.std(fd_1)} #round(hlth_df[f].sum(),6)
        stat_df = stat_df.append(new_row, ignore_index=True)
    if binary_class==False:
        stat_df[['Healthy_Sum', 'Healthy_Mean', 'Healthy_STD', 'Disorder_Sum', 'Disorder_Mean', 'Disorder_STD']] = np.nan 
    return stat_df



'''
######################################
Calculate MI (mutual information), Normalised MI (NMI) and Entropy for a dataset with many features  
'''
def get_MI_NMI_and_entropy_for_dataset(tmp_df, class_name, binary_class=True):   #for two/binary class only 
    unique_class = sorted(tmp_df[class_name].unique().tolist())
    feats = tmp_df.drop(columns=[class_name]).values 
    target = tmp_df[class_name].values 
    feature_names = tmp_df.drop(columns=[class_name]).columns 
    is_binary = len(np.unique(target)) == 2 
    
    # print('===>>>', feats.shape, target.shape, feature_names) 
    print('===>>>', binary_class, is_binary)  

    # mi_stat_df = pd.DataFrame(columns=['Features', 'MI', 'Entropy', 'NMI']) 

    # Mutual Information (classification only)
    mi_scores = mutual_info_classif(feats, target, discrete_features=False)

    # Entropy
    entropies = []
    for i in range(feats.shape[1]):
        hist, _ = np.histogram(feats[:, i], bins=10, density=True)
        ent = entropy(hist + 1e-9, base=2)
        entropies.append(ent)

    # Normalized Mutual Information
    nmi_scores = mi_scores / np.array(entropies)

    mi_stat_df = pd.DataFrame({
        'Features': feature_names, 
        'MI': mi_scores,
        'Entropy': entropies,
        'NMI': nmi_scores
    })
    
    # if binary_class==False:
    #     mi_stat_df[['MI', 'Entropy', 'NMI']] = np.nan 
    return mi_stat_df



'''
######################################
Calculate and return all statistical measures and test 
'''
def get_all_statistical_information(tmp_df, class_name, binary_class=True): 
#     print('start') 
    stat_df = get_mean_SD_values(tmp_df.copy(), class_name, binary_class=binary_class)
    stat_df
#     print('start 222', stat_df.shape) 
    all_p_and_auc_df = calculate_p_and_auc_for_dataset(tmp_df.copy(), class_name, binary_class=binary_class) 
    all_p_and_auc_df
#     print('start 333', all_p_and_auc_df.shape) 
    stat_significance_df = get_statistical_significant_using_wilcoxon_and_mannwhitney_u_test(tmp_df.copy(), class_name, binary_class=binary_class)
    stat_significance_df
#     print('start 444', stat_significance_df.shape) 
    mi_stat_df = get_MI_NMI_and_entropy_for_dataset(tmp_df.copy(), class_name, binary_class=binary_class) 

    all_stat_info = pd.merge(stat_df, all_p_and_auc_df, on="Features", how="left")
    all_stat_info = pd.merge(all_stat_info, stat_significance_df, on="Features", how="left")
    all_stat_info = pd.merge(all_stat_info, mi_stat_df, on="Features", how="left") 
#     print('start 555', all_stat_info.shape) 
    
#     exclude_cols = [class_name] 
# #     all_cols = all_stat_info.columns 
# #     all_stat_info.loc[:, ~all_cols.isin(exclude_cols)] = all_stat_info.loc[:, ~all_cols.isin(exclude_cols)].apply(lambda col: col.fillna(col.mean())) 
#     all_stat_info.loc[:, ~all_stat_info.columns.isin(exclude_cols)] = all_stat_info.loc[:, ~all_stat_info.columns.isin(exclude_cols)].apply(lambda col: col.fillna(col.mean())) 
#     print('start 666', all_stat_info.shape) 
    
    return all_stat_info








##




