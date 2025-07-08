'''
######################################
Importing necessary modules
'''
import PSA_Imports 
import os, sys 
import glob, re, copy 
import numpy as np 
import pandas as pd 
# import matplotlib 
# import matplotlib.pyplot as plt 




'''
######################################
Fill missing values in matrix    
'''
def fill_missing_stage_values(arr, indx):
    
    for i in indx: 
        column_to_be_added = np.zeros(arr.shape[0])
        arr2 = np.insert(arr, i, column_to_be_added, axis=0)
        column_to_be_added = np.zeros(arr.shape[0]+1)
        arr3 = np.insert(arr2, i, column_to_be_added, axis=1)
        arr = arr3.copy()
    return arr.tolist()



'''
######################################
Create all zero matrix before calculation begins 
'''
def create_zero_matrix(n, st):
    mat = None
    if n>1:
        inmat = [0 for _ in range(n)]
        tmpmat = inmat.copy()
        for i in range(st-1):
            inmat = [copy.deepcopy(inmat) for _ in range(n)]
        mat = copy.deepcopy(inmat)
    else:
        print("\nNo transition is set...")
    return mat



'''
######################################
Calculate transition matrix: transitoin count and duration 
'''
def create_transition_matrix(transitions, duration, tran_count, tran_dura, tran_step=2):
    # now get transition count and duration
    for i, (t, d) in enumerate(zip(transitions[:-(tran_step-1)], duration[:-(tran_step-1)])):
        stats = f"{t}"
        # print(t)        
        tmp_count = f"tran_count[{t}]" 
        tmp_dura = f"tran_dura[{t}]" 
        for j in range(tran_step-1):
            nt = transitions[i+j+1]
            stats += f"->{nt}"
            # print(nt)            
            tmp_count = f"{tmp_count}[{nt}]"
            tmp_dura = f"{tmp_dura}[{nt}]"  
            if (tran_step>2) and j<(tran_step-2):
                d += duration[i+j+1]
            
        exec(f"tc={tmp_count}+{1}")
        exec(f"td={tmp_dura}+{d}")  
        tmp_count = f"{tmp_count}={tmp_count}+{1}"
        tmp_dura = f"{tmp_dura}={tmp_dura}+{d}"  
        # print("=======\n", tmp_count, tmp_dura)
        exec(tmp_count)
        exec(tmp_dura)           
        # exec("print(tc)")
        # print(t, stats, tmp_count, tmp_dura, "\n================\n")
        # print(stats, tc, td, "\n================\n")
        # print("=======\n", tran_count, tran_dura)
    return tran_count, tran_dura



'''
######################################
Calculate transition matrix: transitoin probability 
'''
# #### Probability based on total transition from the same stage 
# def create_transition_probability_for_a_matrix(tran_p):    
#     tran_proba = tran_p.copy() 
#     for row in tran_proba:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row] 
#     return tran_proba


# # #### Probability based on total transition from all the stage
# def create_transition_probability_for_a_matrix(tran_p): 
#     tran_p = np.array(tran_p) 
# #     tran_proba = np.zeros(tran_p.shape) #.copy() 
#     total_count = tran_p.sum()
#     tran_proba = np.round(tran_p/total_count, 4).tolist() 
# #     for row in tran_proba:
# # #         s = sum(row)
# #         if s > 0:
# #             row[:] = [f/s for f in row] 
#     return tran_proba



'''
######################################
Calculate transition matrix: transitoin probability | from all the stage/same stage 
'''
# #### Probability based on total transition from all the stage/same stage 
def create_transition_probability_for_a_matrix(tran_p, prob_cal_from_all): 
    tran_proba = list()
    if prob_cal_from_all: 
        tran_p = np.array(tran_p) 
        total_count = tran_p.sum()
        tran_proba = np.round(tran_p/total_count, 4).tolist() 
    else: 
        tran_proba = tran_p.copy() 
        for row in tran_proba:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row] 
    return tran_proba



'''
######################################
Create transition matrix: transitoin probability | from all the stage/same stage using previous method / considering multiple stage transitions 
'''
def create_transition_probability(tran_p, prob_cal_from_all):    
    tran_proba = []     
    if type(tran_p[0][0])!=list:
        ttp = create_transition_probability_for_a_matrix(tran_p, prob_cal_from_all)
        # tran_proba.append(ttp)
        return ttp
    for i, tp in enumerate(tran_p):
        ttp = create_transition_probability(tp, prob_cal_from_all)   
        tran_proba.append(ttp)
    return tran_proba



'''
######################################
Print any transition matrix: square matrix 
'''
def print_a_matrix(tran_info, decimal):
    # print(tran_info)
    num_digits = len(str( np.amax(np.array(tran_info)) ))
    formatter_str = '{0:'+str(num_digits+4)+'.4f}' if decimal else '{0:'+str(num_digits+2)+'}' 
    for row in tran_info: 
        print(' '.join(formatter_str.format(x) for x in row)) 
        

        
'''
######################################
Print transition matrix: nested matrix for multiple stage transitions 
'''  
def print_nested_matrix(big_mat, decimal, msg=None):
    if type(big_mat[0][0])!=list:
        print_a_matrix(big_mat, decimal)
        return 
    for i, mat in enumerate(big_mat):
        mm = f"stage{i}->" if not msg else f"{msg}stage{i}->"
        print(mm)
        print_nested_matrix(mat, decimal, msg=mm)        

        

'''
######################################
Print transition information: to be added to the header of the transition matrix and printing; also check for wrong transition matrix formation, etc 
'''
def print_transition_info(tran_info, sleep_stage_labels_dict, decimal=False, tran_step=2): 
    original_sleep_stages = sleep_stage_labels_dict 
    tr_st = "->".join(f"step{i}" for i in range(1, tran_step+1))
    print(f"{tran_step} Step transition ({tr_st}). The index of the data corresponds to the sleep stages: {original_sleep_stages}")
    if tran_step<2:
        print("Wrong transition matrix...")
    else:
        print_nested_matrix(tran_info, decimal)
        


'''
######################################
Calculate and print all the transition matrix: transitoin count, duration and probability; including multiple stage transitions 
'''
# Markov transition probability matrix
# https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python#:~:text=In%20order%20to%20obtain%20a,transpose()%20.
def transition_count_and_probability_matrix(all_annot_df, sleep_stage_labels_dict, missing_labels, sleep_label_column, duration_column, prob_cal_from_all, tran_step=2, verbose=0):
    transitions = all_annot_df[sleep_label_column].values  
    duration = all_annot_df[duration_column].values 
    original_sleep_stages = list(sleep_stage_labels_dict.keys())
    ind = list([original_sleep_stages.index(ss) for ss in missing_labels])
    ind.sort()
    print("1111---", ind, missing_labels, original_sleep_stages, transitions.shape, np.unique(transitions))
    tran_count = None 
    tran_dura = None 
    tran_proba = None
    # n = 1+ max(transitions) #number of states
    n = len(list(sleep_stage_labels_dict.keys()))
    
    tran_count = create_zero_matrix(n, tran_step)
    tran_dura = create_zero_matrix(n, tran_step)
    print("000---", n, tran_step, np.unique(transitions), tran_count, tran_dura)
    u_stg = np.unique(transitions) 
    print(f"Transition steps: {tran_step}, Original stages: {sleep_stage_labels_dict}, Unique sleep stages available: {u_stg} | {[original_sleep_stages[si] for si in u_stg]}")
    
    # now get transition count and duration
    tran_count, tran_dura = create_transition_matrix(transitions, duration, tran_count, tran_dura, tran_step=tran_step)
    
    # now convert to probabilities
    tran_proba = create_transition_probability(copy.deepcopy(tran_count), prob_cal_from_all)
    ##########################################################################################
    if verbose>=2:
        # print(f"\nTransition Count: {(tran_step)}")
        print(f"\nTransition Count\n====================================")
        print_transition_info(tran_count, sleep_stage_labels_dict, tran_step=tran_step)  

        # print(f"\nTransition Duration: {(tran_step)}")
        print(f"\nTransition Duration\n====================================")
        print_transition_info(tran_dura, sleep_stage_labels_dict, tran_step=tran_step)  

        # print(f"\nTransition Probability: {(tran_step)}")
        print(f"\nTransition Probability\n====================================")
        print_transition_info(tran_proba, sleep_stage_labels_dict, decimal=True, tran_step=tran_step)
    
    #Deal with missing data    
    #original_sleep_stages = list(sleep_stage_labels_dict.keys())
    #print("2222-", len(tran_count), len(tran_proba), len(tran_dura))
    
    return tran_count, tran_proba, tran_dura



##########################################
'''
######################################
Sleep stage name value conversion: returns sleep stage names based on values   
'''
# def sleep_stage_name_to_value_conversion(stage_names, sleep_stage_names_and_values_for_graph):
#     tmp_stage_names = sleep_stage_names_and_values_for_graph.copy()
#     for sname in stage_names:
#         # print(sname)
#         if sname not in list(sleep_stage_names_and_values_for_graph.keys()):
#             tmp_stage_names[sname] = 6
#     return tmp_stage_names

def sleep_stage_name_to_value_conversion(stage_names, sleep_stage_names_and_values_for_graph, rk2aasm=None):
    tmp_stage_names = {} 
    for sname in stage_names:
        sname1 = sname.lower() 
        if sname1 in list(sleep_stage_names_and_values_for_graph.keys()):
            tmp_stage_names[sname] = sleep_stage_names_and_values_for_graph[sname1]
            
    print('---->##', stage_names, tmp_stage_names)
    return tmp_stage_names



'''
######################################
Map RK to AASM sleep stages and their values   
'''
def rk_to_aasm_converter(rk2aasm, stg_lable_dict):
    aasm_stg_lable_dict = {}
    for stage, label in stg_lable_dict.items(): 
        new_stage = rk2aasm[stage] 
#         # for RK to {W:0, N1:1, N2:2, N3:3, REM:5}
#         if new_stage not in aasm_stg_lable_dict or aasm_stg_lable_dict[new_stage] > label:
#             aasm_stg_lable_dict[new_stage] = label
        # for RK to {W:0, N1:1, N2:2, N3:3, REM:4}
        if new_stage not in aasm_stg_lable_dict:
            aasm_stg_lable_dict[new_stage] = 4 if new_stage=="REM" else label
    return aasm_stg_lable_dict 



'''
######################################
Organise any transition matrix with appropriate header/column names  
'''
def organise_a_matrix(tran_info, stage_names, from_stages=None, verbose=0):
    if from_stages==None:
        for i, ti in enumerate(tran_info):
            ti.insert(0, stage_names[i+1])
    else:
        for i, ti in enumerate(tran_info):
            ti.insert(0, f"{from_stages}{stage_names[i+1]}")
    t_mat = pd.DataFrame(columns=stage_names, data=tran_info)
    if verbose>=1:
        print(t_mat)
    return t_mat



'''
######################################
Organise big transition matrix with appropriate header/column names using above method 
'''
def organise_transition_matrix(big_mat, sleep_stage_labels_dict, msg=None, tran_step=2, timer=2, stage_names=[], verbose=0):
    original_sleep_stages_names = list(sleep_stage_labels_dict.keys())
    tran_mat_df = pd.DataFrame()
    original_sleep_stages = sleep_stage_labels_dict
    if type(big_mat[0][0])!=list:
        t_mat = organise_a_matrix(big_mat, stage_names=stage_names, from_stages=msg, verbose=verbose)
        return t_mat
    for i, mat in enumerate(big_mat):
        mm = f"{original_sleep_stages_names[i]}->" if not msg else f"{msg}{original_sleep_stages_names[i]}->"  
        print(f"### Transition from: {mm}")
        t_mat = organise_transition_matrix(mat, sleep_stage_labels_dict, msg=mm, tran_step=tran_step, timer=timer-1, stage_names=stage_names)
        tran_mat_df = pd.concat([tran_mat_df, t_mat])
    return tran_mat_df



'''
######################################
Create transitoin names and organise big transition matrix with appropriate header/column names using above method 
'''
def organise_sleep_stage_transition_matrix(tran_info, sleep_stage_labels_dict, tran_step=2, verbose=0): 
    transition_matrix_df = pd.DataFrame()
    stage_names = ["From"]+list(sleep_stage_labels_dict.keys()) 
    print(stage_names)
    
    original_sleep_stages = sleep_stage_labels_dict 
#     original_sleep_stages = list(sleep_stage_labels_dict.keys())
    tr_st = "->".join(f"step{i}" for i in range(1, tran_step+1))
    print(f"{tran_step} Step transition ({tr_st}). The index of the data corresponds to the sleep stages: {original_sleep_stages}")
    if tran_step<2:
        print("Wrong transition matrix...")
    else:
        transition_matrix_df = organise_transition_matrix(tran_info, sleep_stage_labels_dict, tran_step=tran_step, timer=tran_step, stage_names=stage_names, verbose=verbose)
#         transition_matrix_df.index = stage_names
    
    return transition_matrix_df




##################################################
############### Callable methods #################
'''
######################################
Create, format/organise and save all transition matrix/information 
'''
def create_and_save_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, prob_cal_from_all, tran_step=2, annot_type="annot", rk2aasm=None, verbose=0):
    file_name = f"{file_type}_{annot_type}"
    modified_label = "Sleep_Stage_Number"
    all_annot_df[modified_label] = all_annot_df[sleep_label_column]
    available_stages = list(all_annot_df[modified_label].unique())
    
    stg_name_val_dict = sleep_stage_names_and_values_for_graph 
    stg_lable_dict = copy.deepcopy(sleep_stage_labels_dict) 
    aasm_fname = "" 
    if rk2aasm is not None:
        all_annot_df[modified_label] = all_annot_df[modified_label].replace(rk2aasm) 
        aasm_fname = "_AASM" 
        stg_lable_dict = rk_to_aasm_converter(rk2aasm, copy.deepcopy(sleep_stage_labels_dict))
        
    original_sleep_stages = list(stg_lable_dict.keys())
    missing_labels = list( set(original_sleep_stages).difference(set(available_stages)) )
    
#     tmp_stage_names = sleep_stage_name_to_value_conversion(available_stages, sleep_stage_names_and_values_for_graph)
    tmp_stage_names = sleep_stage_name_to_value_conversion(available_stages, stg_name_val_dict, rk2aasm=rk2aasm)
    print("===>>", file_name, available_stages, sleep_label_column, missing_labels, tmp_stage_names, stg_lable_dict) 
#     all_annot_df.replace({modified_label: tmp_stage_names}, inplace=True)
    all_annot_df.replace({modified_label: stg_lable_dict}, inplace=True)
    
#     show_results = True if verbose>=3 else False 
#     count, proba, dura = transition_count_and_probability_matrix(all_annot_df, sleep_stage_labels_dict, missing_labels, sleep_label_column=modified_label, duration_column=duration_column, prob_cal_from_all=prob_cal_from_all, tran_step=tran_step, verbose=verbose) 
    count, proba, dura = transition_count_and_probability_matrix(all_annot_df, stg_lable_dict, missing_labels, sleep_label_column=modified_label, duration_column=duration_column, prob_cal_from_all=prob_cal_from_all, tran_step=tran_step, verbose=verbose) 
    
    print("Processing for saving\n===================================")
    
    file_path = f"{save_path}/{file_name}_tranStep{tran_step}{aasm_fname}"
    
    print(f"\nTransition Count\n====================================")
    count_df = organise_sleep_stage_transition_matrix(count, stg_lable_dict, tran_step=tran_step, verbose=verbose)
    count_df.fillna(0, inplace=True) 
    print(f"\nTransition Duration\n====================================")
    dura_df = organise_sleep_stage_transition_matrix(dura, stg_lable_dict, tran_step=tran_step, verbose=verbose)
    dura_df.fillna(0, inplace=True) 
    print(f"\nTransition Probability\n====================================")
    proba_df = organise_sleep_stage_transition_matrix(proba, stg_lable_dict, tran_step=tran_step, verbose=verbose)
    proba_df.fillna(0, inplace=True) 
    
    print("Started saving\n===================================")
    count_df.to_csv(f"{file_path}_count.csv", index=False)
    dura_df.to_csv(f"{file_path}_dura.csv", index=False)
    proba_df.to_csv(f"{file_path}_proba.csv", index=False)
    print("Finished saving\n===================================")
    
    return count_df, proba_df, dura_df



'''
######################################
Create, format/organise and save all transition matrix/information for sleep disorder categories 
'''
def create_and_save_category_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, prob_cal_from_all, tran_step=2, annot_type="annot", rk2aasm=None, verbose=0):
#     file_name = f"{file_type}_{annot_type}" 
    file_name = f"{file_type}" 
    disease_names = all_annot_df['Category'].unique().tolist() 
    all_matrices = {}
    
    for dis in disease_names:
        print(f"Creating and saving the transition matrices for {file_type} = {dis}")
        tmp_annot_df = all_annot_df[(all_annot_df["Category"]==dis)].copy()
        file_name1 = f"{file_name}_{dis}"              
        tmp_annot_df = tmp_annot_df.reset_index(drop=True) 
        count, proba, dura = create_and_save_transition_information(tmp_annot_df, sleep_label_column=sleep_label_column, duration_column=duration_column, save_path=save_path, file_type=file_name1, sleep_stage_labels_dict=sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph=sleep_stage_names_and_values_for_graph, prob_cal_from_all=prob_cal_from_all, tran_step=tran_step, annot_type=annot_type, rk2aasm=rk2aasm, verbose=verbose) 
        all_matrices[dis] = (count, proba, dura)
    return all_matrices



'''
######################################
Create, format/organise and save all transition matrix/information for all subjects 
'''
def create_and_save_subject_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, prob_cal_from_all, tran_step=2, annot_type="annot", subj_pattern=0, rk2aasm=None, verbose=0):
#     file_name = f"{file_type}_{annot_type}" 
    file_name = f"{file_type}" 
    list_of_subjs = all_annot_df["Subject_ID"].unique().tolist() 
    all_matrices = {} 
    
    for sub in list_of_subjs: 
        print(f"Creating and saving the transition matrices for {file_type} = {sub}")
        tmp_annot_df = all_annot_df[(all_annot_df["Subject_ID"]==sub)].copy()                
        tmp_annot_df = tmp_annot_df.reset_index(drop=True) 
        cat = tmp_annot_df["Category"].unique().tolist()[0] 
        file_name1 = f"{file_name}_{cat}_{sub}"
        count, proba, dura = create_and_save_transition_information(tmp_annot_df, sleep_label_column=sleep_label_column, duration_column=duration_column, save_path=save_path, file_type=file_name1, sleep_stage_labels_dict=sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph=sleep_stage_names_and_values_for_graph, prob_cal_from_all=prob_cal_from_all, tran_step=tran_step, annot_type=annot_type, rk2aasm=rk2aasm, verbose=verbose) 
#         all_matrices[file] = (count, proba, dura) 
        all_matrices[sub] = (count, proba, dura) 
    return all_matrices



# ### OLD 2
# def create_and_save_subject_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, prob_cal_from_all, tran_step=2, annot_type="annot", subj_pattern=0, verbose=0):
#     file_name = f"{file_type}_{annot_type}" 
#     list_of_files = all_annot_df["File_Name"].unique().tolist() 
#     all_matrices = {}
    
#     list_of_subj = []
#     list_of_subj = [ss[:-2] for ss in list_of_files] #get list of subjects from list of files/records 
#     list_of_subj = list_of_files if subj_pattern==0 else [ss[:-subj_pattern] for ss in list_of_files] 
#     list_of_subj = list( np.unique( np.array(list_of_subj) ) )
# #     print(list_of_subj)
    
#     for init_file in list_of_subj: 
#         tmp_annot_df = pd.DataFrame() 
        
#         if subj_pattern==0: ### One subject one record/file 
#             print(f"Single file || Creating and saving the transition matrices for {init_file}")
#             tmp_annot_df = all_annot_df[(all_annot_df["File_Name"]==init_file)].copy()
#         else:  ### One subject multiple records/files 
#             tmp_digits = subj_pattern*'\\d'
#             r = re.compile(f"^{init_file}{tmp_digits}$")
#             file_set = list( filter(r.match, list_of_files) ) 
#             print(f"Multiple files || Creating and saving the transition matrices for fileset: {init_file} | {file_set}") 
#             for file in file_set:
#                 print(f"Creating and saving the transition matrices for {file}")
#                 tmp_df = all_annot_df[(all_annot_df["File_Name"]==file)].copy()
#                 tmp_annot_df = pd.concat([tmp_annot_df, tmp_df])
                
#         tmp_annot_df = tmp_annot_df.reset_index(drop=True) 
#         file_name1 = f"{file_name}_{init_file}"
#         count, proba, dura = create_and_save_transition_information(tmp_annot_df, sleep_label_column=sleep_label_column, duration_column=duration_column, save_path=save_path,
#                                                                     file_type=file_name1, sleep_stage_labels_dict=sleep_stage_labels_dict, 
#                                                                     sleep_stage_names_and_values_for_graph=sleep_stage_names_and_values_for_graph, prob_cal_from_all=prob_cal_from_all, 
#                                                                     tran_step=tran_step, annot_type=annot_type, verbose=verbose) 
# #         all_matrices[file] = (count, proba, dura) 
#         all_matrices[init_file] = (count, proba, dura) 
#     return all_matrices


# ### OLD 
# def create_and_save_subject_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, tran_step=2, annot_type="annot", sub_pattern=0):
#     file_name = f"{file_type}_{annot_type}"
#     all_matrices = {}
#     list_of_files = all_annot_df["File_Name"].unique().tolist() 
    
#     for file in list_of_files:
#         print(f"Creating and saving the transition matrices for {file}")
#         tmp_annot_df = all_annot_df[(all_annot_df["File_Name"]==file)].copy()
#         file_name1 = f"{file_name}_{file}"
#         count, proba, dura = create_and_save_transition_information(tmp_annot_df, sleep_label_column=sleep_label_column, duration_column=duration_column, 
#                                                                     save_path=save_path, file_name=file_name1, sleep_stage_labels_dict=sleep_stage_labels_dict, tran_step=tran_step, annot_type=annot_type) 
#         all_matrices[file] = (count, proba, dura)
#     return all_matrices




'''
######################################
Create, format/organise and save all transition matrix/information for all files/records 
'''
def create_and_save_record_transition_information(all_annot_df, sleep_label_column, duration_column, save_path, file_type, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, prob_cal_from_all, tran_step=2, annot_type="annot", rk2aasm=None, verbose=0):
#     file_name = f"{file_type}_{annot_type}" 
    file_name = f"{file_type}" 
    all_matrices = {}
    list_of_files = all_annot_df["File_Name"].unique().tolist() 
    
    for file in list_of_files:
        print(f"Creating and saving the transition matrices for {file_type} = {file}")
        tmp_annot_df = all_annot_df[(all_annot_df["File_Name"]==file)].copy()
        file_name1 = f"{file_name}_{file}"
        count, proba, dura = create_and_save_transition_information(tmp_annot_df, sleep_label_column=sleep_label_column, duration_column=duration_column, save_path=save_path, file_type=file_name1, sleep_stage_labels_dict=sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph=sleep_stage_names_and_values_for_graph, prob_cal_from_all=prob_cal_from_all,
                                                                    tran_step=tran_step, annot_type=annot_type, rk2aasm=rk2aasm, verbose=verbose) 
        all_matrices[file] = (count, proba, dura)
    return all_matrices




