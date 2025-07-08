'''
######################################
Importing necessary modules
'''
import PSA_Imports 
import os, glob, re, copy 
import numpy as np 
import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt 




'''
######################################
Get annotation sequences from a dataframe for a file    
'''
def sequencer_func(x):
    result = {"File_Name":x["File_Name"].head(1).values[0], "Category":x["Category"].head(1).values[0], "Subject_ID":x["Subject_ID"].head(1).values[0], "Record_ID":x["Record_ID"].head(1).values[0], 
              "Sleep_Stage":x["Sleep_Stage_Number"].head(1).values[0], "Stage_Count": x.shape[0], "Duration": x["Duration"].sum()}
    return pd.Series(result, name="index")



def get_annotation_sequence(annot_df):
#     print(annot_df.columns)
    annot_df["Sleep_Stage_Number"] = annot_df["Sleep_Stage"]
    annot_df["Count"] = (annot_df["Sleep_Stage_Number"] != annot_df["Sleep_Stage_Number"].shift(1)).cumsum()
    annot_seq_df = annot_df.groupby("Count",as_index=False).apply(sequencer_func)
    ind = annot_seq_df.columns.values.tolist().index("Sleep_Stage")  
#     print('inside===>', ind, annot_seq_df.columns)    
    annot_seq_df.insert(loc=ind, column="Sequence", value=annot_seq_df["Count"].values.tolist())
#     print('inside 1111===>', annot_seq_df.columns)
    annot_seq_df = annot_seq_df.drop("Count", axis=1)
#     print('inside 2222===>', annot_seq_df.columns)
    return annot_seq_df



'''
######################################
Get annotation sequences from all files 
'''
def get_all_annotation_sequence(result_directory, annot_directory, all_annot_df, list_of_files, trimW_states, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, rk2aasm):
    all_annot_seq_df = pd.DataFrame()
#     for rk2aasm in [None, RK_to_AASM_stage_mapper]: 
    for file in list_of_files:
        annot_df = get_annot_data_from_filelist(result_directory, annot_directory, all_annot_df, file, trimW_states, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, rk2aasm=rk2aasm) 
#         print('11111--->', annot_df.columns) 
        annot_seq_df = get_annotation_sequence(annot_df.copy()) 
#         print('22222--->', annot_seq_df.columns) 

        # print('======', "shape", annot_seq_df.shape)
        all_annot_seq_df = pd.concat([all_annot_seq_df, annot_seq_df], axis=0)

#     print(all_annot_seq_df.columns) 
    all_annot_seq_df.reset_index(drop=True, inplace=True) 
    return all_annot_seq_df



'''
######################################
Convert sleep stage name to values for plotting   
''' 
def sleep_stage_name_to_value_conversion(stage_names, sleep_stage_names_and_values_for_graph, rk2aasm=None):
    tmp_stage_names = {} 
    for sname in stage_names:
        sname1 = sname.lower() 
        if sname1 in list(sleep_stage_names_and_values_for_graph.keys()):
            tmp_stage_names[sname] = sleep_stage_names_and_values_for_graph[sname1]
            
#     print('---->##', stage_names, tmp_stage_names)
    return tmp_stage_names



'''
######################################
Get single annot data from dataframe or file to plot and save hyphogram  
'''
def get_annot_data_from_filelist(result_directory, annot_directory, all_annot_df, annot_file_name, trimW_states, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, rk2aasm=None):
    annot_exist = all_annot_df is not None         
    if annot_exist:
        annot_df = all_annot_df[(all_annot_df["File_Name"]==annot_file_name)].copy() 
    else:
        annot_csv = f"{annot_directory}/{annot_file_name}_annot.csv" 
        annot_df = pd.read_csv(annot_csv)
        
    stg_lable_dict = copy.deepcopy(sleep_stage_labels_dict) 
    if rk2aasm is not None:
        annot_df["Sleep_Stage"] = annot_df["Sleep_Stage"].replace(rk2aasm) 
        stg_lable_dict = rk_to_aasm_converter(rk2aasm, copy.deepcopy(sleep_stage_labels_dict))
#     print('Extra 11--->', annot_df.columns) 
    annot_df["Sleep_Stage_Number"] = annot_df["Sleep_Stage"]
    
    tmp_stage_names = sleep_stage_name_to_value_conversion(list(annot_df["Sleep_Stage_Number"].unique()), sleep_stage_names_and_values_for_graph, rk2aasm=rk2aasm)
#     print('===>##', sleep_stage_labels_dict, stg_lable_dict, tmp_stage_names)
    # pprint(tmp_stage_names)
    annot_df.replace({"Sleep_Stage_Number": tmp_stage_names}, inplace=True)
    annot_df.replace({"Sleep_Stage_Number": stg_lable_dict}, inplace=True)
    
#     ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
#     ## This is not required since the triming is done already
#     if trimW_states:
#         st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_file_name, annot_df.copy())
#         annot_df = annot_df[st_ind : en_ind]
#     print('Extra 22--->', annot_df.columns) 
    return annot_df 



'''
######################################
Map RK to AASM sleep stages and their values   
'''
def rk_to_aasm_converter(rk2aasm, stg_lable_dict):
    aasm_stg_lable_dict = {}
    for stage, label in stg_lable_dict.items(): 
        new_stage = rk2aasm[stage] 
#         # for {W:0, N1:1, N2:2, N3:3, REM:5}
#         if new_stage not in aasm_stg_lable_dict or aasm_stg_lable_dict[new_stage] > label:
#             aasm_stg_lable_dict[new_stage] = label
        # for {W:0, N1:1, N2:2, N3:3, REM:4}
        if new_stage not in aasm_stg_lable_dict:
            aasm_stg_lable_dict[new_stage] = 4 if new_stage=="REM" else label
    return aasm_stg_lable_dict 



'''
######################################
Get transition sequences from all files 
'''
def get_transition_sequence_from_annotation_sequences(all_trans_seq_df):  
    all_trans_seq_df = all_trans_seq_df.reset_index(drop=True) 
    all_trans_seq_df['next_code'] = all_trans_seq_df.groupby('File_Name')['Sleep_Stage'].shift(-1, fill_value="W")
    all_cols = all_trans_seq_df.columns.values.tolist() 
    indx = all_cols.index('Sleep_Stage') 
    all_trans_seq_df.insert(loc=indx+1, column="Next_Stage", value=all_trans_seq_df["next_code"].values.tolist())
    all_trans_seq_df = all_trans_seq_df.drop("next_code", axis=1)
    all_trans_seq_df = get_annotation_sequence(all_trans_seq_df.copy()) 
    all_trans_seq_df['Sequence'] = all_trans_seq_df.groupby(['File_Name']).cumcount(['Sequence']).values 
    return all_trans_seq_df




