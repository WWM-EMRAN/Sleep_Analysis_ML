'''
######################################
Importing necessary modules
'''

import PSA_Imports 
import os, glob, re 
import shutil
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import mne




'''
######################################
Explore the contents/files in the directory
'''
def get_list_of_paths_from_a_directory(directory, path_type=None, containes=None, extension=None, exclude=None):
    '''
    directory: valid path string, path_type: p_file|p_dir, containes: string, extension: valid string file extension 
    '''
    os_path = os.path
    list_of_paths = []
        
    path_keywords = "*"
    if containes:
        path_keywords = f"{path_keywords}{containes}*"
    
    if extension:
        path_keywords = f"{path_keywords}.{extension}"
        
    complete_path = f"{directory}/{path_keywords}"
    print(f"============> {path_keywords}, {path_type}, {complete_path}")
    
    all_paths = glob.glob(complete_path) 
    print(f"Total files 111: {len(all_paths)}")
    all_temp_paths = None
    list_of_paths = None
    
    if path_type:
        if path_type=="p_file":
            all_temp_paths = [path.replace("\\", "/") for path in all_paths if (os_path.exists(path) and os_path.isfile(path))]
        if path_type=="p_dir":
            all_temp_paths = [path.replace("\\", "/") for path in all_paths if (os_path.exists(path) and os_path.isdir(path))]   
    else:
        all_temp_paths = [path.replace("\\", "/") for path in all_paths]
    print(f"Total files 222: {len(all_temp_paths)}")
#     print(f"Total files 222: {len(all_temp_paths)}", all_temp_paths, exclude)
        
    if exclude:
        # print(all_temp_paths)
        # print(len(all_temp_paths), exclude)
        # list_of_paths = [path for path in all_temp_paths for ex in exclude if ex not in path]
        # list_of_paths = [path for ex in exclude for path in all_temp_paths if ex not in path]
        list_of_paths = [path for path in all_temp_paths if not any((ex in path) for ex in exclude)]
        # list_of_paths = [path for ex in exclude if any(ex not in path for path in all_temp_paths)]
        # any(substring in string for substring in substring_list)
        # print(len(list_of_paths))
    else:
        list_of_paths = all_temp_paths.copy()
    print(f"Total files 333: {len(list_of_paths)}")
    
    return list_of_paths



'''
######################################
Get all selected directory and filter what file or folder to copy   
'''
def get_files_and_directories(source_directory, files_or_folder_to_copy): 
    path_lst = get_list_of_paths_from_a_directory(source_directory, path_type=None, containes=None, extension=None, exclude=None)  
    path_lst = [path.replace('\\', '/') for path in path_lst if any(sub in path for sub in files_or_folder_to_copy)] 
    path_lst = [path.split('/')[-1] for path in path_lst] 
    return path_lst 



'''
######################################
Copy files and folders for annotation and related contents   
'''
def files_or_folders_to_copy_from_sameStage_to_allStage(result_root_directory, prob_cal_from_all_lst, tran_category, files_or_folder_to_copy, force=False):    
    prob_cal_from_all = prob_cal_from_all_lst [0] 
    result_subdirectory = tran_category [int(prob_cal_from_all)] 
    source_directory = f"{result_root_directory}/{result_subdirectory}" 
    print(source_directory) 
    
    prob_cal_from_all = prob_cal_from_all_lst [1] 
    result_subdirectory = tran_category [int(prob_cal_from_all)] 
    destination_directory = f"{result_root_directory}/{result_subdirectory}" 
    print(destination_directory) 
    
    path_lst = get_files_and_directories(source_directory, files_or_folder_to_copy) 
    print(path_lst) 
    source_paths = [f"{source_directory}/{path}" for path in path_lst] 
    destination_paths = [f"{destination_directory}/{path}" for path in path_lst]  
    
    for s_path, d_path in zip(source_paths, destination_paths):
        if os.path.isfile(s_path):
            if not os.path.exists(d_path) or force: 
                if os.path.exists(d_path) and force:
                    os.remove(d_path) 
                shutil.copy2(s_path, destination_directory) 
                print(f"File '{s_path}' has been copied to '{destination_directory}'.")
            else:
                print(f"Destination file '{dest_folder_path}' already exists.") 
        elif os.path.isdir(s_path):
            if not os.path.exists(d_path) or force: 
                if os.path.exists(d_path) and force:
                    shutil.rmtree(d_path) 
                shutil.copytree(s_path, d_path) 
                print(f"Folder '{s_path}' has been copied to '{destination_directory}'.")
            else:
                print(f"Destination folder '{dest_folder_path}' already exists.") 
        else:
            print(f"The path '{path}' does not exist or is not accessible.") 



'''
######################################
Trim leading and trailing Wake (W) from hypnogram  - helper  
'''
def split_array_based_on_same_stages(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)



'''
######################################
Trim leading and trailing Wake (W) from hypnogram  
'''
def trim_additional_leadingAndTrailing_Wake_stages(file_n, data_f, sleep_stage_labels_dict): 
    print(f'Triming leading and trailing W for data shape: {data_f.shape} file: {file_n}')
    data_f
#     sleep_stage_name_dict = {'W':'W', 'S1':'S', 'S2':'S', 'S3':'S', 'S4':'S', 'R':'S', 'REM':'S', 'MT':'S', 'ART':'S',
#                             'N1':'S', 'N2':'S', 'N3':'S'}
    sleep_stage_name_dict = {k:('W' if k=='W' else 'S') for k,v in sleep_stage_labels_dict.items()} 
    sleep_stage_name_dict['ART'] = 'S' 
    sleep_stage_name_dict['?'] = 'S' 
    sleep_stage_dict = {'W':0, 'S':1}

    data_f['Sleep_Stage'].replace(sleep_stage_name_dict, inplace=True)
    data_f['stgs'] = data_f['Sleep_Stage'].values
    data_f['stgs'].replace(sleep_stage_dict, inplace=True)
    data_f 

    stgs = data_f['stgs'].values
    stgs
    stg_lbls = data_f['Sleep_Stage'].values
    stg_lbls
#     print('QQQQ-> ', stgs, data_f['stgs'].unique())

    splitted_array = split_array_based_on_same_stages(stgs, stepsize=0) 
    splitted_array 
    

    indx = 0 
    dat_list = [] 

    for arr in splitted_array:
        # print(len(arr), end=' ')
        count = len(arr)
        row = {'stage':stg_lbls[indx], 'indx':indx, 'count':count} 
        dat_list.append(row)
        indx += len(arr)

    tdf = pd.DataFrame(dat_list)
    tdf 
#     print(tdf.sort_values(by=['stage', 'count']))

#     tmp_indxs = sorted( tdf[(tdf['stage']=='W')].sort_values(by=['count'], ascending=[False])[:2]['indx'].values.tolist() )
#     cnts = sorted( tdf[(tdf['stage']=='W')].sort_values(by=['count'], ascending=[False])[:2]['count'].values.tolist() )
    tmp = tdf[(tdf['stage']=='W')].sort_values(by=['count'], ascending=[False])[:2].sort_values(by=['indx'])
    tmp_indxs = tmp['indx'].values.tolist()
    cnts = tmp['count'].values.tolist()
    print(tmp_indxs, cnts)
    indxs = tmp_indxs.copy() 
    indxs.insert(1,indxs[0]+cnts[0]-1)
    indxs
    indxs.append(indxs[-1]+cnts[1]-1)
    indxs
    indxs.insert(0,0)
    indxs
    indxs.append(len(data_f)-1)
    print(indxs)

    st_ind, en_ind = 0, len(data_f)-1
    
    # check condition if the Wake is 1.5hr=90min or more then consider otherwise keep them | Average sleep cycle duration 90-110 (>1.5hr) minutes
    if cnts[0]>=1*90*2 and cnts[1]>=1*90*2:
        st, md, en = indxs[1]-indxs[0], indxs[3]-indxs[2], indxs[5]-indxs[4] 
        print('Both gaps >90 mins', st, md, en) 
        if st>=md and st>=en: 
            st_ind, en_ind = indxs[0], indxs[1]  
        elif md>=st and md>=en: 
            st_ind, en_ind = indxs[2]+1, indxs[3] 
        elif en>=st and en>=md: 
            st_ind, en_ind = indxs[4]+1, indxs[5]+1  
    elif  cnts[0]>=1*90*2:        
        st, en = indxs[3]-indxs[0], indxs[5]-indxs[4] 
        print('First gap >90 mins', st, en) 
        if st>=en: 
            st_ind, en_ind = indxs[0],  indxs[3] 
        else: 
            st_ind, en_ind = indxs[4]+1, indxs[5]+1  
    elif  cnts[1]>=1*90*2:      
        st, en = indxs[1]-indxs[0], indxs[5]-indxs[2] 
        print('Last gap >90 mins', st, en) 
        if st>=en: 
            st_ind, en_ind = indxs[0],  indxs[1] 
        else: 
            st_ind, en_ind = indxs[2]+1, indxs[5]+1  
    else:
        print('Both gaps <90 mins, thus taking the entire signal')

    print('--->', st_ind, en_ind)
    # take max 20 minutes of Wake from leading and trailing Wakes | 20 minutes = 40 ea, 30sec segments or 40 samples | Average people sleep between 10-20 minutes 
    st_ind, en_ind = st_ind-20*2, en_ind+20*2 
    if st_ind<0:
        st_ind = 0
    if en_ind>len(data_f)-1:
        en_ind = len(data_f)-1
    
    print('===>', st_ind, en_ind)
    return st_ind, en_ind



'''
######################################
Read all annotations from annotation path and trim when applied 
'''
def get_all_annotations(list_of_annot_csvs, trimW_states):
    all_annot_df = pd.DataFrame()
    for annot_csv in list_of_annot_csvs:
        annot_df = pd.read_csv(annot_csv)
#         ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
#         if trimW_states:
#             st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_csv, annot_df.copy())
#             annot_df = annot_df[st_ind : en_ind]
        all_annot_df = pd.concat([all_annot_df, annot_df], axis=0)
    all_annot_df.reset_index(drop=True, inplace=True) 
    return all_annot_df



'''
######################################
Convert sleep stage name to values for plotting   
'''
def sleep_stage_name_to_value_conversion(stage_name, sleep_stage_names_and_values_for_graph):
    stage_name = stage_name.replace("  ", " ") 
    stage_name = stage_name.lower()  
    stage_numbers_dict = sleep_stage_names_and_values_for_graph 
    stage_num = 6 
    corrected_name = "ART"
    long_stage_name_list = list(stage_numbers_dict.keys())
    if stage_name in long_stage_name_list: 
        stage_num = stage_numbers_dict[stage_name] 
    return stage_num


def sleep_stage_name_corrector(stage_name, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, stage_number=False): 
    stage_num = sleep_stage_name_to_value_conversion(stage_name, sleep_stage_names_and_values_for_graph)         
    inv_dict = {v:k for k,v in sleep_stage_labels_dict.items()} 
    inv_dict[6] = "ART" 
    inv_dict[7] = "?" 
    corrected_name = inv_dict[stage_num]  
            
    return corrected_name




'''
######################################
CAP Sleep dataset 
'''
'''
######################################
Parse single annotation file for CAP dataset 
'''
def parse_annotation_for_CAP_Sleep(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    print(f"Annotation conversion for file: {annot_txt} \n")
    annot_f = annot_txt.split('.')[-2]
    file_name = annot_f.split("/")[-1] 
    
    cat, sub, rec = file_name.split("_") 
    
    annot_csv = f"{annot_directory}/{file_name}_annot.csv"
    annot_csv
    
    print(annot_f, file_name, annot_csv)

    contents = ""
    less_cnt = 0
    
    with open(annot_txt, "r") as in_file, open(annot_csv, "w") as out_file:
        line = in_file.readline()
        while line:
            line = in_file.readline()
            tab_cnt = line.count("\t")
            # print(line, tab_cnt)
            # if tab_cnt!=(5-less_cnt):
            if tab_cnt<4:
                continue
            elif tab_cnt==4: 
                less_cnt = 1
            line = line.replace("\t", ",")
            
            if extras: 
                line = line.replace("\n", "")
                arr = line.split(",")
                # print(arr)
                ind = (3-less_cnt)
                ev = arr[ind].split("-")
                # print(ev)
                if len(ev)==1:
                    ev[0] = "Event_Type"
                    ev.append("Event_Level")
                ind = (5-less_cnt)
                loc = arr[ind].split("-")
                # print(loc)
                if len(loc)==1:
                    if loc[0]=="Location":
                        loc[0] = "Signal_Type"
                        loc.append("Channel")
                    else:
                        loc.append("UNKNOWN")
                elif len(loc)==2:
                    loc[1] = f"{loc[0]}-{loc[1]}"
                    loc[0] = loc[0][:3]
                else:
                    loc[1] = f"{loc[1]}-{loc[2]}"
                    
                line = f"{','.join(arr)},{','.join(ev)},{','.join(loc[:2])}\n"
            
            arr = line.split(",")
            if "Sleep Stage" in arr:
                arr.insert(0, "Record_ID")
                arr.insert(0, "Subject_ID")
                arr.insert(0, "Category")
                arr.insert(0, "File_Name")
                if "Duration[s]" in arr:
                    indx = arr.index("Duration[s]")
                    arr[indx] = "Duration [s]"
            else:
                match = re.match(r"([a-z]+)([0-9]+)", file_name, re.I)
                if match:
                    items = match.groups()
                arr.insert(0, rec)
                arr.insert(0, sub)
                arr.insert(0, cat)
                arr.insert(0, file_name)
                
                arr[4] = sleep_stage_name_corrector(arr[4], sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict) 
                
#             print("==->", arr)
            # Newly added
#             new_cols = [col[:(col.index("["))-1] if "[" in col else col.replace(" ", "_") for col in arr] 
            new_cols = [col[:(col.index("["))-1] if isinstance(col, str) and "[" in col else col.replace(" ", "_") if isinstance(col, str) else col for col in arr]
#             print("My arr=====>", new_cols)
            line = ",".join(new_cols)
            #print("===>", line)
            contents += line
            writer = out_file.writelines(line)
    if show:
        print(contents)
        
    ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
    if trimW_states:
        annot_df = pd.read_csv(annot_csv, index_col=None) 
        st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_csv, annot_df.copy(), sleep_stage_labels_dict)
        annot_df = annot_df[st_ind : en_ind]
        annot_df.to_csv(annot_csv, index=False)
    print(f"Annotation saved to file: {annot_csv} \n")

    

'''
######################################
Parse all annotation files for CAP dataset 
'''
def parse_all_annotation_for_CAP_Sleep(list_of_annot_txts, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    for annot_txt in list_of_annot_txts:
        parse_annotation_for_CAP_Sleep(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=extras, show=show)




'''
######################################
EDFX Sleep dataset 
'''
'''
######################################
Parse single annotation file for EDFX dataset 
'''
def parse_annotation_for_Sleep_EDFX(annot_edf, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, remove_unscored=True):
    print(f"Annotation conversion for file: {annot_edf} \n") 
    annot_f = annot_edf.split('.')[-2]
    file_name = annot_f.split("/")[-1] 
    file_name = file_name.split("-")[-2] 
    
#     annot_f = annot_edf.split('.')[-2]
#     file_name = annot_f.split("/")[-1].split("-")[0] 
    
    annot_csv = f"{annot_directory}/{file_name}_annot.csv"
    annot_csv
    
    cat, sub, rec = file_name.split("_") 
    
    raw_annot_data = mne.read_annotations(annot_edf)
    
    data = {"Sleep_Stage": [d.split(" ")[-1] for d in raw_annot_data.description.tolist()], "onset": raw_annot_data.onset.tolist(), "Duration": raw_annot_data.duration.tolist()}
    edf_annot_df = pd.DataFrame.from_dict(data)
    
    if remove_unscored:
        edf_annot_df = edf_annot_df[(edf_annot_df["Sleep_Stage"]!='?')]
    all_corrected_stages = [] 
    for s in edf_annot_df["Sleep_Stage"].values.tolist():
        all_corrected_stages.append(sleep_stage_name_corrector(s, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict)) 
    edf_annot_df["Sleep_Stage"] = all_corrected_stages
    
    edf_annot_df["Event_Type"] = edf_annot_df["Sleep_Stage"]
    stg_detail = get_short_stage_from_EDFX() 
    edf_annot_df = edf_annot_df.replace({'Event_Type': stg_detail[0]})    
    edf_annot_df["Event_Level"] = edf_annot_df["Sleep_Stage"]
    edf_annot_df = edf_annot_df.replace({'Event_Level': stg_detail[1]})
    # ### considered unknown state '?' means 'W' state and 'time' means 'W'
    edf_annot_df = edf_annot_df.replace({"Sleep_Stage": stg_detail[2]})  
    edf_annot_df["Event"] = edf_annot_df["Event_Type"] +"-"+ edf_annot_df["Event_Level"] 
    edf_annot_df["num_s"] = edf_annot_df["Duration"]/30
    edf_annot_df
    
    edf_annot_df = edf_annot_df.loc[edf_annot_df.index.repeat(edf_annot_df["num_s"])]
    edf_annot_df['Duration'] = np.where(edf_annot_df['Duration'] > 1, edf_annot_df['Duration']/edf_annot_df['num_s'], edf_annot_df['Duration'])  
    
    edf_annot_df['Duration_s'] = edf_annot_df.groupby(['Sleep_Stage', 'onset'])['Duration'].cumsum() 
    edf_annot_df['onset_s'] = np.where(edf_annot_df['Duration_s'] > 1, edf_annot_df['Duration_s']+edf_annot_df['onset'], edf_annot_df['onset']) 
    edf_annot_df['onset_s'] = edf_annot_df.onset_s.shift(1) 
    
    edf_annot_df.reset_index(inplace=True)
    edf_annot_df.at[0, 'onset_s'] = edf_annot_df.at[0, 'onset']  
    edf_annot_df['Time'] = pd.to_datetime(edf_annot_df['onset_s'], unit='s').dt.strftime('%H:%M:%S')
    # edf_annot_df = edf_annot_df[['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Event Type', 'Event Level']] 
    # ['File_Name', 'Category', 'Subject_ID', 'Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Location', 'Event Type', 'Event Level', 'Signal Type', 'Channel'] 
    # ['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Event Type', 'Event Level']
    # ['File_Name', 'Category', 'Subject_ID', 'Position', 'Location', 'Signal Type', 'Channel'] 
    
    rows = edf_annot_df.shape[0] 
    edf_annot_df['File_Name'] = [file_name]*rows 
    edf_annot_df['Category'] = [cat]*rows 
#     edf_annot_df['Subject_ID'] = [file_name[2:5]]*rows   # ([int(file_name[3:5])+100] if "SC" in file_name else [int(file_name[3:5])+200])*rows   
    edf_annot_df['Subject_ID'] = [sub]*rows
    edf_annot_df['Record_ID'] = [rec]*rows 
    edf_annot_df['Position'] = ['UNKNOWN']*rows 
    edf_annot_df['Location'] = ['UNKNOWN']*rows 
    edf_annot_df['Signal_Type'] = ['UNKNOWN']*rows  
#     edf_annot_df['Signal_Type'] = [file_name[-2:]]*rows  
    edf_annot_df['Channel'] = ['UNKNOWN']*rows 
    
    # cols = ['File_Name', 'Category', 'Subject_ID', 'Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Location'] 
    # if extras:
    #     cols = ['File_Name', 'Category', 'Subject_ID', 'Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Location', 'Event Type', 'Event Level', 'Signal Type', 'Channel'] 
    cols = ['File_Name', 'Category', 'Subject_ID', 'Record_ID', 'Sleep_Stage', 'Position', 'Time', 'Event', 'Duration', 'Location'] 
    if extras:
        cols = ['File_Name', 'Category', 'Subject_ID', 'Record_ID', 'Sleep_Stage', 'Position', 'Time', 'Event', 'Duration', 'Location', 'Event_Type', 'Event_Level', 'Signal_Type', 'Channel'] 
    
    edf_annot_df = edf_annot_df[cols] 
    # ['File_Name', 'Category', 'Subject_ID', 'Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Location', 'Event Type', 'Event Level', 'Signal Type', 'Channel'] 
    # ['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration [s]', 'Event Type', 'Event Level']
    # ['File_Name', 'Category', 'Subject_ID', 'Position', 'Location', 'Signal Type', 'Channel'] 
    
    print(annot_f, file_name, annot_csv)
    edf_annot_df.to_csv(annot_csv, index=False)
        
    ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
    if trimW_states:
        # annot_df = pd.read_csv(annot_csv, index_col=None) 
        annot_df = edf_annot_df.copy()  
        st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_csv, annot_df.copy(), sleep_stage_labels_dict)
        annot_df = annot_df[st_ind : en_ind]
        annot_df.to_csv(annot_csv, index=False)
    print(f"Annotation saved to file: {annot_csv} \n")
    return 

    

def get_short_stage_from_EDFX():
    short_stage = [] 
    tmp = {'?': 'Unknown', 'time':'SLEEP', 'W':'SLEEP', '1':'SLEEP', '2':'SLEEP', '3':'SLEEP', '4':'SLEEP', 'R':'SLEEP'} 
    short_stage.append(tmp)  
    tmp = {'?': 'Unknown', 'time':'S0', 'W':'S0', '1':'S1', '2':'S2', '3':'S3', '4':'S4', 'R':'REM'}  
    short_stage.append(tmp) 
    tmp = {'?': 'ART', 'time':'W', '1':'S1', '2':'S2', '3':'S3', '4':'S4', 'R':'R'}  
    short_stage.append(tmp) 
    return short_stage 



'''
######################################
Parse all annotation files for EDFX dataset 
'''
def parse_all_annotation_for_Sleep_EDFX(list_of_annot_edfs, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=True, remove_unscored=True):
    for annot_edf in list_of_annot_edfs:
        parse_annotation_for_Sleep_EDFX(annot_edf, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=extras, remove_unscored=remove_unscored) 




'''
######################################
SDRC Sleep dataset 
'''
'''
######################################
Parse single annotation file for SDRC dataset 
'''
def parse_annotation_for_SDRC(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    print(f"Annotation conversion for file: {annot_txt} \n")
    annot_f = annot_txt.split('.')[-2]
    file_name = annot_f.split("/")[-1] 
    
    cat, sub, rec = file_name.split("_") 
    
    annot_csv = f"{annot_directory}/{file_name}_annot.csv"
    annot_csv
    
    print(annot_f, file_name, annot_csv)

    contents = ""
    less_cnt = 0
    
    with open(annot_txt, "r") as in_file, open(annot_csv, "w") as out_file:
        line = in_file.readline() 
        line_cnt = 0 
        while line:
            line = in_file.readline()
            line_cnt += 1 
            if line_cnt<7:
                continue 
            sem_cnt = line.count(";")
#             print('NNNNN ', line, sem_cnt)
            # if sem_cnt!=(5-less_cnt):
            if sem_cnt!=1:
                continue
#             line = line.replace("; A", "; ART")
#             line = line.replace(";A", "; ART")
            line = line.replace(";", ",")
            
            arr = [it.strip() for it in line.split(",")] 
#             print('test 000', arr)
            # File_Name	Category	Subject_ID	Sleep_Stage	Position	Time	Event	Duration	Location	Event_Type	Event_Level	Signal_Type	Channel 
            # 'UNKNOWN' 
            if line_cnt==7 and (("A" in arr) or ("ART" in arr)):
                first_line = "File_Name,Category,Subject_ID,Record_ID,Sleep_Stage,Position,Time,Event,Duration,Location,Event_Type,Event_Level,Signal_Type,Channel\n" 
#                 print(first_line)
                contents += first_line 
                writer = out_file.writelines(first_line)  
                    
#             match = re.match(r"([a-z]+)([0-9]+)", file_name, re.I)
#             if match:
#                 items = match.groups()
#             print('test 111', arr)
            arr.insert(0, rec)
            arr.insert(0, sub)
            arr.insert(0, cat)
            arr.insert(0, file_name)
#             print('test 222', arr)
            
            ### Adding new columns  
            slp_val = sleep_stage_name_corrector(arr[6], sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict)
            arr.pop(5) 
            arr.insert(4, arr[5]) 
#             print('test 333', arr)
            full_type_stg = slp_val if slp_val=='ART' else 'SLEEP' 
            arr.pop() 
            arr.insert(5, 'UNKNOWN') 
#             print('test 444', arr)
            slp_val = 'A' if slp_val=='ART' else slp_val
            arr.append(f'{full_type_stg}-{slp_val}')
            arr.append(str(30))
            arr.append('UNKNOWN')
            arr.append(full_type_stg)
            arr.append(slp_val)
            arr.append('UNKNOWN')
            arr.append('UNKNOWN')
            arr[4] = sleep_stage_name_corrector(arr[4], sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict) 
            
            
#             print("==->", arr)
#             for i, v in enumerate(arr): 
#                 arr[i] = str(arr[i])
#                 print(f'{i}, {v}') 
            # Newly added
            new_cols = [col[:(col.index("["))-1] if "[" in col else col.replace(" ", "_") for col in arr] 
#             new_cols = [col[:(col.index("["))-1] if isinstance(col, str) and "[" in col else col.replace(" ", "_") if isinstance(col, str) else col for col in arr]
#             print("=====>", arr, new_cols, line)
            line = ",".join(new_cols) 
            line = line+'\n'
            #print("===>", line)
            contents += line
            writer = out_file.writelines(line)
    if show:
        print(contents)
        
    ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
    if trimW_states:
        annot_df = pd.read_csv(annot_csv, index_col=None) 
        st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_csv, annot_df.copy(), sleep_stage_labels_dict)
        annot_df = annot_df[st_ind : en_ind]
        annot_df.to_csv(annot_csv, index=False)
    print(f"Annotation saved to file: {annot_csv} \n") 



# def get_short_stage_from_SDRC(long_stage):
#     if not isinstance(long_stage, str): 
#         long_stage = str(long_stage) 
#     long_stage = long_stage.replace("  ", " ")
#     lname = ['A', 'Art', 'Mt', 'Movement', 'R', 'Rem', 'Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'] 
#     sname = ['ART', 'ART', 'MT', 'MT', 'REM', 'REM', 'W', 'S1', 'S2', 'S3', 'S4']  
# #     lname = ['A', 'Art', 'Mt', 'Movement', 'R', 'Rem', 'Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'] 
# #     sname = ['ART', 'ART', 'MT', 'MT', 'REM', 'REM', 'W', 'N1', 'N2', 'N3', 'N3']  
#     short_stage = sname [ lname.index(long_stage) ] if (long_stage not in sname) else long_stage
#     return short_stage 



'''
######################################
Parse all annotation files for SDRC dataset 
'''   
def parse_all_annotation_for_SDRC(list_of_annot_txts, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    for annot_txt in list_of_annot_txts:
        parse_annotation_for_SDRC(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=extras, show=show)
#         print("Here stopped...") 
#         break 




'''
######################################
ISRUC Sleep dataset 
'''
'''
######################################
Parse single annotation file for ISRUC dataset 
'''
def parse_annotation_for_ISRUC(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    print(f"Annotation conversion for file: {annot_txt} \n")
    annot_f = annot_txt.split('.')[-2]
    file_name = annot_f.split("/")[-1] 
    
    cat, sub, rec = file_name.split("_") 
    
    annot_csv = f"{annot_directory}/{file_name}_annot.csv"
    annot_csv
    
    print(annot_f, file_name, annot_csv)

    contents = ""
    less_cnt = 0
    
    with open(annot_txt, "r") as in_file, open(annot_csv, "w") as out_file:
#         line = in_file.readline() 
        line_cnt = 0 
        while True:
            line = in_file.readline()
            if not line:
                break 
            if line_cnt == 0:
                first_line = "File_Name,Category,Subject_ID,Record_ID,Sleep_Stage,Position,Time,Event,Duration,Location,Event_Type,Event_Level,Signal_Type,Channel\n" 
#                 print(first_line)
                contents += first_line 
                writer = out_file.writelines(first_line)
                line_cnt += 1
                continue 
        
            sstage = line.strip() 
#             sstage = get_short_stage_from_ISRUC(sstage) 
            sstage = sleep_stage_name_corrector(sstage, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict) 
            arr = [file_name, cat, sub, rec, sstage, 'UNKNOWN'] 
            evnt_name = "ART" if sstage=="ART" else "SLEEP" 
            evnt_symb = "A" if sstage=="ART" else sstage 
            arr.append('UNKNOWN') 
            arr.append(f"{evnt_name}-{evnt_symb}") #ev 
            arr.append(str(30)) 
            arr.append('UNKNOWN') 
            arr.append(evnt_symb) 
            arr.append('UNKNOWN') 
            
            arr[4] = sleep_stage_name_corrector(arr[4], sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict) 
#             print("==->", arr, line)
            # Newly added
#             new_cols = [col[:(col.index("["))-1] if "[" in col else col.replace(" ", "_") for col in arr] 
            new_cols = [col[:(col.index("["))-1] if isinstance(col, str) and "[" in col else col.replace(" ", "_") if isinstance(col, str) else col for col in arr]
            # print("Ext=====>", arr, new_cols, line)
            line = ",".join(new_cols) 
            line = line+'\n'
            #print("===>", line)
#             print("Ext=====>", line_cnt, arr, new_cols, line)
            contents += line
            writer = out_file.writelines(line)
            line_cnt += 1 
    if show:
        print(contents)
        
    ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
    if trimW_states:
        annot_df = pd.read_csv(annot_csv, index_col=None) 
        st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_csv, annot_df.copy(), sleep_stage_labels_dict)
        annot_df = annot_df[st_ind : en_ind]
        annot_df.to_csv(annot_csv, index=False)
    print(f"Annotation saved to file: {annot_csv} \n") 

    

# def get_short_stage_from_ISRUC(long_stage):
#     if not isinstance(long_stage, str): 
#         long_stage = str(long_stage) 
#     long_stage = long_stage.replace("  ", " ")
#     lname = ['A', 'Art', 'Mt', 'Movement', 'R', 'Rem', 'Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
#             '0', '1', '2', '3', '4', '5', '6', '', '?'] 
#     sname = ['ART', 'ART', 'MT', 'MT', 'REM', 'REM', 'W', 'S1', 'S2', 'S3', 'S4',
#             'W', 'S1', 'S2', 'S3', 'S4', 'REM', 'MT', 'MT', 'MT']  
# #     lname = ['A', 'Art', 'Mt', 'Movement', 'R', 'Rem', 'Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
# #             '0', '1', '2', '3', '4', '5', '6', '', '?'] 
# #     sname = ['ART', 'ART', 'MT', 'MT', 'REM', 'REM', 'W', 'N1', 'N2', 'N3', 'N3',
# #             'W', 'N1', 'N2', 'N3', 'N3', 'REM', 'MT', 'MT', 'MT']  
#     short_stage = sname [ lname.index(long_stage) ] if (long_stage not in sname) else long_stage
#     return short_stage 



'''
######################################
Parse all annotation files for ISRUC dataset 
'''
def parse_all_annotation_for_ISRUC(list_of_annot_txts, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=False, show=False):
    for annot_txt in list_of_annot_txts:
        parse_annotation_for_ISRUC(annot_txt, annot_directory, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, extras=extras, show=show)
#         print("Here stopped...") 
#         break 


        
        

