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
def get_annot_data_for_hypnogram(result_directory, annot_directory, all_annot_df, annot_file_name, trimW_states, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, rk2aasm=None):
    annot_exist = all_annot_df is not None         
    if annot_exist:
        annot_df = all_annot_df[(all_annot_df["File_Name"]==annot_file_name)].copy() 
    else:
        annot_csv = f"{annot_directory}/{annot_file_name}_annot.csv" 
        annot_df = pd.read_csv(annot_csv)
        
    stg_name_val_dict = sleep_stage_names_and_values_for_graph 
    stg_lable_dict = copy.deepcopy(sleep_stage_labels_dict) 
    aasm_fname = "" 
    if rk2aasm is not None:
        annot_df["Sleep_Stage"] = annot_df["Sleep_Stage"].replace(rk2aasm) 
        aasm_fname = "_AASM" 
        stg_lable_dict = rk_to_aasm_converter(rk2aasm, copy.deepcopy(sleep_stage_labels_dict))
#         stg_name_val_dict = {k:3 if v==4 else k:v for k,v in sleep_stage_names_and_values_for_graph.items()} 
    annot_df["Sleep_Stage_Number"] = annot_df["Sleep_Stage"]
    
    tmp_stage_names = sleep_stage_name_to_value_conversion(list(annot_df["Sleep_Stage_Number"].unique()), stg_name_val_dict, rk2aasm=rk2aasm)
#     print('===>##', sleep_stage_labels_dict, stg_lable_dict, tmp_stage_names)
#     annot_df.replace({"Sleep_Stage_Number": tmp_stage_names}, inplace=True)
    annot_df.replace({"Sleep_Stage_Number": stg_lable_dict}, inplace=True)
    
#     ## trimW_states: pre-process to remove leading and trailing Ws {'_TrimW' if trimW_states else ''}
#     ## This is not required since the triming is done already
#     if trimW_states:
#         st_ind, en_ind = trim_additional_leadingAndTrailing_Wake_stages(annot_file_name, annot_df.copy())
#         annot_df = annot_df[st_ind : en_ind]
    save_file_name = f"{result_directory}/Hypnograms{'_TrimW' if trimW_states else ''}/{annot_file_name}{aasm_fname}_hypno" 
    #print('======', save_file_name)
    return save_file_name, annot_df, stg_lable_dict



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
Plot single hypnogram and save it  
'''
def plot_hypnogram(sleep_stage_pattern, yticks, ytick_labels, 
                   title="Whole night sleep hypnogram", xlabel="Epoch [30-sec]", ylabel="Sleep Stage", graph_save_filename=f"./Results/Hypnograms"):    
    plt.rcParams["figure.figsize"] = (20,6) 
    fig, ax = plt.subplots()
    plt.plot(sleep_stage_pattern)
    ax.set_yticks( yticks )
    ax.set_yticklabels( ytick_labels )
#     plt.title(title)
    print('XXXX | plot_hypnogram===> ', xlabel) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'{graph_save_filename}.png', dpi=300)
    plt.savefig(f'{graph_save_filename}.pdf', dpi=300)
    plt.savefig(f'{graph_save_filename}.eps', dpi=300)
    plt.show()



'''
######################################
Draw single hypnogram and save it and use plot hypnogram function 
'''
def draw_hypnogram(save_file_name, annot_df, sleep_stage_labels_dict): 
    graph_name = f"{(save_file_name.split('/')[-1])}"
    
    # sleep_stage_labels_dict = {'W':0, 'S1':1, 'S2':2, 'S3':3, 'S4':4, 'REM':5, 'MT':6}
#     print('xxxx ====>', annot_df.columns, annot_df.head())
    sleep_stage_pattern = annot_df["Sleep_Stage_Number"].values
#     print('0000 ====>', np.unique(sleep_stage_pattern), sleep_stage_pattern.shape, sleep_stage_labels_dict)
    # Remove movement artifact 'MT' or 'ART'
    mask = np.isin(sleep_stage_pattern, np.array([6, 7])) 
    sleep_stage_pattern = sleep_stage_pattern[~mask] 
#     print('1111 ====>', np.unique(sleep_stage_pattern), sleep_stage_pattern.shape)
    
    yticks = list(sleep_stage_labels_dict.values())
    ytick_labels = list(sleep_stage_labels_dict.keys())
    yticks = yticks#[:-1]
    ytick_labels = ytick_labels#[:-1]
    
    title=f"Whole night sleep hypnogram for- {graph_name}"
    xlabel = "Epoch [30-sec]"
    ylabel = "Sleep Stage"
    print('XXXX | draw_hypnogram===> ', xlabel) 
    plot_hypnogram(sleep_stage_pattern, yticks=yticks, ytick_labels=ytick_labels, title=title, xlabel=xlabel, ylabel=ylabel, graph_save_filename=save_file_name)


  
'''
######################################
Defining graph properties
'''
def setup_mpl_graph_properties(
    show_grid=True, border_top=True, border_bottom=True, border_left=True, border_right=True,
    plot_style='seaborn-whitegrid', fig_size=(16, 8), font_size=30, font_family='serif', font_style='normal', font_weight='bold', text_color='black', bg_face_color='white' 
):
    plt.close('all')
    print(f'Font size: {font_size}')

    # What format to use, pgf is the LaTeX friendly format
    use_latex = False #True #False #True

#     if use_latex:
#         matplotlib.use('pgf')
#     else:
#         matplotlib.use('nbagg')

    # Plot style:
    # str: 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark',
    # 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
    # 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test_patch'
    plt.style.use(plot_style)
#     plt.style.use('classic')

    # Parameter dictionary
    params = {} #dict()

    # Remove/show grid lines
    params['axes.grid'] = show_grid  #bool: true/false

    # Graph frame lines setup
    params['axes.spines.top'] = border_top  #bool: true/false
    params['axes.spines.bottom'] = border_bottom  #bool: true/false
    params['axes.spines.left'] = border_left  #bool: true/false
    params['axes.spines.right'] = border_right  #bool: true/false

    # Axis setup
    params['axes.facecolor'] = bg_face_color #str: white (background color)
#     params['axes.titlecolor'] = text_color #str: black, red, green, etc
    params['axes.titlesize'] = (font_size+5) #int(points): 30p
    params['axes.titleweight'] = 'normal' #font_weight #str: normal, bold
#     params['axes.labelcolor'] = text_color #str: black, red, green, etc
    params['axes.labelsize'] = (font_size+5) #font_size #int(points): 30p
    params['axes.labelweight'] = 'normal' #font_weight #str: normal, bold

    # Figure and dimension setup
    params['figure.figsize'] = fig_size #touple: (width, height) - (16, 8)

    # Legend setup
    params['legend.fontsize'] = (font_size*0.70)

    # Font setup
    params['font.size'] = font_size     #int(points): 30p
    params['font.family'] = font_family  #str: serif, sens-serif
    params['font.style'] = font_style    #str: normal, italic
#     params['font.weight'] = 'normal'  #str: normal, bold
    params['font.weight'] = 'normal' #font_weight  #str: normal, bold

    # Text color setup
#     params['text.color'] = text_color #str: black, red, green, etc

#     # Tick setup
#     params['xtick.color'] = text_color #str: black, red, green, etc
    params['xtick.labelsize'] = (font_size)     #int(points): 30p
#     params['ytick.color'] = text_color #str: black, red, green, etc
    params['ytick.labelsize'] = (font_size)     #int(points): 30p

    # Setup for latex components
    if use_latex:
        params['pgf.texsystem'] = 'pdflatex'
        params['text.usetex'] = True
        params['pgf.rcfonts'] = False
# #         rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#         params['pgf.preamble'] = r'\usepackage{sfmath} \boldmath'
#         params['pgf.preamble'] = r'\usepackage{amsmath} \unboldmath'

    # Setup all parameters
    matplotlib.rcParams.update(params)

    return



def draw_all_hypnograms(result_directory, annot_directory, all_annot_df, list_of_files, trimW_states, sleep_stage_labels_dict, sleep_stage_names_and_values_for_graph, RK_to_AASM_stage_mapper):    
    for rk2aasm in [None, RK_to_AASM_stage_mapper]:
        for file in list_of_files:
            save_file_name, annot_df, stg_lable_dict = get_annot_data_for_hypnogram(result_directory, annot_directory, all_annot_df, file, trimW_states, sleep_stage_names_and_values_for_graph, sleep_stage_labels_dict, rk2aasm=rk2aasm) 
            print('======', save_file_name, "shape", annot_df.shape)
            draw_hypnogram(save_file_name, annot_df, stg_lable_dict)




