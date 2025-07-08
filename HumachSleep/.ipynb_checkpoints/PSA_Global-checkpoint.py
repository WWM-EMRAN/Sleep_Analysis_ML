"""
File Name: HumachLab_Global.py || PSC_Global 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 27/07/2021 2:54 am
"""



"""
This file provides all the system information related to processors (CPU and GPU) and OS details 
"""


### OS
import os
import platform
### GPU
import GPUtil
### CPU
import cpuinfo
import psutil

### GPU code
### GPU code - Comment it if no gpu available or not linux system or no support for RapidsAI package
# if HAS_GPU:
#     from cuml import RandomForestClassifier as gpuRandomForestClassifier
# https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/

GPUs = GPUtil.getGPUs()
HAS_GPU = True if len(GPUs)>0 else False
sys_ind:int



def get_cpu_details():
    cpu_info_to_return = dict()
    cpu_info_dict = cpuinfo.get_cpu_info()

    print(
        f'\n===========================================\nProcessor (CPU) details: \n___________________________________________',
        end='\n')

    print(f'{cpu_info_dict}')
    cpu_info_to_display = ['brand_raw', 'arch_string_raw', 'arch', 'count', 'python_version']
    # cpu_info_to_display = cpu_info_dict.keys()

    for key in cpu_info_to_display:
        val = cpu_info_dict[key]
        print(f'{key.capitalize()} = {val}', end='\n')
        cpu_info_to_return[key] = val

    print(
        f'___________________________________________\nProcessor (CPU) usage: \n___________________________________________',
        end='\n')

    cpu_usage = ['CPU_usage', 'RAM_usage', 'Total_RAM', 'Used_RAM', 'Available_RAM']
    cup = psutil.cpu_percent()
    mem_stat = psutil.virtual_memory()
    print(f'{mem_stat}')
    cpu_info_to_return[cpu_usage[0]] = cup
    rup = mem_stat[2]
    cpu_info_to_return[cpu_usage[1]] = rup
    tr = round(mem_stat[0]/2.**30, 1)
    cpu_info_to_return[cpu_usage[2]] = tr
    ar = round(mem_stat[1]/2.**30, 1)
    cpu_info_to_return[cpu_usage[3]] = round(tr-ar, 1)
    cpu_info_to_return[cpu_usage[4]] = ar

    for key in cpu_usage:
        val = cpu_info_to_return[key]
        print(f'{key.capitalize()} = {val}', end='\n')

    return  cpu_info_to_return



def get_os_details():
    all_flatforms = ['Darwin', 'Windows', 'Linux']
    sys_os = platform.system()
    global sys_ind
    sys_ind = 0
    print(
        f'\n===========================================\nList of OS platforms and codes\n___________________________________________',
        end='\n')
    for i, pltform in enumerate(all_flatforms):
        print(f'{i} {pltform}', end='\n')
    if sys_os in all_flatforms:
        sys_ind = all_flatforms.index(sys_os)
        print(f'===> "{sys_ind} - {sys_os}" OS is detected.')  # procsr
    else:
        raise Exception("Unknown platform is detected!")

    return sys_ind, all_flatforms[sys_ind]



def get_gpu_details(mxLoad=0.5, mxMemory=0.5, show_logs=True):
    ### GPU code
    global GPUs
    global HAS_GPU
    classifier = None

    if show_logs:
        print(
            f'\n===========================================\nProcessor (GPU) details: \n___________________________________________',
            end='\n')

    GPUs = GPUtil.getGPUs()
    tot_gpus = len(GPUs)
    HAS_GPU = True if len(GPUs) > 0 else False
    avl_GPUIDs = GPUtil.getAvailable(order='load', limit=tot_gpus, maxLoad=mxLoad, maxMemory=mxMemory,
                                     includeNan=False, excludeID=[], excludeUUID=[]) #order=first, load, memory
    tot_avl_gpus = len(avl_GPUIDs)
    if show_logs:
        print(
            f'For GPU based tasks. There are {tot_gpus} GPUs in the system and {tot_avl_gpus} are available. \nAvailable GPU IDs with MaxLoad>={mxLoad} and MaxMem>={mxMemory} are: {avl_GPUIDs}',
            end='\n')

    if show_logs:
        print(
            f'___________________________________________\nProcessor (GPU) usage: \n___________________________________________',
            end='\n')

    if show_logs:
        GPUtil.showUtilization()

    ret_gpu1 = avl_GPUIDs[0] if tot_avl_gpus>0 else None 
    print(f'\n===========================================\n') 

    return avl_GPUIDs, ret_gpu1



def get_system_info():
    os_srl, os_name = get_os_details()
    cpu_details = get_cpu_details()
    all_avl_gpu, best_avl_gpu = get_gpu_details()
    return os_srl, os_name, cpu_details, all_avl_gpu, best_avl_gpu





############ OLD code ..............................
def check_system_os_name():
    all_flatforms = ['Darwin', 'Windows', 'Linux']
    sys_os = platform.system()
    global sys_ind
    sys_ind = 0
    print(f'\n===========================================\nList of OS platforms and codes\n___________________________________________', end='\n')
    for i, pltform in enumerate(all_flatforms):
        print(f'{i} {pltform}', end='\n')
    if sys_os in all_flatforms:
        sys_ind = all_flatforms.index(sys_os)
        procsr = platform.processor()
        print(f'===> "{sys_ind} - {sys_os}" OS is detected.') #procsr
    else:
        raise Exception("Unoknown platform is detected!")




