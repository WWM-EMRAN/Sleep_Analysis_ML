'''
######################################
Importing necessary modules
'''
import os 
# import HumachLab_Global 
# HumachLab_Global.get_system_info() 

from HumachSleep import PSA_Global
# import PSA_Global 
PSA_Global.get_system_info() 

print(f"Current/Project Directory: {os.getcwd()}")
print(f"HumachLab Directory: {os.getcwd()}\HumachLab")
print(f"HumachSleep Directory: {os.getcwd()}\HumachSleep")

# import PSA_Global 
# PSA_Global.get_system_info() 

# PSA_Imports.get_system_path() 


# def get_system_path(): 
#     HumachLab_Global.get_system_info() 
#     print(f"Current/Project Directory: {os.getcwd()}")
#     print(f"HumachLab Directory: {os.getcwd()}\HumachLab")
#     print(f"HumachSleep Directory: {os.getcwd()}\HumachSleep")

