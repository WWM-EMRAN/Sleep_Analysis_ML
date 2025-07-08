'''
######################################
Importing necessary modules
'''
# import HumachLab_Global 
# HumachLab_Global.get_system_info() 

import os
import sys
import datetime

os.chdir(os.getcwd())
sys.path.append(f"{os.getcwd()}\HumachLab")
sys.path.insert(0, os.path.abspath('./HumachLab'))
sys.path.insert(0, os.path.abspath('./HumachSleep'))
# print(f"Current/Project Directory: {os.getcwd()}")
# print(f"HumachLab Directory: {os.getcwd()}\HumachLab")
# print(f"HumachSleep Directory: {os.getcwd()}\HumachSleep")

import glob

import re
import pickle
import json

import copy
from pprint import pprint

import itertools as it
from functools import reduce

import math
import numbers

import numpy as np
import pandas as pd

import mne

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rc, rcParams
# %matplotlib inline
import seaborn as sns

# from HumachLab import * 
from HumachSleep import * 
# from HumachLab.HumachLab_Global import *

# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (20,6)



from IPython.display import Markdown, display
%matplotlib inline


