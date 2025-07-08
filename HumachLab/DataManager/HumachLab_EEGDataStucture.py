# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

from mne.io.edf.edf import RawEDF

# from dataclasses import dataclass

#%%
### Data structure for Patient
# @dataclass
class Patient:

    def __init__(self, patient_no):
        self.patient_no = patient_no
        self.sam_rate = 0
        self.channels = [] #dict
        self.removable_channels_ = [] #dict
        self.num_records = 0
        self.record_numbers = []
        self.records = []
        self.num_seizure_records = 0
        self.seizure_record_numbers = []
        pass



#%%
### Data structure for each record of the Patient
class Record:

    def __init__(self, record_no):
        self.record_no = record_no
        self.patient_no = 0
        self.file_name = ''
        self.file_path = ''
        self.channel_id = 0 # position of channel set in patient info 'channels'
        self.record_times = None # (start, end) '12:04:35' converted to 43259 sec
        self.record_duration = 0 # sec
        self.num_seizures = 0
        self.seizure_times = [] #[(start, end)]
        self.seizure_durations = []
        self.data : RawEDF = None


