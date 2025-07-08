# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""

###
import itertools
import ast
import copy
import re
import mne
import pandas as pd
import numpy as np
import os
import inspect, os.path
from mne.io.edf.edf import RawEDF
# import matlab.engine

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.SignalProcessor.HumachLab_FeatureExtractor import HumachLab_FeatureExtractor
    from HumachLab.Utility.HumachLab_Utility import Humachlab_Utility
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###



class HumachLab_SignalProcessor:

    ### Excluding channels
    _excluded_channel_values = []
    ### How much of the event in a segment should be considered as event
    _event_threshold_percentage = 0

    def __init__(self, logger, excluded_channel_values=[], _event_threshold_percentage=0, class_name='class', feat_file_name = 'fchbmit_', seg_file_name = 'seg_chbmit_'):
        self.logger = logger
        self._excluded_channel_values = excluded_channel_values
        self._event_threshold_percentage = _event_threshold_percentage
        self.class_name = class_name
        self.feat_file_name = feat_file_name
        self.seg_file_name = seg_file_name
        # self.main_meta_columns = ['pat_id', 'rec_id', 'seg_id', 'ch_id', 'ch_name', class_name]
        self.main_meta_columns = [class_name]
        return


    def manage_matlab_python_engine(self, existing_eng=None):
        import pkgutil
        import os, sys
        from pathlib import Path

        eggs_loader = pkgutil.find_loader('matlab')
        found = eggs_loader is not None


        mat_bld_path = str(Path.home())
        mat_bld_path = mat_bld_path.replace("\\\\", "\\")
        mat_bld_path = mat_bld_path.replace("\\", "/")
        mat_bld_path += '/matlab_build/lib'

        thisfilename = inspect.getframeinfo(inspect.currentframe()).filename
        thispath = os.path.dirname(os.path.abspath(thisfilename))
        thispath = thispath.replace("\\\\", "\\")
        thispath = thispath.replace("\\", "/")
        # print('>>>', thispath, mat_bld_path)

        if existing_eng is None:
            eng = None
            if found:
                import matlab.engine
                eng = matlab.engine.start_matlab()
            elif (not found) and (os.path.exists(mat_bld_path)):
                # sys.path.append(thispath)
                # eng.addpath(thispath, nargout=0)
                # eng.cd(thispath)
                sys.path.append(mat_bld_path)
                import matlab.engine
                eng = matlab.engine.start_matlab()
            else:
                self.logger.info(f'No matlab is installed...')
            eng.addpath(thispath, nargout=0)
            return eng
        else:
            existing_eng.quit()

        return



    ### Raw data management
    ###
    ### Convert specific edf file to csv file
    ###
    def _convert_specific_edf_from_file(self, data_file, removable_dummy_channels):
        # with tqdm(total=100, desc='Converting EDF file..') as pbar:
        self.logger.info(f'---> Getting EDF data...')
        self.logger.info(f'---> {data_file}')
        removable_dummy_channels = list(removable_dummy_channels.values())

        removable_dummy_channels = list(set(removable_dummy_channels).difference(set(self._excluded_channel_values)))

        # self.logger.info(f'@@@@@@@@@>> \n{self._excluded_channel_values} \n{removable_dummy_channels}')
        edf_data = mne.io.read_raw_edf(data_file, preload=True, exclude=self._excluded_channel_values)
        original_chns = edf_data.ch_names
        # self.logger.info(f'$$$$$$$$$>> \n{self._excluded_channel_values} \n{removable_dummy_channels} \n{original_chns}')

        removable_dummy_channels = list(set(original_chns).intersection(set(removable_dummy_channels)))
        # self.logger.info(f'%%%%%%%%>> \n{self._excluded_channel_values} \n{removable_dummy_channels} \n{original_chns}')

        channels_to_keep = list(set(original_chns).difference(set(removable_dummy_channels)))
        # self.logger.info(f'##############>> \n{self._excluded_channel_values} \n{removable_dummy_channels} \n{original_chns} \n{channels_to_keep}')

        if len(channels_to_keep) == 0:
            edf_data = None
        else:
            edf_data.drop_channels(removable_dummy_channels)

        self.logger.info(f'===> EDF data completed!!!')
        return (edf_data)



    # def _convert_specific_edf_from_file(self, data_file, removable_dummy_channels):
    #     # with tqdm(total=100, desc='Converting EDF file..') as pbar:
    #     self.logger.info(f'---> Getting EDF data...')
    #     removable_dummy_channels = list(removable_dummy_channels.values())
    #     # self.logger.info(f'{removable_dummy_channels}')
    #     preload = False
    #     preload = True if len(removable_dummy_channels)>0 else False
    #     edf_data = None
    #
    #     edf_data = mne.io.read_raw_edf(data_file, preload=preload, exclude=self._excluded_channel_values)
    #
    #     # self.logger.info(f'##############>> {self._excluded_channel_values} {removable_dummy_channels} \n{edf_data.ch_names}')
    #
    #     # self.logger.info(f'{edf_data.__dict__}')
    #     removable_dummy_channels = [ch for ch in removable_dummy_channels if ch not in self._excluded_channel_values]
    #
    #     channels_to_keep = list(set(edf_data.ch_names)-set(removable_dummy_channels))
    #     self.logger.info(f'##############>> \n'{self._excluded_channel_values} \n{removable_dummy_channels} \n{edf_data.ch_names} \n{channels_to_keep}')
    #
    #     if len(channels_to_keep)==0:
    #         edf_data = None
    #     else:
    #         edf_data.drop_channels(removable_dummy_channels)
    #
    #     # self.logger.info(f'=================>> {edf_data.ch_names}')
    #
    #     self.logger.info(f'===> EDF data completed!!!')
    #     return (edf_data)



    ### Removing duplicate channels if needed
    def _atoi(self, text):
        return int(text) if text.isdigit() else text

    def _natural_keys(self, text):
        return [self._atoi(c) for c in re.split(r'(\d+)', text)]


    def _remove_unnecessary_channels(self, chans, rem_chans):
        new_chans = {}
        i = 1
        for key, val in chans.items():
            if (val not in rem_chans.values()):
                new_key = 'Channel {}'.format(i)
                new_chans[new_key] = val
                i += 1

        return new_chans


    def _select_necessary_channels(self, chans, picked_chans): ### Still to implement
        new_chans = {}
        i = 1
        for key, val in chans.items():
            if (val in picked_chans.values()):
                new_key = 'Channel {}'.format(i)
                new_chans[new_key] = val
                i += 1

        return new_chans



    ###
    ### Channel and Misc management
    ###
    def find_similar_channels_for_all_patients(self, patient_list: list):
        channel_list = []

        for patient in patient_list:
            i = 0
            for channel in patient.channels:
                channel = self._remove_unnecessary_channels(channel, patient.removable_channels_[patient.channels.index(channel)])
                rec_list = [rec.record_no for rec in patient.records if rec.channel_id==i]
                channel_list.append([str(channel), patient.patient_no, i, rec_list])
                i += 1

        channel_group_list = [list(item[1]) for item in itertools.groupby(sorted(channel_list), key=lambda x: x[0])]

        # self.logger.info(f'>> {len(channel_group_list)} = {channel_group_list}')
        channel_list = []
        for ch_gr in channel_group_list:
            tmp_lst = []
            i = 0
            for ch in ch_gr:
                if i == 0:
                    ch[0] = ast.literal_eval(ch[0])
                    tmp_lst.append(ch[0])
                    tmp_lst.append([])
                tmp_lst[1].append((ch[1], ch[2], ch[3]))
                i += 1
            channel_list.append(tmp_lst)
        # self.logger.info(f'{len(channel_list)} --> {channel_list}')

        return channel_list



    def select_signal_of_channels(self, patient_list:list, picked_channels:list):
        data_list = [] ###

        patients_list_copy = []
        for pat in patient_list:
            pat_copy = copy.deepcopy(pat)

            tmp_data = [] ###
            tmp_data.append(pat_copy.patient_no) ###
            tmp_data.append([]) ###
            for rec in pat_copy.records:
                dat = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])
                dat = dat.pick_channels(picked_channels)
                rec.data = dat
                tmp_data[1].append((rec.record_no, [dat])) ###

            data_list.append(tmp_data) ###
            patients_list_copy.append(pat_copy)

        # return patients_list_copy ###
        return data_list ###


    def segment_signal_of_all_channels(self, patient_list:list, st_time, end_time=0, duration=0, picked_channels:list=[]):
        data_list = []  ###

        patients_list_copy = []
        for pat in patient_list:
            pat_copy = copy.deepcopy(pat)

            tmp_data = [] ###
            tmp_data.append(pat_copy.patient_no) ###
            tmp_data.append([]) ###
            for rec in pat_copy.records:
                if end_time==0 and duration==0:
                    self.logger.info(f'Specify signal segmentation measures. Proper start and end time or duration is needed.')
                else:
                    if end_time==0:
                        end_time = st_time + duration

                    # dat:RawEDF = rec.data ###
                    dat:RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id]) ###
                    if len(picked_channels)>0:
                        dat = dat.pick_channels(picked_channels)
                    dat = dat.crop(tmin=st_time, tmax=end_time)

                    rec.data = dat
                    tmp_data[1].append((rec.record_no, [dat])) ###

            data_list.append(tmp_data) ###
            patients_list_copy.append(pat_copy)

        # return patients_list_copy ###
        return data_list ###



    def segment_signal_of_all_channels_for_seizure(self, patient_list:list, picked_channels:list=[]):
        data_list = []  ###
        nonseizure_data = [] ###-
        seizure_data = [] ###-

        patients_list_copy = []
        for pat in patient_list:
            pat_copy = copy.deepcopy(pat)

            tmp_data = []  ###
            tmp_data.append(pat_copy.patient_no)  ###
            tmp_data.append([])  ###
            for rec in pat_copy.records:
                dat: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])  ###
                if len(picked_channels) > 0:
                    dat = dat.pick_channels(picked_channels)

                # tmp_data[1].append(rec.record_no)  ###
                nonseiz = [] ###-
                seiz = [] ###-
                st = 0 ###-
                end = rec.record_duration-0.005 ### to avoid last second's fraction ###-
                sz_cnt = 0 ###-
                # self.logger.info(f'Record duration: {rec.record_times} - {rec.record_duration} = {rec.num_seizures}')
                for seiz_times in rec.seizure_times:
                    sz_st = seiz_times[0]
                    sz_end = seiz_times[1]

                    # self.logger.info(f'Seizure duration: {seiz_times} - {rec.seizure_durations[rec.seizure_times.index(seiz_times)]}')

                    if not(st==sz_st):
                        # self.logger.info(f'non- {st} {sz_st}')
                        dt = dat.copy()
                        dt = dt.crop(tmin=st, tmax=sz_st)
                        nonseiz.append(dt)

                    # self.logger.info(f'seiz- {sz_st} {sz_end}')
                    dt = dat.copy()
                    dt = dt.crop(tmin=sz_st, tmax=sz_end)
                    seiz.append(dt)
                    sz_cnt += 1
                    if sz_cnt<rec.num_seizures:
                        st = sz_end
                    else:
                        dt = dat.crop(tmin=sz_end, tmax=end)
                        nonseiz.append(dt)

                nonseizure_data.append(nonseiz)
                seizure_data.append(seiz)
                tmp_data[1].append((rec.record_no, nonseiz, seiz)) ###


            data_list.append(tmp_data)  ###
            patients_list_copy.append(pat_copy)

        # return patients_list_copy, nonseizure_data, seizure_data  ###
        return data_list  ###


    # ###### Main Tasks Raw (Not in use)
    #
    # def create_features_dataset_from_raw_signals_segment(self, patients:list, segment_duration_in_seconds:int=5, overlap_duration_in_seconds:int=0):
    #
    #     patients_copy = copy.deepcopy(patients)
    #
    #     for pat in patients_copy:
    #         pat_copy = copy.deepcopy(pat)
    #         # self.logger.info(f'{pat_copy.__dict__}')
    #         for rec in pat_copy.records:
    #             # self.logger.info(f'{rec.__dict__}')
    #             data: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])
    #             # data.plot()
    #             # data.pick_channels(['FP1-F7', 'F7-T7'])
    #             # data.plot()
    #
    #             self.logger.info(f'{pat_copy.channels} {pat_copy.removable_channels_[rec.channel_id]}')
    #             for ch in data.ch_names:
    #                 # if ch.count('-')>1:
    #                 #     ch = ch[:ch.rfind('-')]
    #                 self.logger.info(f'"{ch}"')
    #                 dat: RawEDF = data.copy().pick_channels([ch])
    #                 # dat.plot()
    #
    #                 # finished = false
    #                 seg_st = 0
    #                 seg_end = segment_duration_in_seconds
    #                 end_time = len(data)/int(pat_copy.sam_rate)-0.01
    #                 # self.logger.info(f'{seg_st} {seg_end}')
    #                 wind_mov = segment_duration_in_seconds-overlap_duration_in_seconds
    #
    #                 self.logger.info(f'Seizure Times: {rec.seizure_times}')
    #
    #                 # while not finished:
    #                 while seg_st<end_time:
    #                     # self.logger.info(f'{seg_st} {seg_end}')
    #                     # This is a window of data from which the features are extracted
    #                     dt = dat.copy().crop(tmin=seg_st, tmax=seg_end)
    #                     # dt.plot()
    #                     # self.logger.info(f'{type(dt)} ---> {dt.__dict__}')
    #
    #                     # This dataframe is reading one extra datapoint
    #                     df = dt.to_data_frame()
    #                     self.logger.info(f'{type(df)} ===> {df.head()} {df.tail()} {len(df)}')
    #
    #                     # Is the segment an event | Thi is the label of the data
    #                     if self._is_the_segment_an_event(rec, seg_st, seg_end, segment_duration_in_seconds):
    #                         self.logger.info(f'---> Seizure found in this segment: {seg_st} {seg_end}')
    #
    #                     seg_st += wind_mov
    #                     nxt_seg = seg_end + wind_mov
    #                     seg_end = min(nxt_seg, end_time)
    #
    #     return



########################################################################################################################
    ###### Main Tasks

    def _is_the_segment_an_event(self, rec, samp_rate, seg_st, seg_end, seg_duration):

        # self.logger.info(f'++++ {rec.seizure_times} {seg_st} {seg_end} {seg_duration}')
        segment_is_event = 0
        tmp_st = seg_st
        tmp_end = seg_end
        for seiz_time in rec.seizure_times:
            seiz_st, seiz_end = seiz_time
            seiz_st = int(seiz_st)*samp_rate
            seiz_end = int(seiz_end)*samp_rate
            seiz_duration = seiz_end-seiz_st

            # self.logger.info(f'### {seiz_st} {seiz_end}')
            if (seiz_st<=seg_st and seg_st<seiz_end) or (seiz_st<seg_end and seg_end<=seiz_end):
                tmp_st = max(seg_st, seiz_st)
                tmp_end = min(seg_end, seiz_end)

                tmp_prob = (((tmp_end-tmp_st)/seg_duration)*100)
                # self.logger.info(f'prob, {tmp_st} {tmp_end} {tmp_prob} {self._event_threshold_percentage}')

                # Start of the seizure
                if seiz_st>=seg_st and (seg_end-seiz_st)<=seg_duration:
                    segment_is_event = 2
                # End of the seizure
                elif seg_st<seiz_end and (seiz_end-seg_st)<=seg_duration:
                    segment_is_event = 3
                elif tmp_prob >= self._event_threshold_percentage:
                    segment_is_event = 1
                else:
                    segment_is_event = 0

        return segment_is_event


    def create_signals_segment(self, segment_save_location:str, patients:list, segment_duration_in_seconds:int=5, overlap_duration_in_seconds:int=0, overlap_duration_in_percentage:int=0, overlap_duration_in_samples:int=0):

        self.logger.info(f'--------------------------------------------\n### Segment generation started\n--------------------------------------------')
        patients_copy = copy.deepcopy(patients)

        for pat in patients_copy:
            util2 = Humachlab_Utility()
            util2.start_timer()

            pat_copy = copy.deepcopy(pat)
            # self.logger.info(f'{pat_copy.__dict__}')

            # ##########################################################
            # About segmentation and overlapping
            samp_rate = int(pat_copy.sam_rate)
            segment_duration_in_samples = (segment_duration_in_seconds*samp_rate)

            if overlap_duration_in_seconds > 0:
                overlap_duration_in_samples = (overlap_duration_in_seconds*samp_rate)
            elif overlap_duration_in_percentage>0:
                overlap_duration_in_samples = ((overlap_duration_in_percentage*segment_duration_in_seconds)*samp_rate)

            self.logger.info(f'Segment Duration: {segment_duration_in_seconds}, Sample Rate: {samp_rate}')
            self.logger.info(f'Durations: {segment_duration_in_samples}, Overlap: {overlap_duration_in_samples}')
            # ==========================================================

            for rec in pat_copy.records:
                util3 = Humachlab_Utility()
                util3.start_timer()
                # self.logger.info(f'{rec.__dict__}')
                # ##########################################################
                # Reading full raw data file
                data: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])

                if not data:
                    continue

                data_frame = data.to_data_frame()
                self.logger.info(f'Record shape: {data_frame.shape}')

                # chnl = list(pat_copy.channels[rec.channel_id].values())
                chnl = pat_copy.channels[rec.channel_id].values()
                self.logger.info(f'THIS IS NEW= {data.ch_names} \n{chnl}')

                # self.logger.info(f'{pat_copy.channels} {pat_copy.removable_channels_[rec.channel_id]}')
                for ch in data.ch_names:
                    util4 = Humachlab_Utility()
                    util4.start_timer()
                    self.logger.info(f'Started for channel: "{ch}"')
                    data_frame_channel = data_frame[[ch]]
                    index = data_frame_channel.index

                    # finished = false
                    # ##########################################################
                    # Segmentation based on sample, calculation
                    seg_st = 0
                    seg_end = segment_duration_in_samples
                    end_time = len(index)
                    # self.logger.info(f'{seg_st} {seg_end}')
                    wind_mov = segment_duration_in_samples-overlap_duration_in_samples

                    self.logger.info(f'Seizure Times: {rec.seizure_times}')
                    i=1

                    # while not finished:
                    # ##########################################################
                    # Start segmenting data
                    extracted_features = pd.DataFrame()
                    feature_file_name = ''
                    while seg_st<end_time:
                        # self.logger.info(f'{seg_st} {seg_end}')
                        # This is a window of data from which the features are extracted
                        row_st = seg_st
                        row_end = seg_end

                        # ##########################################################
                        # Segmented data
                        data_frame_segment = data_frame_channel.iloc[row_st:row_end, :]

                        # ##########################################################
                        # Class determination of being seizure or non-seizure
                        is_seizure = self._is_the_segment_an_event(rec, samp_rate, seg_st, seg_end, segment_duration_in_samples)

                        is_seizure = 1 if is_seizure>0 else 0

                        # ##########################################################
                        # Extracting features to be saved to the file
                        # self.logger.info(f'==> {extracted_features}')
                        if not os.path.isdir(segment_save_location):
                            os.mkdir(segment_save_location)
                        feature_file_name = f'{segment_save_location}{self.seg_file_name}'
                        feature_file_name += f'{pat.patient_no:02d}_'
                        feature_file_name += f'{rec.record_no:02d}_'
                        feature_file_name += f'{(data.ch_names).index(ch):02d}_{ch}_{i}_{is_seizure}'
                        # feature_file_name += '{:02d}_{}_{}_{}'.format((chnl).index(ch), ch, i, is_seizure)
                        feature_file_name += ".csv"

                        self.logger.info(f'{feature_file_name} | {i} ===> {row_st} {row_end}')

                        data_frame_segment.to_csv(feature_file_name, index=False)

                        i += 1
                        seg_st += wind_mov
                        nxt_seg = seg_end + wind_mov
                        seg_end = min(nxt_seg, end_time)

                    self.logger.info(f'Segments are written to file {feature_file_name}...')
                    # --------------------------------------------
                    util4.end_timer()
                    elt = util4.time_calculator()
                    self.logger.info(f'*** Feature extraction finished for channel: {ch} in {elt}')
                util3.end_timer()
                elt = util3.time_calculator()
                self.logger.info(f'@@@ Feature extraction finished for record: {rec.record_no} in {elt}')
            util2.end_timer()
            elt = util2.time_calculator()
            self.logger.info(f'### Feature extraction finished for patient: {pat.patient_no} in {elt}')
        self.logger.info(f'--------------------------------------------\n### Segment generation ended\n--------------------------------------------')

        return



    # def create_features_dataset_from_signals_segment(self, patients:list, segment_duration_in_seconds:int=5, overlap_duration_in_seconds:int=0, overlap_duration_in_percentage:int=0, overlap_duration_in_samples:int=0):
    #
    #     patients_copy = copy.deepcopy(patients)
    #
    #     if overlap_duration_in_samples > 0:
    #         overlap_duration_in_seconds = overlap_duration_in_samples / patients_copy.sam_rate
    #     elif overlap_duration_in_percentage>0:
    #         overlap_duration_in_seconds = overlap_duration_in_percentage*overlap_duration_in_seconds
    #
    #     for pat in patients_copy:
    #         pat_copy = copy.deepcopy(pat)
    #         # self.logger.info(f'{pat_copy.__dict__}')
    #         for rec in pat_copy.records:
    #             # self.logger.info(f'{rec.__dict__}')
    #             data: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])
    #             data_frame = data.to_data_frame()
    #             self.logger.info(f'Record shape: {data_frame.shape}')
    #
    #             self.logger.info(f'{pat_copy.channels} {pat_copy.removable_channels_[rec.channel_id]}')
    #             for ch in data.ch_names:
    #                 self.logger.info(f'"{ch}"')
    #                 data_frame_channel = data_frame[[ch]]
    #
    #                 # finished = false
    #                 seg_st = 0
    #                 seg_end = segment_duration_in_seconds
    #                 end_time = len(data)/int(pat_copy.sam_rate)
    #                 # self.logger.info(f'{seg_st} {seg_end}')
    #                 wind_mov = segment_duration_in_seconds-overlap_duration_in_seconds
    #
    #                 self.logger.info(f'Seizure Times: {rec.seizure_times}')
    #                 i=1
    #
    #                 # while not finished:
    #                 extracted_features = pd.DataFrame()
    #                 while seg_st<end_time:
    #                     # self.logger.info(f'{seg_st} {seg_end}')
    #                     # This is a window of data from which the features are extracted
    #                     samp_rate = int(pat_copy.sam_rate)
    #                     row_st = (seg_st*samp_rate)
    #                     row_end = (seg_end*samp_rate)
    #                     self.logger.info(f'{i} ===> {row_st} {row_end}')
    #                     i += 1
    #                     data_frame_segment =data_frame_channel.iloc[row_st:row_end, :]
    #
    #                     # --------------------------------------------
    #                     # Is the segment an event | This is the label of the data
    #                     is_seizure = self._is_the_segment_an_event(rec, seg_st, seg_end, segment_duration_in_seconds)
    #
    #                     # Extract features and labels from the data --------------------------------------------
    #                     feat_ext_obj = HumachLab_FeatureExtractor(data_frame_segment)
    #                     extr_feat = feat_ext_obj.get_all_features(feature_types=1)
    #                     extr_feat[self.class_name] = is_seizure
    #
    #                     # if seg_st==0:
    #                     #     extracted_features = pd.DataFrame(columns=extr_feat.columns)
    #
    #                     extracted_features = extracted_features.append(extr_feat, ignore_index = True)
    #                     #--------------------------------------------
    #
    #
    #                     seg_st += wind_mov
    #                     nxt_seg = seg_end + wind_mov
    #                     seg_end = min(nxt_seg, end_time)
    #
    #                 # Extract features and labels save in a file --------------------------------------------
    #                 # self.logger.info(f'==> {extracted_features}')
    #                 feature_file_name = './FeaturesData/fchb_'
    #                 feature_file_name += '{:02d}_'.format(pat.patient_no)
    #                 feature_file_name += '{:02d}_'.format(rec.record_no)
    #                 feature_file_name += '{:02d}_{}'.format((data.ch_names).index(ch), ch)
    #                 feature_file_name += ".csv"
    #                 # self.logger.info(f'==> {feature_file_name}')
    #                 extracted_features.to_csv(feature_file_name, index=False)
    #                 self.logger.info(f'Features written to file {feature_file_name}...')
    #                 # --------------------------------------------
    #
    #     return


    def create_features_dataset_from_signals_segment(self, feature_save_location:str, patients:list, segment_duration_in_seconds:int=5, overlap_duration_in_seconds:int=0, overlap_duration_in_percentage:int=0, overlap_duration_in_samples:int=0):

        # # ### If data correction is needed:
        # self.correct_feature_data(feature_save_location, patients, segment_duration_in_seconds, overlap_duration_in_seconds, overlap_duration_in_percentage, overlap_duration_in_samples)
        # return

        self.logger.info(f'--------------------------------------------\n### Feature extraction started\n--------------------------------------------')
        # self.logger.info(f'--------------------- {len(patients)}')
        patients_copy = copy.deepcopy(patients)
        mat_eng = self.manage_matlab_python_engine()

        for pat in patients_copy:
            util2 = Humachlab_Utility()
            util2.start_timer()

            pat_copy = copy.deepcopy(pat)
            # self.logger.info(f'{pat_copy.__dict__}')

            # ##########################################################
            # About segmentation and overlapping
            samp_rate = int(pat_copy.sam_rate)
            segment_duration_in_samples = (segment_duration_in_seconds*samp_rate)

            if overlap_duration_in_seconds > 0:
                overlap_duration_in_samples = (overlap_duration_in_seconds*samp_rate)
            elif overlap_duration_in_percentage>0:
                overlap_duration_in_samples = ((overlap_duration_in_percentage*segment_duration_in_seconds)*samp_rate)

            self.logger.info(f'Segment Duration: {segment_duration_in_seconds}, Overlap Duration: {overlap_duration_in_seconds}, Sample Rate: {samp_rate}')
            self.logger.info(f'Durations: {segment_duration_in_samples}, Overlap: {overlap_duration_in_samples}')
            # ==========================================================

            for rec in pat_copy.records:
                util3 = Humachlab_Utility()
                util3.start_timer()
                # self.logger.info(f'{rec.__dict__}')
                # ##########################################################
                # Reading full raw data file
                data: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])

                if not data:
                    continue

                data_frame = data.to_data_frame()
                self.logger.info(f'Record shape: {data_frame.shape}')

                chnl = list(pat_copy.channels[rec.channel_id].values())
                self.logger.info(f'THIS IS NEW= {data.ch_names} \n{chnl}')
                # continue

                # self.logger.info(f'{pat_copy.channels} {pat_copy.removable_channels_[rec.channel_id]}')
                for ch in data.ch_names:
                    util4 = Humachlab_Utility()
                    util4.start_timer()
                    self.logger.info(f'Started for channel: "{ch}"')
                    data_frame_channel = data_frame[[ch]]
                    index = data_frame_channel.index

                    # ##########################################################
                    # Extracting features to be saved to the file
                    # self.logger.info(f'==> {extracted_features}')
                    if not os.path.isdir(feature_save_location):
                        os.mkdir(feature_save_location)
                    feature_file_name = f'{feature_save_location}{self.feat_file_name}'
                    feature_file_name += f'{pat.patient_no:02d}_'
                    feature_file_name += f'{rec.record_no:02d}_'
                    feature_file_name += f'{(chnl).index(ch):02d}_{ch}'
                    feature_file_name += ".csv"

                    # finished = false
                    # ##########################################################
                    # Segmentation based on sample, calculation
                    seg_st = 0
                    seg_end = segment_duration_in_samples
                    end_time = len(index)
                    # self.logger.info(f'{seg_st} {seg_end}')
                    wind_mov = segment_duration_in_samples-overlap_duration_in_samples

                    self.logger.info(f'Seizure Times: {rec.seizure_times}, \nMove window:{wind_mov}')

                    # ####################################################
                    # Create Feature extraction object

                    feat_ext_obj = HumachLab_FeatureExtractor(self.logger, ch,
                                                              manage_exceptional_data=0, signal_frequency=samp_rate,
                                                              sample_per_second=segment_duration_in_samples,
                                                              filtering_enabled=True)
                    feat_ext_obj.matlab_engine = mat_eng
                    feature_names_to_calculate = feat_ext_obj.get_new_features_to_calculate()

                    # ##########################################################
                    # Get list of features already extracted, so that no multiple worst be done
                    saved_data = pd.DataFrame()
                    existing_csv_columns = []
                    calculate_feats_again = True

                    if os.path.isfile(feature_file_name):
                        # saved_data = pd.read_csv(feature_file_name, nrows=1)
                        # existing_csv_columns = saved_data.columns.tolist()[(len(self.main_meta_columns)) :]

                        ### Calculate again the file having partially feature extracted or all the same columns existed
                        saved_data = pd.read_csv(feature_file_name)
                        existing_csv_columns = saved_data.columns.tolist() #[(len(self.main_meta_columns)):]
                        feature_names_to_calculate = feat_ext_obj.get_new_features_to_calculate(already_existing_features=existing_csv_columns[(len(self.main_meta_columns)):], feature_types=0)
                        calculate_feats_again = False
                        future_num_segs_from_edf = int(np.ceil((data_frame_channel.shape[0]-overlap_duration_in_samples)/wind_mov)) #MODIFY
                        self.logger.info(f'====> Number of segments in edf: {saved_data.shape[0]} in feature file: {future_num_segs_from_edf} | Read again: {saved_data.shape[0] != future_num_segs_from_edf}')

                        ### Calculate again the file having partially feature extracted
                        if saved_data.shape[0] != future_num_segs_from_edf:
                            # print(f'here to remove...')
                            self.logger.info(f'Prob: {pat.patient_no}-{rec.record_no}')
                            with open("prob.txt", "a+") as file_object:
                                file_object.write(f'{pat.patient_no}-{rec.record_no}\n')
                            # continue;
                            existing_csv_columns = []
                            feature_names_to_calculate = feat_ext_obj.get_new_features_to_calculate()
                            calculate_feats_again = True
                            try:
                                os.remove(feature_file_name)
                            except:
                                self.logger.info(f'Error removing existing feature file...')
                                exit(0)
                        # print('XXXXXXXXXX', len(existing_csv_columns), (len(feature_names_to_calculate)))
                        # if len(existing_csv_columns) < (len(feature_names_to_calculate)+1):
                        if len(feature_names_to_calculate)>0:
                            # print(f'here to add new feats...')
                            # existing_csv_columns = []
                            calculate_feats_again = True


                    # while not finished:
                    # ##########################################################
                    # Start segmenting data
                    new_features_ordered_list = feat_ext_obj.all_feature_list
                    extracted_features = pd.DataFrame()
                    i = 1

                    while ((seg_st<end_time) and calculate_feats_again):
                        # self.logger.info(f'{seg_st} {seg_end}')
                        # This is a window of data from which the features are extracted
                        row_st = seg_st
                        row_end = seg_end
                        self.logger.info(f'{feature_file_name} | {i:>4} ===> {row_st:>6} {row_end:>6}')
                        # ##########################################################
                        # Segmented data
                        data_frame_segment = data_frame_channel.iloc[row_st:row_end, :]
                        # self.logger.info(f'{type(data_frame_segment)} === {data_frame_segment}')

                        # ##########################################################
                        # Class determination of being seizure or non-seizure
                        is_seizure = self._is_the_segment_an_event(rec, samp_rate, seg_st, seg_end, segment_duration_in_samples)
                        new_data = {self.class_name: [is_seizure]}
                        extr_feat = pd.DataFrame(new_data)

                        # ##########################################################
                        # Extracting features from the segment
                        # self.logger.info(f'{data_frame_segment.head()}')
                        # feat_ext_obj = HumachLab_FeatureExtractor(self.logger, data_frame_segment, ch, manage_exceptional_data=0, signal_frequency=samp_rate, sample_per_second=segment_duration_in_samples, filtering_enabled=True)
                        # feat_ext_obj.matlab_engine = mat_eng
                        # new_feats = feat_ext_obj.get_all_features(already_existing_features=existing_csv_columns, feature_types=0)
                        new_feats = feat_ext_obj.get_all_features(data_frame_segment, feature_names_to_calculate)
                        extr_feats = extr_feat.join([new_feats])

                        # self.logger.info(f'{extr_feats}')
                        # if seg_st==0:
                        #     extracted_features = pd.DataFrame(columns=extr_feat.columns)

                        extracted_features = extracted_features.append(extr_feats, ignore_index = True)

                        # ### Observe extra null or nan entries
                        # if (new_feats.isnull().values.any()) or (i==40):
                        #     print(f'NaN Data and Features to write for segment {i}')
                        #     new_feats.to_csv(f'./tst_feats_{i}.csv')
                        #     data_frame_segment.to_csv(f'./tst_data_{i}.csv')
                        #--------------------------------------------

                        i += 1
                        seg_st += wind_mov
                        nxt_seg = seg_end + wind_mov
                        seg_end = min(nxt_seg, end_time)

                    # ##########################################################
                    # Checking for the features already saved, then marge with new features if any
                    addable_columns_from_new_features = extracted_features.columns.tolist()
                    if os.path.isfile(feature_file_name):
                        # saved_data = pd.read_csv(feature_file_name)
                        # existing_csv_columns = saved_data.columns.tolist()#[1:]
                        new_features_columns = addable_columns_from_new_features.copy() #extracted_features.columns.tolist()
                        # print(f'new feats cols: {new_features_columns} existing feats: {existing_csv_columns}')
                        addable_columns_from_new_features = [i for i in (new_features_columns + existing_csv_columns) if i not in existing_csv_columns]
                        # print(f'addable new feats cols: {addable_columns_from_new_features}')
                        addable_features = extracted_features[addable_columns_from_new_features]
                        # print(f'addable features: {len(addable_features)}')
                        # self.logger.info(f'{addable_columns_from_new_features} {addable_features}')
                        extracted_features = saved_data.join(addable_features)
                        # self.logger.info(f'{extracted_features}')

                    # existing_csv_columns = existing_csv_columns[(len(self.main_meta_columns)) :]
                    # self.logger.info(f'{existing_csv_columns} {new_features_ordered_list} {len(existing_csv_columns)} {len(new_features_ordered_list)} and {len(existing_csv_columns)} == {sum([1 for i, j in zip(existing_csv_columns, new_features_ordered_list) if i == j])}')

                    # self.logger.info(f'Features--- {extracted_features}')
                    # print(f'new feat order list: {len(new_features_ordered_list)}, addable new features: {len(addable_columns_from_new_features)}')

                    reordering_cond = (len(existing_csv_columns) == (len(new_features_ordered_list)+1) and not(len(existing_csv_columns) == (sum([1 for i, j in zip(existing_csv_columns, new_features_ordered_list) if i == j])+1)))
                    reordering_cond = not(len(existing_csv_columns) == (sum([1 for i, j in zip(existing_csv_columns, new_features_ordered_list) if i == j])+1))
                    # ''' Open up to  just see features not to write in the file: if new feature is encountered or order is changed
                    if len(addable_columns_from_new_features)>0 or reordering_cond:
                        #Reordering datafrmae
                        s1 = set(new_features_ordered_list)
                        s2 = set(extracted_features.columns.tolist())
                        if len(list((s1.union(s2).difference(s1.intersection(s2))))) == 0:
                            extracted_features = extracted_features[new_features_ordered_list]

                        # if reordering_cond:
                        #     print(f'reordering....')
                        new_features_ordered_list.insert(0, self.class_name)
                        extracted_features = extracted_features[new_features_ordered_list]

                        extracted_features.to_csv(feature_file_name, index=False)
                        self.logger.info(f'Features written to file {feature_file_name}...')
                    else:
                        self.logger.info(f'Features file {feature_file_name} already exist with all features...')
                    # --------------------------------------------
                    util4.end_timer()
                    elt = util4.time_calculator()
                    self.logger.info(f'*** Feature extraction finished for channel: {ch} in {elt}')
                util3.end_timer()
                elt = util3.time_calculator()
                self.logger.info(f'@@@ Feature extraction finished for record: {rec.record_no} in {elt}')
            util2.end_timer()
            elt = util2.time_calculator()
            self.logger.info(f'### Feature extraction finished for patient: {pat.patient_no} in {elt}')
        self.logger.info(f'--------------------------------------------\n### Feature extraction ended\n--------------------------------------------')
        # '''
        self.manage_matlab_python_engine(existing_eng=mat_eng)

        return


    def correct_feature_data(self, feature_save_location:str, patients:list, segment_duration_in_seconds:int=5, overlap_duration_in_seconds:int=0, overlap_duration_in_percentage:int=0, overlap_duration_in_samples:int=0):

        self.logger.info(f'--------------------------------------------\n### Feature Correction started\n--------------------------------------------')
        task_id = 1 #1=replace row, 2=remove row, 3=replace column, 4=remove column
        if task_id==0:
            return

        # self.logger.info(f'--------------------- {len(patients)}')
        patients_copy = copy.deepcopy(patients)
        mat_eng = self.manage_matlab_python_engine()

        for pat in patients_copy:
            util2 = Humachlab_Utility()
            util2.start_timer()

            pat_copy = copy.deepcopy(pat)
            # self.logger.info(f'{pat_copy.__dict__}')

            # ##########################################################
            # About segmentation and overlapping
            samp_rate = int(pat_copy.sam_rate)
            segment_duration_in_samples = (segment_duration_in_seconds*samp_rate)

            if overlap_duration_in_seconds > 0:
                overlap_duration_in_samples = (overlap_duration_in_seconds*samp_rate)
            elif overlap_duration_in_percentage>0:
                overlap_duration_in_samples = ((overlap_duration_in_percentage*segment_duration_in_seconds)*samp_rate)

            self.logger.info(f'Segment Duration: {segment_duration_in_seconds}, Sample Rate: {samp_rate}')
            self.logger.info(f'Durations: {segment_duration_in_samples}, Overlap: {overlap_duration_in_samples}')
            # ==========================================================

            for rec in pat_copy.records:
                util3 = Humachlab_Utility()
                util3.start_timer()
                # self.logger.info(f'{rec.__dict__}')
                # ##########################################################
                # Reading full raw data file

                data: RawEDF = self._convert_specific_edf_from_file(rec.file_path, pat_copy.removable_channels_[rec.channel_id])

                if not data:
                    continue

                data_frame = data.to_data_frame()
                self.logger.info(f'Record shape: {data_frame.shape}')

                chnl = list(pat_copy.channels[rec.channel_id].values())
                self.logger.info(f'THIS IS NEW= {data.ch_names} \n{chnl}')
                # continue

                # self.logger.info(f'{pat_copy.channels} {pat_copy.removable_channels_[rec.channel_id]}')
                for ch in data.ch_names:
                    util4 = Humachlab_Utility()
                    util4.start_timer()
                    self.logger.info(f'Started for channel: "{ch}"')
                    data_frame_channel = data_frame[[ch]]
                    index = data_frame_channel.index

                    # ##########################################################
                    # Extracting features to be saved to the file
                    # self.logger.info(f'==> {extracted_features}')
                    if not os.path.isdir(feature_save_location):
                        os.mkdir(feature_save_location)
                    feature_file_name = f'{feature_save_location}{self.feat_file_name}'
                    feature_file_name += f'{pat.patient_no:02d}_'
                    feature_file_name += f'{rec.record_no:02d}_'
                    feature_file_name += f'{(chnl).index(ch):02d}_{ch}'
                    feature_file_name += ".csv"
                    print('feat file name: ', feature_file_name)

                    # ##########################################################
                    # Get list of features already extracted, so that no multiple worst be done
                    existing_csv_columns = []
                    saved_data = None
                    if os.path.isfile(feature_file_name):
                        saved_data = pd.read_csv(feature_file_name)
                        # saved_data = pd.read_csv(feature_file_name, nrows=1)
                        # existing_csv_columns = saved_data.columns.tolist()[(len(self.main_meta_columns)) :]
                    else:
                        continue

                    k = 0

                    if task_id==3:
                        continue

                    if task_id==4:
                        rem_col_name = 'spectralEntropy'
                        colss = list(saved_data.columns)
                        if rem_col_name in colss:
                            del saved_data[rem_col_name]
                            print(f'{rem_col_name} is removed')
                            k=1

                    if task_id==1:
                        rep_value = np.nan

                        # finished = false
                        # ##########################################################
                        # Segmentation based on sample, calculation
                        seg_st = 0
                        seg_end = segment_duration_in_samples
                        end_time = len(index)
                        # self.logger.info(f'{seg_st} {seg_end}')
                        wind_mov = segment_duration_in_samples-overlap_duration_in_samples

                        self.logger.info(f'Seizure Times: {rec.seizure_times}')
                        i = 1

                        # while not finished:
                        # ##########################################################
                        # Start segmenting data
                        new_features_ordered_list = []
                        extracted_features = pd.DataFrame()

                        while seg_st<end_time:
                            # self.logger.info(f'{seg_st} {seg_end}')
                            # This is a window of data from which the features are extracted
                            row_st = seg_st
                            row_end = seg_end
                            self.logger.info(f'{feature_file_name} | {i:>4} ===> {row_st:>6} {row_end:>6}')
                            # ##########################################################
                            # Segmented data
                            data_frame_segment = data_frame_channel.iloc[row_st:row_end, :]
                            # self.logger.info(f'{type(data_frame_segment)} === {data_frame_segment}')

                            # ##########################################################
                            # Class determination of being seizure or non-seizure
                            is_seizure = self._is_the_segment_an_event(rec, samp_rate, seg_st, seg_end, segment_duration_in_samples)
                            new_data = {self.class_name: [is_seizure]}
                            extr_feat = pd.DataFrame(new_data)

                            j = i-1
                            xdf = saved_data.iloc[[j]]

                            if xdf.isnull().values.any():
                                # ##########################################################
                                # Extracting features from the segment
                                # self.logger.info(f'{data_frame_segment.head()}')
                                feat_ext_obj = HumachLab_FeatureExtractor(self.logger, ch,
                                                                          manage_exceptional_data=1,
                                                                          signal_frequency=samp_rate,
                                                                          sample_per_second=segment_duration_in_samples,
                                                                          filtering_enabled=True)
                                feat_ext_obj.matlab_engine = mat_eng
                                feature_names_to_calculate = feat_ext_obj.get_new_features_to_calculate()

                                # feat_ext_obj = HumachLab_FeatureExtractor(self.logger, ch, manage_exceptional_data=0, signal_frequency=samp_rate, sample_per_second=segment_duration_in_samples, filtering_enabled=True)
                                # feat_ext_obj.matlab_engine = mat_eng
                                new_feats = feat_ext_obj.get_all_features(data_frame_segment, feature_names_to_calculate)
                                new_features_ordered_list = feat_ext_obj.all_feature_list
                                extr_feats = extr_feat.join([new_feats])

                                saved_data = pd.concat([saved_data.iloc[:j], extr_feats, saved_data.iloc[j + 1:]]).reset_index(drop=True)
                                k += 1

                                print(f'Row replaced {i}')

                                # self.logger.info(f'{extr_feats}')
                                # if seg_st==0:
                                #     extracted_features = pd.DataFrame(columns=extr_feat.columns)

                                # extracted_features = extracted_features.append(extr_feats, ignore_index = True)

                            # ### Observe extra null or nan entries
                            # if (new_feats.isnull().values.any()) or (i==40):
                            #     print(f'NaN Data and Features to write for segment {i}')
                            #     new_feats.to_csv(f'./tst_feats_{i}.csv')
                            #     data_frame_segment.to_csv(f'./tst_data_{i}.csv')
                            #--------------------------------------------

                            i += 1
                            seg_st += wind_mov
                            nxt_seg = seg_end + wind_mov
                            seg_end = min(nxt_seg, end_time)

                    if k>0:
                        saved_data.to_csv(feature_file_name, index=False)
                        self.logger.info(f'Features re-written to file {feature_file_name}...')
                    else:
                        print('no need to change')

                    util4.end_timer()
                    elt = util4.time_calculator()
                    self.logger.info(f'*** Feature correction finished for channel: {ch} in {elt}')
                util3.end_timer()
                elt = util3.time_calculator()
                self.logger.info(f'@@@ Feature correction finished for record: {rec.record_no} in {elt}')
            util2.end_timer()
            elt = util2.time_calculator()
            self.logger.info(f'### Feature correction finished for patient: {pat.patient_no} in {elt}')
        self.logger.info(f'--------------------------------------------\n### Feature correction ended\n--------------------------------------------')
        # '''
        self.manage_matlab_python_engine(existing_eng=mat_eng)

        return


