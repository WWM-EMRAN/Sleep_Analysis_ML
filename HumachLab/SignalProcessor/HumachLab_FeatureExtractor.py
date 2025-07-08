# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:55:47 2020
@author: WWM Emran (wwm.emran@gmail.com)
Copyright and support: HumachLab (humachlab@gmail.com)
"""


###
import math
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import math
import collections
from scipy.stats import entropy as scipyEntropy
from scipy.signal import butter, lfilter, welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
# from scipy.fft import fft
from scipy import fft, fftpack
# from math import log, floor
import math

from antropy import *
import pyeeg

# from numba import jit
# from math import factorial, log
# from sklearn.neighbors import KDTree
# from scipy.signal import periodogram, welch, butter, lfilter

# from utils import _linear_regression, _log_n

import matlab.engine
# from utils import _embed

### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.SignalProcessor.HumachLab_FeatureDetails import *
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###




###
class HumachLab_FeatureExtractor:

    feature_list = None
    manage_exceptional_data = None

    def __init__(self, logger, channel_name, manage_exceptional_data=0, signal_frequency = 256, sample_per_second=1280, filtering_enabled=False, lowcut=1, highcut=48):
        self.logger = logger
        self.channel_name = channel_name
        self.manage_exceptional_data = manage_exceptional_data

        self.td_linear_statistical = HumachLab_FeatureDetails.td_linear_statistical
        self.td_nonlinear_entropy = HumachLab_FeatureDetails.td_nonlinear_entropy
        self.td_nonlinear_complexity_and_fractal_dimensions = HumachLab_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
        # 'hurstExponent',
        self.td_nonlinear_samp_entropy_profiling = HumachLab_FeatureDetails.td_nonlinear_samp_entropy_profiling
        # other means gamma frequency
        self.fd_linear_statistical = HumachLab_FeatureDetails.fd_linear_statistical
        self.fd_linear_statistical_binwise = HumachLab_FeatureDetails.fd_linear_statistical_binwise
        self.fd_spectral_band_power = HumachLab_FeatureDetails.fd_spectral_band_power

        self.band_frequency_list = HumachLab_FeatureDetails.band_frequency_list

        self.band = (0, signal_frequency)

        # # CHB-MIT
        # self.feature_list = self.td_linear_statistical + self.td_nonlinear_entropy + self.td_nonlinear_complexity_and_fractal_dimensions + \
        #                     self.td_nonlinear_samp_entropy_profiling + self.fd_linear_statistical + self.fd_linear_statistical_binwise + \
        #                     self.fd_spectral_band_power
        # # Alfred
        self.feature_list = [] + self.td_linear_statistical# + self.fd_linear_statistical
        self.feature_list = self.td_linear_statistical + self.td_nonlinear_entropy + self.td_nonlinear_complexity_and_fractal_dimensions + \
                            self.td_nonlinear_samp_entropy_profiling + self.fd_linear_statistical + self.fd_linear_statistical_binwise + \
                            self.fd_spectral_band_power

        self.all_feature_list = []
        self.signal_frequency = signal_frequency
        self.sample_per_second = sample_per_second
        self.filtering_enabled = filtering_enabled
        self.lowcut = lowcut
        self.highcut = highcut
        if self.filtering_enabled:
            self.lowcut = lowcut
            self.highcut = highcut

        self.fd_data_dict = None
        self.entropy_profile = None
        self.matlab_engine = None

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

        if existing_eng is None:
            eng = None
            if found:
                import matlab.engine
                eng = matlab.engine.start_matlab()
            elif (not found) and (os.path.exists(mat_bld_path)):
                sys.path.append(mat_bld_path)
                import matlab.engine
                eng = matlab.engine.start_matlab()
            else:
                self.logger.info(f'No matlab is installed...')
            return eng
        else:
            existing_eng.quit()

        return


    def get_new_features_to_calculate(self, already_existing_features=[], feature_types=0):
        # Select features based on type
        if feature_types==0:
            feature_names = self.feature_list
        elif feature_types==1:
            feature_names = self.feature_list[:58] #self.feature_list[18:19] #self.feature_list[:18] #self.feature_list[4:10] #self.feature_list[:5] #self.feature_list[:16]
        else:
            feature_names = self.feature_list

        self.all_feature_list = feature_names
        # self.logger.info(f'{self.all_feature_list}')

        feature_names = [i for i in (feature_names + already_existing_features) if i not in already_existing_features]
        return feature_names


    ### Getting all the features
    #############################################################
    def get_all_features(self, data_frame_segment, feature_names):
        feature_values = []

        seg_values = data_frame_segment[self.channel_name].values.flatten() #np.array(data_frame_segment) # data_frame_segment.values.flatten()
        # seg_values = np.round(seg_values, decimals=20) # ### Don't know why but some features are getting NaN if this is not given, especially SpertralEntropy
        # seg_values = 1000*seg_values

        # Generate corresponding features
        for feat in feature_names:
            # print(feat)
            method = None
            final_feat = None
            try:
                final_feat = feat
                final_data = seg_values

                #Reuse appropriate method call
                if feat.startswith('fd_'):
                    if (feat in (self.fd_linear_statistical)) or (feat in (self.fd_linear_statistical_binwise)):
                        #FFT data for frequency domain features
                        data_dict = self.fd_data_dict
                        if (self.fd_data_dict is None):
                            data_dict = self.fd_spectralAmplitude(seg_values)
                            self.fd_data_dict = data_dict

                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)
                        if fnl>1:
                            final_feat = final_feat_list[1]
                            final_data = list(data_dict.values())

                            tmp = data_dict.keys()

                            if fnl > 2:
                                tmp = [i for i in tmp if i in range(self.band_frequency_list[final_feat_list[2]][0], self.band_frequency_list[final_feat_list[2]][1])]
                                final_data = [data_dict[x] for x in tmp]

                    elif (feat in (self.fd_spectral_band_power)):
                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)

                        if fnl > 1:
                            final_feat = f'{final_feat_list[0]}_{final_feat_list[1]}'
                            self.band = (0, self.signal_frequency)

                            if fnl > 2:
                                self.band = (self.band_frequency_list[final_feat_list[2]][0], self.band_frequency_list[final_feat_list[2]][1])

                elif feat.startswith('entropyProfiled_'):
                    enProf = self.entropy_profile.copy() if self.entropy_profile is not None else None
                    if self.entropy_profile is None:
                        enProf = self._call_entropy_profiling_algorithm(final_data)
                        dat = np.asarray(enProf)
                        dat2 = [0.0]
                        if len(enProf)>1:
                            dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        enProf = np.array(dat2)
                        self.entropy_profile = enProf.copy()

                    final_feat = (feat.split('_'))
                    # fnl = len(final_feat)
                    final_feat = final_feat[1]

                    final_data = enProf

                # print(feat, final_feat, seg_values, final_data)
                # self.logger.info(f'Calling... {final_feat} for feature {feat}')
                method = getattr(self, final_feat)

            except AttributeError:
                self.logger.info(f'Method for feature: {final_feat} is not implemented.')
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, final_feat))
                # return

            feat_val = 0
            # result = np.all(final_data == final_data[0])
            result = final_data
            if type(final_data) == list:
                #                 print('XXXX', final_data)
                if len(final_data) == 0:
                    result = True
                else:
                    result = np.all(final_data == final_data[0])
            else:
                result = np.all(final_data == final_data[0])

            if not result:
                feat_val = method(final_data)
                # # Handling nan data
                # if feat_val == np.nan:
                #     feat_val = 0

            # self.logger.info(f'{feat} -- {feat_val} -- {type(feat_val)}')
            feat_val = round(feat_val, 2)
            # feat_val = round(feat_val, 16)
            # self.logger.info(f'{feat} -- {feat_val}')
            feature_values.append([feat_val])

        # self.logger.info(f'{feature_names} -- {feature_values}')
        numpy_array = np.array(feature_values)
        numpy_array = numpy_array.T

        # self.logger.info(f'{len(feature_names)} {numpy_array.shape}')
        # self.logger.info(f'{numpy_array}')
        # self.logger.info(f'data--- {numpy_array} {feature_names}')

        all_features = pd.DataFrame()
        if len(feature_names)>0 and len(numpy_array)>0:
            all_features = pd.DataFrame(numpy_array, columns=feature_names)
        # self.logger.info(f'{all_features}')

        # ##########################################################
        # all_features = all_features[all_features != np.inf]
        # print(f'Data and type: ', type(all_features), all_features['spectralEntropy'])
        # if all_features.isnull().values.any():
        #     print(f'Infinity found: ', all_features)

        # Exceptional data management
        if self.manage_exceptional_data == 0:
            all_features = all_features[all_features != np.inf]
        elif self.manage_exceptional_data == 1:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(0)
        elif self.manage_exceptional_data == 2:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.dropna()
        elif self.manage_exceptional_data == 3:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(all_features.mean())

        return all_features

###########################################################################
### Time Domain Features

    ### Total value of the segment
    def total(self, data):
        tot = np.sum(data)
        return tot

    ### Summation value of the segment
    def summation(self, data):
        avg = np.sum(data)
        return avg

    ### Average value of the segment
    def average(self, data):
        avg = np.mean(data)
        return avg

    ### Minimum value of the segment
    def minimum(self, data):
        min = np.min(data)
        return min

    ### Maximum value of the segment
    def maximum(self, data):
        max = np.max(data)
        return max

    ### Mean value of the segment
    def mean(self, data):
        mean = np.mean(data)
        return mean

    ### Median value of the segment
    def median(self, data):
        med = np.median(data)
        return med

    ### Standard Deviation value of the segment
    def standardDeviation(self, data):
        std = np.std(data)
        return std

    ### Variance value of the segment
    def variance(self, data):
        var = np.var(data)
        return var

    ### kurtosis value of the segment
    def kurtosis(self, data):
        # kurtosis(y1, fisher=False)
        kur = sp.stats.kurtosis(data)
        return kur

    ### skewness value of the segment
    def skewness(self, data):
        skw = sp.stats.skew(data)
        return skw

    ### peak_or_Max value of the segment
    def peakOrMax(self, data):
        peak = self.maximum(data)
        return peak

    ### numberOfPeaks value of the segment
    def numberOfPeaks(self, data):
        # peaks, _ = find_peaks(x, distance=20)
        # peaks2, _ = find_peaks(x, prominence=1)  # BEST!
        # peaks3, _ = find_peaks(x, width=20)
        # peaks4, _ = find_peaks(x, threshold=0.4)
        numPeak = len(sig.find_peaks(data))
        return numPeak

    ### numberOfZeroCrossing value of the segment
    def numberOfZeroCrossing(self, data):
        # numZC1 = np.where(np.diff(np.sign(data)))[0]
        numZC = num_zerocross(data) #Antropy package
        return numZC #len(numZC1)

    ### positiveToNegativeSampleRatio value of the segment
    def positiveToNegativeSampleRatio(self, data):
        pnSampRatio = (np.sum(np.array(data) >= 0, axis=0)) / (np.sum(np.array(data) < 0, axis=0))
        return pnSampRatio

    ### positiveToNegativeSampleRatio value of the segment
    def positiveToNegativePeakRatio(self, data):
        pnPeakRatio = (len(sig.find_peaks(data))) / (len(sig.find_peaks(-data)))
        return pnPeakRatio

    ### meanAbsoluteValue value of the segment
    def meanAbsoluteValue(self, data):
        meanAbsVal = self.mean(abs(data))
        return meanAbsVal



############ Entropy
    # ### Collected from Antropy and pyeeg package

    def approximateEntropy(self, data):
        ae = app_entropy(data)
        return ae

    def sampleEntropy(self, data):
        se = sample_entropy(data)
        return se

    def permutationEntropy(self, data):
        pe = perm_entropy(data)
        return pe

    def spectralEntropy(self, data):
        sf = self.signal_frequency
        # se = spectral_entropy(data, sf)
        se = spectral_entropy(data, sf, method='welch')
        return se

    def singularValueDecompositionEntropy(self, data):
        svd_e = svd_entropy(data)
        return svd_e

    ############ Fracta dimension
    # Collected from Antropy and pyeeg packages

    def hjorthMobility(self, data):
        hjm, _ = hjorth_params(data)
        return hjm

    def hjorthComplexity(self, data):
        _, hjc = hjorth_params(data)
        return hjc

    def hurstExponent(self, data):
        hre = pyeeg.hurst(data)
        return hre

    def fisherInfo(self, data, tau=1, m=2):
        fsi = pyeeg.fisher_info(data, tau, m)
        return fsi

    def lempelZivComplexity(self, data):
        lzc = lziv_complexity(data)
        return lzc

    def petrosianFd(self, data):
        pfd = petrosian_fd(data)
        return pfd

    def katzFd(self, data):
        kfd = katz_fd(data)
        return kfd

    def higuchiFd(self, data):
        hfd = higuchi_fd(data)
        return hfd

    def detrendedFluctuation(self, data):
        dfl = detrended_fluctuation(data)
        return dfl


    ## ########
    # ### Collected from internet sources
    def fuzzyEntropy(self, data, m=2, tau=1, r=0.2):
        # r = float(0.2 * data.std())

        sig_seg_df_list = data.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            # eng = matlab.engine.start_matlab()
            eng = self.manage_matlab_python_engine()
            self.matlab_engine = eng
            is_gen = True

        # self.logger.info(f'{len(sig_seg_df_list)} {sig_seg_df_list}')
        fuzz_ent = eng.fuzzyEn(sig_seg_df_list, m, tau, r, nargout=1)
        # if isinstance(ent2, float):
        #     ent2 = [ent2]
        # ent = list(ent2)
        # self.logger.info(f'=== {len(ent)} {ent}')

        if is_gen:
            # eng.quit()
            self.manage_matlab_python_engine(existing_eng=eng)
        return fuzz_ent


    def distributionEntropy(self, data, m=2, M=500):

        sig_seg_df_list = data.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            eng = self.manage_matlab_python_engine()
            self.matlab_engine = eng
            is_gen = True

        # self.logger.info(f'{len(sig_seg_df_list)} {sig_seg_df_list}')
        dist_ent = eng.distributionEn(sig_seg_df_list, m, M, nargout=1)
        # if isinstance(ent2, float):
        #     ent2 = [ent2]
        # ent = list(ent2)
        # self.logger.info(f'=== {len(ent)} {ent}')

        if is_gen:
            self.manage_matlab_python_engine(existing_eng=eng)

        return dist_ent


    def distributionEntropy4(self, data, m=4):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy6(self, data, m=6):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy8(self, data, m=8):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy10(self, data, m=10):
        return self.distributionEntropy(data, m=m)


    def shannonEntropy(self, data, m=2):
        bases = collections.Counter([tmp_base for tmp_base in data])
        # define distribution
        dist = [i / sum(bases.values()) for i in bases.values()]

        # use scipy to calculate entropy
        entropy_value = scipyEntropy(dist, base=m)

        return entropy_value


    def _x_log2_x(self, data):
        """ Return x * log2(x) and 0 if x is 0."""
        results = data * np.log2(data)
        if np.size(data) == 1:
            if np.isclose(data, 0.0):
                results = 0.0
        else:
            results[np.isclose(data, 0.0)] = 0.0
        return results


    def renyiEntropy(self, data, alpha=2):
        assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(alpha)  # DEBUG
        if np.isinf(alpha):
            # XXX Min entropy!
            return - np.log2(np.max(data))
        elif np.isclose(alpha, 0):
            # XXX Max entropy!
            return np.log2(len(data))
        elif np.isclose(alpha, 1):
            # XXX Shannon entropy!
            return - np.sum(self._x_log2_x(data))
        else:
            return (1.0 / (1.0 - alpha)) * np.log2(np.sum(data ** alpha))


    ############ Entropy Profiling
    # Collected from @author: radhagayathri
    def _call_entropy_profiling_algorithm(self, data, m=2):

        sig_seg_df_list = data.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            eng = self.manage_matlab_python_engine()
            self.matlab_engine = eng
            is_gen = True

        # self.logger.info(f'{len(sig_seg_df_list)} {sig_seg_df_list}')
        ent2 = eng.sampEnProfiling(sig_seg_df_list, m, nargout=1)
        if isinstance(ent2, float):
            ent2 = [ent2]
        ent = list(ent2)
        # self.logger.info(f'=== {len(ent)} {ent}')

        if is_gen:
            self.manage_matlab_python_engine(existing_eng=eng)

        return ent


### Frequency Domain Features

    ##Bandpass filtering
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        tpl_res = butter(order, [low, high], btype='band')
        b, a = tpl_res[0], tpl_res[1]
        return b, a


    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    ##Fast Faurier Transformation
    def _fast_faurier_transformation(self, data, fs, fft_type=2):
        feat_data = None

        if fft_type==0:
            #Normal FFT using scipy
            fft_data = fft.fft(data)
            freqs = fft.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==1:
            #FFT for Amplitude calculation using fftpack
            fft_data = fftpack.fft(data)
            freqs = fftpack.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==2:
            #FFT for Amplitude calculation using fftpack
            fft_data = np.abs(np.fft.fft(data))
            freqs = np.fft.fftfreq(len(data), d=1.0 / fs)
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}

        return feat_data


    ### Original Frequency domain features
    def fd_spectralAmplitude(self, data):
        filtered_data = data
        sample_per_second = self.sample_per_second

        if self.filtering_enabled:
            lowcut = self.lowcut
            highcut = self.highcut
            filtered_data = self._butter_bandpass_filter(data, lowcut, highcut, sample_per_second, order=6)

        feat_data = self._fast_faurier_transformation(data, sample_per_second)

        return feat_data


    ### Power of a signal or signal band
    ### @Author: raphaelvallat (Author of Antropy package) Source:- https://raphaelvallat.com/bandpower.html
    def fd_bandPower(self, data, method='multitaper', window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
        Requires MNE-Python >= 0.14.
        Parameters
        ----------
        data : 1d-array :- Input signal in the time-domain.
        sf : float :- Sampling frequency of the data.
        band : list :- Lower and upper frequencies of the band of interest.
        method : string :- Periodogram method: 'welch' or 'multitaper'
        window_sec : float :- Length of each window in seconds. Useful only if method == 'welch'. If None, window_sec = (1 / min(band)) * 2.
        relative : boolean :-If True, return the relative power (= divided by the total power of the signal). If False (default), return the absolute power.
        ------
        Return
        ------
        bp : float :- Absolute or relative band power.
        ------
        Use
        ------
        # Multitaper delta power
        bp = fd_bandpower(data, sf, [0.5, 4], 'multitaper')
        311.559, and 0.790 (for relative band power)
        """
        sf = self.signal_frequency
        band = self.band

        band = np.asarray(band)
        low, high = band
        # Compute the modified periodogram (Welch)
        freqs, psd = 0, 0
        if method == 'welch':
            if window_sec is not None:
                nperseg = window_sec * sf
            else:
                nperseg = (2 / low) * sf
            freqs, psd = welch(data, sf, nperseg=nperseg)
        elif method == 'multitaper':
            psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp





    # ### Minimum value of the segment
    # def fd_minimum(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     min = np.min(data)
    #     return min
    #
    # ### Maximum value of the segment
    # def fd_maximum(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     max = np.max(data)
    #     return max
    #
    # ### Mean value of the segment
    # def fd_mean(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     mean = np.mean(data)
    #     return mean
    #
    # ### Median value of the segment
    # def fd_median(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     med = np.median(data)
    #     return med
    #
    # ### Summation value of the segment
    # def fd_summation(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     avg = np.sum(data)
    #     return avg
    #
    # ### Average value of the segment
    # def fd_average(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     avg = self.mean(data)
    #     return avg
    #
    # ### Standard Deviation value of the segment
    # def fd_standardDeviation(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     std = np.std(data)
    #     return std
    #
    # ### Variance value of the segment
    # def fd_variance(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     var = np.var(data)
    #     return var
    #
    # ### kurtosis value of the segment
    # def fd_kurtosis(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     # kurtosis(y1, fisher=False)
    #     kur = sp.stats.kurtosis(data)
    #     return kur
    #
    # ### skewness value of the segment
    # def fd_skewness(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     skw = sp.stats.skew(data)
    #     return skw

### Time-Frequency Domain Features

### Wevlate Domain Features




