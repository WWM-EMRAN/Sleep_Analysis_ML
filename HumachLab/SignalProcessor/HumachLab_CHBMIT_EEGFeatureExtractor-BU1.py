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
from scipy.stats import entropy

from numba import jit
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch, butter, lfilter
# from scipy.fft import fft
from scipy import fft, fftpack
from math import log, floor

from utils import _linear_regression, _log_n

from utils import _embed

# import matlab.engine



###
class HumachLab_CHBMIT_EEGFeatureExtractor:

    feature_list = None
    manage_exceptional_data = None

    def __init__(self, data_frame_segment, manage_exceptional_data=0, signal_frequency = 256, sample_per_second=1280, filtering_enabled=False, lowcut=1, highcut=48):
        self.data_frame_segment = data_frame_segment
        self.manage_exceptional_data = manage_exceptional_data

        td_statistical_features = ['maximum', 'minimum', 'mean', 'median', 'standardDeviation', 'variance', 'kurtosis',
                                   'skewness', 'numberOfZeroCrossing', 'positiveToNegativeSampleRatio', 'meanAbsoluteValue']
        td_nonlinear_features = ['permutationEntropy', 'spectralEntropy', 'singularValueDecompositionEntropy', 'approximateEntropy',
                                 'sampleEntropy', 'fuzzyEntropy', 'distributionEntropy', 'shannonEntropy', 'renyiEntropy', 'lempelZivComplexity']
        dist_en_variations = ['distributionEntropy4', 'distributionEntropy6', 'distributionEntropy8', 'distributionEntropy10']
        # td_nonlinear_features2_entropy_profiling = ['entropyProfiledTotalSampleEntropy', 'entropyProfiledAverageSampleEntropy', 'entropyProfiledMaximumSampleEntropy', 'entropyProfiledMinimumSampleEntropy',
        #                                             'entropyProfiledMedianSampleEntropy', 'entropyProfiledStandardDeviationSampleEntropy', 'entropyProfiledVarianceSampleEntropy',
        #                                             'entropyProfiledKurtosisSampleEntropy', 'entropyProfiledSkewnessSampleEntropy']
        td_nonlinear_features2_entropy_profiling = ['entropyProfiled_total_sampleEntropy', 'entropyProfiled_average_sampleEntropy', 'entropyProfiled_maximum_sampleEntropy',
                                                    'entropyProfiled_minimum_sampleEntropy', 'entropyProfiled_median_sampleEntropy', 'entropyProfiled_standardDeviation_sampleEntropy',
                                                    'entropyProfiled_variance_sampleEntropy', 'entropyProfiled_kurtosis_sampleEntropy', 'entropyProfiled_skewness_sampleEntropy']
        td_nonlinear_features3_fractal_dimensions = ['petrosianFd', 'katzFd', 'higuchiFd', 'detrendedFluctuation']
        fd_statistical_features = ['fd_maximum', 'fd_minimum', 'fd_mean', 'fd_median', 'fd_standardDeviation', 'fd_variance', 'fd_kurtosis', 'fd_skewness',
                                   'fd_maximum_alpha', 'fd_minimum_alpha', 'fd_mean_alpha', 'fd_median_alpha', 'fd_standardDeviation_alpha', 'fd_variance_alpha', 'fd_kurtosis_alpha', 'fd_skewness_alpha',
                                   'fd_maximum_beta', 'fd_minimum_beta', 'fd_mean_beta', 'fd_median_beta', 'fd_standardDeviation_beta', 'fd_variance_beta', 'fd_kurtosis_beta', 'fd_skewness_beta',
                                   'fd_maximum_delta', 'fd_minimum_delta', 'fd_mean_delta', 'fd_median_delta', 'fd_standardDeviation_delta', 'fd_variance_delta', 'fd_kurtosis_delta', 'fd_skewness_delta',
                                   'fd_maximum_theta', 'fd_minimum_theta', 'fd_mean_theta', 'fd_median_theta', 'fd_standardDeviation_theta', 'fd_variance_theta', 'fd_kurtosis_theta', 'fd_skewness_theta',
                                   'fd_maximum_other', 'fd_minimum_other', 'fd_mean_other', 'fd_median_other', 'fd_standardDeviation_other', 'fd_variance_other', 'fd_kurtosis_other', 'fd_skewness_other']

        # self.feature_list = td_statistical_features + td_nonlinear_features + dist_en_variations + td_nonlinear_features2_entropy_profiling + td_nonlinear_features3_fractal_dimensions + fd_statistical_features
        self.feature_list = td_statistical_features + td_nonlinear_features + td_nonlinear_features2_entropy_profiling + td_nonlinear_features3_fractal_dimensions + fd_statistical_features

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
                print('No matlab is installed...')
            return eng
        else:
            existing_eng.quit()

        return


    ### Getting all the features
    #############################################################
    def get_all_features(self, already_existing_features=[], feature_types=0):
        feature_names = None
        feature_values = []


        # Select features based on type
        if feature_types==0:
            feature_names = self.feature_list
        elif feature_types==1:
            feature_names = self.feature_list[:58] #self.feature_list[18:19] #self.feature_list[:18] #self.feature_list[4:10] #self.feature_list[:5] #self.feature_list[:16]
        else:
            feature_names = self.feature_list

        self.all_feature_list = feature_names
        # print(self.all_feature_list)

        feature_names = [i for i in (feature_names + already_existing_features) if i not in already_existing_features]

        seg_values = self.data_frame_segment.values.flatten()

        # Generate corresponding features
        for feat in feature_names:
            method = None
            try:
                final_feat = feat
                final_data = seg_values

                #Reuse appropriate method call
                if feat.startswith('fd_'):
                    #FFT data for frequency domain features
                    data_dict = self.fd_data_dict
                    if self.fd_data_dict is None:
                        data_dict = self.fd_spectralAmplitude(seg_values)
                        self.fd_data_dict = data_dict

                    final_feat = (feat.split('_'))
                    fnl = len(final_feat)
                    if fnl>1:
                        final_feat = final_feat[1]
                        final_data = list(data_dict.values())

                        tmp = data_dict.keys()
                        if feat.endswith('alpha'):
                            tmp = [i for i in tmp if i in range(8, 14)]
                        elif feat.endswith('beta'):
                            tmp = [i for i in tmp if i in range(14, 31)]
                        elif feat.endswith('delta'):
                            tmp = [i for i in tmp if i in range(0, 5)]
                        elif feat.endswith('theta'):
                            tmp = [i for i in tmp if i in range(5, 8)]
                        elif feat.endswith('other'):
                            tmp = [i for i in tmp if i in range(31, 51)]

                        if fnl > 2:
                            final_data = [data_dict[x] for x in tmp]

                elif feat.startswith('entropyProfiled_'):
                    enProf = self.entropy_profile
                    if self.entropy_profile is None:
                        enProf = self._call_entropy_profiling_algorithm(final_data)
                        dat = np.asarray(enProf)
                        dat2 = [0.0]
                        if len(enProf)>1:
                            dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        enProf = np.array(dat2)
                        self.entropy_profile = enProf

                    final_feat = (feat.split('_'))
                    # fnl = len(final_feat)
                    final_feat = final_feat[1]

                    final_data = enProf

                # print(f'Calling... {final_feat} for feature {feat}')
                method = getattr(self, final_feat)

            except AttributeError:
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, final_feat))
                return

            feat_val = 0
            result = np.all(final_data == final_data[0])
            if not result:
                feat_val = method(final_data)

            feat_val = round(feat_val, 2)
            # print(feat, '--', feat_val)
            feature_values.append([feat_val])

        # print(feature_names, '--', feature_values)
        numpy_array = np.array(feature_values)
        numpy_array = numpy_array.T

        # print(len(feature_names), numpy_array.shape)
        # print(numpy_array)
        # print('data---', numpy_array, feature_names)

        all_features = pd.DataFrame()
        if len(feature_names)>0 and len(numpy_array)>0:
            all_features = pd.DataFrame(numpy_array, columns=feature_names)
        # print(all_features)

        # ##########################################################
        # Exceptional data management
        if self.manage_exceptional_data == 1:
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

    ### Summation value of the segment
    def summation(self, data):
        avg = np.sum(data)
        return avg

    ### Average value of the segment
    def average(self, data):
        avg = self.mean(data)
        return avg

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
        numZC = np.where(np.diff(np.sign(data)))[0]
        return len(numZC)

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
    # Collected from @author: msrahman

    # all = ['permutationEntropy', 'spectralEntropy', 'singularValueDecompositionEntropy', 'approximateEntropy', 'sampleEntropy', 'lempelZivComplexity']

    def permutationEntropy(self, x, order=2, delay=1, normalize=False):
        """Permutation Entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times)
        order : int
            Order of permutation entropy. Default is 3.
        delay : int
            Time delay (lag). Default is 1.
        normalize : bool
            If True, divide by log2(order!) to normalize the entropy between 0
            and 1. Otherwise, return the permutation entropy in bit.

        Returns
        -------
        pe : float
            Permutation Entropy.

        Notes
        -----
        The permutation entropy is a complexity measure for time-series first
        introduced by Bandt and Pompe in 2002.

        The permutation entropy of a signal :math:`x` is defined as:

        .. math:: H = -\\sum p(\\pi)log_2(\\pi)

        where the sum runs over all :math:`n!` permutations :math:`\\pi` of order
        :math:`n`. This is the information contained in comparing :math:`n`
        consecutive values of the time series. It is clear that
        :math:`0 ≤ H (n) ≤ log_2(n!)` where the lower bound is attained for an
        increasing or decreasing sequence of values, and the upper bound for a
        completely random system where all :math:`n!` possible permutations appear
        with the same probability.

        The embedded matrix :math:`Y` is created by:

        .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

        .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T

        References
        ----------
        Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
        natural complexity measure for time series." Physical review letters
        88.17 (2002): 174102.

        Examples
        --------
        Permutation entropy with order 2

        >>> from entropy import permutationEntropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(permutationEntropy(x, order=2))
        0.9182958340544896

        Normalized permutation entropy with order 3

        >>> from entropy import permutationEntropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutationEntropy(x, order=3, normalize=True))
        0.5887621559162939
        """
        x = np.array(x)
        # print(x)
        ran_order = range(order)
        # print(x)
        hashmult = np.power(order, ran_order)
        # Embed x and sort the order of permutations
        sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
        # Associate unique integer to each permutations
        hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
        # Return the counts
        _, c = np.unique(hashval, return_counts=True)
        # Use np.true_divide for Python 2 compatibility
        p = np.true_divide(c, c.sum())
        pe = -np.multiply(p, np.log2(p)).sum()
        if normalize:
            pe /= np.log2(factorial(order))
        return pe


    def spectralEntropy(self, x, sf=256, method='fft', nperseg=None, normalize=False):
        """Spectral Entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times)
        sf : float
            Sampling frequency, in Hz.
        method : str
            Spectral estimation method:

            * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
            * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
        nperseg : int or None
            Length of each FFT segment for Welch method.
            If None (default), uses scipy default of 256 samples.
        normalize : bool
            If True, divide by log2(psd.size) to normalize the spectral entropy
            between 0 and 1. Otherwise, return the spectral entropy in bit.

        Returns
        -------
        se : float
            Spectral Entropy

        Notes
        -----
        Spectral Entropy is defined to be the Shannon entropy of the power
        spectral density (PSD) of the data:

        .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) log_2[P(f)]

        Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
        frequency.

        References
        ----------
        Inouye, T. et al. (1991). Quantification of EEG irregularity by
        use of the entropy of the power spectrum. Electroencephalography
        and clinical neurophysiology, 79(3), 204-210.

        https://en.wikipedia.org/wiki/Spectral_density

        https://en.wikipedia.org/wiki/Welch%27s_method

        Examples
        --------
        Spectral entropy of a pure sine using FFT

        >>> from entropy import spectralEntropy
        >>> import numpy as np
        >>> sf, f, dur = 100, 1, 4
        >>> N = sf * dur # Total number of discrete samples
        >>> t = np.arange(N) / sf # Time vector
        >>> x = np.sin(2 * np.pi * f * t)
        >>> np.round(spectralEntropy(x, sf, method='fft'), 2)
        0.0

        Spectral entropy of a random signal using Welch's method

        >>> from entropy import spectralEntropy
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> x = np.random.rand(3000)
        >>> spectralEntropy(x, sf=100, method='welch')
        6.980045662371389

        Normalized spectral entropy

        >>> spectralEntropy(x, sf=100, method='welch', normalize=True)
        0.9955526198316071
        """
        x = np.array(x)
        # Compute and normalize power spectrum
        if method == 'fft':
            _, psd = periodogram(x, sf)
        elif method == 'welch':
            _, psd = welch(x, sf, nperseg=nperseg)
        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        if normalize:
            se /= np.log2(psd_norm.size)
        return se


    def singularValueDecompositionEntropy(self, x, order=3, delay=1, normalize=False):
        """Singular Value Decomposition entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times)
        order : int
            Order of SVD entropy (= length of the embedding dimension).
            Default is 3.
        delay : int
            Time delay (lag). Default is 1.
        normalize : bool
            If True, divide by log2(order!) to normalize the entropy between 0
            and 1. Otherwise, return the permutation entropy in bit.

        Returns
        -------
        svd_e : float
            SVD Entropy

        Notes
        -----
        SVD entropy is an indicator of the number of eigenvectors that are needed
        for an adequate explanation of the data set. In other words, it measures
        the dimensionality of the data.

        The SVD entropy of a signal :math:`x` is defined as:

        .. math::
            H = -\\sum_{i=1}^{M} \\overline{\\sigma}_i log_2(\\overline{\\sigma}_i)

        where :math:`M` is the number of singular values of the embedded matrix
        :math:`Y` and :math:`\\sigma_1, \\sigma_2, ..., \\sigma_M` are the
        normalized singular values of :math:`Y`.

        The embedded matrix :math:`Y` is created by:

        .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

        .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T

        Examples
        --------
        SVD entropy with order 2

        >>> from entropy import singularValueDecompositionEntropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(singularValueDecompositionEntropy(x, order=2))
        0.7618909465130066

        Normalized SVD entropy with order 3

        >>> from entropy import singularValueDecompositionEntropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(singularValueDecompositionEntropy(x, order=3, normalize=True))
        0.6870083043946692
        """
        x = np.array(x)
        mat = _embed(x, order=order, delay=delay)
        W = np.linalg.svd(mat, compute_uv=False)
        # Normalize the singular values
        W /= sum(W)
        svd_e = -np.multiply(W, np.log2(W)).sum()
        if normalize:
            svd_e /= np.log2(order)
        return svd_e


    def _app_samp_entropy(self, x, order, metric='chebyshev', approximate=True):
        """Utility function for `app_entropy`` and `sample_entropy`.
        """
        _all_metrics = KDTree.valid_metrics
        if metric not in _all_metrics:
            raise ValueError('The given metric (%s) is not valid. The valid '
                             'metric names are: %s' % (metric, _all_metrics))
        phi = np.zeros(2)
        r = 0.2 * np.std(x, axis=-1, ddof=1)

        # compute phi(order, r)
        _emb_data1 = _embed(x, order, 1)
        if approximate:
            emb_data1 = _emb_data1
        else:
            emb_data1 = _emb_data1[:-1]
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(order + 1, r)
        emb_data2 = _embed(x, order + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        if approximate:
            phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
            phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
        else:
            phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
            phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
        return phi


    # @jit#('f8(f8[:], i4, f8)', nopython=True)

    @staticmethod
    @jit(nopython=True)
    def _numba_sampen(x, mm=2, r=0.2):
        """
        Fast evaluation of the sample entropy using Numba.
        """
        n = x.size
        n1 = n - 1
        mm += 1
        mm_dbld = 2 * mm

        # Define threshold
        r *= x.std()

        # initialize the lists
        run = [0] * n
        run1 = run[:]
        r1 = [0] * (n * mm_dbld)
        a = [0] * mm
        b = a[:]
        p = a[:]

        for i in range(n1):
            nj = n1 - i

            for jj in range(nj):
                j = jj + i + 1
                if abs(x[j] - x[i]) < r:
                    run[jj] = run1[jj] + 1
                    m1 = mm if mm < run[jj] else run[jj]
                    for m in range(m1):
                        a[m] += 1
                        if j < n1:
                            b[m] += 1
                else:
                    run[jj] = 0
            for j in range(mm_dbld):
                run1[j] = run[j]
                r1[i + n * j] = run[j]
            if nj > mm_dbld - 1:
                for j in range(mm_dbld, nj):
                    run1[j] = run[j]

        m = mm - 1

        while m > 0:
            b[m] = b[m - 1]
            m -= 1

        b[0] = n * n1 / 2
        a = np.array([float(aa) for aa in a])
        b = np.array([float(bb) for bb in b])
        p = np.true_divide(a, b)
        return -log(p[-1])


    def approximateEntropy(self, x, order=2, metric='chebyshev'):
        """Approximate Entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times).
        order : int
            Embedding dimension. Default is 2.
        metric : str
            Name of the distance metric function used with
            :py:class:`sklearn.neighbors.KDTree`. Default is
            `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_.

        Returns
        -------
        ae : float
            Approximate Entropy.

        Notes
        -----
        Approximate entropy is a technique used to quantify the amount of
        regularity and the unpredictability of fluctuations over time-series data.
        Smaller values indicates that the data is more regular and predictable.

        The value of :math:`r` is set to :math:`0.2 * \\texttt{std}(x)`.

        Code adapted from the `mne-features <https://mne.tools/mne-features/>`_
        package by Jean-Baptiste Schiratti and Alexandre Gramfort.

        References
        ----------
        Richman, J. S. et al. (2000). Physiological time-series analysis
        using approximate entropy and sample entropy. American Journal of
        Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

        Examples
        --------
        >>> from entropy import approximateEntropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(approximateEntropy(x, order=2))
        2.0754913760787277
        """
        phi = self._app_samp_entropy(x, order=order, metric=metric, approximate=True)
        return np.subtract(phi[0], phi[1])


    def sampleEntropy(self, x, order=2, metric='chebyshev'):
        """Sample Entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times).
        order : int
            Embedding dimension. Default is 2.
        metric : str
            Name of the distance metric function used with
            :py:class:`sklearn.neighbors.KDTree`. Default is
            `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_.

        Returns
        -------
        se : float
            Sample Entropy.

        Notes
        -----
        Sample entropy is a modification of approximate entropy, used for assessing
        the complexity of physiological time-series signals. It has two advantages
        over approximate entropy: data length independence and a relatively
        trouble-free implementation. Large values indicate high complexity whereas
        smaller values characterize more self-similar and regular signals.

        The sample entropy of a signal :math:`x` is defined as:

        .. math:: H(x, m, r) = -log\\frac{C(m + 1, r)}{C(m, r)}

        where :math:`m` is the embedding dimension (= order), :math:`r` is
        the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
        :math:`C(m + 1, r)` is the number of embedded vectors of length
        :math:`m + 1` having a
        `Chebyshev distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        inferior to :math:`r` and :math:`C(m, r)` is the number of embedded
        vectors of length :math:`m` having a Chebyshev distance inferior to
        :math:`r`.

        Note that if ``metric == 'chebyshev'`` and ``len(x) < 5000`` points,
        then the sample entropy is computed using a fast custom Numba script.
        For other distance metric or longer time-series, the sample entropy is
        computed using a code from the
        `mne-features <https://mne.tools/mne-features/>`_ package by Jean-Baptiste
        Schiratti and Alexandre Gramfort (requires sklearn).

        References
        ----------
        Richman, J. S. et al. (2000). Physiological time-series analysis
        using approximate entropy and sample entropy. American Journal of
        Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

        Examples
        --------
        Sample entropy with order 2.

        >>> from entropy import sampleEntropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sampleEntropy(x, order=2))
        2.192416747827227

        Sample entropy with order 3 using the Euclidean distance.

        >>> from entropy import sampleEntropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sampleEntropy(x, order=3, metric='euclidean'))
        2.7246543561542453
        """
        x = np.asarray(x, dtype=np.float64)
        if metric == 'chebyshev' and x.size < 5000:
            return self._numba_sampen(x, mm=order, r=0.2)
        else:
            phi = self._app_samp_entropy(x, order=order, metric=metric, approximate=False)
            return -np.log(np.divide(phi[1], phi[0]))


    def fuzzyEntropy(self, x, m=2):

        r = float(0.2 * x.std())

        sig_seg_df_list = x.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            eng = matlab.engine.start_matlab()
            self.matlab_engine = eng
            is_gen = True

        # print(len(sig_seg_df_list), sig_seg_df_list)
        fuzz_ent = eng.fuzzyEn(sig_seg_df_list, m, 1, r, nargout=1)
        # if isinstance(ent2, float):
        #     ent2 = [ent2]
        # ent = list(ent2)
        # print('===', len(ent), ent)

        if is_gen:
            eng.quit()
        return fuzz_ent


    def distributionEntropy(self, x, m=2, M=500):

        sig_seg_df_list = x.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            eng = self.manage_matlab_python_engine()
            self.matlab_engine = eng
            is_gen = True

        # print(len(sig_seg_df_list), sig_seg_df_list)
        dist_ent = eng.distributionEn(sig_seg_df_list, m, M, nargout=1)
        # if isinstance(ent2, float):
        #     ent2 = [ent2]
        # ent = list(ent2)
        # print('===', len(ent), ent)

        if is_gen:
            self.manage_matlab_python_engine(existing_eng=eng)

        return dist_ent


    def distributionEntropy4(self, x, m=4):
        return self.distributionEntropy(x, m=m)
    def distributionEntropy6(self, x, m=6):
        return self.distributionEntropy(x, m=m)
    def distributionEntropy8(self, x, m=8):
        return self.distributionEntropy(x, m=m)
    def distributionEntropy10(self, x, m=10):
        return self.distributionEntropy(x, m=m)


    def shannonEntropy(self, x, m=2):
        """Shannon Entropy.

        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times).

        Returns
        -------
        se : float
            Shannon Entropy.

        Notes
        -----
        Entropy or Information entropy is the information theory’s basic quantity and the expected value for
        the level of self-information. Entropy is introduced by Claude Shannon and hence it is named so after him.

        Shannon entropy is a self-information related introduced by him. The self-information related value
        quantifies how much information or surprise levels are associated with one particular outcome.
        This outcome is referred to as an event of a random variable. The Shannon entropy quantifies the
        levels of “informative” or “surprising” the whole of the random variable would be and all its possible
        outcomes are averaged. Information entropy is generally measured in terms of bits which are also known as
        Shannons or otherwise called bits and even as nats.

        I(x) = -logp(x)

        Now, we can quantify the level of uncertainty in a whole probability distribution using the equation
        of Shannon entropy as below:

        H(x) = (i is element of x) sum(p(i) log_2(1/p(i)))

        References
        ----------
        https://onestopdataanalysis.com/shannon-entropy/

        Examples
        --------
        Shannon entropy with order 2.

        >>> from entropy import shannonEntropy
        >>> import numpy as np
        >>> import collections
        >>> import math
        >>> np.random.seed(1234567)
        >>> x = "ATCGTAGTGAC"
        >>> print(shannonEntropy(x))
        1.9808259362290785
        Shannon entropy with order 2.

        >>> from entropy import shannonEntropy
        >>> import numpy as np
        >>> import collections
        >>> import math
        >>> np.random.seed(1234567)
        >>> x = dataframe/numpyarray
        >>> print(shannonEntropy(x))
        1.9808259362290785
        """

        bases = collections.Counter([tmp_base for tmp_base in x])
        # define distribution
        dist = [i / sum(bases.values()) for i in bases.values()]

        # use scipy to calculate entropy
        entropy_value = entropy(dist, base=m)

        return entropy_value


    def _x_log2_x(self, x):
        """ Return x * log2(x) and 0 if x is 0."""
        results = x * np.log2(x)
        if np.size(x) == 1:
            if np.isclose(x, 0.0):
                results = 0.0
        else:
            results[np.isclose(x, 0.0)] = 0.0
        return results


    def renyiEntropy(self, x, alpha=2):
        assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(
            alpha)  # DEBUG
        if np.isinf(alpha):
            # XXX Min entropy!
            return - np.log2(np.max(x))
        elif np.isclose(alpha, 0):
            # XXX Max entropy!
            return np.log2(len(x))
        elif np.isclose(alpha, 1):
            # XXX Shannon entropy!
            return - np.sum(self._x_log2_x(x))
        else:
            return (1.0 / (1.0 - alpha)) * np.log2(np.sum(x ** alpha))


    # @jit#('u8(unicode_type)', nopython=True)

    @staticmethod
    @jit(nopython=True)
    def _lz_complexity(binary_string):
        """
        Internal Numba implementation of the Lempel-Ziv (LZ) complexity.
        https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lziv_complexity.py
        """
        u, v, w = 0, 1, 1
        v_max = 1
        length = len(binary_string)
        complexity = 1
        while True:
            if binary_string[u + v - 1] == binary_string[w + v - 1]:
                v += 1
                if w + v >= length:
                    complexity += 1
                    break
            else:
                v_max = max(v, v_max)
                u += 1
                if u == w:
                    complexity += 1
                    w += v_max
                    if w >= length:
                        break
                    else:
                        u = 0
                        v = 1
                        v_max = 1
                else:
                    v = 1
        return complexity


    def lempelZivComplexity(self, sequence, normalize=False):
        """
        Lempel-Ziv (LZ) complexity of (binary) sequence.

        .. versionadded:: 0.1.1

        Parameters
        ----------
        sequence : str or array
            A sequence of character, e.g. ``'1001111011000010'``,
            ``[0, 1, 0, 1, 1]``, or ``'Hello World!'``.
        normalize : bool
            If ``True``, returns the normalized LZ (see Notes).

        Returns
        -------
        lz : int or float
            LZ complexity, which corresponds to the number of different
            substrings encountered as the stream is viewed from the
            beginning to the end. If ``normalize=False``, the output is an
            integer (counts), otherwise the output is a float.

        Notes
        -----
        LZ complexity is defined as the number of different substrings encountered
        as the sequence is viewed from begining to the end.

        Although the raw LZ is an important complexity indicator, it is heavily
        influenced by sequence length (longer sequence will result in higher LZ).
        Zhang and colleagues (2009) have therefore proposed the normalized LZ,
        which is defined by

        .. math:: LZn = \\frac{LZ}{(n / \\log_b{n})}

        where :math:`n` is the length of the sequence and :math:`b` the number of
        unique characters in the sequence.

        References
        ----------
        .. [1] Lempel, A., & Ziv, J. (1976). On the Complexity of Finite Sequences.
               IEEE Transactions on Information Theory / Professional Technical
               Group on Information Theory, 22(1), 75–81.
               https://doi.org/10.1109/TIT.1976.1055501

        .. [2] Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized
               Lempel-Ziv complexity and its application in bio-sequence analysis.
               Journal of Mathematical Chemistry, 46(4), 1203–1212.
               https://doi.org/10.1007/s10910-008-9512-2

        .. [3] https://en.wikipedia.org/wiki/Lempel-Ziv_complexity

        .. [4] https://github.com/Naereen/Lempel-Ziv_Complexity

        Examples
        --------
        >>> from entropy import lempelZivComplexity
        >>> # Substrings = 1 / 0 / 01 / 1110 / 1100 / 0010
        >>> s = '1001111011000010'
        >>> lempelZivComplexity(s)
        6

        Using a list of integer / boolean instead of a string:

        >>> # 1 / 0 / 10
        >>> lempelZivComplexity([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        3

        With normalization:

        >>> lempelZivComplexity(s, normalize=True)
        1.5

        Note that this function also works with characters and words:

        >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        >>> lempelZivComplexity(s), lempelZivComplexity(s, normalize=True)
        (26, 1.0)

        >>> s = 'HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD!'
        >>> lempelZivComplexity(s), lempelZivComplexity(s, normalize=True)
        (11, 0.38596001132145313)
        """
        assert isinstance(sequence, (str, list, np.ndarray))
        assert isinstance(normalize, bool)
        if isinstance(sequence, (list, np.ndarray)):
            sequence = np.asarray(sequence)
            if sequence.dtype.kind in 'bfi':
                # Convert [True, False] or [1., 0.] to [1, 0]
                sequence = sequence.astype(int)
            # Convert to a string, e.g. "10001100"
            s = ''.join(sequence.astype(str))
        else:
            s = sequence

        if normalize:
            # 1) Timmermann et al. 2019
            # The sequence is randomly shuffled, and the normalized LZ
            # is calculated as the ratio of the LZ of the original sequence
            # divided by the LZ of the randomly shuffled LZ. However, the final
            # output is dependent on the random seed.
            # sl_shuffled = list(s)
            # rng = np.random.RandomState(None)
            # rng.shuffle(sl_shuffled)
            # s_shuffled = ''.join(sl_shuffled)
            # return _lz_complexity(s) / _lz_complexity(s_shuffled)
            # 2) Zhang et al. 2009
            n = len(s)
            base = len(''.join(set(s)))  # Number of unique characters
            base = 2 if base < 2 else base
            return self._lz_complexity(s) / (n / log(n, base))
        else:
            return self._lz_complexity(s)


    ############ Entropy Profiling
    # Collected from @author: radhagayathri
    def _call_entropy_profiling_algorithm(self, x, m=2):

        sig_seg_df_list = x.tolist()

        # Calling Matlab code from Python
        is_gen = False
        eng = self.matlab_engine
        if self.matlab_engine is None:
            eng = self.manage_matlab_python_engine()
            self.matlab_engine = eng
            is_gen = True

        # print(len(sig_seg_df_list), sig_seg_df_list)
        ent2 = eng.sampEnProfiling(sig_seg_df_list, m, nargout=1)
        if isinstance(ent2, float):
            ent2 = [ent2]
        ent = list(ent2)
        # print('===', len(ent), ent)

        if is_gen:
            self.manage_matlab_python_engine(existing_eng=eng)

        return ent


    ############ Fracta dimension
    # Collected from @author: msrahman

    def petrosianFd(self, x):
        """Petrosian fractal dimension.

        Parameters
        ----------
        x : list or np.array
            One dimensional time series.

        Returns
        -------
        pfd : float
            Petrosian fractal dimension.

        Notes
        -----
        The Petrosian fractal dimension of a time-series :math:`x` is defined by:

        .. math:: P = \\frac{\\log_{10}(N)}{\\log_{10}(N) +
                  \\log_{10}(\\frac{N}{N+0.4N_{\\delta}})}

        where :math:`N` is the length of the time series, and
        :math:`N_{\\delta}` is the number of sign changes in the signal derivative.

        Original code from the `pyrem <https://github.com/gilestrolab/pyrem>`_
        package by Quentin Geissmann.

        References
        ----------
        .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and
           recognition of different preictal EEG patterns, in , Proceedings of the
           Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
           pp. 212-217.

        .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.

        Examples
        --------
        >>> import numpy as np
        >>> from entropy import petrosian_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(petrosian_fd(x))
        1.0505385662721405
        """
        n = len(x)
        # Number of sign changes in the first derivative of the signal
        diff = np.ediff1d(x)
        N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

    def katzFd(self, x):
        """Katz Fractal Dimension.

        Parameters
        ----------
        x : list or np.array
            One dimensional time series.

        Returns
        -------
        kfd : float
            Katz fractal dimension.

        Notes
        -----
        The Katz fractal dimension is defined by:

        .. math:: K = \\frac{\\log_{10}(n)}{\\log_{10}(d/L)+\\log_{10}(n)}

        where :math:`L` is the total length of the time series and :math:`d`
        is the
        `Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
        between the first point in the series and the point that provides the
        furthest distance with respect to the first point.

        Original code from the `mne-features <https://mne.tools/mne-features/>`_
        package by Jean-Baptiste Schiratti and Alexandre Gramfort.

        References
        ----------
        .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
               dimension algorithms. IEEE Transactions on Circuits and Systems I:
               Fundamental Theory and Applications, 48(2), 177-183.

        .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
               the computation of EEG biomarkers for dementia." 2nd International
               Conference on Computational Intelligence in Medicine and Healthcare
               (CIMED2005). 2005.

        Examples
        --------
        >>> import numpy as np
        >>> from entropy import katz_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(katz_fd(x))
        5.121395665678078
        """
        x = np.array(x)
        dists = np.abs(np.ediff1d(x))
        ll = dists.sum()
        ln = np.log10(np.divide(ll, dists.mean()))
        aux_d = x - x[0]
        d = np.max(np.abs(aux_d[1:]))
        return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))

    # @jit('float64(float64[:], int32)')
    def _higuchi_fd(self, x, kmax):
        """Utility function for `higuchi_fd`.
        """

        n_times = x.size
        lk = np.empty(kmax)
        x_reg = np.empty(kmax)
        y_reg = np.empty(kmax)
        for k in range(1, kmax + 1):
            lm = np.empty((k,))
            for m in range(k):
                ll = 0
                n_max = floor((n_times - m - 1) / k)
                n_max = int(n_max)
                for j in range(1, n_max):
                    ll += abs(x[m + j * k] - x[m + (j - 1) * k])
                ll /= k
                ll *= (n_times - 1) / (k * n_max)
                lm[m] = ll

            if lm[0] == 0:
                return 0

            # Mean of lm
            m_lm = 0
            for m in range(k):
                m_lm += lm[m]
            m_lm /= k
            lk[k - 1] = m_lm
            x_reg[k - 1] = log(1. / k)

            # if lm[0]==0:
            #     print('Signal: ', x)
            #     import matplotlib.pyplot as plt
            #     plt.plot(x)
            #     plt.show()

            #print('************', lm, m_lm)
            y_reg[k - 1] = log(m_lm)
        higuchi, _ = _linear_regression(x_reg, y_reg)
        return higuchi


    def higuchiFd(self, x, kmax=10):
        """Higuchi Fractal Dimension.

        Parameters
        ----------
        x : list or np.array
            One dimensional time series.
        kmax : int
            Maximum delay/offset (in number of samples).

        Returns
        -------
        hfd : float
            Higuchi fractal dimension.

        Notes
        -----
        Original code from the `mne-features <https://mne.tools/mne-features/>`_
        package by Jean-Baptiste Schiratti and Alexandre Gramfort.

        This function uses Numba to speed up the computation.

        References
        ----------
        .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
           basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
           (1988): 277-283.

        Examples
        --------
        >>> import numpy as np
        >>> from entropy import higuchi_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(higuchi_fd(x))
        2.0511793572134467
        """
        x = np.asarray(x, dtype=np.float64)
        kmax = int(kmax)
        return self._higuchi_fd(x, kmax)

    # @jit('f8(f8[:])', nopython=True)
    def _dfa(self, x):
        """
        Utility function for detrended fluctuation analysis
        """
        N = len(x)
        nvals = _log_n(4, 0.1 * N, 1.2)
        walk = np.cumsum(x - x.mean())
        fluctuations = np.zeros(len(nvals))

        for i_n, n in enumerate(nvals):
            d = np.reshape(walk[:N - (N % n)], (N // n, n))
            ran_n = np.array([float(na) for na in range(n)])
            d_len = len(d)
            slope = np.empty(d_len)
            intercept = np.empty(d_len)
            trend = np.empty((d_len, ran_n.size))
            for i in range(d_len):
                slope[i], intercept[i] = _linear_regression(ran_n, d[i])
                y = np.zeros_like(ran_n)
                # Equivalent to np.polyval function
                for p in [slope[i], intercept[i]]:
                    y = y * ran_n + p
                trend[i, :] = y
            # calculate standard deviation (fluctuation) of walks in d around trend
            flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
            # calculate mean fluctuation over all subsequences
            fluctuations[i_n] = flucs.sum() / flucs.size

        # Filter zero
        nonzero = np.nonzero(fluctuations)[0]
        fluctuations = fluctuations[nonzero]
        nvals = nvals[nonzero]
        if len(fluctuations) == 0:
            # all fluctuations are zero => we cannot fit a line
            dfa = np.nan
        else:
            dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
        return dfa

    def detrendedFluctuation(self, x):
        """
        Detrended fluctuation analysis (DFA).

        Parameters
        ----------
        x : list or np.array
            One-dimensional time-series.

        Returns
        -------
        alpha : float
            the estimate alpha (:math:`\\alpha`) for the Hurst parameter.

            :math:`\\alpha < 1`` indicates a
            stationary process similar to fractional Gaussian noise with
            :math:`H = \\alpha`.

            :math:`\\alpha > 1`` indicates a non-stationary process similar to
            fractional Brownian motion with :math:`H = \\alpha - 1`

        Notes
        -----
        `Detrended fluctuation analysis
        <https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis>`_
        is used to find long-term statistical dependencies in time series.

        The idea behind DFA originates from the definition of self-affine
        processes. A process :math:`X` is said to be self-affine if the standard
        deviation of the values within a window of length n changes with the window
        length factor :math:`L` in a power law:

        .. math:: \\texttt{std}(X, L * n) = L^H * \\texttt{std}(X, n)

        where :math:`\\texttt{std}(X, k)` is the standard deviation of the process
        :math:`X` calculated over windows of size :math:`k`. In this equation,
        :math:`H` is called the Hurst parameter, which behaves indeed very similar
        to the Hurst exponent.

        For more details, please refer to the excellent documentation of the
        `nolds <https://cschoel.github.io/nolds/>`_
        Python package by Christopher Scholzel, from which this function is taken:
        https://cschoel.github.io/nolds/nolds.html#detrended-fluctuation-analysis

        Note that the default subseries size is set to
        entropy.utils._log_n(4, 0.1 * len(x), 1.2)). The current implementation
        does not allow to manually specify the subseries size or use overlapping
        windows.

        The code is a faster (Numba) adaptation of the original code by Christopher
        Scholzel.

        References
        ----------
        .. [1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
               H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
               DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.

        .. [2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
               V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
               “Detrended fluctuation analysis: A scale-free view on neuronal
               oscillations,” Frontiers in Physiology, vol. 30, 2012.

        Examples
        --------
        >>> import numpy as np
        >>> from entropy import detrended_fluctuation
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(detrended_fluctuation(x))
        0.761647725305623
        """
        x = np.asarray(x, dtype=np.float64)
        return self._dfa(x)


### Frequency Domain Features

    ##Bandpass filtering
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
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




