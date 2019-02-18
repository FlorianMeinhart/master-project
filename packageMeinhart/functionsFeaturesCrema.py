import scipy
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from sklearn.decomposition import PCA
#import functionsMasterProjectMeinhart as fmpm



'''Function to write features to csv-file'''
def write_crema_features_to_csv(feature_csv_file, features=None, label='non'):
    '''
    Function to add 24 features (according to Crema) to csv-file.
    If no features are provided (features=None) a new csv-file
    is generated with the header for the features.
    (--> label;aX_ampFreqDist_01_05;aX_rms_06;aX_mean_07;...)
    
    Parameters
    ----------
    feature_csv_file : string
        File to save the features.
        
    features : dict
        Dictionary with the following keys for the 24 features:
                 'aX_ampFreqDist_01_05',
                 'aX_rms_06',
                 'aX_mean_07',
                 'aX_std_08',
                 'aX_powerbands_09_18',
                 'aX_maxValAutocorr_19',
                 'aX_numPromPeaks_20',
                 'aX_numWeakPeaks_21',
                 'aX_valFirstPeak_22',
                 'aX_kurtosis_23',
                 'aX_intQuatRange_24',

        same for 'aYZPC1_ampFreqDist_01_05'
                 ...
                 
        and for  'gPC1_ampFreqDist_01_05'
                 ...
   
    label: string
        Abbreviation of the exercise (e.g. 'RF').
        default: 'non'
    
    Returns
    -------
    no returns
    '''
    
    # names of features
    feature_name_list = ['_ampFreqDist_01',
                         '_ampFreqDist_02',
                         '_ampFreqDist_03',
                         '_ampFreqDist_04',
                         '_ampFreqDist_05',
                         '_rms_06',
                         '_mean_07',
                         '_std_08',
                         '_powerbands_09',
                         '_powerbands_10',
                         '_powerbands_11',
                         '_powerbands_12',
                         '_powerbands_13',
                         '_powerbands_14',
                         '_powerbands_15',
                         '_powerbands_16',
                         '_powerbands_17',
                         '_powerbands_18',
                         '_maxValAutocorr_19',
                         '_numPromPeaks_20',
                         '_numWeakPeaks_21',
                         '_valFirstPeak_22',
                         '_kurtosis_23',
                         '_intQuatRange_24']
    
    # putting the header of the feature-file together if features is None
    if features is None:
        # first column contains labels
        header_string = 'label;'
        
        for sig in ['ax','aYZPC1','gPC1']:
            for feature_name in feature_name_list:
                header_string +=  sig + feature_name + ';'
        
        # remove last separator (;)
        idx_last_sep = header_string.rfind(";")
        header_string =  header_string[:idx_last_sep]
        
        # write header to file
        with open(feature_csv_file, 'w') as feature_file:
            feature_file.writelines(header_string + '\n')
        
        return
    
    # string to write features to csv-file
    feature_string = label + ';' # first column contains labels
    
    # putting the feature string together
    for sig in ['aX','aYZPC1','gPC1']:
            
            ampFreqDist_flag = False # flag for the five ampFreqDist features --> check if already done
            powerband_flag = False # flag for the ten powerband features --> check if already done
            
            for feature_name in feature_name_list:
                
                # if ampFreqDist features --> consider list of features
                if 'ampFreqDist' in feature_name:
                    if ampFreqDist_flag is False:
                        ampFreqDist_flag = True
                        feature_string += ';'.join(map(str, features[sig+'_ampFreqDist_01_05'])) + ';'
                
                # if powerband features --> consider list of features
                elif 'powerbands' in feature_name:
                    if powerband_flag is False:
                        powerband_flag = True
                        feature_string += ';'.join(map(str, features[sig+'_powerbands_09_18'])) + ';'
                    
                else:
                    feature_string +=  str(features[sig+feature_name]) + ';'
    
    # remove last separator (;)
    idx_last_sep = feature_string.rfind(";")
    feature_string =feature_string[:idx_last_sep]
    
    # append features to file
    with open(feature_csv_file, 'a') as feature_file:
        feature_file.writelines(feature_string + '\n')



'''Function to select a window of a certain time from a dictionary with signals'''
def select_window(signals, window_length=5, start_time=None, start_index=0, sampling_rate=256):
    '''
    This function returns the signal data of a certain time range,
    selected via start time (or start index) and window length.
    If stop index is out of range, None is returned.
    All signals in the incoming dict must have the same length.
    
    Parameters
    ----------
    signals : dicitionary
        the values of all items have to be 1- or 2-dimensional arrays
        
    window_length : float or int
        window length in seconds to select data
        
    start_time : float or int
        start time in seconds where data selection starts
        
    start_index :  int
        only used, if start_time is None;
        start index where data selection starts
        
    sampling_rate : float or int
        sampling rate of signal data
    
    Returns
    -------
    dict --> same keys as input dictionary (signals),
        contains selected data
    '''
    
    signal_keys = [*signals]
    
    # if start time is given --> convert to start index
    if start_time is not None:
        start_index = int(start_time * sampling_rate)
    
    # calculate stop index
    stop_index = start_index + int(window_length * sampling_rate)
    
    out_dict = {}
    
    # check if stop index is out of range (compare with all signals)
    for sig in signal_keys:
        if stop_index > len(signals[sig]):
            # return None if stop index is out of range
            return None
        else:
            #check dimensions of item values for slicing
            if np.array(signals[sig]).ndim == 1:
                # save selected 1-dimensional array to out dicitionary
                out_dict[sig] = signals[sig][start_index:stop_index]
            elif np.array(signals[sig]).ndim == 2:
                # save selected 2-dimensional array to out dicitionary
                out_dict[sig] = signals[sig][start_index:stop_index,:]
            else:
                # raise error if dimensions are wrong
                raise NameError('Wrong dimensions --> dict key: ' + sig)
    
    return out_dict


'''Function to perform dimension reduction (PCA) for signals according to Crema'''
def signal_dim_reduc(signals):
    '''
    This function performs a dimension reduction by means of PCA
    with respect to the input signals (acceleration, angular velocity).
    
    Parameters
    ----------
    signals : dicitionary
        'signals' must have the keys 'Acc' and 'Gyr';
        the values of both items have to be arrays of shape n x 3,
        where the 1st column contains the x values, the 2nd the y values and the 3rd the z values
    
    Returns
    -------
    dict --> 3 keys: aX, aYZPC1, gPC1
        aX:     the x-axis accelerometer signal
        aYZPC1: the projection of the y and z accelerometer signals onto the first PC 
        gPC1:   the projection of the three gyroscope signals onto its first PC
    '''
    
    # aX, the x-axis accelerometer signal
    aX = signals['Acc'][:,0] # x axis of acc. signal

    # aYZPC1, the projection of the y and z accelerometer signals onto the first PC 
    # (this captures movement perpendicular to the arm, which allows deriving information 
    # from the y and z axes despite the unknown rotation of the armband)
    pca_aYZPC1 = PCA(n_components=1)
    aYZPC1 = pca_aYZPC1.fit_transform(signals['Acc'][:,1:]) # y and z axes of acc. signal
    
    # gPC1, the projection of the three gyroscope signals onto its first PC 
    # (this is the movement along the axis that demonstrates the most variance within this window)
    pca_gPC1 = PCA(n_components=1)
    gPC1 = pca_gPC1.fit_transform(signals['Gyr']) # all axes of gyr. signal
    
    out_dict = {}
    out_dict['aX'] = aX
    out_dict['aYZPC1'] = np.ravel(aYZPC1) #  ensure that array has one dimension with np.ravel
    out_dict['gPC1'] = np.ravel(gPC1) #  ensure that array has one dimension with np.ravel
    
    return out_dict


'''Amplitude frequency distribution within five classes'''
def amp_freq_dist_classes(signal, num_classes=5, min_val=None, max_val=None):
    '''
    Function to count the frequency of amplitude values of a signal
    corresponding to a certain number of classes.
    Class boundaries are linearly spaced from min_val to max_val.
    
    Parameters
    ----------
    signal : array (1-dim)
        Signal which is used at counting frequency of amplitude values.
        
    num_class :  int
         Number of classes for the frequency distribution.
         
    min_val : None or float
        Lower limiting value of the the class for the lowest values.
        --> if None: min of signal values is taken
        
    max_val : None or float
        Upper limiting value of the the class for the highest values.
        --> if None: max of signal values is taken
        
    Returns
    -------
    list 
        One element for each class, which represents the number of
        values of the signal within the corresponding class.
    '''
    
    # take the min/max of the signal if the corresponding parameter is None
    if min_val is None:
        min_val = min(signal)
    if max_val is None:
        max_val = max(signal)
    
    # define class boundaries (linearly spaced)
    class_bound = np.linspace(min_val, max_val, num_classes+1) # num_class+1, because one bound. more than classes
    
    # list to count values in classes
    class_count = []
    
    #  count values in classes
    for ii in range(num_classes):
        class_count.append(len([x for x in signal if x >= class_bound[ii] and x <= class_bound[ii+1]]))
        
    return class_count


'''Power bands'''
def bandpower(signal, fs, fmin, fmax):
    '''
    This function calculates the bandpower of a selected frequency range.
    
    Parameters
    ----------
    signal : array (1-dim)
        Signal for which the bandpower is calculated.
        
    fs : float or int
        Sampling rate of the input signal [Hz].
        
    fmin : float or int
        Lower limiting frequency of the power band [Hz].
    
    fmax : float or int
        Upper limiting frequency of the power band [Hz].
    
    Returns
    -------
    float
        Bandpower of the choosen frequency range.
    '''
    f, Pxx = scipy.signal.periodogram(signal, fs=fs) # Pxx has units of V**2/Hz if x is measured in V
    ind_min = scipy.argmax(f > fmin) - 1 # get lower limiting frequency index
    ind_max = scipy.argmax(f > fmax) - 1 # get upper limiting frequency index
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max]) # integrate selected power densitiy spectrum & return


def power_band_sequence(signal, num_power_bands=10, start_freq=0.1, stop_freq=25, sampling_rate=256):
    '''
    This function calculates a number of powerbands, linearly spaced from start freq. to stop freq.
    The function 'bandpower' is used.
    
    Parameters
    ----------
    signal : array (1-dim)
        Signal for which the power bands are calculated.
        
    num_power_bands : int
        Number of power bands to be calculated.
        
    start_freq : float or int
        Lower limiting frequency of the lowest power band [Hz].
    
    stop_freq : float or int
        Upper limiting frequency of the highest power band [Hz].
        
    sampling_rate : float or int
        Sampling rate of the signals [Hz].
    
    Returns
    -------
    list
        Bandpower values of the different frequency ranges.
    '''
    
    # split the range from start freq. to stop freq. into linearly spaced bands
    freq_range = np.linspace(start_freq, stop_freq, num_power_bands + 1) # e.g. 10 bands spaced linearly from 0.1-25 Hz
    
    # list for bandpower values
    power_bands = []
    
    # calculation for each power band
    for ii in range(num_power_bands):
         power_bands.append(bandpower(signal,
                                      fs = sampling_rate, 
                                      fmin = freq_range[ii],
                                      fmax = freq_range[ii+1]))
    return power_bands


'''Autocorrelation (max. value, num. of prominent & weak peaks, value of first peak after zero-crossing)'''
def autocorrelation_features(signal, thres_prominent=0.5, min_time_shift_prominent=0.1, 
                             thres_weak=0.1, min_time_shift_weak=0.1, sampling_rate=256, show_plot=False):
    '''
    This function calculates a number of powerbands, linearly spaced from start freq. to stop freq.
    The function 'bandpower' is used.
    
    Parameters
    ----------
    signal : array (1-dim)
        Signal for which the autocorrelation is performed.
        
    thres_prominent : float
        Threshold for prominent peak finding, but with respect to the normalized autocorrelation
        where the max value is 1. Hence, thres_prominent has to be bewteen 0 and 1.
        
    min_time_shift_prominent : float or int
        Minimum lag between two prominent peaks [s].
        Indices calculated by means of sampling rate.
    
    thres_weak : float
        Threshold for weak peak finding, but with respect to the normalized autocorrelation
        where the max value is 1. Hence, thres_weak has to be bewteen 0 and 1.
        
    min_time_shift_weak : float or int
        Minimum lag between two weak peaks [s].
        Indices calculated by means of sampling rate.
    
    sampling_rate : float or int
        Sampling rate of the input signal [Hz].
        
    show_plot : boolean
        If True, plot normalized autocorreation and peaks.
    
    Returns
    -------
    list
        4 elements:
        list[0]:   max value of the autocorrelation
        list[1]:   number of prominent peaks
        list[2]:   number of weak peaks
        list[3]:   value of first peak after zero-crossing
    '''
    
    # perform the autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    
    # take only the right half of the spectrum (due to symmetry)
    autocorr = autocorr[int(len(autocorr)/2):]
    
    # get the max value of the autocorrelation
    autocorr_max = max(autocorr)
    
    # normalize the autocorr. to get a max value of 1
    autocorr_norm = autocorr / max(autocorr)
    
    # convert the minimal time shifts for the peak detection to indices
    min_dist_prominent = min_time_shift_prominent * sampling_rate
    min_dist_weak = min_time_shift_weak * sampling_rate
    
    # get indices of prominent peaks (the highest value at fully overlapping is not considered as peak)
    idx_prominent = peakutils.indexes(autocorr_norm, thres=thres_prominent, min_dist=min_dist_prominent, thres_abs=True)
    num_idx_prominent = len(idx_prominent)
      
    # take only the weak peaks which belong to a smaller value than thres_prominent
    idx_weak = peakutils.indexes(autocorr_norm, thres=thres_weak, min_dist=min_dist_weak, thres_abs=True)
    idx_weak = [x for x in idx_weak if autocorr_norm[x] < thres_prominent]
    num_idx_weak = len(idx_weak)
    
    # value of first peak after zero-crossing
    val_1st_peak = 0
    for idx in np.sort(np.concatenate((idx_prominent, idx_weak))):
        if np.any(autocorr[:int(idx)] < 0):
            val_1st_peak = autocorr[int(idx)]
            break
    
    # plot autocorreation and peaks if desired
    if show_plot:
        plt.figure(figsize=(10,5))
        plt.plot(autocorr_norm)
        plt.plot(idx_prominent, autocorr_norm[idx_prominent],'r+', label='prominent', markersize=10)
        plt.plot(idx_weak, autocorr_norm[idx_weak],'g+', label='weak', markersize=10)
        plt.legend()
        plt.title('Normalized Autocorrelation', fontsize=15)
        plt.xlabel('index shift', fontsize=13)
        plt.grid(True)
        plt.show();
    
    return [autocorr_max, num_idx_prominent, num_idx_weak, val_1st_peak]
    

'''Function to generate all 24 features for each of the 3 signals (aX, aYZPC1, gPC1)'''
def generate_features_Crema(signals,
                            ampFreqDist_num_classes=5,
                            ampFreqDist_min_val=None,
                            ampFreqDist_max_val=None,
                            powerbands_num=10,
                            powerbands_start_freq=0.1,
                            powerbands_stop_freq=25,
                            autocorrelation_thres_prominent=0.2,
                            autocorrelation_min_time_shift_prominent=0.1,
                            autocorrelation_thres_weak=0.05,
                            autocorrelation_min_time_shift_weak=0.1,
                            sampling_rate=256):
    '''
    This function generates all 24 features according to Crema for an incoming signal.
    (see "Characterization of a Wearable System for Automatic Supervision of Fitness Exercises")
    
    Parameters
    ----------
    signals : dict
        Dictionary with signals (1-dim) for which the features are generated.
        
    ampFreqDist_num_class : int
        --> Feature: amplitude frequency distribution
        Number of classes for the frequency distribution.
         
    ampFreqDist_min_val : None or float
        --> Feature: amplitude frequency distribution
        Lower limiting value of the the class for the lowest values.
        --> if None: min of signal values is taken
        
    ampFreqDist_max_val : None or float
        --> Feature: amplitude frequency distribution
        Upper limiting value of the the class for the highest values.
        --> if None: max of signal values is taken
    
    powerbands_num : int
        --> Feature: powerbands
        Number of power bands to be calculated.
        
    powerbands_start_freq : float or int
        --> Feature: powerbands
        Lower limiting frequency of the lowest power band [Hz].
    
    powerbands_stop_freq : float or int
        --> Feature: powerbands
        Upper limiting frequency of the highest power band [Hz].
        
    autocorrelation_thres_prominent : float
        --> Feature: autocorrelation
        Threshold for prominent peak finding, but with respect to the normalized autocorrelation
        where the max value is 1. Hence, thres_prominent has to be bewteen 0 and 1.
        
    autocorrelation_min_time_shift_prominent : float or int
        --> Feature: autocorrelation
        Minimum lag between two prominent peaks [s].
        Indices calculated by means of sampling rate.
    
    autocorrelation_thres_weak : float
        --> Feature: autocorrelation
        Threshold for weak peak finding, but with respect to the normalized autocorrelation
        where the max value is 1. Hence, thres_weak has to be bewteen 0 and 1.
        
    autocorrelation_min_time_shift_weak : float or int
        --> Feature: autocorrelation
        Minimum lag between two weak peaks [s].
        Indices calculated by means of sampling rate.
    
    sampling_rate : float or int
        Sampling rate of the input signal [Hz].
    
    Returns
    -------
    dict
        Dictionary with features for each signal.
        Dict keys consist of signal keys of incoming dict, feature names and feature number.
        (e.g. aX_maxValAutocorr_19)
    '''
    feature_dict = {}
    
    for sig in [*signals]:
        feature_dict[sig+'_ampFreqDist_01_05'] = amp_freq_dist_classes(signals[sig],
                                                                   num_classes=ampFreqDist_num_classes,
                                                                   min_val=ampFreqDist_min_val,
                                                                   max_val=ampFreqDist_max_val)
        
        feature_dict[sig+'_rms_06'] = np.sqrt(np.mean(signals[sig]**2))
        
        feature_dict[sig+'_mean_07'] = np.mean(signals[sig])
        
        feature_dict[sig+'_std_08'] = np.std(signals[sig])
        
        feature_dict[sig+'_powerbands_09_18'] = power_band_sequence(signals[sig], 
                                                                    num_power_bands=powerbands_num, 
                                                                    start_freq=powerbands_start_freq, 
                                                                    stop_freq=powerbands_stop_freq,
                                                                    sampling_rate=sampling_rate)
        
        autcorr_feat = autocorrelation_features(signals[sig], 
                                                thres_prominent=autocorrelation_thres_prominent, 
                                                min_time_shift_prominent=autocorrelation_min_time_shift_prominent, 
                                                thres_weak=autocorrelation_thres_weak, 
                                                min_time_shift_weak=autocorrelation_min_time_shift_weak, 
                                                sampling_rate=sampling_rate)
        
        feature_dict[sig+'_maxValAutocorr_19'] = autcorr_feat[0]
        
        feature_dict[sig+'_numPromPeaks_20'] = autcorr_feat[1]
        
        feature_dict[sig+'_numWeakPeaks_21'] = autcorr_feat[2]
        
        feature_dict[sig+'_valFirstPeak_22'] = autcorr_feat[3]
        
        feature_dict[sig+'_kurtosis_23'] = scipy.stats.kurtosis(signals[sig])
        
        feature_dict[sig+'_intQuatRange_24'] = scipy.stats.iqr(signals[sig])
        
    return feature_dict   
