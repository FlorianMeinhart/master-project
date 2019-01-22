'''
Classes to handle physio data:

- PhysioData_SectionFeatures()
- PhysioData_WindowingProcedure()

'''
import pandas as pd
import numpy as np
import sqlite3
import time
import os
import itertools
import skinematics as skin
from scipy.signal import butter, lfilter
from scipy.ndimage.filters import maximum_filter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider, Button


#--------------------------------------------------------------------------------
# Class and corresponding functions for feature generation using section means.
#--------------------------------------------------------------------------------

def load_data_from_database(data_base_path='DataBase_Physio_with_nonEx.db'):
    '''
    Function to load the following data from data base:
        - subject IDs
        - exercise abbreviations
        - number of repetitions
        - sequence numbers
        - start times
        - stop times
        - csv-file name
    
    Parameters
    ----------
    data_base_path : string
        Path to data base.
    
    Returns
    -------
    DataFrame
        DataFrame with the listet information (see above).
    '''
    # Connect to an existing database
    conn = sqlite3.connect(data_base_path)
    cur = conn.cursor()

    # sql command to extract data
    query_sql = """
        SELECT e.subject_id,
        p.abbreviation,
        e.num_rep,
        r.sequence_num,
        r.start_time, r.stop_time,
        e.csv_file
        FROM subjects s
        INNER JOIN exercises e
        ON s.id = e.subject_id
        INNER JOIN paradigms p
        ON p.id = e.paradigm_id
        INNER JOIN repetitions r
        ON e.id = r.exercise_id
        """
    
    # get data from data base and close connection
    all_data_points_df = pd.read_sql_query(query_sql, conn)
    conn.close()
    
    return all_data_points_df


def select_data_points_from_df(all_data_points_df,
                               subject_ids=-1,
                               subject_ids_complementary=[],
                               reps=-1,
                               abbrs=-1,
                               with_non_Ex=True,
                               sub_id_key='subject_id',
                               num_rep_key='num_rep',
                               abbreviation_key='abbreviation'):
    '''
    Function to select data points from a DataFrame based on subject IDs,
    number of repetitions and exercise abbreviations.
    
    Parameters
    ----------
    all_data_points_df : pandas DataFrame
        DataFrame with all data points.
    
    subject_ids : int or list
        Subject IDs to select (e.g. [1, 2, 3]).
        --> default -1: Select all subjects not in subject_ids_complementary.
        --> if subject_ids is an empty list: empty DataFrame is returned
        
    subject_ids_complementary : int or list
        If subject_ids is -1 --> select all subjects not in subject_ids_complementary.
        
    reps : int or list
        Repetition numbers to select (e.g. [5, 10]).
        --> default -1: Select all repetitions.
        
    abbrs : int or list
        Exercise abbreviations to select (e.g. ['RF', 'SA']).
        --> default -1: Select all exercise abbreviations.
    
    with_non_Ex : boolean
        If False --> omit non exercise data (data points with zero repetitions).
        
    sub_id_key : string
        Key of the DataFrame for subject IDs.
        
    num_rep_key : string
        Key of the DataFrame for repetition numbers.
        
    abbreviation_key : string
        Key of the DataFrame for exercise abbreviations.
    
    
    Returns
    -------
    DataFrame
        DataFrame with selected data points.
    '''
    
    data_points_df = all_data_points_df.copy()
    
    # check if subject_ids is an empty list --> return and empty DataFrame in this case
    if isinstance(subject_ids, list) and not subject_ids:
        return pd.DataFrame()
    
    # select the subject IDs
    # if subject_ids is -1 --> select data from all subjects that are NOT in subject_ids_complementary
    if subject_ids is -1:
        if not isinstance(subject_ids_complementary, list): # if not list --> make list
            subject_ids_complementary = [subject_ids_complementary]
        data_points_df = data_points_df.loc[~data_points_df[sub_id_key].isin(subject_ids_complementary)]
        
    elif subject_ids is not -1:
        if not isinstance(subject_ids, list): # if not list --> make list
            subject_ids = [subject_ids]
        data_points_df = data_points_df.loc[data_points_df[sub_id_key].isin(subject_ids)]

    # select the repetition numbers
    if reps is not -1:
        if not isinstance(reps, list): # if not list --> make list
            reps = [reps]
        if with_non_Ex is True:
            reps.append(0) # zero repetitions correspond to non exercise data
        data_points_df = data_points_df.loc[data_points_df[num_rep_key].isin(reps)]

    elif with_non_Ex is False:
        data_points_df = data_points_df.loc[data_points_df[num_rep_key] != 0]
        
    # select the exercise abbreviations
    if abbrs is not -1:
        if not isinstance(abbrs, list): # if not list --> make list
            abbrs = [abbrs]
        data_points_df = data_points_df.loc[data_points_df[abbreviation_key].isin(abbrs)]

    return data_points_df
	
	
def split_range_into_sections(signal_data, num_sec=10, signals=['Acc','Gyr'], start_index=0, stop_index=None):
    '''
    This function splits a selected range of the input signals into a defined number 
    of equally distributed sections. For each signal and section the mean is calculated,
    and afterwards returned by means of a dictionary.
    
    Parameters
    ----------
    signal_data : dict
        Dictionary with the signals in the 'signals' argument as keys.
        The signal arrays must have same length.
    
    num_sec : int
        Number of sections to split the signals.
        
    signals : list
        Keys to select signals in the signal_data dictionary.
        If no keys are provided, all keys of the signal_data
        dictionary are taken.
        
    start_index : int
        Start index of selected range (default=0).
    
    stop_index : int
        Stop index of selected range.
        If not given --> take length of signal data.
    
    
    Returns
    -------
    Dictionary with section means for each signal --> keys are same as the selected
    in the list "signals".
    '''
    
    # if no signals are given as keys, select all keys of the input dictionary
    if not signals:
        signals = [*signal_data]
    
    # number of input data points of each signal (signals have to be of the same length --> take index 0)
    len_signals = np.shape(signal_data[signals[0]])[0]
    
    # check if stop index is given
    if stop_index is None:
        stop_index = len_signals
    
    # get indices of the sections (+1 due to start and end index of each section)
    sec_ind = np.linspace(start_index, stop_index, num_sec+1).round().astype(int)
    
    # dicitonary to save sections means for each signal
    section_means = {}

    for signal in signals:
        # generate row with zeros in order to use np.vstack afterwards
        section_means[signal] = np.zeros([1, np.shape(signal_data[signal])[1]])

        # add the mean of each section
        for ii in range(len(sec_ind)-1):
            section_means[signal] = np.vstack([section_means[signal], 
                                               np.mean(signal_data[signal][sec_ind[ii]:sec_ind[ii+1]], axis=0)])

        # delete the first row with the zeros
        section_means[signal] = np.delete(section_means[signal], 0, axis=0)
        
    return section_means
	
	
def print_progress_func(current_num, max_num, prev_prog, add_info=None):
    '''
    Function to print progress [%] in a loop.
    
    Parameters
    ----------
    current_num : int
        Number of the current run in a loop.
        
    max_num : int
        Maximum number of runs in a loop.
        
    prev_prog : int
        Previous progress, to print only if necessary.
        
    add_info : str
        Additional information to print instead of "Progress".
    
    
    Returns
    -------
    int
        Previous progress, important for next run.
    '''
    new_prog = int(current_num/max_num*100)
    
    if new_prog > prev_prog:
        clear_output(wait=True)
        
        if isinstance(add_info, str):
            print(add_info + ' {:3d}%'.format(new_prog))
        else:
            print('Progress: {:3d}%'.format(new_prog))
        
    return new_prog
	
	
def rotate_signal(signal_data, axis=0, rot_angle=90, signals=['Acc','Gyr']):
    '''
    Function to rotate signals around x, y or z-axis.
    
    Parameters
    ----------
    signal_data : dict
        Dictionary with the signals in the 'signals' argument as keys.
        The signal arrays must have three columns (x, y, z).
        
    axis : int
        Axis for rotation:
        0, 1 or 2 --> x, y or z
        
    rot_angle : int or float
        Rotation angle in degree.
        
    signals : list of strings
        Names of the signals, which shall be considered for rotation (e.g. ['Acc','Gyr']).


    Returns
    -------
    Dictionary with rotated selected signals.
    (Same structure as input signal dictionary.)

    '''
    # if no signals are given as keys, select all keys of the input dictionary
    if signals is None:
        signals = [*signal_data]
    
    # create rotation matrix
    R = skin.rotmat.R(axis=axis, angle=rot_angle)
    
    # dictionary for rotated data
    rot_signal_data = {}
    
    # rotate the signals
    for sig in signals: 
        rot_signal_data[sig] = (R @ signal_data[sig].T).T
        
    return rot_signal_data
	
	
def add_noise_to_signal(signal_data, target_snr_db=20, signals=['Acc','Gyr'], signal_orientations=['x','y','z']):
    '''
    Function to add Additive White Gaussian Noise (AWGN) to all signals with a defined SNR.
    
    Used formulas:
    SNR = P_signal / P_noise
    SNR_db = 10 * log10(P_signal / P_noise)
    SNR_db = P_signal_db - P_noise_db
    
    Parameters
    ----------
    signal_data : dict
        Dictionary with the signals in the 'signals' argument as keys.
        
    target_snr_db : int or float
        Target signal to noise ration in db.
        
    signals : list of strings
        Names of the signals, which shall be considered for rotation (e.g. ['Acc','Gyr']).
        
    signal_orientations : list of strings
        Orientations of the signals (e.g. ['x','y','z']).
    
    
    Returns
    -------
    Dictionary with noisy signals.
    (Same structure as input signal dictionary.)

    '''
    # if no signals are given as keys, select all keys of the input dictionary
    if signals is None:
        signals = [*signal_data]
    
    # dictionary for noisy data
    noisy_signal_data = {}
    
    # adding noise using target SNR
    for sig in signals:
        
        # fill in old values
        noisy_signal_data[sig] = np.zeros(np.shape(signal_data[sig]))
        
        for ii in range(len(signal_orientations)):
            
            # get power of the signal [watts] (with removed offset)
            P_signal_watts = (signal_data[sig][:,ii]-np.mean(signal_data[sig][:,ii])) ** 2
            P_signal_mean_watts = np.mean(P_signal_watts) # get mean
            P_signal_mean_db = 10 * np.log10(P_signal_mean_watts) # convert to db
            
            P_noise_mean_db = P_signal_mean_db - target_snr_db # get corresponding noise power
            P_noise_mean_watts = 10 ** (P_noise_mean_db/10) # convert from db to watts
            noise_mean_std = np.sqrt(P_noise_mean_watts) # std of noise (P_noise_mean_watts is variance)
            
            # generate sample of white noise (power = variance = P_noise_mean_watts)
            noise = np.random.normal(0, noise_mean_std, len(signal_data[sig][:,ii]))
            
            # add noise to original signal
            noisy_signal_data[sig][:,ii] = signal_data[sig][:,ii] + noise

    return noisy_signal_data
	
	
def generate_section_features_from_separate_repetitions(data_points_df,
            num_sections=10,
            csv_data_dir='E:\Physio_Data_Split_Ex_and_NonEx',
            csv_skiprows=0,
            csv_separator=',',
            signal_abbrs=['Acc','Gyr'],
            rot_axis=0,
            rot_angle=0,
            add_noise=False,
            target_snr_db=20,
            signal_orientations=['x','y','z'],
            labels_abbr2num_dict={'RF':0,'RO':1,'RS':2,'LR':3,'BC':4,'TC':5,'MP':6,'SA':7,'P1':8,'P2':9,'NE':10},
            sampling_rate=256,
            abbreviation_key='abbreviation',
            start_time_key='start_time',
            stop_time_key='stop_time',
            csv_file_key='csv_file',
            print_progress=True,
            progress_info='Generate features...'):
    '''
    Function to generate section mean features from separate repetitions, 
    which are given by the input DataFrame.
    
    Parameters
    ----------
    data_points_df : DataFrame
        DataFrame with information about data points (see load_data_from_database()).
        
    num_sections : int
        Number of sections to split the signals.
        
    csv_data_dir : string
        Directory of signal data csv-files.
        
    csv_skiprows : int
        Number of rows to skip for signal data csv-files.
        
    csv_separator : string
        Separator for signal data csv-files.
        
    signal_abbrs : list of strings
        Abbreviations of the signals (e.g. ['Acc','Gyr']).
    
    rot_axis : int or list of int
        Axis for rotation:
        0, 1 or 2 --> x, y or z
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_angle,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    rot_angle : int or float or list of int or float
        Rotation angle in degree.
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_axis,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    add_noise : boolean
        If True --> noise is added to signals.
        
    target_snr_db : int or float
        Signal to noise ratio in db for the generated noisy signals.
    
    signal_orientations : list of strings
        Orientations of the signals (e.g. ['x','y','z']).
        
    labels_abbr2num_dict : dict
        Dictionary to convert exercise abbreviations to number (e.g. ={'RF':0,'RO':1,'RS':2, ... }).
        
    sampling_rate : int or float
        Sampling rate of the signals in Hz.
    
    abbreviation_key : strings
        Exercise abbreviation key for DataFrame which contains data base entries.
        
    start_time_key : strings
        Start time key for DataFrame which contains data base entries.
        
    stop_time_key : strings
        Stop time key for DataFrame which contains data base entries.
        
    csv_file_key : strings
        csv-file key for DataFrame which contains data base entries.
        
    print_progress : boolean
        If True --> print progress at feature generation.
        
    progress_info : strings
        Additional information to print with progress.
        
    
    Returns
    -------
    X_df, y_df
        
        X_df ... DataFrame with section means of each signal
               e.g.  Acc_x_01    Acc_x_02    Acc_x_03  ...
               0    -0.939115   -0.851133   -0.074181  ...
               1    -0.928223   -1.003425   -0.495449  ...
               2    -0.896511   -0.949733   -0.381539  ...
               ...  ...         ...         ...
        
        y_df ... DataFrame with labels
               e.g.        ex_abbr    ex_num
                        0    RF         0
                        1    RF         0
                        2    RO         1
    '''
    
    # dictionary to convert number to exercise abbreviation
    labels_num2abbr_dict = {num: abbr for abbr, num in labels_abbr2num_dict.items()}
                                                   
    # create DataFrame for labels
    y_df = pd.DataFrame(np.zeros((len(data_points_df), 2), dtype=np.int8), columns=['ex_abbr', 'ex_num']) 
    
    # generate the column names of the feature matrix (depending on number of section means)
    X_columns = []
    for sig in signal_abbrs:
        for xyz in signal_orientations:
            for sec_num in range(num_sections):
                # append the current column name
                X_columns.append(sig + '_' + xyz + '_{:02d}'.format(sec_num+1))

    # create DataFrame for features
    X_df = pd.DataFrame(np.zeros((len(data_points_df), len(X_columns))), columns=X_columns)    

    # location counter for the feature DataFrame in order to append rows
    loc_count = 0

    # variables for progress printing
    if print_progress:
        prog_count = 0
        max_count = len(data_points_df.csv_file.unique()) # number of unique csv-files
        prev_progress = 0 # previous progress

    # going through all csv-files (unique --> only once for each file)
    for current_csv_file in data_points_df.csv_file.unique():

        # join file path
        file_path = os.path.join(csv_data_dir, current_csv_file)

        # load the signal data of the current file
        selected_data_df = pd.read_csv(file_path, skiprows=csv_skiprows, sep=csv_separator)
        
        # write data with selected signals to dict
        selected_data = {}
        for sig in signal_abbrs:
            selected_data[sig] = selected_data_df.filter(regex=sig+'*').values
            
        # rotate the signals
        if not isinstance(rot_axis, list): # if not list --> make list
            rot_axis = [rot_axis]
        if not isinstance(rot_angle, list): # if not list --> make list
            rot_angle = [rot_angle]
        # going through all rotation axes and rotation angles
        for current_rot_axis, current_rot_angle in zip(rot_axis, rot_angle):
            # apply rotation only if rotation angle is not zero
            if current_rot_angle != 0:
                selected_data = rotate_signal(selected_data, 
                                              axis=current_rot_axis, 
                                              rot_angle=current_rot_angle, 
                                              signals=signal_abbrs)
            
        # add noise to signal if corresponding parameter is True
        if add_noise is True:
            selected_data = add_noise_to_signal(selected_data,
                                                target_snr_db=target_snr_db, 
                                                signals=signal_abbrs, 
                                                signal_orientations=signal_orientations)
    
    
        # data frame with all repetitions of the current file
        current_data_points = data_points_df.loc[data_points_df[csv_file_key] == current_csv_file]

        # going through all repetitions of the current file and calculating the section means
        for ii in range(len(current_data_points)):

            # reset indices of current data frame in order to go through all rows 
            # and get start and stop indices via sampling rate
            start_idx = int(float(current_data_points.reset_index().loc[ii,start_time_key]) * sampling_rate)
            stop_idx = int(float(current_data_points.reset_index().loc[ii,stop_time_key]) * sampling_rate)

            # calculate the corresponding section means of the current repetition    
            section_means = split_range_into_sections(signal_data = selected_data,
                                                      num_sec = num_sections,
                                                      signals = signal_abbrs,
                                                      start_index = start_idx,
                                                      stop_index = stop_idx)
            
            # append the features to the DataFrame
            X_df.loc[loc_count] = np.concatenate([section_means[sig].transpose().flatten() for sig in signal_abbrs])

            # append current label (string + integer)
            current_ex_abbr = current_data_points.reset_index().loc[ii,abbreviation_key]
            y_df.loc[loc_count] = [current_ex_abbr, labels_abbr2num_dict[current_ex_abbr]]

            loc_count += 1

        # print progress of feauture generation
        if print_progress:
            prog_count += 1
            prev_progress = print_progress_func(prog_count, max_count, prev_progress, add_info=progress_info)
    
    clear_output()
    
    return X_df, y_df
	
	
class PhysioData_SectionFeatures():
    '''
    Class for feature generation using section means.
    There are various selectable options --> see Parameters. 
    
    Parameters
    ----------
    num_sections : int
        Number of equally partitioned sections to split the single repetitions of the signals.
        
    test_subject_ids : int or list (of int)
        Subject IDs to select for testing (e.g. [1, 2, 3]).
        --> default -1: Select all subjects.
        --> if test_subject_ids is an empty list: empty DataFrame is returned by corresponding method.
        
    train_subject_ids : int or list
        Subject IDs to select for training (e.g. [1, 2, 3]).
        --> default -1: Select all subjects not in test_subject_ids.
        --> if train_subject_ids is an empty list: empty DataFrame is returned by corresponding method.
        
    test_rep_nums : int or list
        Repetition numbers to select for testing (e.g. [5, 10]).
        --> default -1: Select all repetitions.
        
    train_rep_nums : int or list
        Repetition numbers to select for training (e.g. [5, 10]).
        --> default -1: Select all repetitions.
        
    test_ex_abbrs : int or list
        Exercise abbreviations to select for testing (e.g. ['RF', 'SA']).
        --> default -1: Select all exercise abbreviations.
        
    train_ex_abbrs : int or list
        Exercise abbreviations to select for training (e.g. ['RF', 'SA']).
        --> default -1: Select all exercise abbreviations.
    
    with_non_Ex : boolean
        If False --> omit non exercise data (data points with zero repetitions).
        
    rot_axis_test_data : int or list of int
        Axis (axes) for rotation:
        0, 1 or 2 --> x, y or z
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_angle,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    rot_angle_test_data : int or float or list of int or float
        Rotation angle(s) in degree.
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_axis,
        otherwise the shorter list of the two is taken and all other values are omitted.)
    
    add_noise_test_data : boolean
        If True --> Additive White Gaussian Noise (AWGN) is added to signals of data for testing.
    
    snr_db : int or float
        Desired signal to noise ratio in db for the generated noisy test signals.
    
    csv_data_dir : string
        Directory of signal data csv-files.
        
    csv_skiprows : int
        Number of rows to skip for signal data csv-files.
        
    csv_separator : string
        Separator for signal data csv-files.
    
    data_base_path : string
        Path to data base (containing at least the following):
            - subject IDs
            - exercise abbreviations
            - number of repetitions
            - sequence numbers
            - start times
            - stop times
            - csv-file name
        
    print_progress : boolean
        If True --> print progress at feature generation.
    
    signal_abbrs : list of strings
        Abbreviations of the signals (e.g. ['Acc','Gyr']).
    
    signal_orientations : list of strings
        Orientations of the signals (e.g. ['x','y','z']).
        
    labels_abbr2num_dict : dict
        Dictionary to convert exercise abbreviations to number (e.g. ={'RF':0,'RO':1,'RS':2, ... }).
    
    sub_id_key : string
        Key of the DataFrame for subject IDs.
        
    num_rep_key : string
        Key of the DataFrame for repetition numbers.
        
    abbreviation_key : string
        Key of the DataFrame for exercise abbreviations.
        
    start_time_key : strings
        Start time key for DataFrame which contains data base entries.
        
    stop_time_key : strings
        Stop time key for DataFrame which contains data base entries.
        
    csv_file_key : strings
        csv-file key for DataFrame which contains data base entries.
        
    sampling_rate : int or float
        Sampling rate of the signals in Hz.
        

    Attributes
    ----------
    X_test_df : DataFrame
        Features for testing.
    
    y_test_df : DataFrame
        Labels for testing.
    
    X_train_df : DataFrame
        Features for training.
    
    y_train_df : DataFrame
        Labels for testing.
    
    
    test_data_points_df : DataFrame
        Data points for testing from data base.
    
    train_data_points_df : DataFrame
        Data points for training from data base.
    
    all_data_points_df : DataFrame
        All data points from data base.


    Methods
    -------
    get_X_test_df()
        Returns features for testing as DataFrame.
    
    get_y_test_df() :
        Returns labels for testing as DataFrame.
    
    get_X_train_df() :
        Returns features for training as DataFrame.
    
    get_y_train_df() :
        Returns labels for testing as DataFrame.
    
    
    X_test():
        Returns feature matrix for testing as np.array.
    
    y_test():
        Returns numeric labels for testing as np.array.
    
    X_train():
        Returns feature matrix for training as np.array.
    
    y_train():
        Returns numeric labels for training as np.array.
    
    
    get_test_data_points()
        Returns data points for testing from data base as DataFrame.
    
    get_train_data_points()
        Returns data points for training from data base as DataFrame.
    
    get_all_data_points()
        Returns all data points from data base as DataFrame.
    '''
    def __init__(self,
                 num_sections=10,
                 test_subject_ids=-1,
                 train_subject_ids=-1,
                 test_rep_nums=-1,
                 train_rep_nums=-1,
                 test_ex_abbrs=-1,
                 train_ex_abbrs=-1,
                 with_non_Ex=True,
                 rot_axis_test_data=0,
                 rot_angle_test_data=0,
                 add_noise_test_data=False,
                 add_noise_train_data=False,
                 snr_db=20,
                 csv_data_dir='E:\Physio_Data_Split_Ex_and_NonEx',
                 csv_skiprows=0,
                 csv_separator=',',
                 data_base_path='E:\Physio_Data\DataBase_Physio_with_nonEx.db',
                 print_progress=True,
                 signal_abbrs=['Acc','Gyr'],
                 signal_orientations=['x','y','z'],
                 labels_abbr2num_dict={'RF':0,'RO':1,'RS':2,'LR':3,'BC':4,'TC':5,'MP':6,'SA':7,'P1':8,'P2':9,'NE':10},
                 sub_id_key='subject_id',
                 num_rep_key='num_rep',
                 abbreviation_key='abbreviation',
                 start_time_key='start_time',
                 stop_time_key='stop_time',
                 csv_file_key='csv_file',
                 sampling_rate=256):
        """
        Parameters
        ----------
        --> See class docstring.
        """
        
        # load all data from data points
        self.all_data_points_df = load_data_from_database(data_base_path)
        
        # load data points for testing if list is not empty
        self.test_data_points_df =  select_data_points_from_df(self.all_data_points_df,
                                                               subject_ids=test_subject_ids,
                                                               subject_ids_complementary=[],
                                                               reps=test_rep_nums,
                                                               abbrs=test_ex_abbrs,
                                                               with_non_Ex=with_non_Ex,
                                                               sub_id_key=sub_id_key,
                                                               num_rep_key=num_rep_key,
                                                               abbreviation_key=abbreviation_key)
        
        # load data points for training
        self.train_data_points_df = select_data_points_from_df(self.all_data_points_df,
                                                               subject_ids=train_subject_ids,
                                                               subject_ids_complementary=test_subject_ids,
                                                               reps=train_rep_nums,
                                                               abbrs=train_ex_abbrs,
                                                               with_non_Ex=with_non_Ex,
                                                               sub_id_key=sub_id_key,
                                                               num_rep_key=num_rep_key,
                                                               abbreviation_key=abbreviation_key)
        
        # generate features for testing if corresponding DataFrame is not empty
        if not self.test_data_points_df.empty:
            self.X_test_df, self.y_test_df =   generate_section_features_from_separate_repetitions(
                                               data_points_df=self.test_data_points_df,
                                               num_sections=num_sections,
                                               csv_data_dir=csv_data_dir,
                                               csv_skiprows=csv_skiprows,
                                               csv_separator=csv_separator,
                                               signal_abbrs=signal_abbrs,
                                               rot_axis=rot_axis_test_data,
                                               rot_angle=rot_angle_test_data,
                                               add_noise=add_noise_test_data,
                                               target_snr_db=snr_db,
                                               signal_orientations=signal_orientations,
                                               labels_abbr2num_dict=labels_abbr2num_dict,
                                               sampling_rate=sampling_rate,
                                               abbreviation_key=abbreviation_key,
                                               start_time_key=start_time_key,
                                               stop_time_key=stop_time_key,
                                               csv_file_key=csv_file_key,
                                               print_progress=print_progress,
                                               progress_info='Generate features for testing...')
            
        # otherwise create empty DataFrames for test features and labels
        else:
            self.X_test_df = pd.DataFrame()
            self.y_test_df = pd.DataFrame()
        
        # generate features for training if corresponding DataFrame is not empty
        if not self.train_data_points_df.empty:
            self.X_train_df, self.y_train_df = generate_section_features_from_separate_repetitions(
                                               data_points_df=self.train_data_points_df,
                                               num_sections=num_sections,
                                               csv_data_dir=csv_data_dir,
                                               csv_skiprows=csv_skiprows,
                                               csv_separator=csv_separator,
                                               signal_abbrs=signal_abbrs,
                                               add_noise=add_noise_train_data,
                                               target_snr_db=snr_db,
                                               signal_orientations=signal_orientations,
                                               labels_abbr2num_dict=labels_abbr2num_dict,
                                               sampling_rate=sampling_rate,
                                               abbreviation_key=abbreviation_key,
                                               start_time_key=start_time_key,
                                               stop_time_key=stop_time_key,
                                               csv_file_key=csv_file_key,
                                               print_progress=print_progress,
                                               progress_info='Generate features for training...')
            
        # otherwise create empty DataFrames for train features and labels
        else:
            self.X_train_df = pd.DataFrame()
            self.y_train_df = pd.DataFrame()
    
    
    # methods to get features
    def get_X_test_df(self):
        return self.X_test_df
    
    def get_y_test_df(self):
        return self.y_test_df
    
    def get_X_train_df(self):
        return self.X_train_df
    
    def get_y_train_df(self):
        return self.y_train_df
    
    
    # methods to get feature values only
    def X_test(self):
        return self.X_test_df.values
    
    def y_test(self):
        return self.y_test_df.values[:,1].flatten().astype('int')
    
    def X_train(self):
        return self.X_train_df.values
    
    def y_train(self):
        return self.y_train_df.values[:,1].flatten().astype('int')
    
    
    # methods to get data points (DataFrames)
    def get_test_data_points(self):
        return self.test_data_points_df
    
    def get_train_data_points(self):
        return self.train_data_points_df
    
    def get_all_data_points(self):
        return self.all_data_points_df

    
    
    
    
    
    
    
    
    
    

#--------------------------------------------------------------------------------
# Class and corresponding functions to apply windowing procedure for new data
#--------------------------------------------------------------------------------

# The following functions from the class above are used again:
#   - print_progress_func()
#   - rotate_signal()
#   - add_noise_to_signal()


def butter_lowpass(cutoff, fs, order=5):
    '''
    Function to get filter coefficients for butterworth filter.
    
    Parameters
    ----------
    cutoff : int or float
        Cutoff-frequency of the applied filter.
    
    fs : int or float
        Sampling rate in Hz.
    
    order : int
        Order of the applied filter.
    
    
    Returns
    -------
    Filter coefficients for butterworth filter (a, b).
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    Filter data with butterworth filter.
    
    Parameters
    ----------
    data : array or matrix like
        N-dimensional input array (if matrix --> one signal per column).
    
    cutoff : int or float
        Cutoff-frequency of the applied filter.
    
    fs : int or float
        Sampling rate in Hz.
    
    order : int
        Order of the applied filter.
    
    
    Returns
    -------
    Filtered data (matrix or array-like)
    '''
    
    b, a = butter_lowpass(cutoff, fs, order=order) # from scipy.signal
    
    # filter data along one-dimension
    y = lfilter(b, a, data, axis=0) # from scipy.signal
    return y


def get_sensor_data(in_file, 
                    signals=['Acc','Gyr','Mag'], 
                    sampling_rate=256, 
                    start_time=None, 
                    stop_time=None, 
                    skip_rows=0, 
                    sep=',',
                    return_time_array=True,
                    add_info='no info'):
    '''
    Function to read sensor data from a file, in order to return data from selected sensors and time range.
    
    Parameters
    ----------
    in_file: string
        Directory and file name of data (e.g. 'Subject_01/subject01.csv').
    
    signals: list of stings
        Sensor signal abbreviations (have to be equal to the first letters of the data column names!).
    
    sampling_rate: int or float
        Sampling rate of the measured signals in Hz.
    
    start_time : int or float
        Start time for selecting data in sec (if None --> start from beginning).
    
    stop_time : int or float
        Stop time for selecting data in sec (if None --> until end of data).
    
    csv_skiprows : int
        Number of rows to skip for pandas read_csv() function.
    
    csv_separator : char
        Seperator for pandas read_csv() function.
    
    return_time_array : boolean
        If True: out dict has an item (np.array) containing the time (key: "time").
    
    add_info: string
        Additional info to plot if error occurs.
    
    
    Returns
    -------
    Dictionary with selected data and time array [s]
    '''
    
    data = pd.read_csv(in_file, skiprows=skip_rows, sep=sep)
    
    num_steps = np.shape(data.values)[0] # total number of data points
    
    if start_time is None:
        start_index = 0
    else:
        start_index = round(start_time * sampling_rate)
        
    if stop_time is None:
        stop_index = num_steps
    else:
        stop_index = round(stop_time * sampling_rate)
        
    if start_index < 0 or stop_index > num_steps or start_index >= stop_index:
        print('Error at selecting data from given time range. (' + add_info + ')')
        return {}
        
    data_dict = {}
    for signal in signals:
        data_dict[signal] = data.filter(regex=signal+'*').values[start_index:stop_index]
    
    if return_time_array:
        data_dict['time'] = np.arange(num_steps)[start_index:stop_index] / sampling_rate
    
    return data_dict


def signal_windowing_via_indices(test_subject_path,
                                 number_sections=10,
                                 sig_names=['Acc','Gyr'],
                                 signal_orientations=['x','y','z'],
                                 sampling_rate=256,
                                 cutoff=10,
                                 order=6,
                                 win_start_inc=0.2,
                                 win_stretch_inc=0.2,
                                 win_min_len=1,
                                 win_max_len=5,
                                 win_start=0,
                                 win_last_start=None,
                                 print_progress=True,
                                 progress_info='Generate feature map...',
                                 rot_axis=0,
                                 rot_angle=0,
                                 add_noise=False,
                                 target_snr_db=20,
                                 csv_skiprows=0,
                                 csv_separator=','):
    '''
    This function applies a defined windowing procedure in order to split a signal 
    into different sections, which can be then taken as features for machine learning.
    The different section values are determined by taking the index in the middle of
    the corresponding section.
    In order to avoid extreme outliers a butterworth filter is used before sectioning.
    
    Parameters
    ----------
    test_subject_path : str
        Path to the csv-file of the test subject data.
        
    number_sections: int
        Number of sections to split each window.
        
    sig_names : list of strings
        Signal names, used as keys for signal dictionaries.
        
    signal_orientations : list of strings
        Orientations of the signals (e.g. ['x','y','z']).
        
    sampling_rate : int or float
        Sampling rate of the signals.
        
    cutoff : int or float
        Cutoff frequency of the butterworh filter.
        
    order : int
        Order of the butterworth filter.
        
    win_start_inc : int or float
        Start increment for the window [s].
        
    win_stretch_inc : int or float
        Stretch increment for the window [s].
    
    win_min_len : int or float
        Minimum window length [s].
    
    win_max_len : int or float
        Maximum window length [s].
    
    win_start : int or float
        Start time of the window [s].
    
    win_last_start : int or float or None
        Last start time of the window [s].
        If None, set to time where the minimum window length just fits into the sensor data.
    
    print_progress : boolean
        If True --> print progress at feature generation.
    
    progress_info : str
        Information to print with progress.
        
    rot_axis : int or list of int
        Axis for rotation:
        0, 1 or 2 --> x, y or z
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_angle,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    rot_angle : int or float or list of int or float
        Rotation angle in degree.
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_axis,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    add_noise : boolean
        If True --> noise is added to signals.
        
    target_snr_db : int or float
        Signal to noise ratio in db for the generated noisy signals.
    
    csv_skiprows : int
        Number of rows to skip for pandas read_csv() function.
    
    csv_separator : char
        Seperator for pandas read_csv() function.
    
    
    Returns
    -------
    list
        list[0] : numpy.ndarray
            Matrix with sectioned signal data.
                (Number of columns = number of features)
                (Number of rows = number of data points)
        
        list[1] : numpy.ndarray
            Array with all possible window start points for the choosen parameters [s].
        
        list[2] : numpy.ndarray
            Array with all possible window lengths for the choosen parameters [s].
        
        list[3] : int
            Length of the original signals (number of indices).
    '''


    # get data from selected file
    sensor_data = get_sensor_data(in_file=test_subject_path,
                                  signals=sig_names,
                                  sampling_rate=sampling_rate,
                                  skip_rows=csv_skiprows,
                                  sep=csv_separator)
    
    # rotate the signals
    if not isinstance(rot_axis, list): # if not list --> make list
        rot_axis = [rot_axis]
    if not isinstance(rot_angle, list): # if not list --> make list
        rot_angle = [rot_angle]
    # going through all rotation axes and rotation angles
    for current_rot_axis, current_rot_angle in zip(rot_axis, rot_angle):
        # apply rotation only if rotation angle is not zero
        if current_rot_angle != 0:
            sensor_data = rotate_signal(sensor_data, 
                                        axis=current_rot_axis, 
                                        rot_angle=current_rot_angle, 
                                        signals=sig_names)

    # add noise to signal if corresponding parameter is True
    if add_noise is True:
        sensor_data = add_noise_to_signal(sensor_data,
                                          target_snr_db=target_snr_db, 
                                          signals=sig_names, 
                                          signal_orientations=signal_orientations)

    # filter data with butterworth filter and save to new dictionary
    sensor_data_filt = {}
    for signal in sig_names:
        sensor_data_filt[signal] = butter_lowpass_filter(sensor_data[signal], 
                                                         cutoff=cutoff, 
                                                         fs=sampling_rate, 
                                                         order=order)

    # signal length: all sensor data must have same length --> Acc, Gyr, ...
    # --> but to ensure that indices are not out of range in case of wrong input data
    # let's take the smallest stop index of the different signals
    signal_len = float('inf')
    for sig in sig_names:
        if np.shape(sensor_data_filt[sig])[0] < signal_len:
            signal_len = np.shape(sensor_data_filt[sig])[0]

    # last window start is None--> set to time where the minimum window length just fits into the sensor data
    if win_last_start is None:
        win_last_start = signal_len/sampling_rate - win_min_len
    
    # array with all possible window start points
    all_window_start_points = np.arange(win_start, win_last_start, win_start_inc)
    # include win_last_start if the last start point plus the increment is equal to that value (adding end point)
    # (round due to small discrepancy)
    if round(all_window_start_points[-1] + win_start_inc, 5) == win_last_start:
        all_window_start_points = np.append(all_window_start_points, win_last_start)
    
    # array with all possible window lengths
    all_window_lengths = np.arange(win_min_len, win_max_len, win_stretch_inc)
    # include win_max_len if the max window length plus the increment is equal to that value (adding end point)
    # (round due to small discrepancy)
    if round(all_window_lengths[-1] + win_stretch_inc, 5) == win_max_len:
        all_window_lengths = np.append(all_window_lengths, win_max_len)
    
    # number of different window start points
    num_start_points = len(all_window_start_points)
    
    # number of different window sizes
    num_win_sizes = len(all_window_lengths)

    # matrix with all generated features (number_sections*6 = number of features --> Acc, Gyr (x,y,z))
    feature_map = np.zeros([num_start_points * num_win_sizes, number_sections*6])
    
    # counter for current position (row) in the feature map
    count = 0
    
    # variables for progress printing
    max_count = len(feature_map)
    prev_progress = 0 # previous progress

    # going through all window start points
    for ii, win_pos in enumerate(all_window_start_points):

        # going through all window lengths
        for jj, win_len in enumerate(all_window_lengths):

            # calculate start and stop index (type: float --> conversion to int happens afterwards)
            start_index = win_pos * sampling_rate
            stop_index = start_index + (win_len * sampling_rate)

            # check if stop index is out of range
            if stop_index >= signal_len:
                stop_index = signal_len-1 # set equal to last index

            # get indices of the sections
            section_indices, step = np.linspace(start_index, stop_index, number_sections, endpoint=False, retstep=True)

            #  + step/2 in order to get the indices in the middle of the sections
            section_indices = (section_indices + step/2).round().astype(int)

            # putting the feature map together
            #feature_map[count,:] = np.concatenate((sensor_data_filt[sig_names[0]][section_indices,:].transpose(), 
            #                                       sensor_data_filt[sig_names[1]][section_indices,:].transpose())).flatten().reshape(1, -1)
            feature_map[count,:] = np.concatenate([sensor_data_filt[sig][section_indices,:].transpose().flatten() for sig in sig_names])

            count += 1
        
        # print progress of feauture map generation
        if print_progress:
            prev_progress = print_progress_func(count, max_count, prev_progress, add_info=progress_info)
    
    return [feature_map, all_window_start_points, all_window_lengths, signal_len]


def detect_prob_map_peaks(prob_matrix,
                          win_start_inc,
                          num_win_sizes,
                          threshold_prob=0.5, 
                          footprint_length=1.5):
    '''
    Function to detect the local peaks of a probability map.
    
    Parameters
    ----------
    prob_matrix : 2d-array
        Matrix with predicted probabilities.
        
    threshold_prob : int or float
        Find only peaks with a minimum probability (threshold).
        
    footprint_length : int or float
        Length of the footprint for the maximum_filter in order to find peaks [s].
        
    win_start_inc : int or float
        Window start increment [s].
        
    num_win_sizes : int
        Number of different window sizes.


    Returns
    -------
    array 
        array[0] ... peak time indices
        array[1] ... peak window length indices
        e.g. ([[ 390, 723, 1331, ...], [4, 4, 10, ...]], dtype=int64)
    '''
    
    # length and height of the footprint for the maximum_filter (see below)
    footprint_length_indices = int(footprint_length / win_start_inc)
    footprint_height = num_win_sizes * 2  # take twice the number of all window sizes for footprint height
    
    footprint=np.ones((footprint_length_indices,footprint_height))
    
    # applying a maximum filter and generating a boolean map for local maxima
    local_max = maximum_filter(prob_matrix, footprint=footprint)==prob_matrix
    
    # removing all maxima below the threshold
    local_max = (prob_matrix>=threshold_prob) & local_max
    
    # check if there are several points with the same probability at one local maxima (within footprint length)
    #   --> remove them, otherwise we get more than one local maxima
    peak_indices_check = np.argwhere(local_max)
    if len(peak_indices_check) > 1:
        for ii in range(len(peak_indices_check)-1):
            row_ind, col_ind = peak_indices_check[ii]
            row_ind_next, col_ind_next = peak_indices_check[ii+1]
            if row_ind_next-row_ind <= footprint_length_indices/2:
                local_max[row_ind,col_ind] = False
    
    # get the maxima indices of the probability map
    peak_indices = np.argwhere(local_max).transpose()
    
    return peak_indices


def evaluate_peaks(peak_ind,
                   prob_matrix,
                   win_start_inc,
                   exercise_abbrs_peak_eval,
                   max_time_between_peaks=10,
                   min_peaks_per_block=3):
    '''
    Function to evaluate the detected peaks.
    (see function detect_prob_map_peaks(prob_matrix))
    
    --> assign peaks to repetition blocks with min two repetitions
    --> if blocks are overlapping, keep only the block with the highest predicted probabilities (sum)
    
    Parameters
    ----------
    peak_ind : dict
        Exercise-abbreviations as keys (e.g. 'RF', 'RO', ...)
        --> values: 2d-array 
        array[0] ... peak time indices
        array[1] ... peak window length indices
        e.g. ([[ 390, 723, 1331, ...], [4, 4, 10, ...]], dtype=int64)
        
    prob_matrix : dict
        Exercise-abbreviations as keys (e.g. 'RF', 'RO', ...)
        --> values: 2d-array 
        Matrices with predicted probabilities.
        
    win_start_inc : int or float
        Window start increment.
        
    exercise_abbrs_peak_eval : list of strings
        Exercises considered for peak evaluation.
        
    max_time_between_peaks : int or float
        Maximum time between two peaks in the same block [s].
        
    min_peaks_per_block : int
        Minimum number of peaks per block.


    Returns
    -------
    dict
        Dictionary with exercise abbreviations as keys --> repetition blocks
        
        Example: rep_blocks['RF'][0] (np.narray)  (first block of exercise 'RF'):
        [[4121, 9],
         [4135, 11],
         [4150, 11],
         [4166, 9],
         [4179, 10],
         [4193, 10],
         [4207, 10],
         [4221, 12],
         [4236, 13],
         [4251, 13]]
           --> 1st column: indices corresponding to horizontal axis (window start position)
           --> 2nd column: indices corresponding to vertical axis (window stretching)
           --> 10 rows --> 10 repetitions in this block
    '''
    
    # define the maximum time between two peaks in a block
    max_ind_between_peaks = int(max_time_between_peaks / win_start_inc)
    
    exercise_abbrs_peak_eval = [*peak_ind]
    
    # assign peaks to repetition blocks
    rep_blocks = {}
    for ex in exercise_abbrs_peak_eval:
        rep_blocks[ex] = []
        new_block = True # remember if current peak belongs to a new block
        
        # going through all time indices of the peaks of the current exercise
        for current_peak_time_ind, current_peak_win_ind in zip(peak_ind[ex][0], peak_ind[ex][1]):
            
            # if the current time index belongs to a new block --> append new block
            if new_block is True:
                rep_blocks[ex].append(np.array([[current_peak_time_ind, current_peak_win_ind]]))
                new_block = False
            
            # check if previous peak is within acceptable temporal distance in order to belong to the same block
            elif current_peak_time_ind - rep_blocks[ex][-1][-1,0] <= max_ind_between_peaks:
                # append (stack) the current peak to the last block
                rep_blocks[ex][-1] = np.vstack((rep_blocks[ex][-1], 
                                               np.array([[current_peak_time_ind, current_peak_win_ind]])))
                
            # append a new block
            else:
                rep_blocks[ex].append(np.array([[current_peak_time_ind, current_peak_win_ind]]))
    
    
    # check if the repetition blocks have a minimum number of peaks (min_peaks_per_block)
    valid_rep_blocks = {}
    for ex in exercise_abbrs_peak_eval:
        valid_rep_blocks[ex] = []
        # going through all blocks of the current exercise
        for rep_block in rep_blocks[ex]:
            # retain the block only if there is a minimum number of peaks
            if np.shape(rep_block)[0] >= min_peaks_per_block:
                valid_rep_blocks[ex].append(rep_block)
    
    
    # if blocks are overlapping --> retain only the block with the highest predicted probabilities (sum)
    #    --> the more peaks in the block, the higher the sum of probabilities (in general)
    blocks_to_remove = []
    
    # check all combinations of two exercises
    for ex1, ex2 in itertools.combinations(exercise_abbrs_peak_eval, 2):
        for ii in range(len(valid_rep_blocks[ex1])):
            for jj in range(len(valid_rep_blocks[ex2])):
                start_1 = valid_rep_blocks[ex1][ii][0,0] # time index of the first peak in the current block 1
                stop_1 = valid_rep_blocks[ex1][ii][-1,0] # time index of the last peak in the current block 1
                start_2 = valid_rep_blocks[ex2][jj][0,0] # time index of the first peak in the current block 2
                stop_2 = valid_rep_blocks[ex2][jj][-1,0] # time index of the last peak in the current block 2

                # check if the two blocks overlap
                if (start_1 >= start_2 and start_1 <= stop_2) or (stop_1 >= start_2 and stop_1 <= stop_2) \
                or (start_2 >= start_1 and start_2 <= stop_1) or (stop_2 >= start_1 and stop_2 <= stop_1):

                    # selecet the corresponding probability values of prob_matrix and sum them up
                    sum_prob_block_1 = prob_matrix[ex1][valid_rep_blocks[ex1][ii][:,0], 
                                                        valid_rep_blocks[ex1][ii][:,1]].sum()

                    sum_prob_block_2 = prob_matrix[ex2][valid_rep_blocks[ex2][jj][:,0], 
                                                        valid_rep_blocks[ex2][jj][:,1]].sum()

                    # compare the sum of the probabilities of the two blocks
                    if sum_prob_block_1 < sum_prob_block_2:
                        blocks_to_remove.append([ex1, ii])
                    else:
                        blocks_to_remove.append([ex2, jj])
    
    # ensure that there are no duplicates in the nested list
    blocks_to_remove_unique = []
    for sublist in blocks_to_remove:
        if sublist not in blocks_to_remove_unique:
            blocks_to_remove_unique.append(sublist)
    
    # by removing the blocks take the reversed sorted order of the block index
    #    --> so it is possible to remove all blocks without "refreshing" the indices
    #        (if one block is removed, higher indices of all other blocks are changing)
    for ex, block_ind in sorted(blocks_to_remove_unique, key=lambda x: x[1])[::-1]:
        valid_rep_blocks[ex].pop(block_ind)
        
    return valid_rep_blocks


def convert_time_format(min_sec, sampling_rate=None, time_offset=0, max_index=None, convert_to_s=False):
    '''
    Function converts a string with the time format 'min:sec' (e.g. 5:17.2)
    to a corresponding index, considering the sampling rate.
    If index would be negative, 0 is returned.
    If convert_to_s is True --> convert to seconds instead.
    
    Parameters
    ----------
    min_sec : string
        Time data, defined format: 'min:sec'
    
    sampling_rate : float or int
        Sampling rate for the index calculation. [Hz]
        
    time_offset : float of int
        Time offset, considered at the index calculation. [s]
        
    max_index : int
        Maximum valid index.
        If provided and calculated index is out of range,
        max_index is returned instead.
        
    convert_to_s : boolean
        If True --> convert to seconds.
    
    
    Returns
    -------
    int or float
        Corresponding index or value [s] to parameter 'min_max'.
    '''
    
    # split time string and convert to float
    minutes = float(min_sec.split(':')[0])
    seconds = float(min_sec.split(':')[1])
    
    # start and stop time in seconds
    time_s = minutes*60 + seconds + time_offset
    
    if convert_to_s is True:
        return time_s
    
    # get corresponding index
    index = round(time_s * sampling_rate)
    
    # ensure that index is not below 0
    if index < 0:
        index = 0
    
    # ensure that index is in valid range if max index is given
    if max_index is not None and index > max_index:
        index = max_index
            
    return index


def indices_to_time(start_index, stop_index, win_start_inc):
    '''
    Function convert indices to time string.
    
    Parameters
    ----------
    start_index : int
        
    stop_index : int
    
    win_start_inc : int or float
    
    
    Returns
    -------
    str
        String with start and stop time (e.g. '14:39.6 - 15:19.4').
    '''
    
    start_time_text = '{0:02}:{1:04.1f}'.format(int(start_index*win_start_inc/60), 
                                               (start_index*win_start_inc)%60)
    stop_time_text = '{0:02}:{1:04.1f}'.format(int(stop_index*win_start_inc/60), 
                                               (stop_index*win_start_inc)%60)
    return start_time_text + ' - ' + stop_time_text


def fill_prediction_matrix(pred_probs,
                           exercise_abbrs,
                           num_start_points,
                           num_win_sizes,
                           all_window_start_points,
                           all_window_lengths):
    
    '''
    Function to write the predicted probabilities into a dictionary 
    with prediction matrices as elements for each exercise (key).
    
    Parameters
    ----------
    pred_probs : numpy.ndarray
        Matrix with all predicted probabilities.
        (Number of rows: number of data points)
        (Number of columns: number of labels)
        
    exercise_abbrs : list of strings
        Abbreviations of exercises.
        
    num_start_points : int
        Number of window start points.
        
    num_win_sizes : int
        Number of different window sizes.
        
    all_window_start_points : array
        Array with all window start points.
        
    all_window_lengths : array
        Array with all different window lengths.
    
    
    Returns
    -------
    str
        String with start and stop time (e.g. '14:39.6 - 15:19.4').
    '''
    
    count = 0 # counter for the current row of the matrix with the predicted probabilities

    # dictionary with matrices to save predicted values for all classes
    prob_matrix_dict = {}
    for ex in exercise_abbrs:
        prob_matrix_dict[ex] = np.zeros([num_start_points, num_win_sizes])

    # going through all window start points
    for ii, win_pos in enumerate(all_window_start_points):

        # going through all window lengths 
        for jj, win_len in enumerate(all_window_lengths):

            for kk, ex in enumerate(exercise_abbrs):
                prob_matrix_dict[ex][ii,jj] = pred_probs[count,kk]
            
            count += 1

    return prob_matrix_dict


class PhysioData_WindowingProcedure():
    '''
    Class for feature generation according to a certain windowing procedure.
    There are various selectable options --> see Parameters. 
    
    Parameters
    ----------
    test_subject_dir : string
        Directory to the csv-file of the test subject data.
    
    test_subject_file : string
        Name of the csv-file.
        
    number_sections: int
        Number of sections to split each window.
        
    sig_names : list of strings
        Signal names, used as keys for signal dictionaries.
        
    signal_orientations : list of strings
        Orientations of the signals (e.g. ['x','y','z']).
        
    sampling_rate : int or float
        Sampling rate of the signals.
        
    cutoff : int or float
        Cutoff frequency of the butterworh filter.
        
    order : int
        Order of the butterworth filter.
        
    win_start_inc : int or float
        Start increment for the window [s].
        
    win_stretch_inc : int or float
        Stretch increment for the window [s].
    
    win_min_len : int or float
        Minimum window length [s].
    
    win_max_len : int or float
        Maximum window length [s].
    
    win_start_min_sec : string
        Start time of the window ['min:sec'] (e.g. '05:30.0').
    
    win_last_start_min_sec : string or None
        Last start time of the window ['min:sec'] (e.g. '10:30.0').
        If None, set to time where the minimum window length just fits into the sensor data.
    
    print_progress : boolean
        If True --> print progress at feature generation.
    
    progress_info : str
        Information to print with progress.
        
    rot_axis : int or list of int
        Axis for rotation:
        0, 1 or 2 --> x, y or z
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_angle,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    rot_angle : int or float or list of int or float
        Rotation angle in degree.
        --> if list: sequence of rotations
        (Length of list has to match with the length of rot_axis,
        otherwise the shorter list of the two is taken and all other values are omitted.)
        
    add_noise : boolean
        If True --> noise is added to signals.
        
    target_snr_db : int or float
        Signal to noise ratio in db for the generated noisy signals.
    
    csv_skiprows : int
        Number of rows to skip for pandas read_csv() function.
    
    csv_separator : char
        Seperator for pandas read_csv() function.
        
    exercise_abbrs : list of strings
        Exercise abbreviations (sequence matters).
        
    exercise_abbrs_peak_eval : list of strings
        Exercises to consider for peak evaluation (e.g. omit non-exercise).
        

    Methods
    -------
    get_feature_map()
        Returns the feature map.
        
    evaluate_probability_matrix()
        Method to evaluate a probability matrix.
        --> Parameters: See docstring of method.
        
    print_rep_blocks()
        (!) Call this method only after evaluate_probability_matrix().
        Method to print the found repetition blocks of each exercise 
        with time range and number of repetitons.
        --> Parameters: See docstring of method.
        
    plot_probability_matrices_and_peaks()
        (!) Call this method only after evaluate_probability_matrix()
        Method to plot the probability matrix as well as
        the evaluated peaks (repetitions).
        --> Parameters: See docstring of method.
        
    '''
    def __init__(self,
                 test_subject_dir  = r'E:\Physio_Data\Subject_01',
                 test_subject_file = 'subject01.csv',
                 number_sections=10,
                 signal_abbrs=['Acc','Gyr'],
                 signal_orientations=['x','y','z'],
                 sampling_rate=256,
                 cutoff=10,
                 order=6,
                 win_start_inc=0.2,
                 win_stretch_inc=0.2,
                 win_min_len=1,
                 win_max_len=5,
                 win_start_min_sec='00:00.0',
                 win_last_start_min_sec=None,
                 print_progress=True,
                 progress_info='Generate feature map...',
                 rot_axis=0,
                 rot_angle=0,
                 add_noise=False,
                 target_snr_db=20,
                 csv_skiprows=0,
                 csv_separator=',',
                 exercise_abbrs=['RF','RO','RS','LR','BC','TC','MP','SA','P1','P2','NE'],
                 exercise_abbrs_peak_eval = ['RF','RO','RS','LR','BC','TC','MP','SA','P1','P2']):
        """
        Parameters
        ----------
        --> See class docstring.
        """
        
        # convert window start position and last window start position to value in seconds
        self.win_start = convert_time_format(win_start_min_sec, convert_to_s=True)
        if win_last_start_min_sec:
            self.win_last_start = convert_time_format(win_last_start_min_sec, convert_to_s=True)
        else:
            self.win_last_start = None
        
        # parameters for the windowing procedure
        self.win_start_inc = win_start_inc
        self.win_stretch_inc = win_stretch_inc
        self.win_min_len = win_min_len
        self.win_max_len = win_max_len
        self.sampling_rate = sampling_rate
        self.exercise_abbrs = exercise_abbrs
        self.exercise_abbrs_peak_eval = exercise_abbrs_peak_eval
        self.win_start_min_sec = win_start_min_sec
        self.win_last_start_min_sec = win_last_start_min_sec
        
        # file (csv) of selected test subject
        self.test_subject_path = os.path.join(test_subject_dir, test_subject_file)
        
        # get the feature map, start points, window lengths and the signal length of the selected data
        self.feature_map, self.all_window_start_points, \
        self.all_window_lengths, self.signal_len = signal_windowing_via_indices(
                                                                         self.test_subject_path,
                                                                         number_sections=number_sections,
                                                                         sig_names=signal_abbrs,
                                                                         signal_orientations=signal_orientations,
                                                                         sampling_rate=sampling_rate,
                                                                         cutoff=cutoff,
                                                                         order=order,
                                                                         win_start_inc=win_start_inc,
                                                                         win_stretch_inc=win_stretch_inc,
                                                                         win_min_len=win_min_len,
                                                                         win_max_len=win_max_len,
                                                                         win_start=self.win_start,
                                                                         win_last_start=self.win_last_start,
                                                                         print_progress=print_progress,
                                                                         progress_info=progress_info,
                                                                         rot_axis=rot_axis,
                                                                         rot_angle=rot_angle,
                                                                         add_noise=add_noise,
                                                                         target_snr_db=target_snr_db,
                                                                         csv_skiprows=csv_skiprows,
                                                                         csv_separator=csv_separator)
        
        # last window start time --> time where the minimum window length just fits into the sensor data
        self.win_last_start = self.signal_len/self.sampling_rate - self.win_min_len
        
        # number of different window start points
        self.num_start_points = len(self.all_window_start_points)
        
        # number of different window sizes
        self.num_win_sizes = len(self.all_window_lengths)

    
    # method to get the feature map
    def get_feature_map(self):
        return self.feature_map
    
    
    def print_rep_blocks(self, print_rep_len_prob=True):
        '''
        Method to print the found repetition blocks of each exercise 
        with time range and number of repetitons.

        Parameters
        ----------
        print_rep_len_prob : boolean
            If Ture --> print individual repetition lengths and predicted probabilities.


        Returns
        -------
        no returns
        '''
        
        # going through all exercises
        for ex in self.exercise_abbrs_peak_eval:

            # plot exercise only if blocks are found
            if len(self.rep_blocks[ex]) > 0:
                print('\nExercise: ' + ex)
                print('Number of blocks: {}\n'.format(len(self.rep_blocks[ex])))

                # going through all repetition blocks of the current exercise
                for block_num in range(len(self.rep_blocks[ex])):
                    print('\tBlock #{}:'.format(block_num+1))
                    print('\t\tRepetitions: {}'.format(np.shape(np.array(self.rep_blocks[ex][block_num]))[0]))

                    # for both indices we have to consider the start position of the first window (win_start)
                    start_index = self.rep_blocks[ex][block_num][0,0] + \
                        convert_time_format(self.win_start_min_sec, sampling_rate=1/self.win_start_inc)
                    stop_index = self.rep_blocks[ex][block_num][-1,0] + \
                        convert_time_format(self.win_start_min_sec, sampling_rate=1/self.win_start_inc)

                    # for the stop index we have to consider the length of the last repetition
                    stop_index += int((self.rep_blocks[ex][block_num][-1,1]*self.win_stretch_inc \
                                       + self.win_min_len) / self.win_start_inc)

                    print('\t\tTime range: ' + indices_to_time(start_index, stop_index, self.win_start_inc))

                    if print_rep_len_prob is True:
                        print('\t\tRepetition lengths [s] and predicted prob.: ')
                        for kk, rep_length_index in enumerate(self.rep_blocks[ex][block_num][:,1]):
                            win_pos_index = self.rep_blocks[ex][block_num][kk,0]
                            print('\t\t\t{0:3d}\t{1:.2f}\t({2:.3f})'.format(kk+1,
                                                      rep_length_index*self.win_stretch_inc + self.win_min_len,
                                                      self.prob_matrix_dict[ex][win_pos_index,rep_length_index]))
    
    
    def evaluate_probability_matrix(self,
                                    pred_probabilities,
                                    max_time_between_peaks=10,
                                    min_peaks_per_block=3,
                                    threshold_prob=0.5,
                                    footprint_length=1.5,
                                    print_rep_len_prob=True):
        '''
        Evaluate a probability matrix in order to find repetition blocks.
        There are various selectable options --> see Parameters.
        After the evaluation the method print_rep_blocks() is called
        to print the found repetition blocks.

        Parameters
        ----------
        pred_probabilities :  np.narray
            Matrix with probabilities to evaluate.
            
        max_time_between_peaks : int or float
            Maximum time between two peaks of the same block [s].
        
        min_peaks_per_block : int
            Minimum number of peaks per block.
            
        threshold_prob : int or float
            Find only peaks with a minimum probability (threshold).
            (Value from 0 ... 1)
        
        footprint_length : int or float
            Length of the footprint for the maximum_filter in order to find peaks [s].
        
        print_rep_len_prob : boolean
            If True --> print individual repetition lengths and predicted probabilities.


        Returns
        -------
        no returns
        '''
        
        self.prob_matrix_dict = fill_prediction_matrix(pred_probabilities,
                                                       exercise_abbrs=self.exercise_abbrs,
                                                       num_start_points=self.num_start_points,
                                                       num_win_sizes=self.num_win_sizes,
                                                       all_window_start_points=self.all_window_start_points,
                                                       all_window_lengths=self.all_window_lengths)

        self.peak_ind_dict = {}
        for ex in self.exercise_abbrs_peak_eval:
            self.peak_ind_dict[ex] = detect_prob_map_peaks(prob_matrix=self.prob_matrix_dict[ex],
                                                           win_start_inc=self.win_start_inc,
                                                           num_win_sizes=self.num_win_sizes,
                                                           threshold_prob=threshold_prob, 
                                                           footprint_length=footprint_length)

        self.rep_blocks = evaluate_peaks(peak_ind=self.peak_ind_dict,
                                         prob_matrix=self.prob_matrix_dict,
                                         win_start_inc=self.win_start_inc,
                                         exercise_abbrs_peak_eval=self.exercise_abbrs_peak_eval,
                                         max_time_between_peaks=max_time_between_peaks,
                                         min_peaks_per_block=min_peaks_per_block)
        
        self.print_rep_blocks(print_rep_len_prob)
        
        
    def plot_probability_matrices_and_peaks(self,
                                            title_text='Predicted Probabilites',
                                            default_settings_smaller_plot=False,
                                            figsize=(18,9),
                                            cross_size=10,
                                            cross_width=1.5,
                                            fontsize_title=24,
                                            yticks_step_in_s=2,
                                            fontsize_yticks=10,
                                            fontsize_ylabels_ex=18,
                                            labelpad_ex=50,
                                            fontsize_actual_classes=12,
                                            fontsize_actual_classes_label=14,
                                            labelpad_actual_classes=50,
                                            fontsize_window_length=16,
                                            xpos_window_length=0.088,
                                            ypos_window_length=0.6,
                                            fontsize_time_xlabel=16,
                                            fontsize_time_xticks=14,
                                            fontsize_pred_prob=16,
                                            xpos_pred_prob=0.91,
                                            ypos_pred_prob=0.7,
                                            colorbar_position_x_y_length_heigth=[0.93, 0.255, 0.01, 0.625],
                                            fontsize_colorbar_ticks=14,
                                            interactive_plot=True,
                                            plot_time_range=False,
                                            start_time='00:00.0',
                                            stop_time='01:00.0',
                                            time_offset_before_and_after=0,
                                            plot_actual_classes=True,
                                            timetable_file_dir = r'E:\Physio_Data\Exercise_time_tables',
                                            timetable_file_name = 'Timetable_subject01.txt',
                                            exercise_timetable_names = {'Raises Front':'RF',
                                                                        'Raises Oblique':'RO',
                                                                        'Raises Side':'RS',
                                                                        'Rotation Wrist':'LR',
                                                                        'Biceps Curls':'BC',
                                                                        'Triceps Curls':'TC',
                                                                        'Military Press':'MP',
                                                                        'Shoulder Adduct.':'SA',
                                                                        'PNF Diagonal 1':'P1',
                                                                        'PNF Diagonal 2':'P2'}
                                           ):
        '''
        Print the probability matrix as well as the found repetitions
        by means of green crosses.

        Parameters
        ----------
        title_text : int or None
            Title of the plot.
        
        figsize : tuple
            Figure size of the plot (e.g. (18,9)).
            
        cross_size : int
            Size of the green crosses, indicating the individual repetitions.

        fontsize_ ... : int or float
            Fontsize of the corresponding text or label for plotting.
            
        plot_actual_classes : boolean
            If True --> show a separate axis with the actual classes from a timetable.
        
        timetable_file_dir : string
            Directory to the timetable file.
            (Only necessary if plot_actual_classes is True.)
            
        timetable_file_name : string
            Name of the txt-file containing the timetable with the actual classes.
            (Only necessary if plot_actual_classes is True.)


        Returns
        -------
        no returns
        '''

        # default settings for a smaller plot
        if default_settings_smaller_plot is True:
            figsize = (10, 10)
            cross_size = 12
            cross_width = 4
            fontsize_title = 18
            yticks_step_in_s = 2
            fontsize_yticks = 10
            fontsize_ylabels_ex = 16
            labelpad_ex = 45
            fontsize_actual_classes = 12
            fontsize_actual_classes_label = 14
            labelpad_actual_classes = 45
            fontsize_window_length = 14
            xpos_window_length = 0.065
            ypos_window_length = 0.6
            fontsize_time_xlabel = 14
            fontsize_time_xticks = 12
            fontsize_pred_prob = 14
            xpos_pred_prob = 0.915
            ypos_pred_prob = 0.62
            colorbar_position_x_y_length_heigth = [0.945, 0.175, 0.01, 0.705]
            fontsize_colorbar_ticks = 12
            interactive_plot = False


        # title of the plot
        self.title_text = title_text
        self.fontsize_title = fontsize_title

        yticks = np.arange(0, self.win_max_len-self.win_min_len+self.win_stretch_inc, yticks_step_in_s) / self.win_stretch_inc
        ylabels = ['{}'.format(yticks[ii] * self.win_stretch_inc + self.win_min_len) for ii in range(len(yticks))]

        # plot one axis less if plot_actual_classes is False
        if plot_actual_classes is False:
            self.fig, self.axis = plt.subplots(len(self.exercise_abbrs),1,figsize=figsize, sharex=True)
        else:
            self.fig, self.axis = plt.subplots(len(self.exercise_abbrs)+1,1,figsize=figsize, sharex=True)

        # image color settings for probabilities
        cmap = plt.cm.seismic
        vmin=0
        vmax=1

        for ax, ex in zip(self.axis, self.exercise_abbrs):
            s = ax.imshow(self.prob_matrix_dict[ex].transpose(), interpolation='nearest', 
                          aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.invert_yaxis()
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=fontsize_yticks)
            ax.set_ylabel(ex, rotation=0, fontsize=fontsize_ylabels_ex)
            ax.yaxis.labelpad = labelpad_ex
            ax.xaxis.set_ticklabels([])

        # dictionary for cross plots (in order to toggle visibility)
        self.cross_plot = {}

        # plot crosses for image peaks
        for ax, ex in zip(self.axis, self.exercise_abbrs_peak_eval):
            self.cross_plot[ex] = []
            for ii in range(len(self.rep_blocks[ex])):
                x_peak = np.array(self.rep_blocks[ex][ii])[:,0]
                y_peak = np.array(self.rep_blocks[ex][ii])[:,1]
                self.cross_plot[ex].append(ax.plot(x_peak, y_peak, '+g',
                                                   markersize=cross_size, markeredgewidth=cross_width))

        if interactive_plot is True:
            self.Button_showCross_ax = plt.axes([0.78, 0.12, 0.05, 0.03])
            self.Button_showCross = Button(self.Button_showCross_ax, 'Show rep.')
            self.Button_showCross.on_clicked(self.toggle_cross)

        self.fig.text(xpos_window_length, ypos_window_length, r'window length $[s]$',
                      fontsize=fontsize_window_length, rotation=90)

        # time axis
        formatter = FuncFormatter(lambda i, x: time.strftime('%M:%S', time.gmtime(i*self.win_start_inc+self.win_start)))
        self.axis[-1].xaxis.set_major_formatter(formatter)
        self.axis[-1].xaxis.set_tick_params(labelsize=fontsize_time_xticks)
        self.axis[-1].set_xlabel(r'time $[min:sec]$', fontsize=fontsize_time_xlabel)

        if interactive_plot is True:
            self.fig.subplots_adjust(bottom=0.2, right=0.9) # make space for buttons and color bar
        else:
            self.fig.subplots_adjust(right=0.9)  # make space only for color bar

        # color bar
        self.cbar_ax = self.fig.add_axes(colorbar_position_x_y_length_heigth)
        self.cbar = self.fig.colorbar(s, cax=self.cbar_ax)
        self.cbar.ax.set_yticklabels(['{}'.format(x) for x in [0,20,40,60,80,100]],
                                     fontsize=fontsize_colorbar_ticks)
        self.fig.text(xpos_pred_prob, ypos_pred_prob, r'predicted probability [%]',
                      fontsize=fontsize_pred_prob, rotation=90)

        if interactive_plot is True:
            # add slider for selections on the x axis
            self.Slider_shiftX_ax = plt.axes([0.125, 0.07, 0.775, 0.025])
            self.Slider_zoomX_ax = plt.axes([0.125, 0.035, 0.775, 0.025])

            axcolor = 'cornflowerblue'
            self.Slider_shiftX = Slider(self.Slider_shiftX_ax, 'time shift [%]', 0.0, 100.0, valinit=0, facecolor=axcolor)
            self.Slider_zoomX = Slider(self.Slider_zoomX_ax, 'time scale [%]', 0.1, 100.0, valinit=100, facecolor=axcolor)
            self.Slider_zoomX_ax.xaxis.set_visible(True)
            self.Slider_zoomX_ax.set_xticks(np.arange(0,105,5))

            self.Slider_shiftX.on_changed(self.updateX)
            self.Slider_zoomX.on_changed(self.updateX)

            # add button to reset view
            self.Button_resetX_ax = plt.axes([0.85, 0.12, 0.05, 0.03])
            self.Button_resetX = Button(self.Button_resetX_ax, 'Reset view')
            self.Button_resetX.on_clicked(self.resetX)

        self.start_index = 0
        self.stop_index = self.num_start_points - 1 # -1 because of image plot --> pixel in the center

        # if just a certain time range should be plotted
        if plot_time_range is True:
            self.start_index = convert_time_format(start_time, sampling_rate=1/self.win_start_inc) \
                                - self.win_start/self.win_start_inc
            self.stop_index  = convert_time_format(stop_time, sampling_rate=1 / self.win_start_inc) \
                                - self.win_start / self.win_start_inc

            # considering the time offset before and after --> convert seconds to indices
            time_offset_before_and_after_ind = time_offset_before_and_after / self.win_start_inc
            self.start_index -= time_offset_before_and_after_ind # subtract the offset from start index
            self.stop_index  += time_offset_before_and_after_ind # add the offset to stop index

        self.fig.suptitle(self.title_text + '\n' + indices_to_time(
                self.start_index + round(self.win_start/self.win_start_inc),  
                self.stop_index + round(self.win_start/self.win_start_inc), 
                self.win_start_inc), fontsize=fontsize_title)

        self.axis[-1].set_xlim(self.start_index, self.stop_index)


        # Plotting the actual classes (exercises) on the last axis:
        if plot_actual_classes is True:

            # file with timetable (csv) of the test subject
            timetable_data_path = os.path.join(timetable_file_dir, timetable_file_name)

            # read in time table
            timetable_data = pd.read_csv(timetable_data_path, skiprows=0, sep='\t', header=None)
            num_exercises = timetable_data.shape[0] # number of exercises

            self.axis[-1].set_yticks([])
            self.axis[-1].set_ylim([0,1])

            # going through all exercises in the timetable
            for ii, ex_name in enumerate(timetable_data.values[:,0]):

                # going through all repetition blocks in the timetable (5, 10 and 15 rep. blocks)
                for rep_col, start_col, stop_col in zip([1,2,3],[4,6,8],[5,7,9]): # corresponding columns
                    rep_num = timetable_data.values[ii,rep_col]
                    
                    # consider win_start for border calculation
                    left_border = convert_time_format(timetable_data.values[ii,start_col], 
                                                        sampling_rate=1/self.win_start_inc) - \
                                                        self.win_start/self.win_start_inc 
                    right_border = convert_time_format(timetable_data.values[ii,stop_col], 
                                                        sampling_rate=1/self.win_start_inc) - \
                                                        self.win_start/self.win_start_inc 
                    # mark the corresponding area
                    self.axis[-1].axvspan(left_border, right_border, color='y', alpha=0.3, lw=0)
                    # write text to the corresponding area
                    
                    # x center of marked area
                    x_center = left_border + (right_border-left_border)/2
                    self.axis[-1].text(x_center, 0.5, str(rep_num) + '\n' + exercise_timetable_names[ex_name], 
                                  horizontalalignment='center', verticalalignment='center',
                                  fontsize=fontsize_actual_classes, clip_on=True)

            self.axis[-1].set_ylabel('Actual Ex.', rotation=0, fontsize=fontsize_actual_classes_label)
            self.axis[-1].yaxis.labelpad = labelpad_actual_classes

        plt.show()
    
    
    # Auxiliary methods for the interactive plot:
    
    def updateX(self,val):
        self.start_index = int(self.Slider_shiftX.val / 100 * self.num_start_points)
        self.stop_index = self.start_index + self.Slider_zoomX.val / 100 * self.num_start_points
        self.axis[-1].set_xlim((self.start_index, self.stop_index))
        self.fig.suptitle(self.title_text + '\n' + indices_to_time(
            self.start_index + round(self.win_start/self.win_start_inc),  
            self.stop_index + round(self.win_start/self.win_start_inc), 
            self.win_start_inc), fontsize=self.fontsize_title)
        plt.draw()
        
    def toggle_cross(self,val):
        # This function is called by a button to hide/show the crosses
        for ex in self.exercise_abbrs_peak_eval:
            for ii in range(len(self.rep_blocks[ex])):
                self.cross_plot[ex][ii][0].set_visible(not self.cross_plot[ex][ii][0].get_visible())
        plt.draw()
        
    def resetX(self,val):
        self.start_index = 0
        self.stop_index = self.num_start_points
        self.axis[-1].set_xlim((self.start_index, self.stop_index))
        self.Slider_shiftX.reset()
        self.Slider_zoomX.reset()
        self.fig.suptitle(self.title_text + '\n' + indices_to_time(
            self.start_index + round(self.win_start/self.win_start_inc),  
            self.stop_index + round(self.win_start/self.win_start_inc), 
            self.win_start_inc), fontsize=self.fontsize_title)
        plt.draw()


