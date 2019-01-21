import numpy as np
import pandas as pd
import skinematics as skin
import matplotlib.pyplot as plt
import scipy 
from scipy.signal import butter, lfilter
from IPython.display import clear_output
import sqlite3
import os
import re


def get_timetable_ex_dict(timetable_file_dir=r'E:\Physio_Data\Exercise_time_tables',
                          timetable_file_name='Timetable_subject01.txt',
                          exercise_timetable_names={'Raises Front':'RF',
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
    Function to get the start and stop times of individual exercise repetition blocks
    according to a timetable (txt-file) of specific format.
    
    Parameters
    ----------
    timetable_file_dir : string
        Directory to time-table file.
    
    timetable_file_name : string
        Time-table file name.
    
    exercise_timetable_names : 
        Dictionary with exercise names according to the timetable as keys,
        and exercise abbreviations as values. (e.g. {'Raises Front':'RF'})
    
    
    Returns
    -------
    dict
        Dictionary with start and stop times in format 'min:sec'
        for each exercise repetition block.
        --> Keys are strings describing the block and start or stop time.
        
        Example:
        timetable_ex_dict['RF_10_start_time'] = '13:44.8'
        timetable_ex_dict['RF_10_stop_time'] = '14:14'
    '''

    # file with timetable (csv) of the test subject
    timetable_data_path = os.path.join(timetable_file_dir, timetable_file_name)

    # read in time table
    timetable_data = pd.read_csv(timetable_data_path, skiprows=0, sep='\t', header=None)
    num_exercises = timetable_data.shape[0] # number of exercises

    timetable_ex_dict = {}

    # going through all exercises in the timetable
    for ii, ex_name in enumerate(timetable_data.values[:,0]):

        # going through all repetition blocks in the timetable (5, 10 and 15 rep. blocks)
        for rep_col, start_col, stop_col in zip([1,2,3],[4,6,8],[5,7,9]): # corresponding columns

            rep_num = timetable_data.values[ii,rep_col]

            start_time = timetable_data.values[ii,start_col]

            stop_time = timetable_data.values[ii,stop_col]

            # write stop and start time to dict
            timetable_ex_dict['{}_{}_start_time'.format(exercise_timetable_names[ex_name], rep_num)] = start_time
            timetable_ex_dict['{}_{}_stop_time'.format(exercise_timetable_names[ex_name], rep_num)] = stop_time
    
    return timetable_ex_dict


def convert_time_format_to_index(min_sec, sampling_rate, time_offset=0, max_index=None):
    '''
    Function converts a string with the time format 'min:sec' (e.g. 5:17.2)
    to a corresponding index, considering the sampling rate.
    If index would be negative, 0 is returned.
    
    Inputs
    ------
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
     
    Returns
    -------
    int
        Corresponding index to parameter 'min_max'.
    '''
    
    # split time string and convert to float
    minutes = float(min_sec.split(':')[0])
    seconds = float(min_sec.split(':')[1])
    
    # start and stop time in seconds
    time_s = minutes*60 + seconds + time_offset
    
    # get corresponding index
    index = round(time_s * sampling_rate)
    
    # ensure that index is not below 0
    if index < 0:
        index = 0
    
    # ensure that index is in valid range if max index is given
    if max_index is not None and index > max_index:
        index = max_index
            
    return index



def get_window_indices(signal_len, window_length=5, start_time=None, start_index=0, sampling_rate=256, auto_end=True):
    '''
    This function returns the indices of a certain time range,
    selected via start time (or start index) and window length.
    
    Parameters
    ----------
    signal_len : int
        signal length (indices)
        
    window_length : float or int
        window length in seconds to select data
        
    start_time : float or int
        start time in seconds where data selection starts
        
    start_index : int
        only used, if start_time is None;
        start index where data selection starts
        
    sampling_rate : float or int
        sampling rate of signal data
        
    auto_end : boolean
        if True --> set stop index to signal length if out of range
    
    Returns
    -------
    list
        start and stop indices,
        or empty list if stop index is out of range and auto_end = False
    '''
    
    # if start time is given --> convert to start index
    if start_time is not None:
        start_index = int(start_time * sampling_rate)
    
    # calculate stop index
    stop_index = start_index + int(window_length * sampling_rate)
    
    if stop_index > signal_len:
        if auto_end is True:
            stop_index = signal_len
        else:
            return []
    
    return [start_index, stop_index]



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



def print_progress(current_num, max_num, prev_prog, add_info=None):
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



def norm_cross_corr(a, b, mode='full'):
    '''
    Calculate normalized cross-correlation of two signals a and b.
    
    Inputs
    ------
    a, b: Input sequences (array-like)
    
    mode: default --> 'full' (see numpy.correlate)
    
    
    Returns
    -------

    Discrete cross-correlation of a and b (ndarray)

    '''
    
    if len(a)<len(b):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
    else:
        a = (a - np.mean(a)) / (np.std(a))
        b = (b - np.mean(b)) / (np.std(b) * len(b))

    return np.correlate(a, b, mode=mode)



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    Filter data with butterworth filter.
    
    Inputs
    ------
    data: An N-dimensional input array (if matrix --> one signal per column)
    
    cutoff: Cutoff-frequency of the applied filter
    
    fs: Sampling rate
    
    order: Order of the applied filter
    
    
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
    
    Inputs
    ------
    in_file: directory and file name of data (e.g. 'Subject_01/subject01.csv')
    
    signals: list of sensor signal abbreviations (have to be equal to the first letters of the data column names!)
    
    sampling_rate: sampling rate of the measured signals in Hz
    
    start_time: start time for selecting data in sec (if None --> start from beginning)
    
    stop_time: stop time for selecting data in sec (if None --> until end of data)
    
    skip_rows: number of rows to skip for pandas read_csv() function
    
    sep: seperator for pandas read_csv() function
    
    return_time_array : boolean
        If True: out dict has an item (np.array) containing the time (key: "time")
    
    add_info: string --> additional info to plot if error occurs
    
    
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
        #print('Error at selecting data from given time range. (' + add_info + ')')
        return {}
        
    data_dict = {}
    for signal in signals:
        data_dict[signal] = data.filter(regex=signal+'*').values[start_index:stop_index]
    
    if return_time_array:
        data_dict['time'] = np.arange(num_steps)[start_index:stop_index] / sampling_rate
    
    return data_dict



def plot_signal(signal,
                time=None,
                labels=['x','y','z'],
                colors=['tab:blue','tab:orange','tab:green'],
                Title='Acceleration Profile',
                xLabel=r'$time \enspace [s]$',
                yLabel=r'$acc \enspace [g]$',
                g_to_ms2=False,
                legend_loc='best'):
    '''
    Function to plot signals with x, y, and z component.
    
    Inputs
    ------
    signal: matrix (n x 3) with x, y and z signal
    
    time: array with time values
    
    colors: color of plotted lines (string)
    
    xLabel: string of x-label
    
    yLabel: string of y-label
    
    g_to_ms2: if True --> convert signal from [g] to [m/s^2]
    
    
    Returns
    -------
    no returns
    '''
    
    if g_to_ms2 is True:
        yLabel=r'$acc \enspace [\frac{m}{s^2}]$'
        signal = np.array(signal)*9.80665
    
    if time is None:
        time = np.arange(np.shape(signal)[0])
    
    # loop over data, labels and colors
    for ii in range(np.shape(signal)[1]):
        plt.plot(time,signal[:,ii],color=colors[ii],label=labels[ii])

    plt.grid(True)
    plt.title(Title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(loc=legend_loc)
    plt.show()
    


def calc_trajectory(acc_lin_g, vel_ang_degps, sampling_rate=256):
    '''
    This function calculates the trajectory in an upright frame (z-axis aligned with gravity)
    by means of a set of linear acceleration [g] and angular velocity [deg/s] data.
    The sampling rate has to be given in Hz.
    
    Inputs
    ------
    acc_lin_g: linear acceleration [g]
    
    vel_ang_degps: angular velocity [deg/s]
    
    sampling_rate: sampling rate of the measured signals in Hz
    
    
    Returns
    -------
    Dictionary with upright position, 
                    velocity,
                    orientation (vector-part of quaternion) and
                    rotation matrix from initial orientation to upright space
    '''
    
    
    g_mps2 = 9.80665 # gravity [m/s^2]
    
    acc_lin = np.array(acc_lin_g) * g_mps2 # linear acceleration in m/s^2
    
    vel_ang = np.array(vel_ang_degps) * np.pi/180 # angular velocity in rad/s
    
    delta_t = 1/sampling_rate
    
    
    # check number of data points
    if acc_lin.ndim is 1 or vel_ang.ndim is 1:
        num_steps = 1
    elif np.shape(acc_lin)[0] <= np.shape(vel_ang)[0]:
        num_steps = np.shape(acc_lin)[0]
    else:
        num_steps = np.shape(vel_ang)[0]
    
    # reserve memory for position and orientation vectors
    #    --> one step more, because of initial condition
    pos_init_space = np.zeros((num_steps+1, 3))
    vel_init_space = np.zeros((num_steps+1, 3))
    q_vec = np.zeros((num_steps+1, 3)) # vector part of quaternion
         
    
    for ii in range(num_steps):
        
        omega_abs = np.linalg.norm(vel_ang[ii])
        
        # avoid division by zero (or value close to zero)
        if omega_abs < 0.00001 and omega_abs > -0.00001:
            q_vec[ii+1] = q_vec[ii]
        
        else:
            # calculate new quaternion (new orientation)
            delta_q_vec = vel_ang[ii]/omega_abs * np.sin(omega_abs*delta_t/2)
            q_vec[ii+1] = skin.quat.q_mult(q_vec[ii], delta_q_vec)
        
        # get corresponding rotation matrix for new orientation
        R_init_space_obj = skin.quat.convert(q_vec[ii+1] , to='rotmat')
        
        # get current acceleration in initial space
        acc_init_space = np.dot(R_init_space_obj, acc_lin[ii])
        
        # calculate linear velocity for next point in initial space
        vel_init_space[ii+1] = vel_init_space[ii] + acc_init_space*delta_t
        
        # calculate position for next point in initial space
        pos_init_space[ii+1] = pos_init_space[ii] +  vel_init_space[ii]*delta_t + 0.5*acc_init_space*delta_t**2
    
    
    # calculate gravity by means of final position and duration
    acc_gravity = np.array(pos_init_space[-1]) *2 / (delta_t*num_steps)**2
    
    acc_gravity_abs = np.linalg.norm(acc_gravity)
    acc_gravity_norm = np.array(acc_gravity) / acc_gravity_abs # normalize
    
    vec_upright = np.array([0,0,1]) # upright vector

    v = np.cross(acc_gravity_norm, vec_upright)
    s = np.linalg.norm(v) # sine of angle
    c = np.dot(acc_gravity_norm, vec_upright) # cosine of angle

    # skew-symmetric cross-product matrix of v
    Vx = np.array([[ 0,   -v[2], v[1] ],
                   [ v[2], 0 ,  -v[0] ],
                   [-v[1], v[0], 0    ]])

    # calculate matrix for rotation from initial space to upright space
    R_upright_init = np.identity(3) + Vx + np.dot(Vx,Vx) * (1-c)/s**2
    
    # rotate velocity and position profile into upright frame
    vel_upright = np.array([np.dot(R_upright_init, vel_init_space[ii]) for ii in range(num_steps+1)])
    pos_upright = np.array([np.dot(R_upright_init, pos_init_space[ii]) for ii in range(num_steps+1)])
    
    
    # eliminate influence of gravity:  
    for ii in np.arange(num_steps+1):
        vel_upright[ii] = vel_upright[ii] - np.array([0,0,acc_gravity_abs])*(delta_t*ii)
        pos_upright[ii] = pos_upright[ii] - 0.5*np.array([0,0,acc_gravity_abs])*(delta_t*ii)**2
    
    # create dictionary for return
    data_dict = {}
    data_dict['pos'] = pos_upright
    data_dict['vel'] = vel_upright
    data_dict['q_vec'] = q_vec
    data_dict['R_upright_init'] = R_upright_init
    
    return data_dict



def plot_trajectory(pos_data, 
                    scale_plot_section = 0.7,
                    rotmat_upright_init=None, 
                    scale_arrow=0.2, 
                    fig_size=(8,8)):
    '''
    Function to plot trajectory by means of x, y and z position values.
    
    Inputs
    ------
    pos_data: matrix (n x 3) with x, y and z position values
    
    scale_plot_section: define section for plotting (if scale = 1 --> x,y,z-limits = 1 [m])
    
    rotmat_upright_init: if given --> plot initial orientation
    
    scale_arrow: scale arrows of initial orientation
    
    fig_size: tuple with plot figure size, e.g. (7,7)
    
    
    Returns
    -------
    no returns
    '''
    # scale_plot_section: define section for plotting (if scale = 1 --> x,y,z-limits = 1 [m])
    # scale_arrow: scale arrow length
    
    # get position values in x,y,z-direction
    [pos_x, pos_y, pos_z] = [pos_data[:,ii] for ii in range(3)]

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')
    ax.plot(pos_x, pos_y, pos_z)
    ax.set_title('Calculated Trajectory',fontsize='13',fontweight='bold')
    ax.set_xlim(np.array([-1,1])*scale_plot_section)
    ax.set_ylim(np.array([-1,1])*scale_plot_section)
    ax.set_zlim(np.array([-1,1])*scale_plot_section)
    ax.set_xlabel(r'$x \enspace [m]$')
    ax.set_ylabel(r'$y \enspace [m]$')
    ax.set_zlabel(r'$z \enspace [m]$')
    ax.set_aspect('equal')

    if rotmat_upright_init is not None:
        # generate vectors for initial sensor orientation
        [init_x, init_y, init_z] = [rotmat_upright_init[:,ii] for ii in range(3)]
        [init_x, init_y, init_z]= np.array([init_x, init_y, init_z]) * scale_arrow
        origin = [0,0,0]
        X, Y, Z = zip(origin,origin,origin) 
        U, V, W = zip(init_x, init_y, init_z)

        # plot initial sensor orientation
        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.2,color='r')

        # add text for axis labels of initial orientation
        text_dis_scale = 1.1 # to ensure that labels and arrows do not overlap
        for label, pos in zip(['x','y','z'], np.array([init_x, init_y, init_z])*text_dis_scale):
            ax.text(pos[0], pos[1], pos[2], label, color='red', fontsize='11')

    plt.show()
 
    
    
def split_sensor_data(time_file_dir = r'E:\Jupyter_Notebooks\Master_Project_Meinhart\Exercise_time_tables',
                      time_file_name = 'Timetable_subject01.txt',
                      signal_file_dir  = r'E:\Physio_Data\Subject_01',
                      signal_file_name = 'subject01.csv',
                      save_dir  = r'E:\Physio_Data_Split',
                      time_offset_before = 0,
                      time_offset_after = 0,
                      sampling_rate = 256):
    
    '''
    Function splits signal data according to a txt-file with a timetable of predefined format:
    
    Name of exercise, sequence of repetitions, start and stop times (one pair for each number of repetitions)
    
    Example:
    
    Raises Oblique	15	5	10	01:18.6	01:58.3	02:22.1	02:37.1	02:54.8	03:23.3
    PNF Diagonal 2	10	5	15	04:27.1	04:54.3	05:24.5	05:38.9	06:25.8	07:05.1
    Triceps Curls	15	5	10	07:32.3	08:14.8	08:49.5	09:04.9	09:46.1	10:12.6
    Rotation Wrist	5	10	15	10:43.1	10:57.3	11:25.6	11:51.8	12:12.1	12:52.4
    ...
    ...


    Each split section is then written to a csv-file, whose name contains:
    
    Subject number, abbreviation of exercise, number of repetitions
    
    Example: subject01_RF_05.csv
    
    
    Inputs
    ------
    time_file_dir: directory of time timetable file 
    
    time_file_name: name of timetable file
    
    signal_file_dir: directory of signal file
    
    signal_file_name: name of signal file
    
    save_dir: directory to save split data
    
    time_offset_before: opportunity to decrease start times [s]
    
    time_offset_after: opportunity to increase stop times [s]
    
    sampling_rate: sampling rate of the measured signal data
    
    
    Returns
    -------
    no returns
    '''
    
    # dictionary for exercise abbreviations
    exercise_abbr = {}
    exercise_abbr['Raises Front'] = 'RF'
    exercise_abbr['Raises Oblique'] = 'RO'
    exercise_abbr['Raises Side'] = 'RS'
    exercise_abbr['Rotation Wrist'] = 'LR'
    exercise_abbr['Biceps Curls'] = 'BC'
    exercise_abbr['Triceps Curls'] = 'TC'
    exercise_abbr['Military Press'] = 'MP'
    exercise_abbr['Shoulder Adduct.'] = 'SA'
    exercise_abbr['PNF Diagonal 1'] = 'P1'
    exercise_abbr['PNF Diagonal 2'] = 'P2'
    
    # remember the subject number
    subject = re.split('[_.]',time_file_name)[1]
    
    # read in time table
    time_data_path = os.path.join(time_file_dir, time_file_name)
    time_data = pd.read_csv(time_data_path, skiprows=0, sep='\t', header=None)
    num_exercises = time_data.shape[0] # number of exercises

    # read in signal data
    signal_data_path = os.path.join(signal_file_dir, signal_file_name)
    signal_data = pd.read_csv(signal_data_path, skiprows=0, sep=',')
    num_data_points = signal_data.shape[0] #  number of data points
    
    
    # split data according to the timetable and save each exercise to a corresponding csv-file:

    # go through all exercises
    for num_ex in range(num_exercises):

        # for loop for different numbers of repetitions (columns --> 1,2,3)
        for jj in range(3): 

            # selecet time range [min:sec]
            start_min_sec = time_data.values[num_ex,4+2*jj] # 4+.. column --> start time
            stop_min_sec  = time_data.values[num_ex,5+2*jj] # 5+.. column --> stop time

            # split time string and convert to float
            start_min = float(start_min_sec.split(':')[0])
            start_sec = float(start_min_sec.split(':')[1])
            stop_min = float(stop_min_sec.split(':')[0])
            stop_sec = float(stop_min_sec.split(':')[1])

            # start and stop time in seconds
            start_time = start_min*60 + start_sec - time_offset_before # [s]
            stop_time = stop_min*60 + stop_sec + time_offset_after # [s]

            # get corresponding start and stop indices
            start_index = round(start_time * sampling_rate)
            stop_index = round(stop_time * sampling_rate)

            # ensure that indices are in valid range
            if start_index < 0:
                start_index = 0
            if stop_index >= num_data_points:
                stop_index = num_data_points-1 # end index

            # select corresponding signal data (from Pandas DataFrame)
            signal_data_selected = signal_data.iloc[start_index:stop_index+1] # +1 to include stop index

            # put out-file name together (subject number + abbreviation of exercise + number of rep. with leading 0)
            out_file_name = subject \
                             + '_' + exercise_abbr[time_data.values[num_ex,0]] \
                             + '_' + str(time_data.values[num_ex,jj+1]).zfill(2) \
                             + '.csv'

            # join save directory and out-file name
            out_file_path = os.path.join(save_dir, out_file_name)

            # save seleceted data as csv-file
            signal_data_selected.to_csv(out_file_path, sep=',')
            
            
            
def get_data_one_rep(subject_number=1,
                     exercise_abbreviation='RF',
                     number_repetitions=5,
                     sequence_number=1,
                     db_name='DataBase_Pysio.db',
                     csv_dir='E:\Physio_Data_Split_Exercise_done',
                     sampling_rate=256,
                     cutoff=10, 
                     order=6,
                     time_offset_for_filter=0.5,
                     start_time_zero=True):
    '''
    Function to get filtered data of one certain repetition of an exercise by means of a dictionary.
    
    Inputs
    ------
    subject_number: Subject number (ID) (int)
    
    exercise_abbreviation: Abbreviation of exercise (str) (e.g. 'RF')
    
    number_repetitions: Number of repetitions (int)
    
    sequence_number: Sequence number of selected repetition (int)
    
    db_name: Name of database where repetiton info is stored (e.g. 'DataBase_Pysio.db')
    
    csv_dir: Directory where csv-files with signal data are stored
    
    sampling_rate: Sampling rate of recorded data
    
    cutoff:  Cutoff frequency for butterworth lowpass filter
    
    order: order of butterworth lowpass filter
    
    time_offset_for_filter: time offset [s] (+/-) to read data from a bigger time range for filtering
                            in order to avoid bad behaviour of the filter at the beginning and at the end;
                            if time selection is not possible try with half the time offset,
                            if this is also not possible omit the time offset
                            --> BUT: output time array is always the same (additional values are cut afterwards)
    
    start_time_zero: if True, return a time array starting at zero
    
    
    Returns
    -------
    Dictionary with filtered slected data (signals: 'Acc','Gyr','Mag','time')
    '''
    
    # Connect to an existing database
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # sql command to extract data
    query_sql = """
        SELECT r.start_time, r.stop_time, e.csv_file
        FROM subjects s
        INNER JOIN exercises e
        ON s.id = e.subject_id
        INNER JOIN paradigms p
        ON p.id = e.paradigm_id
        INNER JOIN repetitions r
        ON e.id = r.exercise_id
        WHERE s.id = {}
        AND p.abbreviation = '{}'
        AND e.num_rep = {}
        AND r.sequence_num = {}
        """.format(subject_number,
                   exercise_abbreviation,
                   number_repetitions,
                   sequence_number)
    
    # get data from data base and close connection
    df_data_base = pd.read_sql_query(query_sql, conn)
    conn.close()
    
    # draw csv-file name, start time and stop time from selected data
    try:
        index=0
        file_name = df_data_base['csv_file'].values[index]
        start_time = float(df_data_base['start_time'].values[index])
        stop_time = float(df_data_base['stop_time'].values[index])
    
    # if it was not possible to select data return empty dictionary
    except IndexError:
        return {}
        
    
    
    # join the path of the csv-files folder with the file name
    file_path = os.path.join(csv_dir, file_name)
    
    
    # get data from selected csv-file and seleted time range
    data_dict_with_time_offset = get_sensor_data(in_file=file_path,
                                                 signals=['Acc','Gyr','Mag'],
                                                 start_time=start_time - time_offset_for_filter,
                                                 stop_time=stop_time + time_offset_for_filter)
    
    # if time selection is not possible (returns empty dict) try to select with half the time offset
    if not data_dict_with_time_offset:
        time_offset_for_filter = time_offset_for_filter/2
        
        data_dict_with_time_offset = get_sensor_data(in_file=file_path,
                                                     signals=['Acc','Gyr','Mag'],
                                                     start_time=start_time - time_offset_for_filter,
                                                     stop_time=stop_time + time_offset_for_filter)
        
    # if this is also not possible select data without time offset
    if not data_dict_with_time_offset:
        time_offset_for_filter = 0 # needed afterwards!
        
        data_dict_with_time_offset = get_sensor_data(in_file=file_path,
                                                     signals=['Acc','Gyr','Mag'],
                                                     start_time=start_time,
                                                     stop_time=stop_time)
    
    
    # filter data with butterworth filter and save to new dictionary (with time offset)
    data_filt_dict_with_time_offset = {}
    for signal in ['Acc','Gyr','Mag']:
        data_filt_dict_with_time_offset[signal] = butter_lowpass_filter(data_dict_with_time_offset[signal], 
                                                                        cutoff=cutoff, 
                                                                        fs=sampling_rate, 
                                                                        order=order)

    # calculate index offset which corresponds to the time offset
    index_offset = int(time_offset_for_filter * sampling_rate)
    
    # get corresponding data without time offset
    data_filt_dict = {}
    
    # select all if index_offset is zero
    if index_offset == 0:
        data_filt_dict['time'] = data_dict_with_time_offset['time']
        data_filt_dict['Acc'] = data_filt_dict_with_time_offset['Acc']
        data_filt_dict['Gyr'] = data_filt_dict_with_time_offset['Gyr']
        data_filt_dict['Mag'] = data_filt_dict_with_time_offset['Mag']
        
    # otherwise select data according to index_offset
    else:
        data_filt_dict['time'] = data_dict_with_time_offset['time'][index_offset:-index_offset]
        data_filt_dict['Acc'] = data_filt_dict_with_time_offset['Acc'][index_offset:-index_offset]
        data_filt_dict['Gyr'] = data_filt_dict_with_time_offset['Gyr'][index_offset:-index_offset]
        data_filt_dict['Mag'] = data_filt_dict_with_time_offset['Mag'][index_offset:-index_offset]
    
    # set start time=0 if start_time_zero is True
    if start_time_zero:
         data_filt_dict['time'] -= data_filt_dict['time'][0]
    
    # return filtered data
    return data_filt_dict



def print_precision_recall_accuracy(y_pred, y_test):
    '''
    This function prints precision, recall and accuracy for each exercise.
    
    Parameters
    ----------
    y_pred : array
        Prediceted classes (0...10).
    
    y_test : array
        Actual classes (0...10).
    
    Returns
    -------
    None
    '''
    
    # exercise abbreviations
    exercise_abbrs = ['RF','RO','RS','LR','BC','TC','MP','SA','P1','P2','NE']
    
    # dictionary for labels
    label_ex = {'RF':0,'RO':1,'RS':2,'LR':3,'BC':4,'TC':5,'MP':6,'SA':7,'P1':8,'P2':9,'NE':10}

    
    print('Exercise\tPrecision [%]\tRecall [%]\tAccuracy [%]')
    
    for ex in exercise_abbrs:
        TP = sum((y_pred == label_ex[ex]) & (np.array(y_test) == label_ex[ex])) # True Positives
        TN = sum((y_pred != label_ex[ex]) & (np.array(y_test) != label_ex[ex])) # True Negatives
        FP = sum((y_pred == label_ex[ex]) & (np.array(y_test) != label_ex[ex])) # False Positives
        FN = sum((y_pred != label_ex[ex]) & (np.array(y_test) == label_ex[ex])) # False Negatives

        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        accuracy = (TP+TN) / (TP+TN+FP+FN)
        
        print('  '+ ex +'\t\t  {:6.2f}'.format(precision*100)+ \
              '\t  {:6.2f}'.format(recall*100)+'\t  {:6.2f}'.format(accuracy*100))

        
        
def print_misclassified_data_points(y_pred, y_test):
    '''
    This function prints all misclassified data points.
    
    Parameters
    ----------
    y_pred : array
        Predicted classes (0...10).
    
    y_test : array
        Actual classes (0...10).
    
    Returns
    -------
    None
    '''
    
    # exercise abbreviations
    exercise_abbrs = ['RF','RO','RS','LR','BC','TC','MP','SA','P1','P2','NE'] 
    
    # indices of misclassified data points
    ind_misclassified = np.flatnonzero(y_test != y_pred) 

    # print misclassified data points
    print('{0} misclassified ({1} test data points):'.format(sum(y_test != y_pred), len(y_test)))
    for ii in ind_misclassified:
        print(exercise_abbrs[y_test[ii]] + ' classified as ' + exercise_abbrs[y_pred[ii]])

