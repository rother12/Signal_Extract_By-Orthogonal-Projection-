import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import svd
import scipy.signal

import mcgpy
from mcgpy.channel import ChannelActive
from mcgpy.timeseries import TimeSeries
from mcgpy.timeseries import TimeSeriesArray
from mcgpy.channel import ChannelConfig

__credit__='AMCG'
__version__='0.0.1'
__author__ = 'WonSik.Jung <rother12@hanmail.net>'
__all__=['get_Configure_Setting','Check_R_Peak,Averaging_R_peak','projection','Gram_Schmidt','projection_Operation','eye_projection']

''' Reference
Fetal MEG Redistribution by Projection Operators(2004, Jiri Vrba)
'''

def get_Configure_Setting(DataPath):
    '''
    Get Sensor Informataion Positions,Directions,Labels

    Parameters
    ----------
    Datapath : "str","numpy",etc
        Datapath is path of kdf file or Timeseriesarray Data

    Return : "Channel position, Channel directions"
    ------
        1) if the datapath is not Data paths
           return the Default Value(64 Config_KRISS)

    Examples
    --------
    >>> from mcgpy.channel import ChannelActive
    >>> import get_Configure_Setting

    >>> position,direction=get_Configure_Setting(Something)
    >>> position
    ([[-31.0, 70.0, 0.0],
      [4.0, 70.0, 0.0],
      [39.0, 70.0, 0.0],'''

    try:
        config = ChannelConfig(DataPath)
        label = config.get('label')
        positions = config.get('positions')
        directions = config.get('directions')

        ActiveChannel = channelActivate(DataPath)
        ActiveChanneltable = ActiveChannel.get_table()
        ActiveChannelnumber = ActiveChannel.get_number()
        ActiveChannellabel = ActiveChannel.get_label()
    except:
        positions = [[-31.000, 70.000, 0.000], [4.000, 70.000, 0.000], [39.000, 70.000, 0.000], [61.000, 70.000, 0.000],
                     [-66.000, 35.000, 0.000], [-31.000, 35.000, 0.000],
                     [4.000, 35.000, 0.000], [39.000, 35.000, 0.000], [74.000, 35.000, 0.000], [96.000, 35.000, 0.000],
                     [-66.000, 0.000, 0.000], [-31.000, 0.000, 0.000],
                     [4.000, 0.000, 0.000], [39.000, 0.000, 0.000], [74.000, 0.000, 0.000], [96.000, 0.000, 0.000],
                     [-66.000, -35.000, 0.000], [-31.000, -35.000, 0.000],
                     [4.000, -35.000, 0.000], [39.000, -35.000, 0.000], [74.000, -35.000, 0.000],
                     [96.000, -35.000, 0.000], [-31.000, -70.000, 0.000], [4.000, -70.000, 0.000],
                     [39.000, -70.000, 0.000], [61.000, -70.000, 0.000], [-55.000, 46.000, 0.000],
                     [-55.000, 11.000, 0.000], [-55.000, -24.000, 0.000], [-55.000, -46.000, 0.000],
                     [-20.000, 81.000, 0.000], [-20.000, 46.000, 0.000], [-20.000, 11.000, 0.000],
                     [-20.000, -24.000, 0.000], [-20.000, -59.000, 0.000], [-20.000, -81.000, 0.000],
                     [15.000, 81.000, 0.000], [15.000, 46.000, 0.000], [15.000, 11.000, 0.000],
                     [15.000, -24.000, 0.000], [15.000, -59.000, 0.000], [15.000, -81.000, 0.000],
                     [50.000, 81.000, 0.000], [50.000, 46.000, 0.000], [50.000, 11.000, 0.000],
                     [50.000, -24.000, 0.000], [50.000, -59.000, 0.000], [50.000, -81.000, 0.000],
                     [85.000, 46.000, 0.000], [85.000, 11.000, 0.000], [85.000, -24.000, 0.000],
                     [85.000, -46.000, 0.000], [11.000, 35.000, 0.000], [-24.000, 0.000, 0.000],
                     [11.000, 0.000, 0.000], [46.000, 0.000, 0.000], [11.000, -35.000, 0.000],
                     [-35.000, -11.000, 0.000], [0.000, 24.000, 0.000], [0.000, -11.000, 0.000],
                     [0.000, -46.000, 0.00], [35.000, -11.000, 0.000], [46.000, -35.000, 0.00],
                     [35.000, -46.000, 0.000]]

        directions = [[-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [-1.000, 0.000, 0.000],
                      [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, -1.000, 0.000],
                      [1.000, 0.000, 0.000], [1.000, 0.000, 0.000],
                      [1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [1.000, 0.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, -1.000, 0.000], [0.000, -1.000, 0.000],
                      [0.000, -1.000, 0.000], [0.000, -1.000, 0.000], [1.000, 0.000, 0.000], [0.000, -1.000, 0.000]]

    return positions, directions


def Check_R_Peak(DataPath, notch=False, bandpass=False, amplitude_height=0.5):
    # data_positions,data_directions=get_Configure_Setting(DataPath)
    '''
    Get R_Peak Time and Value Informataion

    Parameters
    ----------
    Datapath : "str","numpy",etc
        Datapath is path of kdf file, or Timeseriesarray Data

    notch : "int" or "str"
        Set Notch filter( Signal PreProcessing filter)
        ex: 180

    bandpass : "float" or "str"
        Set band pass filter( Signal PreProcessing filter)
        ex([0.5,200])

    amplitude_height : "float"
        Set Max amplitude_height in Signal, For Catch R_peak


    Return : "Value list, Time list"
    ------
        1) Value_list : Value(Amplitude) in Each R_peak
           return the values 'list'

        2) Time_list : Time(R_peak) in Each R_peak
           return the values 'list'

    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> import Check_R_Peak

    >>> Value list,Time list=Check_R_Peak(fetal_MCG)
    >>> Value list
    [1.2345197197969593,
     1.2008232268503183,
     1.1932767673317954,
     1.1976844623346348,
     1.2145512914884529,
     1.1980357564084518,'''
    data_positions, data_directions = get_Configure_Setting(DataPath)

    try:
        data = TimeSeriesArray(DataPath)
        data1 = TimeSeriesArray(DataPath)


    except:
        data = TimeSeriesArray(source=DataPath, positions=data_positions[:DataPath.shape[0]],
                               directions=data_directions[:DataPath.shape[0]], t0=0)
        data1 = TimeSeriesArray(source=DataPath, positions=data_positions[:DataPath.shape[0]],
                                directions=data_directions[:DataPath.shape[0]], t0=0)

    if notch != False:
        data = data.notch(float(notch))
    elif bandpass != False:
        data = data.bandpass(float(bandpass[0]), float(bandpass[1]))
    else:
        pass

    data_rms = data.to_rms()
    r_peak_test = scipy.signal.find_peaks(data_rms, height=amplitude_height)[0]
    value_stamp = data_rms.value
    time_stamp = data_rms.times
    value_list = []
    time_list = []
    for time_index in r_peak_test:
        for num, time_stamp_list in enumerate(time_stamp.to_value()):
            if time_index == num:
                value_list.append(value_stamp[num])
                time_list.append(num)

    return value_list, time_list


def Averaging_R_peak(DataPath, time_list, split=1):
    '''Get Averaged Signal (By R_peak)

    Parameters
    ----------
    Datapath : "str","numpy",etc
        Datapath is path of kdf file, or Timeseriesarray Data
        If put Timeseriesarray Data it apply 64 channel Config, SampleRate: 1024,Time Duration: 30s

    time_list : "list"
        Set R_peak time(R_peak will be Standard)

    split : "int"
        Set Deleted Signal in start, last (Signal PreProcessing filter)
        ex(1) is Delete first and last Signal

    Return : "avg_signal"
    ----------
        1) avg_signal : Averaged Signal in Each R_peak
           return the values 'numpy' (C*T)

        2) C : Channel, T: Averaged Time

    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> Averaging_R_peak(np.array(f_mcg_combined_data),np.array(Check_R_Peak(f_mcg_combined_data))[1])
    >>>
        array([[0.0271786 , 0.0277465 , 0.02975252, ..., 0.02797596, 0.02973287,
        0.03148546],
       [0.0273628 , 0.02874066, 0.03148462, ..., 0.03187839, 0.03160114,
        0.03343634],
       [0.01547806, 0.01708304, 0.01704547, ..., 0.02137328, 0.02094706,
        0.02045957],'''

    data_positions, data_directions = get_Configure_Setting(DataPath)
    time_list = [int(x) for x in time_list]
    try:
        data = TimeSeriesArray(DataPath)
        data1 = TimeSeriesArray(DataPath)

    except:
        data = TimeSeriesArray(source=DataPath, positions=data_positions[:DataPath.shape[0]],
                               directions=data_directions[:DataPath.shape[0]], sample_rate=1024,
                               times=np.linspace(0, 30, 15000))
        data1 = TimeSeriesArray(source=DataPath, positions=data_positions[:DataPath.shape[0]],
                                directions=data_directions[:DataPath.shape[0]], sample_rate=1024,
                                times=np.linspace(0, 30, 15000))

    time_data = np.array(data1)

    gap = []
    for num, r_index in enumerate(time_list):
        if num != 0:
            gap_value = r_index - time_list[num - 1]
            gap.append(gap_value)

    r_signal_avg_time = np.array(time_list).sum().mean()
    signal_avg_time = np.mean(gap + [time_list[0]])

    k = int(split)

    for num, value in enumerate(gap):
        if num == k and value > 0:
            signal = time_data[:, value + time_list[num] - int(signal_avg_time // 2):value + time_list[num] + int(
                signal_avg_time // 2)]

        if num > k and num <= len(gap) - 1 - k:
            if value > 0:
                signal += time_data[:, value + time_list[num] - int(signal_avg_time // 2):value + time_list[num] + int(
                    signal_avg_time // 2)]
            else:
                signal += time_data[:, time_list[num] - int(signal_avg_time // 2) - value:time_list[num] + int(
                    signal_avg_time // 2) - value]

    signal_num = len(gap) - 2 * k
    avg_signal = signal / signal_num

    return avg_signal


def projection(A, B):
    '''Get projection Result(A Given B)

    Parameters
    ----------
    A : "numpy"
        Vector

    B : "numpy"
        Vector


    Return : "Projected_Result"
    ----------
        1) Projected_Result : 'numpy' (B Project to A)


    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> Averaging_R_peak(np.array(f_mcg_combined_data),np.array(Check_R_Peak(f_mcg_combined_data))[1])

    >>> A=[1,0,0]
    >>> B=[2,2,2]
    >>> projection(A,B)
    array([2., 0., 0.])
    '''

    C = np.array(A) * np.dot(A, B) / (np.dot(A, A))
    return C


def Gram_Schmidt(avg_signal):
    '''Get Gram_Scmidt Vectors

    Parameters
    ----------
    avg_signal : "numpy"
        Averaged Signal in Vectors

    Return : "Projected_Result"
    ----------
        1) Projected_Result : 'numpy' (B Project to A)

    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> Averaging_R_peak(np.array(f_mcg_combined_data),np.array(Check_R_Peak(f_mcg_combined_data))[1])

    >>> A=[1,0,0]
    >>> B=[2,2,2]
    >>> projection(A,B)
    array([2., 0., 0.])
    '''

    orthogonal_vector = np.array(avg_signal[:][:])
    for i in range(avg_signal.shape[0]):
        sum = 0
        for j in range(i):
            if i >= 1:
                sum += projection(orthogonal_vector[:][j], orthogonal_vector[:][i])
            else:
                pass
        orthogonal_vector[:][i] -= sum
        orthogonal_vector[:][i] = projection(orthogonal_vector[:][i], orthogonal_vector[:][i])

    Orthogonal_vector_norm = [np.linalg.norm(x) for x in orthogonal_vector]

    return orthogonal_vector, Orthogonal_vector_norm


def projection_Operation(E):
    '''Get Projection_Operation
    (Based on Hat Matrix in Linear Regression)

    Parameters
    ----------
    E : "numpy" matrix X

    Return : "projection_Operation"
    ----------
        1) Projected_Result : 'numpy' Matrix (Square Matrix)
                                E(E^T*E)^-1 E^T

    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> Averaging_R_peak(np.array(f_mcg_combined_data),np.array(Check_R_Peak(f_mcg_combined_data))[1])

    >>> A=[1,0,0]
    >>> B=[2,2,2]
    >>> projection(A,B)
    array([2., 0., 0.])
    '''

    E = np.array(E)
    E_1 = np.linalg.inv(np.dot(E.T, E))
    E_2 = np.dot(E, E_1)
    E_3 = np.dot(E_2, E.T)

    return E_3


def eye_projection(E, Origin=True):
    '''Get Eye Projection
    (Based on Hat Matrix in Linear Regression)

    Parameters
    ----------
    E : "numpy" matrix X
    eye_projection Matrix

    Origin: "str"
    True:I-E
    False:I-projection_Operation(E)

    Return : "I-E (Or) I-projection_Operation(E)"
    ----------
        1) Projected_Result(Default) : 'numpy' Matrix (Square Matrix)
                                I-E

        2) Projected_Result : 'numpy' Matrix (Square Matrix)
                                I-E(E^T*E)^-1 E^T


    Examples
    --------
    >>> import mcgpy.TimeSeriesArray
    >>> E=[[1,0,0],[0,1,0],[0,0,1]]
    >>>    eye_projection(E,Origin=False)
    >>>
    '''

    project_result = projection_Operation(E)
    k = project_result.shape[0]

    if Origin != True:
        result = np.eye(k, k) - E
    else:
        result = np.eye(k, k) - project_result
    return result

