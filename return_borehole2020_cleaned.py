"""
    This subscript provides the basic borehole built
    
    Steps:
        1. Load data for 2020
        2. Clean the data (i.e. substitute by NaN for an array of
           selected days and inclinometers)
        3. Set colors for plots
	4. Read Surface velocity
	5. Read Cavitometer
	6. Read runoff and rain
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
import read_rain
import matplotlib
from palettable.colorbrewer.sequential import YlGnBu_9 as CBCL
matplotlib.rc('text', usetex = True)
import boreholeclass as BC
from datetime import datetime
#
def borehole_all(time = False, compute_borehole = True, specific_time = ('2020-01-01', '2021-01-01'), BH = 2, read_borehole = False):
    """
        1. Load data for 2020
    """
    path = '/home/juanpelvis/Documents/Inclino/'
    if time:
        start = datetime.strptime(specific_time[0],'%Y-%m-%d')#'2019-04-01','%Y-%m-%d')
        end = datetime.strptime(specific_time[1],'%Y-%m-%d')
    else:
        start = datetime.strptime('2020-01-01','%Y-%m-%d')#'2019-04-01','%Y-%m-%d')
        end = datetime.strptime('2020-10-15','%Y-%m-%d')
    #
    borehole = BC.borehole(path,BH)
    borehole.dx = 48
    borehole.dt = 1
    #
    if read_borehole:
        borehole.load_inclino()
        borehole.return_depth()
        borehole.dudz_lsq = pd.read_csv(path + 'dudz_BH2_forfig4.csv', index_col='TIMESTAMP')
        borehole.dudz_lsq.index = pd.to_datetime(borehole.dudz_lsq.index)
    else:
        if compute_borehole:
            borehole.load_inclino()
            borehole.compute_tilt_az_alt(start,end,dt=borehole.dt)
            borehole.smooth_tilt()
            borehole.return_tilt()
            borehole.read_pressure()
            """
                2. Clean the data (i.e. substitute data per NaN)
            """
            if BH == 2:
                times = [(datetime.strptime('2020-01-19','%Y-%m-%d'),datetime.strptime('2020-01-29','%Y-%m-%d'),[12]),
                         (datetime.strptime('2020-01-31','%Y-%m-%d'),datetime.strptime('2020-02-15','%Y-%m-%d'),[9,10,11,13]),
                         (datetime.strptime('2020-02-01','%Y-%m-%d'),datetime.strptime('2020-02-15','%Y-%m-%d'),[12]),
                         (datetime.strptime('2020-03-07','%Y-%m-%d'),datetime.strptime('2020-03-11','%Y-%m-%d'),[12]),
                         (datetime.strptime('2020-07-03','%Y-%m-%d'),datetime.strptime('2020-07-10','%Y-%m-%d'),[10])]
                borehole.cut_time_period(times)
            borehole.return_tilt()
            borehole.return_azimuth()
            borehole.azimuth = borehole.azimuth.mod(360)
            borehole.return_depth()
            borehole.compute_dudz_lsq()
            if BH == 20:
                borehole.inclinos[9].mask = 0
            borehole.compute_ud_lsq()
    """
        3. Colors
    """
    colors = {}
    colors['surface'] = 'green'#CBCL.mpl_colors[3]
    colors['deformation'] = CBCL.mpl_colors[5]
    colors['sliding'] = CBCL.mpl_colors[8]
    colors['basal'] = CBCL.mpl_colors[8]
    colors['cavitometer'] = 'purple'
    colors['runoff_emosson'] = 'orange'
    colors['runoff_ige'] = 'orange'
    """
        4. Surface:
    """
    file = 'tableau_cleaned_std3D.csv'#'/for_juanpedro/Anuar_data_revisited/vel_arg1_2020_5day_edited.dat'
    aux = pd.read_csv(path + file)#, sep = '\t')
    #usurf.columns = ['TIMESTAMP', 'model ARG1']#'mm/h']
    aux['TIMESTAMP'] = pd.to_datetime( aux['TIMESTAMP'], format = '%Y-%m-%d')#'%Y-%m-%dT%H:%M:%S')
    aux.set_index(['TIMESTAMP'], inplace = True)
    usurf = pd.DataFrame()
    usurf['m/a'] = aux['model ARG1']
    usurf = usurf[usurf.index >= start]
    usurf = usurf[usurf.index < end]
    """
        5. Cavitometer
    """
    cavitometer = pd.read_csv(path+'cavito_halfh_FULL.csv',index_col = 'Timestamp')
    cavitometer.set_index(pd.to_datetime(cavitometer.index),inplace=True)
    cavitometer = cavitometer[cavitometer.index >= start]
    cavitometer = cavitometer[cavitometer.index <= end]
    cavitometer = cavitometer.drop( index = cavitometer.loc[ cavitometer['Ub m.a-1'] < 17].index)
    cavitometer = cavitometer.resample('D').mean()
    """
        6. Runoff and rain
    """
    filepath = '/home/juanpelvis/Documents/Inclino/temperature_precip_ARG_2017-2021/2017-2021/'
    rain = read_rain.read_rain(filepath,start,end)
    runoff1,runoff2 = read_rain.read_runoff(path,start,end)
    runoff1 = runoff1.ewm(span=48*2).mean(center=True)
    runoff2 = runoff2.ewm(span=48*2).mean(center=True)
    """
        End. Return data
    """
    return borehole, colors, usurf, cavitometer, rain, (runoff1,runoff2),start,end
