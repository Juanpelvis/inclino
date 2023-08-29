#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:29:45 2022

@author: juanpelvis
"""
from multiprocessing import Pool

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

LIST_OF_DX_days = [30]# in days

path = '/home/juanpelvis/Documents/Inclino/'
start = datetime.strptime('2020-03-01','%Y-%m-%d')#'2019-04-01','%Y-%m-%d') # I put the 30 January so for 5 days, it starts the 1st February (at 12:30))
end = datetime.strptime('2020-10-31','%Y-%m-%d')
BH = 2
#
print('If you changed DX, mind the starting date!')
def main_process(DX):
    print_all_dt = True
    borehole = BC.borehole(path,BH)
    print('Starting for DX = ' + str(DX) + ' days\n')
    borehole.dx = 48*DX
    if print_all_dt:
        borehole.dt = 48 # 1 per day # 1 per 48 per day
        extra = '_perday' # '48perday'
    else:
        borehole.dt = 48*DX
        extra = ''
    #
    borehole.load_inclino()
    borehole.compute_tilt_az_alt(start,end,dt=borehole.dt)
    borehole.smooth_tilt()
    borehole.return_tilt()
    borehole.read_pressure()
    """
        2. Clean the data (i.e. substitute data per NaN)
    """
    if BH == 20:
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
    print('Computing du/dz for '+ str(DX) + ' days\n')
    for i in borehole.inclinos:
        borehole.inclinos[i].mask = 1
    borehole.compute_dudz_lsq()
    if BH == 20:
        borehole.inclinos[9].mask = 0
        print('Computing u_def for '+ str(DX) + ' days\n')
    borehole.compute_ud_lsq()
    #
    borehole.dudz_lsq.to_csv( path + 'NEW_DUDZ_DX' + str(DX) + 'daysBH2_paperperiod' + extra + 'test.csv')#'4Anuar.csv')
    borehole.ud_lsq.to_csv( path + 'NEW_UD_DX' + str(DX) + 'daysBH2_paperperiod' + extra + 'test.csv')#'4Anuar.csv')
    print('Finished for DX = ' + str(DX) + ' days\n')
    return borehole

p = Pool()
with p:
    p.map( main_process, LIST_OF_DX_days)
    
borehole = main_process(30)