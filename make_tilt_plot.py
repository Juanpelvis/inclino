#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:40:53 2022

@author: juanpelvis
"""
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
import matplotlib
matplotlib.rc('text', usetex = True)
import boreholeclass as BC
from datetime import datetime
sys.path.append('/home/juanpelvis/Documents/python_code/auxpy')
import matplotlib.pyplot as plt
import AUXplot as AUX
import matplotlib.dates as mdates


fig14, figfull, label_pt = AUX.set_figureconditions()
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,minisize, MKS = AUX.give_plotsizes()
font, MARKERmu, STYLEmu, COLORr, thicc, skinny = AUX.set_plotconditions()
plt.rc('font', size=font['size'])          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=minisize)    # legend fontsize
path = '/home/juanpelvis/Documents/Inclino/'
start = datetime.strptime('2021-09-01','%Y-%m-%d')#
end = datetime.strptime('2022-07-01','%Y-%m-%d')#
BH = 14#2
#
borehole = BC.borehole(path,BH)
borehole.dx = 48+1
borehole.dt = 1
#
borehole.load_inclino() # interpolating tilt
borehole.compute_tilt_az_alt(start,end,dt=borehole.dt)
borehole.return_tilt()
borehole.return_depth()
depth = borehole.depth['depth']
initial_date = datetime.strptime('2021-09-25','%Y-%m-%d')
final_date = datetime.strptime('2022-06-15','%Y-%m-%d')
fig, a = plt.subplots(ncols = 4, figsize = (figfull[0], 0.25 * figfull[1]))
tilt_list = [1, 3, 11, 16]
myFmt2 = mdates.DateFormatter('%Y-%m')
for i in range(len(tilt_list)):
    a[i].plot(borehole.tilt['Tilt(' + str(tilt_list[i]) + ')'], )
    a[i].set_title(r'BH' + str(BH) + '\#' + str(tilt_list[i]) + ' at -' + str(depth.loc[tilt_list[i]])+' m')
    #a[i].legend(loc = 'upper left')
    if i == 0:
        a[i].set_ylabel(r'$\theta$, deg', rotation = 90)
    a[i].xaxis.set_major_locator(plt.MaxNLocator(8))
    a[i].xaxis.set_major_formatter(myFmt2)
    a[i].xaxis.set_tick_params(rotation=45)
    a[i].yaxis.set_tick_params(rotation=0)
#    a[i].plot([initial_date, initial_date],[borehole.tilt['Tilt(' + str(tilt_list[i]) + ')'].min(), borehole.tilt['Tilt(' + str(tilt_list[i]) + ')'].max()], 'k:')
    a[i].set_xlim( [initial_date, final_date])
plt.subplots_adjust(wspace=0.27,)
fig.savefig(path + 'initial_tilt_plot_2021.png', dpi = 300, bbox_inches = 'tight')