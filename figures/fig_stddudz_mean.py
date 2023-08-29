#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:46:56 2022

@author: juanpelvis
"""
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
sys.path.append('/home/juanpelvis/Documents/python_code/auxpy')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text', usetex = True)
import boreholeclass as BC
from datetime import datetime
import AUXplot as AUX
import matplotlib.dates as mdates

fig14, figfull, label_pt = AUX.set_figureconditions()
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,minisize, MKS = AUX.give_plotsizes()
font, MARKERmu, STYLEmu, COLORr, thicc, skinny = AUX.set_plotconditions()
# for the inset
path = '/home/juanpelvis/Documents/Inclino/'
start = datetime.strptime('2020-03-01','%Y-%m-%d')
end = datetime.strptime('2021-01-01','%Y-%m-%d')
BH = 2
#
borehole = BC.borehole(path,BH)
borehole.dx = 48+1
borehole.dt = 1
#
borehole.load_inclino() # interpolating tilt
borehole.compute_tilt_az_alt(start,end,dt=borehole.dt)
borehole.return_tilt()
figtilt, axtilt = plt.subplots(figsize = (0.25*figfull[0], 0.33*figfull[1]))
axtilt.plot(borehole.inclinos[12].tilt, label = r'BH2\#12')
axtilt.xaxis.set_tick_params(rotation=30, size=8)
axtilt.set_ylabel(r'$\theta$')
axtilt.legend(loc = 'lower right')
axtilt.xaxis.set_major_locator(plt.MaxNLocator(3))
myFmt = mdates.DateFormatter('%d-%b')
axtilt.xaxis.set_major_formatter(myFmt)
axtilt.annotate(text = '(a)', xy = (borehole.inclinos[12].tilt.index[200], 9.7))
#
depth = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_smooth2days_01Mar15Oct.csv', index_col = 'number')['depth']; dudz = pd.read_csv('/home/juanpelvis/Documents/Inclino/dudz_BH2_forfig4.csv', index_col = 'TIMESTAMP')
#
#depth = pd.read_csv('/home/juanpelvis/Documents/Inclino/NEW_DUDZ_DX30daysBH2.csv', index_col = 'number')['depth']
dudz = pd.read_csv('/home/juanpelvis/Documents/Inclino/NEW_DUDZ_DX30daysBH2.csv', index_col = 'TIMESTAMP')
fig, ax = plt.subplots(figsize = (0.75*fig14[0], fig14[1]))
BHlim = 13
startdudz = datetime.strptime('2020-02-15','%Y-%m-%d').date()
enddudz = datetime.strptime('2020-10-15','%Y-%m-%d').date()
dudz.index = pd.to_datetime(dudz.index).date
dudz = dudz.loc[ startdudz:enddudz:]
#
mean = dudz.mean()
s = dudz.std()/dudz.mean()
s = s[:19]
mean_s = s[:17].mean()
ax.plot(s, [float(i) for i in depth], 'k-', label = r'$s_n$')
ax.plot([mean_s, mean_s], [-250, -60], 'k:', label = r'$\bar{s}_n = ' + str(mean_s)[:4]+'$')
ax.set_xlim([0, 0.8])
ax.set_ylim([-250, 0])
ax.legend()
ax.set_ylabel('Depth')
ax.grid()
ax.set_xlabel(r'$s_n$')
# mean
ax.annotate(xy = (0.05, -20), text = '(b)')
fig.savefig(path + 'std_dudz.png', dpi = 300, bbox_inches = 'tight')
####
periods_to_study = ['2020-03-01', '2020-05-01', '2020-07-01', '2020-09-01',]
figperiods, axperiods = plt.subplots(figsize = (0.75*fig14[0], fig14[1]))
for i, period in enumerate(periods_to_study[:-1]):
    dudzi = dudz.loc[ pd.to_datetime( period ): pd.to_datetime( periods_to_study[i + 1])]
    s = dudzi.std()/dudzi.mean()
    s = s[:19]
    axperiods.plot(s, [float(i) for i in depth], '-', label = r'Period ' + str(period))
    #axperiods.plot(s.iloc[notindices], [float(j) for j in depth.iloc[notindices]], 'k*', label = 'Tiltmeters ignored for\n computing '+ r'$\bar{s}_n$')
    #axperiods.plot([smean, smean], [-250, -135], 'k:', label = r'$\bar{s}_n = ' + str(smean)[:4]+'$')
axperiods.set_xlim([0, 0.8])
axperiods.set_ylim([-250, 0])
axperiods.legend()
axperiods.set_ylabel('Depth')
axperiods.grid()
axperiods.set_xlabel(r'$s_n$')
# mean
mean_s = s[:BHlim].mean()
axperiods.annotate(xy = (0.7, -20), text = '(b)')
#figperiods.savefig(path + 'std_dudz.png', dpi = 300, bbox_inches = 'tight')