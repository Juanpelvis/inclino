#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:36:42 2021

@author: juanpelvis

Produces the timeseries plot
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
sys.path.append('/home/juanpelvis/Documents/python_code/auxpy')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper/for_Adri')
import matplotlib
from matplotlib.gridspec import GridSpec
matplotlib.rc('text', usetex = True)
import boreholeclass as BC
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import return_borehole2020_cleaned as BHdata
import AUXplot as ap
from scipy import stats
# In[]
""" Creating and formatting plots """
fig14, figfull, label_pt = ap.set_figureconditions()
fig, ( axprec, axa ) = plt.subplots(figsize = (1.5*figfull[0], 0.7*figfull[1]), nrows = 2, gridspec_kw={'height_ratios': [1, 4]})
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,minisize, MKS = ap.give_plotsizes()
font, MARKERmu, STYLEmu, COLORr, thicc, skinny = ap.set_plotconditions()
plt.rc('font', size=font['size'])          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=minisize)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# In[]: 
""" Reading data, setting start and end dates """
path = '/home/juanpelvis/Documents/Inclino/'
borehole, colors, udsurf, cavitometer, rain, (runoff1,runoff2),start,end = BHdata.borehole_all(time = True, compute_borehole = False,specific_time = (('2020-02-01', '2020-10-15')), read_borehole = True)
start = datetime.strptime('2020-02-15','%Y-%m-%d').date() # override
end = end.date()
# Rewrites colors
colors['cavitometer'] = 'red'
colors['basal']  ='black'
dudz_2dayssmoothed = borehole.dudz_lsq
runoff_combined = np.max(pd.concat( [runoff1, runoff2], axis = 1), axis = 1) # combines runoff readings since the 'new' reading has an upper limit
""" Overread observed mean """
observed_mean = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_smooth2days_01Mar15Oct.csv', index_col = 'number')
axprec.plot( runoff_combined.resample('D').mean(), color = 'k', label = 'Discharge')
axprec.set_ylabel(r'Discharge~m\textsuperscript{3}~s\textsuperscript{-1}')
axprec.grid()
axprec.set_xticklabels([])
for A in [axprec]:#, axprec_twin]:
    A.set_xlim([start, end])
fig.subplots_adjust( hspace = 0)
#In[]: 
""" Format velocities and plot velocities """
#
DAYS = '1'
freq_rolling_window = 'D'
borehole.ud_lsq = pd.read_csv(path + 'NEW_UD_DX' +  DAYS + 'daysBH2_paperperiod.csv', index_col='number')
borehole.ud_lsq.columns = pd.to_datetime(borehole.ud_lsq.columns).date
# Deformation velocity is velocity at the topmost capteur that is in contact with the ice
deformation_velocity = borehole.ud_lsq.loc[17]
#
common = udsurf.index.intersection( borehole.ud_lsq.iloc[-1].index) # common index
basal_unfiltered = udsurf['m/a'][common] - deformation_velocity.loc[ common]
# In[]:
""" Plot velocities """
al = 1 # alpha color
lw = 2 # linewidth
axa.plot(udsurf['m/a'], color = colors['surface'], alpha = al,lw=lw, label = '$u_s$', marker = '', zorder = 3.7)
axa.plot(deformation_velocity, color = colors['deformation'], alpha = al,lw=lw, label = '$u_d$', zorder = 3.6)
axa.plot(basal_unfiltered, color = colors['basal'], alpha = al, lw=lw,label = '$u_b$', zorder = 3.5)
axa.plot( cavitometer, color = colors['cavitometer'], alpha = al,lw=lw, label = '$u_{cav}$', zorder = 3)
# Plot means
al = 0.5
axa.plot([start, end], [udsurf['m/a'].loc[start:end].mean(), udsurf['m/a'].loc[start:end].mean()], color = colors['surface'], lw = 2, alpha = al,ls = '--', zorder = 3.7)
axa.plot([start, end], [deformation_velocity.loc[start:end].mean(), deformation_velocity.loc[start:end].mean()], color = colors['deformation'], lw = 2, alpha = al,ls = '--', zorder = 3.6)
axa.plot([start, end], [basal_unfiltered.loc[start:end].mean(), basal_unfiltered.loc[start:end].mean()], color = colors['basal'], lw = 2, alpha = al,ls = '--', zorder = 3.5)
axa.plot([start, end], [cavitometer.loc[start:end].mean(), cavitometer.loc[start:end].mean()], color = colors['cavitometer'], lw = 2, alpha = al, ls = '--', zorder = 3)

for a in [axa]:
    a.set_xlim([ start, end])

axa.grid( which = 'major')
axa.legend(loc = 'upper left')
axa.set_ylim([10, 75])
axa.set_ylabel('m/a')
axa.set_xlabel('2020')
myFmt = mdates.DateFormatter('%b-%d')#-%d')
axa.xaxis.set_tick_params(rotation=30, size=8)
axa.xaxis.set_major_formatter(myFmt)
axprec.annotate( xy = ( datetime.strptime('2020-10-05','%Y-%m-%d'), 7), text = '(a)')
axa.annotate( xy = ( datetime.strptime('2020-10-05','%Y-%m-%d'), 71), text = '(b)')
fig.savefig(path + 'comparison_speeds_seasonal_at_' + freq_rolling_window + '.png', dpi = 300, bbox_inches = 'tight')