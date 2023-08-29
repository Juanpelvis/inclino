#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:47:52 2022

@author: juanpelvis
"""
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

path = '/home/juanpelvis/Documents/Inclino/'

LIST_OF_DX_DAYS = [1, 7, 15, 60, 120]
#
dudz = {}
ud = {}
udat6 = {}
udat9 = {}
#
depth = pd.read_csv( path + 'depth.csv', index_col = 'inclino')

for dx in LIST_OF_DX_DAYS:
    dudz[dx] = pd.read_csv( path + 'NEW_DUDZ_DX' + str(dx) + 'daysBH2.csv', index_col='TIMESTAMP')
    dudz[dx].index = pd.to_datetime(dudz[dx].index)
    for col in dudz[dx].columns:
        dudz[dx][col].loc[dudz[dx][col] < 1e-3] = np.nan
    ud[dx]= pd.read_csv( path + 'NEW_UD_DX' + str(dx) + 'daysBH2.csv', index_col = 'number')
    ud[dx].columns = pd.to_datetime(ud[dx].columns)
#
figdudz, axdudz = plt.subplots( figsize = (4, 8))
figudat9, axudat9 = plt.subplots( figsize = (8, 4))
figud, (axud, axudat6, axudat10) = plt.subplots( figsize = (8, 12), nrows = 3)
for dx in LIST_OF_DX_DAYS:
    axdudz.plot( dudz[dx].mean(), depth, '-*', label = 'DUDZ every ' + str(dx) + ' days')
    axud.plot( ud[dx].iloc[-1], '-', label = 'UD every ' + str(dx) + ' days')
    axudat6.plot( ud[dx].iloc[-1] - ud[dx].loc[6], '-', label = 'UD every ' + str(dx) + ' days')
    udat6[dx] = ud[dx].iloc[-1] - ud[dx].loc[6]
    axudat9.plot( ud[dx].iloc[-1] - ud[dx].loc[10], '-', label = 'UD every ' + str(dx) + ' days')
    udat9[dx] = ud[dx].iloc[-1] - ud[dx].loc[10]
#
for dx in LIST_OF_DX_DAYS:
    axudat10.plot(ud[dx].iloc[-1] - ud[dx].loc[10], '-', label = 'UD every ' + str(dx) + ' days')
axudat10.set_ylim([4, 15])
START_PAPER_DATETIME = datetime.strptime('2020-02-01','%Y-%m-%d')#
END_PAPER_DATETIME = datetime.strptime('2020-10-15','%Y-%m-%d')#
ALTERNATIVEEND_PAPER_DATETIME = datetime.strptime('2022-10-15','%Y-%m-%d')#
AUX_PAPER_DATETIME = datetime.strptime('2020-04-15','%Y-%m-%d')#
K = 0
for A in [axud, axudat6, axudat10, axudat9]:
    A.plot([START_PAPER_DATETIME, START_PAPER_DATETIME], [10, 50 - K], 'k:')
    A.plot([END_PAPER_DATETIME, END_PAPER_DATETIME], [10, 50 - K], 'k:')
    A.annotate( s = '', xy=(END_PAPER_DATETIME,50 - K), xytext=(START_PAPER_DATETIME,50- K), arrowprops=dict(arrowstyle='<->'))
    A.annotate( s = 'Period covered in the paper', xy =(AUX_PAPER_DATETIME,50.5- K),)
    A.grid()
    K = K + 18
#
axdudz.set_xlabel('du/dz')    
axdudz.set_ylabel('depth')
axdudz.set_title('From 18 September 2019 until cable breaks')#'20 March 2021')
axud.set_ylabel('')
#
axdudz.legend()
axdudz.set_xlim([ -0.05, 1.4])
axud.legend()
#
axdudz.grid()
#
figdudz_studiedperiod, axdudz_studiedperiod = plt.subplots( figsize = (4, 8))

for dx in LIST_OF_DX_DAYS:
    dudz[dx] = dudz[dx].loc[ dudz[dx].index >= START_PAPER_DATETIME]
    dudz[dx] = dudz[dx].loc[ dudz[dx].index <= ALTERNATIVEEND_PAPER_DATETIME] # ALTERNATIVEEND_PAPER_DATETIME]#
    #for col in dudz[dx].columns:
    #    dudz[dx].loc[dudz[dx][col] < 1e-3] = np.nan
    axdudz_studiedperiod.plot( dudz[dx].mean(), depth, '-*', label = 'DUDZ every ' + str(dx) + ' days')
#
figdudz_compareaverages, ax_dudz_compareaverages = plt.subplots( figsize = (4, 8))
dudz_ALLDAYS_inpaper = pd.read_csv( path + 'NEW_DUDZ_DX271daysBH2_paperperiod.csv', index_col = 'TIMESTAMP')
ax_dudz_compareaverages.plot( dudz_ALLDAYS_inpaper.iloc[0], depth, '-*', color = 'r', label = 'DUDZ from 271 days')
#
dudz_lsq4paper = pd.read_csv(path + 'dudz_BH2_forfig4.csv', index_col='TIMESTAMP')
ax_dudz_compareaverages.plot( dudz_lsq4paper.mean(), depth, '-*', color = 'k', label = 'DUDZ in the paper')
#
axdudz_studiedperiod.set_xlabel('du/dz')    
axdudz_studiedperiod.set_ylabel('depth')
axdudz_studiedperiod.set_title('From 01 February 2020 until cable breaks')
#
ax_dudz_compareaverages.set_xlabel('du/dz')    
ax_dudz_compareaverages.set_ylabel('depth')
ax_dudz_compareaverages.set_title('From 01 February 2020 until 30 October 2020')
ax_dudz_compareaverages.grid()
#
axdudz_studiedperiod.legend()
ax_dudz_compareaverages.legend()
ax_dudz_compareaverages.set_xlim([ -0.05, 1.5])
axdudz_studiedperiod.set_xlim([ -0.05, 1.5])
#
axdudz_studiedperiod.grid()
#
figdudz_studiedperiod.savefig( path + 'dudz_differentDX_andpaper.png', dpi = 300, bbox_inches = 'tight')
figud.savefig( path + 'ud_differentDX_and_depth_sansminiplot.png', dpi = 300, bbox_inches= 'tight')
figdudz_compareaverages.savefig( path + 'dudz_compareaverages.png', dpi = 300, bbox_inches = 'tight')