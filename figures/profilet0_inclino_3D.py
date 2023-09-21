#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:47:15 2020

@author: juanpelvis
"""
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
import read_rain
import aux_inclino2 as AUX
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib
from palettable.colorbrewer.sequential import YlGnBu_9 as CBCL
matplotlib.rc('text', usetex = True)
import boreholeclass as BC
import matplotlib.dates as mdates
import matplotlib.figure as mfigure
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from datetime import datetime
sys.path.append('/home/juanpelvis/Documents/python_code/auxpy')
import AUXplot as ap

fig14, figfull, label_pt = ap.set_figureconditions()
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,minisize, MKS = ap.give_plotsizes()
font, MARKERmu, STYLEmu, COLORr, thicc, skinny = ap.set_plotconditions()
plt.rc('font', size=font['size'])          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=minisize)    # legend fontsize
tabcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
tabcol = {i:tabcolorlist[i-1] for i in range(1,len(tabcolorlist))}
path = '/home/juanpelvis/Documents/Inclino/'
ninclinos={1:18,2:19,3:17,4:19,5:17}
length_last_inclino = {1:3.67,2:19.58,3:5.63,4:6.45,5:17.1}
# distance between devices, taken from the excel
# The last number is the distance between last UNCORRECTED point, and the length of the cable, from the excel
d_bt_capteurs = {1:[20.21,20.67,20.59,20.5,15.42,15.32,15.31,15.42,10.23,10.27,10.24,5.19,2.25,5.05,2.12,2.11,1.15, 12.085818591808],
                 2:[20.48,20.48,20.47,20.59,20.48,15.5,15.31,15.26,15.36,10.14,10.22,10.24,5.07,5.17,5.15,2.13,2.08,1.09,3.20376708810299],
                 3:[20.28,20.57,20.22,15.26,15.4,15.4,15.33,10.09,10.15,10.23,5.14,5.19,5.19,2.12,2.14,1.07,36.585414630233	],
                 4:[20.51,20.47,20.57,20.41,20.57,15.31,15.35,15.48,15.18,10.19,10.14,10.26,5.19,5.17,5.22,2.11,2.1,1.1,15.665190182897],
                 5:[20.08,20.11,20.52,15.34,15.51,15.47,23.86,9.43,10.22,10.22,5.06,5.19,5.17,2.09,2.17,1.12,3.87504719698302]}
mask = {}
"""depth_bed = {1: -261.6999999999998,
 2: -266.3000000000002,
 3: -266.3000000000002,
 4: -266.3000000000002,
 5: -244.5}"""
numericalbed = {1: 2117.0, 2: 2102.6, 3: 2102.6, 4: 2102.6, 5: 2099.8}
surf = {1: 2370.274, 2:2359.998, 3:2357.887, 4:2356.917, 5:2334.383}
depth_bed = {}
for i in range(1,6):
    depth_bed[i] = numericalbed[i]-surf[i]

for i in range(1,6):
    mask[i] = list(np.ones(ninclinos[i]))
"""
mask[5][4-1] = 0
mask[5][3-1] = 0
mask[4][8-1] = 0
mask[4][15-1] = 0
mask[4][16-1] = 0
mask[3][8-1] = 0
mask[2][19-1] = 0
mask[2][12-1] = 0
"""
for i in d_bt_capteurs:
    d_bt_capteurs[i].reverse()
    d_bt_capteurs[i].append(length_last_inclino[i])
vertical_d_bt_capteurs = {}
horizontal_d_bt_capteurs = {}
corrected_depth = {}
corrected_alongX = {}
for i in ninclinos:
    vertical_d_bt_capteurs[i] = []
    horizontal_d_bt_capteurs[i] = []
    corrected_depth[i] = []
    corrected_alongX[i] = []
#figshape,axshape = plt.subplots(figsize=(3.5,8))

""" 3D shapes """
start = datetime.strptime('2019-02-15','%Y-%m-%d')
end = datetime.strptime('2019-10-20','%Y-%m-%d')
boreholes = {}
tilt_0 = {}
az_0 = {}
xyz = {}
timestamp = -1
BHlist = [1,2,3,4,5]
for BH in BHlist:
    borehole = BC.borehole(path,BH)
    borehole.dx = 1
    borehole.dt = 1
    # load
    borehole.load_inclino()
    borehole.compute_tilt_az_alt(start,end,dt=borehole.dt)
    borehole.return_azimuth()
    borehole.return_tilt()
    #
    boreholes[BH] = borehole
    tilt_0[BH] = dict(borehole.tilt.iloc[timestamp])
    az_0[BH] = dict(borehole.azimuth.iloc[timestamp])
    #
    xyz[BH] = {borehole.get_noi()+1 : (0,0,0)}
    t = 0
    az = 0
    x = xyz[BH][borehole.get_noi()+1][0]
    y = xyz[BH][borehole.get_noi()+1][1]
    z = xyz[BH][borehole.get_noi()+1][2]
    for i in range(borehole.get_noi(),-1,-1):
        l = d_bt_capteurs[BH][i]
        #mean_tilt = np.deg2rad(0.5*(t + tilt_0[BH]['Tilt('+str(i)+')']))
        mean_tilt = np.deg2rad(t)
        #mean_azimuth = np.deg2rad(0.5*(az + az_0[BH]['azimuth('+str(i)+')'])) # we weight half at each point
        mean_azimuth = np.deg2rad(90 + az) # alternative
        if i > 0:
            t = tilt_0[BH]['Tilt('+str(i)+')']
            az = az_0[BH]['azimuth('+str(i)+')']
        dx = np.sin(mean_tilt)*l*np.cos(mean_azimuth)
        dy = np.sin(mean_tilt)*l*np.sin(mean_azimuth)
        dz = np.cos(mean_tilt)*l
        xyz[BH][i] = (x+dx,y+dy,z+dz)
        x = xyz[BH][i][0]
        y = xyz[BH][i][1]
        z = xyz[BH][i][2]
        
aspect_ratio = 240/35
xsize = 4
figXY, figXZ, figYZ, figsqrt = {},{},{},{}
altogether = 1
for BH in BHlist:
    for d in [figXY, figXZ, figYZ, figsqrt]:
        d[BH] = {'fig' : None, 'ax' : None}
    #
    if altogether:
        if BH == BHlist[0]:
            figshape,axshape = plt.subplots(figsize=(3.5,8))
            figshapeb,axshapeb = plt.subplots(figsize=(3.5,8))
            figshapec,axshapec = plt.subplots(figsize=(3.5,3.5))
            figshaped,axshaped = plt.subplots(figsize=(3.5,8))
        figXZ[BH]['fig'], figXZ[BH]['ax'] = figshape,axshape
        figYZ[BH]['fig'], figYZ[BH]['ax'] = figshapeb,axshapeb
        figXY[BH]['fig'], figXY[BH]['ax'] = figshapec,axshapec
        figsqrt[BH]['fig'], figsqrt[BH]['ax'] = figshaped,axshaped
    else:
        figshape,axshape = plt.subplots(figsize=(3.5,8))
        figshapeb,axshapeb = plt.subplots(figsize=(3.5,8))
        figshapec,axshapec = plt.subplots(figsize=(3.5,3.5))
        figXZ[BH]['fig'], figXZ[BH]['ax'] = figshape,axshape
        figYZ[BH]['fig'], figYZ[BH]['ax'] = figshapeb,axshapeb
        figXY[BH]['fig'], figXY[BH]['ax'] = figshapec,axshapec
        figsqrt[BH]['fig'], figsqrt[BH]['ax'] = figshaped,axshaped
#
for BH in BHlist:
    [X, Y, Z] = list(zip(*list(xyz[BH].values())))
    X,Y,Z = np.array(X), np.array(Y), -1*np.array(Z)
    figXZ[BH]['ax'].plot(X[:-1],Z[:-1],'-*', color = tabcol[BH], label = BH)
    figYZ[BH]['ax'].plot(Y[:-1],Z[:-1],'-*', color = tabcol[BH], label = BH)
    figXY[BH]['ax'].plot(X[:-1],Y[:-1],'-*', color = tabcol[BH], label = BH)
    figsqrt[BH]['ax'].plot(np.sqrt(np.power(X[:-1],2) + np.power(Y[:-1],2)),Z[:-1],'-*', color = tabcol[BH], label = 'BH'+str(BH))
    #
    figXZ[BH]['ax'].plot(X[-2:],Z[-2:],':', color = tabcol[BH])#, label = BH)
    figYZ[BH]['ax'].plot(Y[-2:],Z[-2:],':', color = tabcol[BH])#, label = BH)
    figXY[BH]['ax'].plot(X[-2:],Y[-2:],':', color = tabcol[BH])#, label = BH)
    figsqrt[BH]['ax'].plot(np.sqrt(np.power(X[-2:],2) + np.power(Y[-2:],2)),Z[-2:],':', color = tabcol[BH],)
    #
    if altogether and BH == BHlist[-1]:
        for ax in [figXZ[BH]['ax'], figYZ[BH]['ax'], figXY[BH]['ax'], figsqrt[BH]['ax']]:
            ax.plot([],[],'v',color = 'k', label = 'Max.\n drilled')
    figXZ[BH]['ax'].plot(X[-1],Z[-1],'v', color = 'k', zorder = 5)#tabcol[BH])#, label = BH)
    figYZ[BH]['ax'].plot(Y[-1],Z[-1],'v', color = 'k', zorder = 5)#"tabcol[BH])#, label = BH)
    figXY[BH]['ax'].plot(X[-1],Y[-1],'v', color = 'k', zorder = 5)#tabcol[BH])#, label = BH)
    figsqrt[BH]['ax'].plot(np.sqrt(np.power(X[-1],2) + np.power(Y[-1],2)),Z[-1],'v', color = 'k', zorder = 5)
    #
    figXZ[BH]['ax'].set_xlabel('Easting m')
    figYZ[BH]['ax'].set_xlabel('Northing m')
    figXY[BH]['ax'].set_xlabel('Easting m')
    figXZ[BH]['ax'].set_ylabel('Depth m')
    figYZ[BH]['ax'].set_ylabel('Depth m')
    figXY[BH]['ax'].set_ylabel('Northing m')
    figsqrt[BH]['ax'].set_ylabel('Depth m')
    figsqrt[BH]['ax'].set_xlabel('Distance to vertical m\n 2x horizontal exaggeration')
    #
    figXZ[BH]['ax'].set_yticks([0, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250])
    figYZ[BH]['ax'].set_yticks([0, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250])
    figsqrt[BH]['ax'].set_yticks([0, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250])
    #
    figXZ[BH]['ax'].set_ylim([-240, 0])
    figYZ[BH]['ax'].set_ylim([-240, 0])
    figsqrt[BH]['ax'].set_ylim([-240, 0])
    #
    for fig in [figXY, figXZ, figYZ, figsqrt]:
        fig[BH]['ax'].set_aspect(aspect = 0.5) 
    figXZ[BH]['ax'].legend(facecolor='white', framealpha=1, loc = 'upper left', bbox_to_anchor = (1.05, 1))
    figsqrt[BH]['ax'].legend(facecolor='white', framealpha=1, loc = 'upper left', bbox_to_anchor = (1.05, 1))
#
for I in [1, 8, 12, 14]:
    figsqrt[2]['ax'].annotate( text = 'BH2\#' + str(I), xy = ( + np.sqrt( np.power(xyz[2][I - 1][0], 2) + np.power(xyz[2][I - 1][1], 2)), -1 * xyz[2][I - 1][2]), xytext = (3 +  np.sqrt( np.power(xyz[2][I - 1][0], 2) + np.power(xyz[2][I - 1][1], 2)), 5 - xyz[2][I - 1][2]) , arrowprops={'arrowstyle':'->'},
                              bbox=dict(boxstyle="square,pad=0.3", fc="white", ec='k', lw=0.5))
    
#
figXZ[2]['fig'].savefig(path + 'initial_shapeXZ.png', dpi = 300, bbox_inches = 'tight')
figYZ[2]['fig'].savefig(path + 'initial_shapeYZ.png', dpi = 300, bbox_inches = 'tight')
figXY[2]['fig'].savefig(path + 'initial_shapeXY.png', dpi = 300, bbox_inches = 'tight')
figsqrt[2]['fig'].savefig(path + 'initial_shapeSQRT.png', dpi = 300, bbox_inches = 'tight')
