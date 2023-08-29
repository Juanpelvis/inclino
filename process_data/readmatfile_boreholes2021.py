#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:09:08 2021

@author: juanpelvis
"""

import numpy as np
import mat4py as mp
from scipy.spatial import distance
import matplotlib.pyplot as plt

pathdown = '/home/juanpelvis/Downloads/'
path = '/home/juanpelvis/Documents/Inclino/'
file = 'VitesseSept2018_Arg_relax(1).mat'
data = mp.loadmat(pathdown+file)

tabcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
tabcol = {i:tabcolorlist[i-1] for i in range(1,len(tabcolorlist))}
tabcol['keller'] = tabcolorlist[-1]
"""
Ok, I have the 3D field...
Now the question in how to work with  it:Plot a 3D map or something, to see what the hell we have: cloud of points or smth

And then try to understand
The coordinates are the same as in the Vincent et al. paper (vertical compression)
"""
x = data['x']
y = data['y']
z = data['z']
""" If I understood correctly (comparing points with same X)

I have 20 vertical layers? or 10? or even 50..."""



BH_positions = {
    1:(	959425.067,	2117688.170,), 
    2 : (959421.807,	2117691.215,),
    3 :	(959265.217	,2117797.799,),
    4 :	(959257.969,	2117801.464),
}
"""
points = []
for i in range(len(x)):
    points.append((x[i][0],2e6 + y[i][0]))
for BH in BH_positions:
    dist = distance.cdist(points, [BH_positions[BH]]).min(axis=1)
    index = np.argmin(dist)
    print('Minimum distance is',min(dist))
    print('at the point', index)
    print('mesh X, Y',x[index], y[index])
    print('real X,Y', BH_positions[BH])
    print()
    
  Minimum distance is 19.156288497591113
  at the point 10223
  mesh X, Y [959410] [117700]
  real X,Y (959425.067, 2117688.17), Z = 2364,530, thickness = 196

  Minimum distance is 14.716707308475014
  at the point 10223
  mesh X, Y [959410] [117700]
  real X,Y (959421.807, 2117691.215), Z = 2364,207, thickness = 196

  Minimum distance is 28.207472237103016
  at the point 10138
  mesh X, Y [959270] [117770]
  real X,Y (959265.217, 2117797.799), Z = 2336.239, thickness = 179

  Minimum distance is 33.68572779395787
  at the point 10138
  mesh X, Y [959270] [117770]
  real X,Y (959257.969, 2117801.464), Z = 2334,745n thickness = 177
"""
indeces = {1:10223, 2:10223, 3:10138, 4:10138,}
points, speed, xyz = {}, {}, {}
for BH in BH_positions:
    points[BH], speed[BH], xyz[BH] = [],[],[]
    for i in range(len(x)):
        if (x[i] == x[indeces[BH]]) and (y[i] == y[indeces[BH]]):
            points[BH].append(i)
            speed[BH].append((data['vx'][i][0], data['vy'][i][0], data['vz'][i][0]))
            xyz[BH].append((data['x'][i][0], data['y'][i][0], data['z'][i][0]))

fig_udz, ax_udz = plt.subplots( figsize = (5, 10))

rmin = 75
BH = 2
points_2, xyz_2, speed_2 = {},{},{}
for BH in BH_positions:
    points_2[BH], xyz_2[BH], speed_2[BH] = [],[],[]
    for i in range(len(x)):
        if np.linalg.norm(np.subtract((data['x'][i][0],data['y'][i][0]),(data['x'][indeces[BH]][0],(data['y'][indeces[BH]][0])))) <= rmin:# and i not in points[BH]:
            points_2[BH].append(i)
            xyz_2[BH].append((data['x'][i][0], data['y'][i][0], data['z'][i][0]))
            speed_2[BH].append((data['vx'][i][0], data['vy'][i][0], data['vz'][i][0]))
BH = 2
[xx,yy,zz] = list(zip(*xyz_2[2]))
[u, v, w] = list(zip(*speed_2[2]))
#DUY = np.gradient(np.array([u, v, w]), yy)
#DUZ = np.gradient(np.array([u, v, w]), zz)
#
fig, axtw = plt.subplots(figsize=(6,14))
#axtw = ax.twiny()
vhor, zz, az, dudz = {},{},{},{}
for BH in BH_positions:
    vhor[BH],zz[BH],az[BH] = [],[],[]
    for i in range(len(speed[BH])):
        vhor[BH].append( np.linalg.norm([speed[BH][i][0], speed[BH][i][1]]))
        zz[BH].append(xyz[BH][i][2])
        az[BH].append(np.rad2deg(np.arctan2(speed[BH][i][1],speed[BH][i][0])))
    dudz[BH] = np.gradient(vhor[BH],zz[BH]) # central point, forward or backward at extremes
    #idx = np.argsort(zz)
    #zz.sort()
    #vhor= [vhor[i] for i in idx]
    #ax.plot(vhor[BH],zz[BH],'*-',label=BH, linewidth=4,markersize=8)
    axtw.loglog(dudz[BH],zz[BH],':*',label='dudz'+str(BH), linewidth=4,markersize=8)
#
#ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.))
#
DUDZ2 = mp.loadmat(path+'dudz_depth2.mat')
DUDZ3 = mp.loadmat(path+'dudz_depth3.mat')
DUDZ4 = mp.loadmat(path+'dudz_depth4.mat')
DUDZkeller = mp.loadmat(path+'dudz_depthkeller.mat')
DUDZkeller['depth'] = DUDZ2['depth'][:-1]
DUDZ = {2: DUDZ2, 3: DUDZ3, 4: DUDZ4, 'keller' : DUDZkeller}
height_zero = {1:2370.274, 2:2359.998, 3:2357.887, 4:2356.917, 5:2334.383, 'keller' : 2359.998}
for i in DUDZ:
    DUDZ[i]['depth'].append(0)
    DUDZ[i]['dudz'].append(0)
    axtw.loglog(DUDZ[i]['dudz'],np.add(height_zero[i], DUDZ[i]['depth']),'--',color=tabcol[i], linewidth=4,markersize=8,label='measured, BH'+str(i))
axtw.legend(loc = 'upper left',bbox_to_anchor=(1.05, 0.85))
#fig.savefig(path+'comparison_model_data_noudloglog.png',dpi=300,bbox_inches = 'tight')