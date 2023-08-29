#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:11:04 2023

@author: juanpelvis
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cmapp = 'inferno'
path = '/home/juanpelvis/Documents/Inclino/'
df = {}
fig, axs = plt.subplots( figsize = (10, 10), ncols = 3, nrows = 3)
for i in range(1, 10):
    ax = axs[(i-1)//3][(i-1)%3]
    file = path + 'L13L33_Aug22_Tilt' + str(i) + '.csv'
    df[i] = pd.read_csv( file, index_col = 'L33')
    CS = ax.contour( df[i], levels = [1,2,5,10,20],  vmin = 1, vmax = 25, colors = 'k')# cmap = cmapp)
    ax.clabel(CS, colors='k', fontsize=8)
    #
    if (i-1)%3 == 0:
        ax.set_yticks(np.arange(0,len(df[i].index), 5))
        ax.set_yticklabels( ["{:.3f}".format(j) for j in df[i].index[0::5] ])
        ax.set_ylabel('dw/dz')
    else:
        ax.set_yticklabels([])
    if (i-1)//3 == 2:
    #
        ax.set_xticks(np.arange(0,len(df[i].columns), 20))
        ax.set_xticklabels( ["{:.1f}".format( float(j)) for j in df[i].columns[0::20] ])
        ax.set_xlabel('du/dz')
    else:
        ax.set_xticklabels([])
    
    ax.set_title('BH2\#' + str(i))
    
#cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
#mesh = ax.pcolormesh( df[i], cmap = cmapp)
#mesh.set_clim(0,25)
#fig.colorbar(mesh, cax=cbar_ax)
#cbar_ax.set_ylabel('Relative error between modeled tilt curves using Keller and Blatter (2012)')
fig.savefig( path + 'keller_welldone_1to9.png', dpi = 150, bbox_inches = 'tight')
#
df2 = {}
fig2, axs2 = plt.subplots( figsize = (10, 10), ncols = 3, nrows = 3)
for i in range(1, 10):
    ax2 = axs2[(i-1)//3][(i-1)%3]
    file = path + 'L13L33_Aug22_Tilt' + str(9+i) + '.csv'
    df2[i] = pd.read_csv( file, index_col = 'L33')
    ax2.contour( df2[i], levels = 50,  vmin = 1, vmax = 25, cmap = cmapp)
    #
    if (i-1)%3 == 0:
        ax2.set_yticks(np.arange(0,len(df2[i].index), 5))
        ax2.set_yticklabels( ["{:.3f}".format(j) for j in df2[i].index[0::5] ])
        ax2.set_ylabel('dw/dz')
    else:
        ax2.set_yticklabels([])
    if (i-1)//3 == 2:
    #
        ax2.set_xticks(np.arange(0,len(df2[i].columns), 20))
        ax2.set_xticklabels( ["{:.1f}".format( float(j)) for j in df2[i].columns[0::20] ])
        ax2.set_xlabel('du/dz')
    else:
        ax2.set_xticklabels([])
    
    ax2.set_title('BH2#' + str(9 + i))
cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.05, 0.7])
mesh2 = ax2.pcolormesh( df2[i], cmap = cmapp)
mesh2.set_clim(0,25)
fig2.colorbar(mesh2, cax=cbar_ax2)
cbar_ax2.set_ylabel('Relative error between modeled tilt curves using Keller and Blatter (2012)')
fig2.savefig( path + 'keller_welldone_10to18.png', dpi = 150, bbox_inches = 'tight')