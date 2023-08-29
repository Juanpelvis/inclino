#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:45:41 2022

@author: juanpelvis

Script for making the map of Argentière
"""
import os
os.environ["PROJ_LIB"] = '/home/juanpelvis/anaconda3/lib/python3.7/site-packages/rasterio/proj_data'

import numpy as np
import mat4py as mp
import pandas as pd
#from scipy.spatial import distance
import matplotlib.pyplot as plt
#import multipagetiff as mtif
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000
#from skimage import io
import rasterio
#import pyproj
import rasterio.plot as rp
from rasterio.enums import Resampling
from matplotlib import cm
import shapefile as shp  # Requires the pyshp package

XY_inclinos = {
1: (959419.686,	2117521.867, 'dodgerblue', 'Boreholes'),
2: (959323.003,	2117572.501, 'dodgerblue','Boreholes'),
3: (959305.421,	2117582.196, 'dodgerblue', 'Boreholes'),	
4: (959294.729,	2117586.937, 'dodgerblue','Boreholes'),
5: (959165.809,	2117701.152, 'dodgerblue','Boreholes'),
6 : (958625, 2118000, 'violet', 'Cavitometer'),
}
DX = {1: 20, 2: -60, 3: 40, 4: -100, 5: 25} # deltax for annotation
DY = {1: 0, 2: -70, 3: 10, 4: 20, 5: 5} # deltay for annotation
tabcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
tabcol = {i:tabcolorlist[i-1] for i in range(1,len(tabcolorlist))}
gps = pd.read_csv('/home/juanpelvis/Documents/Inclino/for_juanpedro/gps.xyz')
for i in range(7, 7 + len(gps)):
    XY_inclinos[i] = (gps['x'].iloc[i - 7] , gps['y'].iloc[i - 7], 'forestgreen', gps['station'].iloc[i - 7])
sf = shp.Reader("/home/juanpelvis/Documents/Inclino/RESOLVE_data/contour_bed/contour_bed.shp")

for shape in []:#sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
    
def read_shapefile(sf):
    #fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]#fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]#converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)#assigning the coordinates
    df = df.assign(coords=shps)
    #
    D = {}
    for i in range(len(df)):
        x,y = list(zip(*df.iloc[i]['coords']))
        D[i] = [x, y, df.iloc[i]['ELEV']]
    return D

def downsize_tif(tif, downscale_factor):
    with rasterio.open(tif) as dataset:
        # resample data to target shape
        bed_down = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * downscale_factor),
                int(dataset.width * downscale_factor)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / bed_down.shape[-1]),
            (dataset.height / bed_down.shape[-2])
        )
        height = bed_down.shape[1]
        width = bed_down.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows, cols)
    return xs,ys,bed_down

D = read_shapefile(sf)
#
# import xarray as xr
# from affine import Affine
# import cartopy.crs as ccrs
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.ticker as mticker
#
pathinc = '/home/juanpelvis/Documents/Inclino/'
path = '/home/juanpelvis/Documents/Inclino/RESOLVE_data/'
file = 'RESOLVE_spatial_data.mat'
data = mp.loadmat(path+file)
tiff_file= 'argentiere_profil4_ebee_130919_LambertII_10cm.tif'
tiff_file_bed = 'bed_arg_serac_mnt15_radar.tif'
#DEM = plt.imread(path+tiff_file)
#geo_tiff = GeoTiff(path+tiff_file)
dataset = rasterio.open(path + tiff_file) 
#
xy_sensors_inLambertII = pd.read_csv(path+'XY_LamII.csv')
xmean, ymean = np.mean(xy_sensors_inLambertII['X']),np.mean(xy_sensors_inLambertII['Y'])
#fig,axx = plt.subplots(figsize = (5,5))
figb, axxb = plt.subplots(figsize = (5,5))
dxgps, dygps = 10, 10
for i in XY_inclinos:
    if i < 7:
        size = 50
    else:
        size = 0.6 * 50
    axxb.scatter(XY_inclinos[i][0], XY_inclinos[i][1], c = XY_inclinos[i][2], edgecolor='black', zorder=10, s = size)
    if i in [1, 4, 5]:
        axxb.annotate('BH'+str(i),(XY_inclinos[i][0]+DX[i], XY_inclinos[i][1]+DY[i]), zorder = 10, weight='bold', fontsize = 10, color = 'gold',)
    elif i in [2, 3]:
        axxb.annotate('BH'+str(i),xy = (XY_inclinos[i][0], XY_inclinos[i][1]), xytext = (XY_inclinos[i][0] + 3*DX[i], XY_inclinos[i][1] +3 *DY[i]), zorder = 10, weight='bold', fontsize = 10, color = 'gold', arrowprops=dict(edgecolor='white', arrowstyle="->", connectionstyle="arc3,rad=.2"),)
    elif i > 7776:
        axxb.annotate(text = gps['station'].iloc[i - 7], xy = (gps['x'].iloc[i - 7] + dxgps, gps['y'].iloc[i - 7] + dygps), weight='bold', fontsize = 6, color = 'lightgreen')
for i in [5, 6]:
    axxb.scatter(0,0,c = XY_inclinos[i][2], edgecolor='black', zorder=10, label = XY_inclinos[i][3],)
xarg1 = 959319.312882318
yarg1 = 2117579.60966177
axxb.annotate('ARG1',xy = (xarg1, yarg1), xytext = (xarg1 - 250, yarg1 - 100), zorder = 10, weight='bold', fontsize = 10, color = 'gold', arrowprops=dict(edgecolor='white', arrowstyle="->", connectionstyle="arc3,rad=.2"),)
axxb.scatter(0,0,c = XY_inclinos[8][2], edgecolor='black', zorder=10, label = 'GPS')
#for BH in [1,2,3,4,5]:
#    shape = pd.read_csv(pathinc + 'initial_shape_BH'+str(BH)+'.csv')
#    axxb.plot(shape['X'] + XY_inclinos[BH][0], shape['Y'] + XY_inclinos[BH][1], color = tabcol[i], zorder = 9)
#axxb.plot(0,0, '-r', linewidth = 2, label = 'Surface elevation')
axxb.plot(0,0, '-k', linewidth = 2, label = 'Thickness m')
#
rp.show(dataset, ax = axxb, zorder=2)

downscale_factor = 1/2

for ax in [axxb]:
    ax.set_xlim([958400, 959800])
    ax.set_ylim([2117200, 2118200])
    ax.set_aspect('equal', adjustable='box')

axxb.legend(loc='lower left',facecolor = 'white', framealpha = 1, fontsize=8)
axxb.minorticks_on()
axxb.xaxis.set_tick_params(rotation=30)

downscale_factor = 1/64
deltax=250
manual_locations_bed = [(959000, 2117450), (958800, 2117700), (958850, 2117800), (959100, 2117850), (959400, 2117450), (959650, 2117350)]#(959550, 2117775)] 

thick = axxb.contour(xmean + deltax + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'], levels = np.arange(50,251,50), alpha = 1, zorder = 4, colors = 'k')
axxb.clabel(thick, thick.levels[::], inline=True, fontsize=8, manual=manual_locations_bed)
figb.savefig(pathinc+'fig1_withshapes.png', dpi = 300, bbox_inches = 'tight')
terrcm = cm.get_cmap('terrain')