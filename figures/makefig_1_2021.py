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
    1:(	959425.067,	2117688.170, 'dodgerblue', 'Boreholes'), 
2 : (959421.807,	2117691.215, 'dodgerblue', 'Boreholes'),
3 :	(959265.217	,2117797.799, 'dodgerblue', 'Boreholes'),
4 :	(959257.969,	2117801.464, 'dodgerblue', 'Boreholes'),
#: (959375, 2117545, 'forestgreen','GPS1'), # after carefully looking at the map :o)
5 : (958625, 2118000, 'violet', 'Cavitometer'),
}
DX = {1: 20, 2: -60, 3: 40, 4: -100, 5: 25} # deltax for annotation
DY = {1: 0, 2: -20, 3: 10, 4: 20, 5: 5} # deltay for annotation
tabcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
tabcol = {i:tabcolorlist[i-1] for i in range(1,len(tabcolorlist))}
gps = pd.read_csv('/home/juanpelvis/Documents/Inclino/for_juanpedro/gps.xyz')
for i in range(6, 6 + len(gps)):
    XY_inclinos[i] = (gps['x'].iloc[i - 6] , gps['y'].iloc[i - 6], 'forestgreen', gps['station'].iloc[i - 6])
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
"""
src = rasterio.open(path+tiff_file_bed)
band1 = src.read(1)
# print('Band1 has shape', band1.shape)
height = band1.shape[0]
width = band1.shape[1]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(src.transform, rows, cols)
"""
#
xy_sensors_inLambertII = pd.read_csv(path+'XY_LamII.csv')
xmean, ymean = np.mean(xy_sensors_inLambertII['X']),np.mean(xy_sensors_inLambertII['Y'])
#fig,axx = plt.subplots(figsize = (5,5))
figb, axxb = plt.subplots(figsize = (5,5))
dxgps, dygps = 10, 10
for i in XY_inclinos:
    if i < 5:
        size = 50
    else:
        size = 0.6 * 50
    axxb.scatter(XY_inclinos[i][0], XY_inclinos[i][1], c = XY_inclinos[i][2], edgecolor='black', zorder=10, s = size)
    if i in [1, 4]:
        axxb.annotate('BH'+str(10+i),(XY_inclinos[i][0]+DX[i], XY_inclinos[i][1]+DY[i]), zorder = 10, weight='bold', fontsize = 10, color = 'gold',)
    elif i in [2, 3]:
        axxb.annotate('BH'+str(10+i),xy = (XY_inclinos[i][0], XY_inclinos[i][1]), xytext = (XY_inclinos[i][0] + 3*DX[i], XY_inclinos[i][1] +3 *DY[i]), zorder = 10, weight='bold', fontsize = 10, color = 'gold', arrowprops=dict(edgecolor='white', arrowstyle="->", connectionstyle="arc3,rad=.2"),)
    elif i > 7776:
        axxb.annotate(text = gps['station'].iloc[i - 7], xy = (gps['x'].iloc[i - 7] + dxgps, gps['y'].iloc[i - 7] + dygps), weight='bold', fontsize = 6, color = 'lightgreen')
for i in [4, 5]:
    axxb.scatter(0,0,c = XY_inclinos[i][2], edgecolor='black', zorder=10, label = XY_inclinos[i][3],)
statis = ['AR6D', 'AR3D', 'AR5D']
for st in statis:
    for i in range(gps.shape[0]):
        if gps.iloc[i]['station'] == st:
            print(' jejeje ')
            xarg = gps.iloc[i]['x']
            yarg = gps.iloc[i]['y']
            axxb.scatter( xarg, yarg, c = 'forestgreen', edgecolor='black', zorder=10, s = 50)
            axxb.annotate(st, xy = (xarg, yarg), xytext = (xarg - 250, yarg - 100), zorder = 10, weight='bold', fontsize = 10, color = 'gold', arrowprops=dict(edgecolor='white', arrowstyle="->", connectionstyle="arc3,rad=.2"),)
axxb.scatter(0,0,c = XY_inclinos[8][2], edgecolor='black', zorder=10, label = 'GPS')
#for BH in [1,2,3,4,5]:
#    shape = pd.read_csv(pathinc + 'initial_shape_BH'+str(BH)+'.csv')
#    axxb.plot(shape['X'] + XY_inclinos[BH][0], shape['Y'] + XY_inclinos[BH][1], color = tabcol[i], zorder = 9)
#axxb.plot(0,0, '-r', linewidth = 2, label = 'Surface elevation')
axxb.plot(0,0, '-k', linewidth = 2, label = 'Thickness m')
#
rp.show(dataset, ax = axxb, zorder=2)
#axxb.contour(xs, ys, band1, levels = np.arange(1500,3000,100), colors ='k', zorder = 5)
#axxb.contour(xmean + 200 + np.array(data['X']), ymean + np.array(data['Y']), data['Zbed_christian'], levels = np.arange(1500,3501,100), alpha = 1, colors='red', zorder=5)
#
downscale_factor = 1/2
#
#x_bed, y_bed, z_bed = downsize_tif(path + tiff_file_bed, downscale_factor)
#BedC = axxb.contour(x_bed, y_bed, z_bed[0], levels = np.arange(2000, 2401, 50) , zorder = 4, colors = 'black', linewidths = 1.5)
#
"""
for i in D:
    if D[i][2]%100:
        #axxb.plot(D[i][0], D[i][1], 'white', zorder=5, linewidth = 2)
        axxb.plot(D[i][0], D[i][1], 'k', zorder=3, linewidth = .5)
    else:
        #axxb.plot(D[i][0], D[i][1], 'white', zorder=5, linewidth = 3)
        axxb.plot(D[i][0], D[i][1], 'k', zorder=3, linewidth = 1.5)
"""
for ax in [axxb]:
    ax.set_xlim([958400, 959800])
    ax.set_ylim([2117200, 2118200])
    ax.set_aspect('equal', adjustable='box')

axxb.legend(loc='lower left',facecolor = 'white', framealpha = 1, fontsize=8)
axxb.minorticks_on()
#transform = Affine.from_gdal(*da.attrs['transform'])
#nx, ny = da.sizes['x'], da.sizes['y']
#x, y = np.meshgrid(np.arange(nx), np.arange(ny)) * transform
#img = io.imread(path+tiff_file)
#imarray = np.array(img)
axxb.xaxis.set_tick_params(rotation=30)
"""
Plot surface iso lines
"""
"""
dem = path+ 'DEM_Lamb2.tif'
src = rasterio.open(dem)
band1 = src.read(1)
band1[band1 > 99999] = 0
height = band1.shape[0]
width = band1.shape[1]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(src.transform, rows, cols)

XX= np.array(xs)
YY = np.array(ys)
#

Plot bed iso lines
srcb = rasterio.open(path + tiff_file_bed)
band1b = srcb.read(1)
band1b[band1b > 99999] = 0
height = band1b.shape[0]
width = band1b.shape[1]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xsb, ysb = rasterio.transform.xy(srcb.transform, rows, cols)
XXb= np.array(xsb)
YYb = np.array(ysb)
CB = axxb.contour(XXb, YYb, band1b, levels = np.arange(2100,2500,50), colors = 'k', zorder = 3, linewidths = 0.5,)
"""
downscale_factor = 1/64
deltax=250
manual_locations_bed = [(959000, 2117450), (958800, 2117700), (958850, 2117800), (959100, 2117850), (959400, 2117450), (959650, 2117350)]#(959550, 2117775)] 

thick = axxb.contour(xmean + deltax + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'], levels = np.arange(50,251,50), alpha = 1, zorder = 4, colors = 'k')
axxb.clabel(thick, thick.levels[::], inline=True, fontsize=8, manual=manual_locations_bed)
"""
x_surf, y_surf, z_surf = downsize_tif(path + 'DEM_Lamb2.tif', downscale_factor)
z_surf[z_surf > 99999] = 0
SurfC = axxb.contour(x_surf, y_surf, z_surf[0], levels = np.arange(2300,2461,50), colors = 'red', zorder = 3, linewidths = 1.5)
"""
#axxb.contour(XX, YY, band1, levels = np.arange(2300,2461,20), colors = 'red', zorder = 3, linewidths = 0.5,)
#axxb.clabel(CB, np.arange(2100,2500,50),  # label every second level
#          inline=True, fontsize=6)
#axxb.clabel(CS, np.arange(2300,2461,50),  # label every second level
#          inline=True, fontsize=6)

#axxb.contour(XX, YY, band1, levels = np.arange(2300,2501,100), colors = 'red', zorder = 3, linewidths = 1.5)
#axxb.contour(XX, YY, band1, levels = np.arange(2300,2501,100), colors = 'red', zorder = 6, linewidths = 2)

"""
manual_locations_bed = [(958800, 2117600), (958975, 2117550), (959050, 2117700), (959200, 2118000), (959450, 2117650), ]#(959550, 2117775)] 
axxb.clabel(BedC, BedC.levels[::2], inline=True, fontsize=8, manual=manual_locations_bed)
manual_locations_surf = [(958825, 2117850), (959150, 2117450), (959650, 2117300)]#, (959200, 2118000), (959450, 2117650), ]#(959550, 2117775)] 
axxb.clabel(SurfC, [2300, 2400], inline=True, fontsize=8, manual=manual_locations_surf)
#axxb.set_xticklabels(axxb.get_xticklabels(),fontdict={'size' : 8})
#axxb.set_yticklabels(axxb.get_yticklabels(),fontdict={'size' : 8})
"""
figb.savefig(pathinc+'fig1_withshapes_2021.png', dpi = 300, bbox_inches = 'tight')
"""
List of variables in data:
    H :                             ???? not thickness
    X :                             Lambert II etendu no
    Xa
    Y :                             Lambert II etendu no
    Ya
    Zbed_christian
    colormnt
    name
    position_select_final
    th_interp
    thickness :                     
    x
    xcrevasse_only
    y
    ycrevasse_only
    z
"""
terrcm = cm.get_cmap('terrain')
""" Project """ 
"""
axxb.contour(xmean + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'], levels = np.arange(0,250,50), alpha = 1, zorder=1)#cmap = 'terrain', zorder=1)
rp.show(dataset, ax = axxb, zorder=2)
deltax = 0
CS = axx.contour(xmean + deltax + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'],levels = [0,50, 100, 150, 200, 250, 300], colors = 'k', zorder=3, linewidths = 2)
axx.contour(xmean + deltax + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'],levels = [0,50, 100, 150, 200, 250, 300], colors = 'w', zorder=2.5, linewidths = 4)
axx.clabel(CS, inline=True, fontsize=10, levels = [])
#ax.imshow(img, origin='lower', extent = EX)
#axx.scatter(200+xmean,ymean,c = 'r', zorder=5)
for ax in [axx, axxb]:
    ax.set_xlim([958300, 960000])axxb.contour(xmean + np.array(data['X']), ymean + np.array(data['Y']), data['thickness'], levels = np.arange(0,250,50), alpha = 1, zorder = 4)

    ax.set_ylim([2116750, 2118500])
    ax.set_aspect('equal', adjustable='box')
fig.savefig(path+'kkk_plus200.png', dpi = 300)
figb.savefig(path+'ortophoto.png', dpi = 600)
"""
"""
ax.set_xlim([-1100,1000])
ax.set_ylim([-1000,1500])
"""
