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
# In[]:
# Similar but step by step, can ignore
figsave, axsave = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]) )
al = 1
axsave.plot(udsurf['m/a'], color = colors['surface'], alpha = al, label = '$u_s$', marker = '')
axsave.legend(loc = 'upper left')
axsave.grid( which = 'major')
axsave.legend(loc = 'upper left')
axsave.set_ylim([10, 75])
axsave.set_ylabel('m/a')
axsave.set_xlabel('2020')
axsave.set_xlim([ start, end])
myFmt = mdates.DateFormatter('%b-%d')#-%d')
axsave.xaxis.set_tick_params(rotation=30, size=8)
axsave.xaxis.set_major_formatter(myFmt)
figsave.savefig( '/home/juanpelvis/Pictures/soutenance/surfvelo.png', dpi = 300, bbox_inches = 'tight')
## and def
figsave, axsave = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]) )
al = 1
axsave.plot(udsurf['m/a'], color = colors['surface'], alpha = al, label = '$u_s$', marker = '')
axsave.plot(deformation_velocity, color = colors['deformation'], alpha = al, label = '$u_d$')
axsave.legend(loc = 'upper left')
axsave.grid( which = 'major')
axsave.set_ylim([10, 75])
axsave.set_ylabel('m/a')
axsave.set_xlabel('2020')
axsave.set_xlim([ start, end])
myFmt = mdates.DateFormatter('%b-%d')#-%d')
axsave.xaxis.set_tick_params(rotation=30, size=8)
axsave.xaxis.set_major_formatter(myFmt)
figsave.savefig( '/home/juanpelvis/Pictures/soutenance/surfdefvelo.png', dpi = 300, bbox_inches = 'tight')
## and def and basal
figsave, axsave = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]) )
al = 1
axsave.plot(udsurf['m/a'], color = colors['surface'], alpha = al, label = '$u_s$', marker = '')
axsave.plot(deformation_velocity, color = colors['deformation'], alpha = al, label = '$u_d$')
axsave.plot(basal_unfiltered, color = colors['basal'], alpha = al, label = '$u_b$')
axsave.legend(loc = 'upper left')
axsave.grid( which = 'major')
axsave.set_ylim([10, 75])
axsave.set_ylabel('m/a')
axsave.set_xlabel('2020')
axsave.set_xlim([ start, end])
myFmt = mdates.DateFormatter('%b-%d')#-%d')
axsave.xaxis.set_tick_params(rotation=30, size=8)
axsave.xaxis.set_major_formatter(myFmt)
figsave.savefig( '/home/juanpelvis/Pictures/soutenance/surfdefbasalvelo.png', dpi = 300, bbox_inches = 'tight')
## all
figsave, axsave = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]) )
al = 1
#
def plot_glissante( velocity, color, axis, WINDOW, lab):
    axis.plot( velocity, color = color, label = lab, lw = 2)
    if WINDOW == '':
        axis.plot( [velocity.mean() for i in velocity], color = color, alpha = .50)
    else: 
        axis.plot( velocity.rolling( center = True, window = WINDOW).mean(), ls = '--', lw = 2, color = color, alpha=0.5)
#
W = 42
plot_glissante( udsurf['m/a'], colors['surface'], axsave, W, '$u_s$')
plot_glissante( deformation_velocity, colors['deformation'], axsave, W, '$u_d$')
plot_glissante( basal_unfiltered, colors['basal'], axsave, W, '$u_b$')
plot_glissante( cavitometer, colors['cavitometer'], axsave, W, '$u_{cav}$')

#axsave.plot(udsurf['m/a'], color = colors['surface'], alpha = al, label = '$u_s$', marker = '')
#axsave.plot(deformation_velocity, color = colors['deformation'], alpha = al, label = '$u_d$')
#axsave.plot(basal_unfiltered, color = colors['basal'], alpha = al, label = '$u_b$')
#axsave.plot( cavitometer, color = colors['cavitometer'], alpha = al, label = '$u_{cav}$')
axsave.grid( which = 'major')
axsave.legend(loc = 'upper left')
axsave.set_ylim([10, 75])
axsave.set_ylabel('m/a')
axsave.set_xlabel('2020')
axsave.set_xlim([ start, end])
myFmt = mdates.DateFormatter('%b-%d')#-%d')
axsave.xaxis.set_tick_params(rotation=30, size=8)
axsave.xaxis.set_major_formatter(myFmt)
figsave.savefig( '/home/juanpelvis/Pictures/soutenance/allvelos.png', dpi = 300, bbox_inches = 'tight')
##
# In[]:
""" From here down below, these are just different ideas to compare velocities.
You can ignore this for the time being. """
# Comparar moyennes glissantes diviendo por la media
figglis, (axglis, alphaplot) = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]), nrows = 2 )
axglis.set_ylabel('$\Delta u_i$ = speed over its mean')
alphaplot.set_ylabel('$\Delta u_s / \Delta u_d$')
alphaplot.set_xlabel('2020')
#
W = 30
def plot_vit_sansmoyen( velocity, WINDOW, start,end,axis, color, label):
    start = datetime.date( start )
    end = datetime.date( end )
    data_to_plot = velocity.rolling( center = True, window = WINDOW).mean()
    axis.plot( data_to_plot.loc[start:end] / data_to_plot.loc[ start : end].mean(), color = color, label = label)
    return data_to_plot, data_to_plot.loc[start:end] / data_to_plot.loc[ start : end].mean()
    #axis.plot( vit.loc[start:end] - vit.loc[ start : end].mean(), color = color)
udsurf_gliss, delta_udsurf_gliss = plot_vit_sansmoyen(udsurf['m/a'], W, start, end, axglis, color = colors['surface'], label = '$\Delta u_s$')
udef_gliss,   delta_udef_gliss   = plot_vit_sansmoyen(deformation_velocity, W, start, end, axglis, color = colors['deformation'], label = '$\Delta u_d$')
ubasal_gliss, delta_ubasal_gliss = plot_vit_sansmoyen(basal_unfiltered, W, start, end, axglis, color = colors['basal'], label = '$\Delta u_b$')
axglis.legend()
#
print('Alpha is Delta ud - Delta ub')
#alpha_to_plot = ( delta_udef_gliss - delta_ubasal_gliss) /( delta_udsurf_gliss - delta_ubasal_gliss )
alpha_to_plot = delta_udsurf_gliss / delta_udef_gliss
# alphaplot.set_ylim([0, 2])
beta_to_plot = delta_udsurf_gliss / delta_ubasal_gliss
#alphaplot.set_ylim([-1, 1])
alphaplot.plot( alpha_to_plot, color = colors['deformation'])
#alphaplot.plot( beta_to_plot, color = colors['basal'])
figpercent, axpercent = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]), nrows = 1 )
axpercent.plot( udef_gliss / udsurf_gliss, color = colors['deformation'], label = '$u_d/u_s$')
axpercent.plot( ubasal_gliss / udsurf_gliss, color = colors['basal'], label = '$u_b/u_s$')
axpercent.legend()
axpercent.set_ylabel('$u_i/u_s$')
figpercent.savefig( path + 'ratio_of_speeds.png', dpi = 300, bbox_inches = 'tight')
figglis.savefig( path + 'DELTAsp.png', dpi = 300, bbox_inches = 'tight')
# In[]:
# Comparar moyennes glissantes restando a la media
for W in np.arange(7, 49, 7):
    figglis, (axglis) = plt.subplots( figsize = (0.75*figfull[0], 0.35*figfull[1]), nrows = 1 )
    axglis.set_ylabel('$\Delta u_i = u_i - mean({u}_i)$~m/a')
    alphaplot.set_ylabel('$\Delta u_s / \Delta u_d$')
    axglis.set_xlabel('2020')
    #
    def plot_vit_sansmoyen( velocity, WINDOW, start,end,axis, color, label):
        start = start#datetime.date( start )
        end = end#datetime.date( end )
        data_to_plot = velocity.rolling( center = True, window = WINDOW).mean()
        axis.plot( data_to_plot.loc[start:end:WINDOW] - data_to_plot.loc[ start : end : WINDOW].mean(), color = color, label = label)
        #axis.plot( data_to_plot.loc[start:end] - data_to_plot.loc[ start : end : WINDOW].mean(), color = color, label = label, alpha = 0.3)
        return data_to_plot, data_to_plot.loc[start:end] - data_to_plot.loc[ start : end].mean()
        #axis.plot( vit.loc[start:end] - vit.loc[ start : end].mean(), color = color)
    udsurf_gliss, delta_udsurf_gliss = plot_vit_sansmoyen(udsurf['m/a'], W, start, end, axglis, color = colors['surface'], label = '$\Delta u_s$')
    udef_gliss,   delta_udef_gliss   = plot_vit_sansmoyen(deformation_velocity, W, start, end, axglis, color = colors['deformation'], label = '$\Delta u_d$')
    #ubasal_gliss, delta_ubasal_gliss = plot_vit_sansmoyen(basal_unfiltered, W, start, end, axglis, color = colors['basal'], label = '$\Delta u_b$')#
    print('Alpha is Delta ud - Delta ub')
    #alpha_to_plot = ( delta_udef_gliss - delta_ubasal_gliss) /( delta_udsurf_gliss - delta_ubasal_gliss )
    alpha_to_plot = delta_udsurf_gliss / delta_udef_gliss
    alphaplot.set_ylim([-2, 2])
    #beta_to_plot = delta_udsurf_gliss / delta_ubasal_gliss
    #alphaplot.set_ylim([-1, 1])
    #alphaplot.plot( alpha_to_plot, color = colors['deformation'])
    from sklearn.linear_model import LinearRegression
    arrays = pd.concat([delta_udef_gliss, delta_udsurf_gliss], axis = 1).dropna()
    y = np.array( arrays['m/a'] ) # surface velocity
    x = np.array( arrays[0]).reshape((-1, 1)) # deformation velocity
    model = LinearRegression().fit( x, y )
    r_sq = model.score(x, y)
    print('R2, coefs')
    print(r_sq)
    print(model.coef_)
    #axglis.plot( delta_udef_gliss * model.coef_ + model.intercept_, 'k--', label = str(model.coef_[0])[:4] + '$u_d$ ' + str(model.intercept_)[:4] )
    #alphaplot.plot( beta_to_plot, color = colors['basal'])
    plot_vit_sansmoyen( basal_unfiltered, W, start, end, axglis, color = colors['basal'], label = '$\Delta u_b$')
    axglis.legend()
    axglis.grid()
    axglis.set_title('R2 on the regression = ' + str(r_sq)[:4] + ', rolling mean every ' + str(W) + ' days')
    myFmt = mdates.DateFormatter('%b-%d')#-%d')
    axglis.xaxis.set_tick_params(rotation=30, size=8)
    axglis.xaxis.set_major_formatter(myFmt)
    figglis.savefig( path + 'deltaspeeds_with_regressionon_udef_' + str(W) + 'days.png', dpi = 300, bbox_inches = 'tight')
# In[]:
# Comparar moyennes glissantes restando a la media y haciendo un modelo de regresion multilineal
figglis, (axglis, alphaplot) = plt.subplots( figsize = (1.5*figfull[0], 0.7*figfull[1]), nrows = 2 )
axglis.set_ylabel('$\Delta u_i$ = speed MINUS its mean')
alphaplot.set_ylabel('$\Delta u_s / \Delta u_d$')
alphaplot.set_xlabel('2020')
#
W = 30
def plot_vit_sansmoyen( velocity, WINDOW, start,end,axis, color, label):
    start = datetime.date( start )
    end = datetime.date( end )
    data_to_plot = velocity.rolling( center = True, window = WINDOW).mean()
    axis.plot( data_to_plot.loc[start:end] - data_to_plot.loc[ start : end].mean(), color = color, label = label)
    return data_to_plot, data_to_plot.loc[start:end] - data_to_plot.loc[ start : end].mean()
    #axis.plot( vit.loc[start:end] - vit.loc[ start : end].mean(), color = color)
udsurf_gliss, delta_udsurf_gliss = plot_vit_sansmoyen(udsurf['m/a'], W, start, end, axglis, color = colors['surface'], label = '$\Delta u_s$')
udef_gliss,   delta_udef_gliss   = plot_vit_sansmoyen(deformation_velocity, W, start, end, axglis, color = colors['deformation'], label = '$\Delta u_d$')
ubasal_gliss, delta_ubasal_gliss = plot_vit_sansmoyen(basal_unfiltered, W, start, end, axglis, color = colors['basal'], label = '$\Delta u_b$')
axglis.legend()
#
from sklearn.linear_model import LinearRegression
y = np.array( delta_udsurf_gliss.dropna())
x = np.array( pd.concat([delta_udef_gliss, delta_ubasal_gliss], axis = 1).dropna())
model = LinearRegression().fit( x, y )
r_sq = model.score(x, y)
print('R2, coefs')
print(r_sq)
print(model.coef_)
print('the model does not work')
#alphaplot.plot( beta_to_plot, color = colors['basal'])
# In[]:
#
ud_continuous = pd.read_csv(path + 'NEW_UD_DX' + DAYS + 'daysBH2_paperperiod_1perday.csv', index_col='number')
#
ud_supercontinuous = pd.read_csv(path + 'NEW_UD_DX1daysBH2_paperperiod_48perday4Anuar.csv', index_col='number')
#
ud_supercontinuous.columns = pd.to_datetime( ud_supercontinuous.columns , format = '%Y-%m-%d')
ud_continuous.columns = pd.to_datetime(ud_continuous.columns).date
import seaborn as sns
data_for_sns = pd.DataFrame()
data_for_sns_p = pd.DataFrame()
#

#plt.subplots_adjust(wspace=0.3, hspace=10)
frequencies_list = ['12W','10W','8W', '7W', '6W', '5W', '4W', '3W', '2W', 'W']
color_frequencies = {'12W' : 'k', '10W' : 'k', '8W' : 'k', '7W' : 'red','6W' : 'red', '5W' : 'k','4W' : 'blue', '3W' : 'yellow', '2W' : 'grey' ,'W' : 'deepskyblue'}
profile_R_depth, axRdepth = plt.subplots( figsize = (3, 5))
twin = axRdepth.twiny()
extra = ''
print('Consider using seaborn regplot with robust = True')
for frequency_resampling in frequencies_list:
    fig_variance_layer, ax_variance_layer = plt.subplots( figsize = (12, 8), ncols = 4, nrows = 5)
    fig_variance_layer.subplots_adjust( hspace = 0.25)
    dic_regs_depth_layer = {}
    # not efficient at all, too tired to make it good
    resampled_dudz = borehole.dudz_lsq.loc[start:end].resample(frequency_resampling).mean()
    resampled_usurf = udsurf.loc[start:end]['m/a'].resample(frequency_resampling).mean()
    for i, IDX in enumerate(borehole.dudz_lsq.columns):
        reg1_aux = stats.linregress(resampled_dudz[IDX], resampled_usurf)
        spear = pd.DataFrame( data = {'dudz' : resampled_dudz[IDX], 'usurf' : resampled_usurf}).corr(method="spearman")
        df_regs = pd.DataFrame( data = { frequency_resampling : { 'start' : resampled_dudz.index[0], 'end' : resampled_dudz.index[0], 'm' : reg1_aux.slope, 'n' : reg1_aux.intercept, 'R2' : reg1_aux.rvalue**2, 'p' : reg1_aux.pvalue, 'depth' : borehole.depth['depth'].iloc[i]}})
        #                               #'Second half' : { 'start' : date[lengroup1], 'end' : date[-1], 'm' : reg2_aux.slope, 'n' : reg2_aux.intercept, 'R2' : reg2_aux.rvalue**2, 'p' : reg2_aux.pvalue}})
        dic_regs_depth_layer[ i ] = df_regs
        #
        axi = ax_variance_layer[i//4][i%4]
        #
        axi.plot( resampled_dudz[IDX], resampled_usurf, 'k*')
        X = [min(resampled_dudz[IDX]), max(resampled_dudz[IDX])]
        axi.plot(X,np.multiply( df_regs[frequency_resampling].loc['m'], X) + df_regs[frequency_resampling].loc['n'], 'k:', label = r'$R^2 = ' + "{:.2f}".format(df_regs[frequency_resampling].loc['R2']) + '$\n $p =' +  "{:.4f}".format(df_regs[frequency_resampling].loc['p']) + '$')#\n spearman=' + "{:.2f}".format(spear['dudz']['usurf']))
        #
        axi.legend()
        axi.set_ylabel(r'$u_s$~m/a')
        axi.set_xlabel(r'$du/dz$ at $z = ' + "{:.0f}".format(borehole.depth['depth'].iloc[i]) +'$~m')
    fig_variance_layer.savefig( path + 'dudz_vs_surfacevelo_perlayer_' + frequency_resampling + str(extra) +'.png', dpi = 300, bbox_inches = 'tight')
    #
    axRdepth.patch.set_visible(False)
    axRdepth.set_zorder(twin.get_zorder()+1)
    Xforplot = [dic_regs_depth_layer[i][frequency_resampling]['R2'] for i in range(borehole.get_noi())]
    pforplot = [dic_regs_depth_layer[i][frequency_resampling]['p'] for i in range(borehole.get_noi())]
    Yforplot =  borehole.depth['depth']
    axRdepth.plot( Xforplot, Yforplot, '-', color = color_frequencies[frequency_resampling],markersize = 3, label = 'Averages of ' + frequency_resampling)
    #
    data_to_add = pd.DataFrame( data = Xforplot, index = Yforplot, columns=[frequency_resampling])
    data_to_add_p = pd.DataFrame( data = pforplot, index = Yforplot, columns=[frequency_resampling])
    #
    plimit = 0.1
    #
    for i in range(borehole.get_noi()):
        if dic_regs_depth_layer[i][frequency_resampling]['p'] < plimit:
            axRdepth.scatter( x = dic_regs_depth_layer[i][frequency_resampling]['R2'] , y =  borehole.depth['depth'].iloc[i], c = color_frequencies[frequency_resampling], zorder = 3)
        elif dic_regs_depth_layer[i][frequency_resampling]['p'] < 2*plimit:
            axRdepth.scatter( x = dic_regs_depth_layer[i][frequency_resampling]['R2'] , y =  borehole.depth['depth'].iloc[i], c = color_frequencies[frequency_resampling], marker = '^', zorder = 3)
            #data_to_add.loc[borehole.depth['depth'].iloc[i]] = np.nan
        elif dic_regs_depth_layer[i][frequency_resampling]['p'] > 2*plimit:
            axRdepth.scatter( x = dic_regs_depth_layer[i][frequency_resampling]['R2'] , y =  borehole.depth['depth'].iloc[i], c = color_frequencies[frequency_resampling], marker = 'x', zorder = 3)
            #data_to_add.loc[borehole.depth['depth'].iloc[i]] = np.nan
    data_for_sns = pd.concat( [data_for_sns, data_to_add], axis = 1,)
    data_for_sns_p = pd.concat( [data_for_sns_p, data_to_add_p], axis = 1,)
data_for_sns.to_csv( path + 'data_regression_perdepth_R2.csv')
data_for_sns_p.to_csv( path + 'data_regression_perdepth_p.csv')
axRdepth.grid()
axRdepth.set_ylabel('depth')
axRdepth.set_xlabel('$R^2$ of $du/dz$ vs $u_s$')
twin.set_xlabel('du/dz')
twin.plot( borehole.dudz_lsq.mean(), borehole.depth['depth'], '-*', color = 'orange', )
#
profile_R_depth.savefig( path + 'Rdudz_vs_depth' + frequency_resampling + '.png', dpi = 300)
#
### Heatmap
figheat, heat = plt.subplots( figsize = (10, 6)) 
sns.heatmap(data = data_for_sns, vmin=0, vmax=1, ax=heat, cbar_kws = {'label' : '$R^2$',}, cmap = 'BuPu') # , cmap=sns.cubehelix_palette(as_cmap=True))
heat.set_xlabel('Time window size (W = week)')
heat.set_ylim(heat.get_ylim()[::-1])
#figheat = heat.get_figure()
# Loop over data dimensions and create text annotations.
for i in range(len(borehole.depth['depth'])):
    for j in range(len(frequencies_list)):
        text = heat.text(0.5+j, 0.5+i, "$p = " + "{:.2f}".format(data_for_sns_p.loc[borehole.depth['depth'].iloc[i]][frequencies_list[j]]) + '$',
                        ha="center", va="center",color="w", fontweight = 'bold')
figheat.savefig( path + 'heat_R2vsdepth.png', dpi = 300, bbox_inches = 'tight')

def detrend( u):
    return (u - u.mean()) / u.std()
fig_detrend, ax_detrend = plt.subplots( figsize = (5, 3), )
ax_detrend.plot( detrend(udsurf), color = colors['surface'], label = '$u_s$',alpha = 0.7)
ax_detrend.plot( detrend( ud_continuous.loc[17]), color = colors['deformation'], label = '$u_d$', alpha = 0.7)
ax_detrend.plot( detrend (basal_unfiltered), color = colors['basal'], label = '$u_b$', alpha = 0.7)
ax_detrend.set_ylim( [-3, 3])
ax_detrend.legend()
ax_detrend.xaxis.set_tick_params(rotation=30, size=8)
ax_detrend.xaxis.set_major_formatter(myFmt)
ax_detrend.set_ylabel(' (Speed - mean ) / std')
fig_detrend.savefig(path + 'comparison_speeds_detrended_' + DAYS + 'days.png')
    
# In[]:
# Let's interpolate the deformation velocity with a polynomial
Ypoly = ud_continuous.iloc[-1][14:].values
Ypoly_days = ud_continuous.iloc[-1][14:].index
window_avg = 40
Ypoly_avg = pd.DataFrame( Ypoly ).rolling( window = window_avg, center = True ).mean().iloc[window_avg//2: -window_avg//2][0]
figpoly, axpoly = plt.subplots( figsize = (5, 3))
figpoly_basal, axpoly_basal = plt.subplots(figsize = (5, 3))
#
Xpoly_avg =  np.arange(ud_continuous.shape[1]-14 - window_avg)
Xpoly =  np.arange(ud_continuous.shape[1]-14)
Xpolybasal = np.arange(14, len( udsurf))
Xpolybasal_dates = udsurf.iloc[14:].index
for n in np.arange(8,17,80):
    # z = np.polyfit( Xpoly_avg, Ypoly_avg, n)
    z = np.polyfit( Xpoly, Ypoly, n)
    f = np.poly1d(z)
    axpoly.plot( 0*window_avg//2 + Xpoly, f(Xpoly), label ='Poly. n =' + str(n))
    axpoly_basal.plot(Xpolybasal_dates, udsurf.loc[ Xpolybasal_dates ]['m/a'] - f(Xpolybasal), label ='Surf. - Def. (polynome n =' + str(n) + ')')

axpoly.plot(Ypoly, 'k--', label = 'Daily def. velocity')
axpoly.plot( Ypoly_avg , '--', color = 'red', label = str(window_avg) + ' days average')
# Surface velocity
axpoly.plot( np.arange(len(udsurf.loc[Xpolybasal_dates])), udsurf.loc[Xpolybasal_dates]['m/a'], 'b--', label = 'Daily surf. velocity')
#axpoly_basal.plot( Xpolybasal_dates, -32 + udsurf.loc[Xpolybasal_dates]['m/a'], 'k--', label = 'Surface velocity')
axpoly_basal.plot( Xpolybasal_dates, basal_unfiltered.loc[ Xpolybasal_dates], 'k--', label = 'Basal velocity')
R2_basalwrt_defpoly = np.power( np.corrcoef(basal_unfiltered.loc[ Xpolybasal_dates], udsurf.loc[ Xpolybasal_dates ]['m/a'] - f(Xpolybasal))[0][1], 2)
axpoly_basal.set_title('R2 = ' + "{:.2f}".format(R2_basalwrt_defpoly))
axpoly_basal.set_ylabel('m/a')
#
axpoly.legend()
axpoly_basal.legend()
axpoly.set_ylabel('m/a')
axpoly.set_xticklabels( Ypoly_days[ [int(i) for i in axpoly.get_xticks()[:-1] ]])
axpoly.set_title('Comparing def.velocity and polynomials')
figpoly.savefig( path + 'polynomialfitting_on_udef.jpg', dpi = 150, bbox_inches = 'tight')
#
print('')
print('Comparing udef with usurf every X days')
for w in [1,8,14,22,28,34,40,48,60]:
    print('')
    R22 = R2_basalwrt_defpoly = np.power( np.corrcoef( ud_continuous.iloc[-1][15:-14].rolling( window = w, center = True).mean()[w//2 : -w//2:w], udsurf.loc[ Xpolybasal_dates ]['m/a'].rolling( window = w, center = True).mean()[w//2:-w//2:w]), 2)
    print(str(w) + ' days')
    print('R2 = ' + "{:.2f}".format(R22[0][1]))
    print('Number of independent points ' + str( len( udsurf.loc[ Xpolybasal_dates ]['m/a'].rolling( window = w, center = True).mean()[w//2:-w//2:w] )))

# In[]:
# Let's interpolate the surface velocity with a polynomial
Ypoly = udsurf['m/a'].iloc[14:].values
Ypoly_days = udsurf['m/a'].iloc[14:].index
window_avg = 40
Ypoly_avg = pd.DataFrame( Ypoly ).rolling( window = window_avg, center = True ).mean().iloc[window_avg//2: -window_avg//2][0]
figpoly, axpoly = plt.subplots( figsize = (5, 3))
figpoly_basal, axpoly_basal = plt.subplots(figsize = (5, 3))
#
Xpoly_avg =  np.arange(udsurf.shape[0]-14 - window_avg)
Xpoly =  np.arange(udsurf.shape[0]-14)
Xpolybasal = np.arange(14, len( udsurf))
Xpolybasal_dates = udsurf['m/a'].iloc[14:].index
for n in np.arange(8,17,80):
    # z = np.polyfit( Xpoly_avg, Ypoly_avg, n)
    z = np.polyfit( Xpoly, Ypoly, n)
    fs = np.poly1d(z)
    axpoly.plot( 0*window_avg//2 + Xpoly, fs(Xpoly), label ='Poly. n =' + str(n))
    axpoly_basal.plot(Xpolybasal[:-14], udsurf.loc[ Xpolybasal_dates ]['m/a'][:-14] - fs(Xpolybasal[:-14]), label ='Surf. - polynome n =' + str(n))

axpoly.plot(Ypoly, 'b--', label = 'Daily surf. velocity')
axpoly.plot( Ypoly_avg , '--', color = 'red', label = str(window_avg) + ' days average')
# Surface velocity
axpoly_basal.plot( np.arange(len(udsurf))[14:-14], udsurf['m/a'][14:-14] - udsurf['m/a'].mean(), 'k--', label = 'Surface velocity variability')
R2_basalwrt_surfpoly = np.power( np.corrcoef( basal_unfiltered.loc[ Xpolybasal_dates[:-14]], udsurf.loc[ Xpolybasal_dates[:-14] ]['m/a'] - fs(Xpolybasal[:-14]))[0][1], 2)
axpoly_basal.set_title('R2 = ' + "{:.2f}".format(R2_basalwrt_surfpoly))
#
axpoly.legend()
axpoly_basal.legend()
axpoly.set_ylabel('m/a')
axpoly.set_xticklabels( Ypoly_days[ [int(i) for i in axpoly.get_xticks()[:-2] ]].date)
axpoly.set_title('Comparing surf.velocity and polynomials')
figpoly.savefig( path + 'polynomialfitting_on_usurf.jpg', dpi = 150, bbox_inches = 'tight')
# In[]:
# timesries anamysis
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

PERIOD = 40
# Multiplicative Decomposition 
#multiplicative_decomposition = seasonal_decompose( udsurf['m/a'], model='multiplicative', period=PERIOD)
# Additive Decomposition
additive_decomposition = seasonal_decompose(udsurf['m/a'], model='additive', period= PERIOD)
additive_decompositiond = seasonal_decompose(ud_continuous.iloc[ -1 ][:], model='additive', period= PERIOD)
# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
#multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
additive_decompositiond.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#
figdetrend, axdetrend = plt.subplots( figsize = (5, 3),)
axdetrend.plot( udsurf['m/a']  - additive_decomposition.trend, label = 'Detrended usurf (' + str(PERIOD) + ') days')
axdetrend.plot( ud_continuous.iloc[ -1 ]  - additive_decompositiond.trend, label = 'Detrended udef (' + str(PERIOD) + ') days')
axdetrend.set_ylabel('m/a')
axdetrend.legend()
axdetrend.set_title('Detrend velocity = Velocity - moving average')
#
plt.plot( (udsurf['m/a']  - additive_decomposition.trend) - (ud_continuous.iloc[ -1 ]  - additive_decompositiond.trend))