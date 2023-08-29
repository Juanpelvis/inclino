import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
matplotlib.rc('text', usetex = True)
import numpy as np
import scipy.optimize as optim
import sys
import return_borehole2021_cleaned as BHdata
sys.path.append('/home/juanpelvis/Documents/python_code/auxpy')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
import AUXplot as AUX
#
path = '/home/juanpelvis/Documents/Inclino/simulation_outputs/'
#
def round_to_value(number,roundto): # from stackoverflow https://stackoverflow.com/a/4265619
    return (round(number / roundto) * roundto)
def function(x,a,A0,b):
    return A0 + np.divide(a, x+b)
def fit_Adrien():
    A = [50, 50 , 200, 400]
    z = [0, -139.7, -203.2, -254]
    return A, z
def Duval_inverse(A,A0,factor = 1.8125):
    """
    Duval uses a different A0 than us, and assumes n = 3
    If we consider A = A0* (1 + factor * w), we have
    w = (A/A0 - 1)/factor
    """
    return (-1 + A/A0)/factor
def SIA(f,n,A,z):
    rho = 900*1e-6
    alpha = np.arctan(0.075)
    g = 9.81#*(362.25*24*60*60)**2
    dudz = 2*A*np.power(f*g*rho*np.sin(alpha)*np.abs(z),n)
    ud = 2*(A/(n+1))*((f*rho*g*np.sin(alpha))**n)*np.power(np.abs(z),n+1)
    tauxz = f*rho*g*np.sin(alpha)*np.abs(z)
    return dudz, ud, tauxz
def strain_heating(stress, gradu):
    Lf = 0.336
    water_gen = stress['sxz'] * gradu['dudz'] / (2 * Lf)
    relative_water = water_gen / 910
    bE = pd.DataFrame(data = {'kga-1m-3' : water_gen, '% generated per year' : relative_water})
    #
    return bE
#
zsurf = {11 : 2364.530, 12: 2364.207, 13 : 2336.239, 14 : 2334.745}
#
BH = 14
# borehole, colors, udsurf, cavitometer, rain, (runoff1,runoff2) = BHdata.borehole_all(BH = BH, time = True, compute_borehole = True,specific_time = (('2021-09-20', '2022-06-15')), read_borehole = False)
# start = datetime.strptime('2021-12-01','%Y-%m-%d')#'2019-04-01','%Y-%m-%d')
# end = datetime.strptime('2022-06-01','%Y-%m-%d')#'2019-04-01','%Y-%m-%d')
# borehole.dudz_lsq = borehole.dudz_lsq[ borehole.dudz_lsq.index >= start]
# borehole.dudz_lsq = borehole.dudz_lsq[ borehole.dudz_lsq.index <= end]
# borehole.dudz_lsq.mean().to_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH' + str(BH) + '_30days_01Dec01May.csv')
# borehole.dudz_lsq.to_csv( '/home/juanpelvis/Documents/Inclino/dudz_BH' + str(BH) + '_30days_01Dec01May.csv')
# dudzBH14_monthlymean = borehole.dudz_lsq.rolling( window = 48*30, center = True).mean()
# dudzBH14_monthlymean.iloc[48*15:-48*15].to_csv( '/home/juanpelvis/Documents/Inclino/dudz_BH' + str(BH) + '_30days_01Nov01Jun_monthlymeans.csv' )

observed_mean11 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH11_30days_01Nov01Jan.csv', index_col = 'number')
observed_mean12 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH12_30days_01Nov15Feb.csv', index_col = 'number')
observed_mean14 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH14_60days_01Dec01May.csv', index_col = 'number')
"""
graduconst3 = pd.read_csv(path+'GradVitForage2_Aconst_n=3_rotated.dat')
graduconst4 = pd.read_csv(path+'GradVitForage2_Aconst_n=4_rotated.dat')
graduconst5 = pd.read_csv(path+'GradVitForage2_Aconst_n=5_rotated.dat')
gradu = pd.read_csv(path+'GradVitForage2_Ainv_n=3_rotated.dat')
observed_mean = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH2_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean3 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH3_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean4 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH4_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean4.iloc[7] = observed_mean4.iloc[8]
lliboutry85 = pd.read_csv('/home/juanpelvis/Documents/Inclino/datapoints_liboutry_webplotdigitizer.csv',)
"""
#
fig14, figfull, label_pt = AUX.set_figureconditions()
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,minisize, MKS = AUX.give_plotsizes()
font, MARKERmu, STYLEmu, COLORr, thicc, skinny = AUX.set_plotconditions()
plt.rc('font', size=font['size'])          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=minisize)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#stress.to_csv(path+'StressForage2_rotated.dat')
#
#stress = pd.read_csv(path+'StressForage2_Ainv_n=3_rotated.dat', sep = ',')
#
#fig, (ax, axstress, ax2) = plt.subplots(ncols = 3, figsize = (1.2*figfull[0], 0.33*figfull[1]), constrained_layout=True)
figzero, (axzero, axzerobis2, thirdax) = plt.subplots(figsize = (2 * fig14[0], fig14[1]), ncols = 3)
plt.subplots_adjust(hspace=1.0, wspace = 0.)
axzerobis = axzerobis2.twiny()
#tabcolorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
colaxzero = {2: 'tab:orange', 3:'tab:green', 4:'tab:red'}
indexes = {2 : np.arange(0, 19),
           3 : np.arange(0, 24),
           4 : np.arange(0, 19),}
indexes_dots = {2 : np.arange(0, 18),
           3 : np.arange(0, 17),
           4 : np.arange(0, 19),
           }
symbols = {2 : '*', 3: '+', 4:'x'}
ax = {2 : axzero, 3 : axzero, 4 : axzerobis}
BHn = {2 : 11, 3 : 12, 4 : 14}
month = {2 : 'Monthly', 3 : 'Monthly', 4: 'Bimonthly'}
k = 3
axboth = thirdax.twinx()
#fig_both, axboth = plt.subplots( figsize = (5, 10))
for df in [observed_mean12, observed_mean14]:#[observed_mean11, observed_mean12, observed_mean14]:#, observed_mean4]:
    ax[k].plot(df['dudz'].iloc[indexes[k]], df['depth'].iloc[indexes[k]], '-', marker = symbols[k], color = colaxzero[k], label = 'BH'+str(BHn[k]), zorder = 4-k*0.1)
    axboth.plot(df['dudz'].iloc[indexes[k]], zsurf[BHn[k]] + df['depth'].iloc[indexes[k]], '-', marker = symbols[k], color = colaxzero[k], label = 'BH'+str(BHn[k]), zorder = 4-k*0.1)
    ax[k].fill_betweenx(df['depth'].iloc[indexes[k]], df['min dudz'].iloc[indexes[k]], df['max dudz'].iloc[indexes[k]], color = colaxzero[k], alpha = 0.2, zorder = 1,label = month[k] + ' min\n and max BH'+str(BHn[k]))#' per\n tilt sensor')
    axboth.fill_betweenx(zsurf[BHn[k]] + df['depth'].iloc[indexes[k]], df['min dudz'].iloc[indexes[k]], df['max dudz'].iloc[indexes[k]], color = colaxzero[k], alpha = 0.2, zorder = 1,label = month[k] + ' min\n and max BH'+str(BHn[k]))#' per\n tilt sensor')
    k = k+1
    #axzero.fill_betweenx([],[],[], color = 'k', alpha = 0.2,  label = 'Monthly min\n and max BH'+str(k))#' per\n tilt sensor')
axzero.set_ylabel('depth')
axzerobis.legend(facecolor='white', framealpha=1, loc = 'upper right', bbox_to_anchor = (1, 0.79))
axzero.legend(facecolor='white', framealpha=1, loc = 'upper right', bbox_to_anchor = (1, 0.79))
axzero.set_xlim([0, 1.1])
axzero.set_ylim([-200, 0])
#
axzerobis.set_xlim([0, 1.1])
axzerobis.set_ylim([-200, 0])
#
axzero.set_xlabel('$du/dz$')
axzerobis.set_xlabel('$du/dz$')
#
axzero.text( s = '(a)', x = 0.9, y = -7.5)
axzerobis.text( s = '(b)', x = 0.9, y = -7.5)
axboth.text( s = '(c)', x = 0.9, y = zsurf[12] + observed_mean12['depth'].iloc[-2])
axboth.set_ylabel('Height')
#
arrowpropsdict = dict(edgecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2")
DELTA = 0
for i in [17, 12]:
    axzerobis.annotate(text = 'BH14\#' + str(i), xy = (observed_mean14['dudz'].iloc[i - 1], observed_mean14['depth'].iloc[i - 1]), 
                    xytext = (0.15 + observed_mean14['dudz'].iloc[i - 1], DELTA + 5 + observed_mean14['depth'].iloc[i - 1]), arrowprops = arrowpropsdict)
    DELTA = -10
axboth.set_xlim([0, 1.1])
axzerobis.set_yticklabels([])
thirdax.set_yticklabels([])
thirdax.set_yticks([])
#
axzero.grid()
axzerobis.grid()
axboth.grid()
#
figzero.savefig('/home/juanpelvis/Documents/Inclino/dudz_mean_BH12and14.png' , dpi = 300, bbox_inches = 'tight')
#
""" Annotate some capteurs """
"""
arrowpropsdict = dict(edgecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2")
axzero.annotate(r'BH2\#6', xy = ( observed_mean['dudz'].loc[6], observed_mean['depth'].loc[6]), xytext = (0.8 , -215), arrowprops = arrowpropsdict)
axzero.annotate(r'BH2\#12', xy = ( observed_mean['dudz'].loc[12], observed_mean['depth'].loc[12]), xytext = (0.4 , -125), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH3\#6', xy = ( observed_mean3['dudz'].loc[6], observed_mean3['depth'].loc[6]), xytext = (0.65 , -145), arrowprops = arrowpropsdict)
#axzero.annotate(r'BH4\#8', xy = ( observed_mean4['dudz'].loc[8], observed_mean4['depth'].loc[8]), xytext = (1 , -190), arrowprops = arrowpropsdict)
axzero.annotate(r'BH4\#6', xy = ( observed_mean4['dudz'].loc[6], observed_mean4['depth'].loc[6]), xytext = (0.05 , -240), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH2\#1', xy = ( observed_mean['dudz'].loc[1], observed_mean['depth'].loc[1]), xytext = (1.15 , -210), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH2\#19', xy = ( observed_mean['dudz'].loc[19], observed_mean['depth'].loc[19]), xytext = (0.85 , -30), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH4\#19', xy = ( observed_mean4['dudz'].loc[19], observed_mean4['depth'].loc[19]), xytext = (1.2 , -20), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH3\#17', xy = ( observed_mean3['dudz'].loc[17], observed_mean3['depth'].loc[17]), xytext = (0.5 , -10), arrowprops = arrowpropsdict, zorder = 20)
#
"""
ax.plot(graduconst3['dudz'], graduconst3['depth'], '-', color = 'navy', label = r'$A$ const, $n$=3')
ax.plot(graduconst4['dudz'], graduconst4['depth'], '-', color = 'dodgerblue', label = r'$A$ const, $n$=4')
ax.plot(graduconst5['dudz'], graduconst5['depth'], '-', color = 'cyan' ,label = r'$A$ const, $n$=5')
#
ax.plot(gradu['dudz'], gradu['depth'], linestyle = '--', color = 'k', label = r'$A(z), n=3$',zorder = 4)
#ax.plot(stress['sxz'], stress['depth'], 'b-', label = 'sxz')
ax.plot(observed_mean['dudz'].iloc[:], observed_mean['depth'].iloc[:], '*-', color = 'tab:orange', label = r'BH2')
ax.fill_betweenx(observed_mean['depth'].iloc[:], observed_mean['min dudz'].iloc[:], observed_mean['max dudz'].iloc[:], color = 'tab:orange', alpha = 0.2, zorder = 1, label = 'Monthly min\n and max BH2')
#
A = 1
dudz_model = 2 * A * stress['sE'] * stress['sE'] * stress['sxz']
#ax.plot(dudz_model, stress['depth'], 'g-', label = 'modeled dudz')
indices_depth_data_closest = -2*round_to_value(observed_mean['depth'], 0.5)
dudz_model_atdata = dudz_model[[int(i) for i in indices_depth_data_closest]]
dudz_model_atdata.index = [i for i in range(1,20)]
fluidity = observed_mean['dudz']/dudz_model_atdata # essentially, I colpute for A = 1 and then compare to see what I should have
ax2.plot(fluidity.iloc[:], stress.iloc[indices_depth_data_closest[:]]['depth'], linestyle = '',marker='*', color = 'orange', label = r'$A$, $W$ inferred', zorder = 3)
#
# Polyfit is not good...
#PolFit = np.polyfit(stress.iloc[indices_depth_data_closest[:-5]]['depth'], fluidity.iloc[:-5],2)
#ax2.plot(np.polyval(PolFit, stress.iloc[indices_depth_data_closest[:-7]]['depth']), stress.iloc[indices_depth_data_closest[:-7]]['depth'], linestyle=':', color = 'orange')
#params = optim.curve_fit(function, stress.iloc[indices_depth_data_closest[:]]['depth'], fluidity.iloc[:], bounds = ([-np.inf, 0, 250],[np.inf, 120, 300]))
#a = params[0][0]
#A0 = params[0][1]
#depth_limit = params[0][2]
#retrieved_A_values = function(stress.iloc[indices_depth_data_closest[:-5]]['depth'],a, A0, depth_limit) # this gives 71 MPa^-3 a^-1 at -172 m
A, z = fit_Adrien()
retrieved_A_values = pd.DataFrame(data = {'A' : A, 'depth' : z})
""" Add point at the top """
"""
retrieved_A_values = pd.DataFrame(data = retrieved_A_values)
retrieved_A_values.rename(columns = {'depth' : 'A'}, inplace = True)
retrieved_A_values['depth'] = stress.iloc[indices_depth_data_closest[:-5]]['depth']
retrieved_A_values.iloc[-1] = [65, 0]
"""
ax2.plot(retrieved_A_values['A'], retrieved_A_values['depth'], linestyle = '-', color = 'k', label = 'Fit', zorder = 3)
""" Paterson and Cuffey and Paterson """
ax2.plot([158.657, 158.657],[-300, 0], 'g--', label = '$A$~P. (1994)')
ax2.plot([77.970,77.970],[-300, 0], 'b:', label = '$A$~C$\&$P\n (2010)')
ax2.set_xlabel(r'$A$ MPa\textsuperscript{-3}~a\textsuperscript{-1}')
ax.set_ylabel(r'Depth m')
#ax.set_yticklabels([])
ax2.set_yticklabels([])
twA = ax2.twiny()
# So we have
# A = A0 + a/(x + depth_limit) , for 'a' a parameter, A0 is b, and depth_limit is c
""" stress plot """
axstress.plot(stress['sxz'], stress['depth'], 'k--', label = r'$\tau_{xz}$', zorder = 3)
axstress.plot(stress['sxy'], stress['depth'], 'k-', zorder = 3)
axstress.plot(stress['sxy'].iloc[::100], stress['depth'].iloc[::100], 'kx', zorder = 3)
axstress.plot([], [], 'kx-', label = r'$\tau_{xy}$',zorder = 3)
axstress.plot(stress['syz'], stress['depth'], 'k-.', label = r'$\tau_{yz}$', zorder = 3)
axstress.plot(stress['sE'], stress['depth'], 'k-', label = r'$\tau_E$', zorder = 3)
axstress.plot(SIA(1, 3, 65, stress['depth'])[2], stress['depth'], linestyle = ':', color = 'forestgreen', label = r'SIA', zorder = 2.9)
axstress.plot(SIA(0.646, 3, 65, stress['depth'])[2], stress['depth'], linestyle = ':', color = 'blue', label = r'fSIA', zorder = 2.9)
""" Vallon """
#
w_retrieved = Duval_inverse(fluidity.iloc[:], 78, 5.8/2.47)
#twA.plot(w_retrieved, observed_mean['depth'], '*', color ='orange', label = '$w$~\% inferred')
#
w = Duval_inverse(retrieved_A_values['A'], 78, 5.8/2.47)
#tw2.plot(w,retrieved_A_values['depth'], 'k-', label = '$w~\%$ fit', zorder = 2.)
twA.set_xlim( Duval_inverse(0, 78, 5.8/2.47), Duval_inverse(700, 78, 5.8/2.47))
#.set_ylim(-260, 0)
twA.set_xticks([0, np.around(Duval_inverse(200, 78, 5.8/2.47), decimals=2), np.around(Duval_inverse(400, 78, 5.8/2.47), decimals=2), np.around(Duval_inverse(600, 78, 5.8/2.47), decimals=2)])
twA.set_xlabel('$W~\%$')
#
filevall = '/home/juanpelvis/Documents/Inclino/WaterContVallon1976.dat'
vallon = pd.read_csv(filevall, sep = ';')
#tw2.plot(vallon['w'], vallon['depth'], 'x', color = 'dodgerblue', label = 'Vallon et al.\n (1976) adj.')
#tw2.set_yticklabels([])
#twA.plot(lliboutry85['w'], -1*lliboutry85['depth'], 'r-.', label = 'Lliboutry\n et al. (1985)')
#ax2.plot([], [], 'r-.', label = '$w$~Lliboutry\n et al. (1985)')
#twA.legend(facecolor='white', framealpha=1, loc = 'upper right')
""" Formatting """
axzero.set_xlabel(r'$du/dz$~a\textsuperscript{-1}')
ax.set_xlabel(r'$du/dz$~a\textsuperscript{-1}')
ax.legend(loc = 'upper right', facecolor='white', framealpha=1,)#  bbox_to_anchor = (1, 0.82))
axzero.set_ylabel(r'Depth m')
axzero.set_xlim([-0.05, 1.8])
axzero.set_ylim([-260, 0])
axzero.set_yticks(np.arange(-250,1,25))
ax.set_xlim([-0.05, 1.8])
ax.set_ylim([-260, 0])
ax2.set_ylim([-260, 0])
ax2.set_xlim([0,700])
# twA.set_xlim([0, 700/80])
# twA.set_xticks([1, 2.5, 5, 7.5])
# twA.set_xlabel(r'$E$')
ax2.legend(loc = 'upper right', facecolor='white', framealpha=1)
axstress.set_ylim([-260, 0])
axstress.set_xlim([-0.03, 0.17])
axstress.set_xticks([0, 0.05, 0.10, 0.15])
axstress.set_yticklabels([])
axstress.set_xlabel('Stress~MPa')
axstress.legend(facecolor='white', framealpha=1)
for a in [axzero, ax, ax2, axstress]:
    a.grid()
  
ext = '.png'
""" Annotate """
def locX(x0,xf,prop):
    return x0 + prop*(xf-x0)
propx = 0.05
#axzero.annotate('(a)', (locX(-0.05, 1.8, propx), -236))
ax.annotate('(a)', (locX(-0.05, 1.8, propx), -236))
axstress.annotate('(b)', (locX(-0.03, 0.20, propx), -236))
t = ax2.annotate('(c)', (locX(0, 700, propx), -236))
t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
#ww = tw2.annotate('(d)', (locX(-0.35, 3.5, propx), -236))
#ww.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
#ww.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
"""
ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )

tw2.add_artist(ab)
"""
fig.savefig('/home/juanpelvis/Documents/Inclino/'+'dudz_stress_A_wfSIA'+ext, dpi = 300, bbox_inches = 'tight')
#figzero.savefig('/home/juanpelvis/Documents/Inclino/'+'dudz_BH234'+ext, dpi = 300, bbox_inches = 'tight')