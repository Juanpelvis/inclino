"""

Creates two plots

One is a single plot, panel is called axzero, corresponds to Figure 2

The other is a multipanel plot, Figure 3, whose panels are called
    - ax
    - axstress
    - axwater

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex = True)
import numpy as np
import scipy.optimize as optim
import sys
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
# In[]: Reading data and plot formatting
""" Read Adri's results: rotated so that X = flow direction """
graduconst3 = pd.read_csv(path+'GradVitForage2_Aconst_n=3_rotated.dat')
graduconst4 = pd.read_csv(path+'GradVitForage2_Aconst_n=4_rotated.dat')
graduconst5 = pd.read_csv(path+'GradVitForage2_Aconst_n=5_rotated.dat')
gradu = pd.read_csv(path+'GradVitForage2_Ainv_n=3_rotated.dat')

""" Read files with MEAN dudz """
observed_mean = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH2_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean3 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH3_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean4 = pd.read_csv('/home/juanpelvis/Documents/Inclino/MEANdudz_BH4_smooth2days_01Mar15Oct.csv', index_col = 'number')
observed_mean4.iloc[7] = observed_mean4.iloc[8]
lliboutry85 = pd.read_csv('/home/juanpelvis/Documents/Inclino/datapoints_liboutry_webplotdigitizer.csv',)
stress = pd.read_csv(path+'StressForage2_Ainv_n=3_rotated.dat', sep = ',')

""" Formatting the plots:"""
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

""" Creating the plots"""
fig, (ax, axstress, axwater) = plt.subplots(ncols = 3, figsize = (1.2*figfull[0], 0.33*figfull[1]), constrained_layout=True)
plt.subplots_adjust(hspace=-1.0, wspace = 0.05)
figzero, axzero = plt.subplots(figsize = (fig14[0], fig14[1]))

""" Formatting colors and symbols for the plots"""
k = 2
colaxzero = {2: 'tab:orange', 3:'tab:green', 4:'tab:red'}
indexes = {2 : np.arange(0, 19),
           3 : np.arange(0, 17),
           4 : np.arange(0, 19),}
indexes_dots = {2 : np.arange(0, 18),
           3 : np.arange(0, 17),
           4 : np.arange(0, 19),
           }
symbols = {2 : '*', 3: '+', 4:'x'}
# In[]: Plotting observations and Adri's results in a single plot and a mutipanel plot
""" Plotting our observations in single plot"""
for df in [observed_mean, observed_mean3, observed_mean4]:
    axzero.plot(df['dudz'].iloc[indexes[k]], df['depth'].iloc[indexes[k]], '-', marker = symbols[k], color = colaxzero[k], label = 'BH'+str(k), zorder = 4-k*0.1)
    axzero.fill_betweenx(df['depth'].iloc[indexes[k]], df['min dudz'].iloc[indexes[k]], df['max dudz'].iloc[indexes[k]], color = colaxzero[k], alpha = 0.2, zorder = 1,label = 'Monthly min\n and max BH'+str(k))#' per\n tilt sensor')
    k = k+1
    #axzero.fill_betweenx([],[],[], color = 'k', alpha = 0.2,  label = 'Monthly min\n and max BH'+str(k))#' per\n tilt sensor')
axzero.set_ylabel('depth')
axzero.legend(facecolor='white', framealpha=1, loc = 'upper right', bbox_to_anchor = (1, 0.79))

""" Annotate some capteurs """
arrowpropsdict = dict(edgecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2")
axzero.annotate(r'BH2\#6', xy = ( observed_mean['dudz'].loc[6], observed_mean['depth'].loc[6]), xytext = (0.8 , -215), arrowprops = arrowpropsdict)
axzero.annotate(r'BH2\#12', xy = ( observed_mean['dudz'].loc[12], observed_mean['depth'].loc[12]), xytext = (0.4 , -125), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH3\#6', xy = ( observed_mean3['dudz'].loc[6], observed_mean3['depth'].loc[6]), xytext = (0.65 , -145), arrowprops = arrowpropsdict)
axzero.annotate(r'BH4\#6', xy = ( observed_mean4['dudz'].loc[6], observed_mean4['depth'].loc[6]), xytext = (0.05 , -240), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH2\#1', xy = ( observed_mean['dudz'].loc[1], observed_mean['depth'].loc[1]), xytext = (1.15 , -210), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH2\#19', xy = ( observed_mean['dudz'].loc[19], observed_mean['depth'].loc[19]), xytext = (0.85 , -30), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH4\#19', xy = ( observed_mean4['dudz'].loc[19], observed_mean4['depth'].loc[19]), xytext = (1.2 , -20), arrowprops = arrowpropsdict, zorder = 20)
axzero.annotate(r'BH3\#17', xy = ( observed_mean3['dudz'].loc[17], observed_mean3['depth'].loc[17]), xytext = (0.5 , -10), arrowprops = arrowpropsdict, zorder = 20)

""" Plotting Adri's results in multipanel plot """
ax.plot(graduconst3['dudz'], graduconst3['depth'], '-', color = 'navy', label = r'$A$ const, $n$=3')
ax.plot(graduconst4['dudz'], graduconst4['depth'], '-', color = 'dodgerblue', label = r'$A$ const, $n$=4')
ax.plot(graduconst5['dudz'], graduconst5['depth'], '-', color = 'cyan' ,label = r'$A$ const, $n$=5')

""" Plotting our results in multipanel plot"""
ax.plot(gradu['dudz'], gradu['depth'], linestyle = '--', color = 'k', label = r'$A(z), n=3$',zorder = 4)
ax.plot(observed_mean['dudz'].iloc[:], observed_mean['depth'].iloc[:], '*-', color = 'tab:orange', label = r'BH2')
ax.fill_betweenx(observed_mean['depth'].iloc[:], observed_mean['min dudz'].iloc[:], observed_mean['max dudz'].iloc[:], color = 'tab:orange', alpha = 0.2, zorder = 1, label = 'Monthly min\n and max BH2')

""" Compute Enhancement factor and plot it """
A = 1
dudz_model = 2 * A * stress['sE'] * stress['sE'] * stress['sxz']
#ax.plot(dudz_model, stress['depth'], 'g-', label = 'modeled dudz')
indices_depth_data_closest = -2*round_to_value(observed_mean['depth'], 0.5)
dudz_model_atdata = dudz_model[[int(i) for i in indices_depth_data_closest]]
dudz_model_atdata.index = [i for i in range(1,20)]
fluidity = observed_mean['dudz']/dudz_model_atdata # essentially, I colpute for A = 1 and then compare to see what I should have
axwater.plot(fluidity.iloc[:], stress.iloc[indices_depth_data_closest[:]]['depth'], linestyle = '',marker='*', color = 'orange', label = r'$A$, $W$ inferred', zorder = 3)

""" Retrieving values of A and plotting them """
A, z = fit_Adrien()
retrieved_A_values = pd.DataFrame(data = {'A' : A, 'depth' : z})
axwater.plot(retrieved_A_values['A'], retrieved_A_values['depth'], linestyle = '-', color = 'k', label = 'Fit', zorder = 3)
""" Paterson and Cuffey and Paterson """
axwater.plot([158.657, 158.657],[-300, 0], 'g--', label = '$A$~P. (1994)')
axwater.plot([77.970,77.970],[-300, 0], 'b:', label = '$A$~C$\&$P\n (2010)')
axwater.set_xlabel(r'$A$ MPa\textsuperscript{-3}~a\textsuperscript{-1}')
ax.set_ylabel(r'Depth m')
axwater.set_yticklabels([])
twA = axwater.twiny()

""" stress plot """
axstress.plot(stress['sxz'], stress['depth'], 'k--', label = r'$\tau_{xz}$', zorder = 3)
axstress.plot(stress['sxy'], stress['depth'], 'k-', zorder = 3)
axstress.plot(stress['sxy'].iloc[::100], stress['depth'].iloc[::100], 'kx', zorder = 3)
axstress.plot([], [], 'kx-', label = r'$\tau_{xy}$',zorder = 3)
axstress.plot(stress['syz'], stress['depth'], 'k-.', label = r'$\tau_{yz}$', zorder = 3)
axstress.plot(stress['sE'], stress['depth'], 'k-', label = r'$\tau_E$', zorder = 3)
axstress.plot(SIA(1, 3, 65, stress['depth'])[2], stress['depth'], linestyle = ':', color = 'forestgreen', label = r'SIA', zorder = 2.9)
axstress.plot(SIA(0.646, 3, 65, stress['depth'])[2], stress['depth'], linestyle = ':', color = 'red', label = r'fSIA', zorder = 2.9)

""" Vallon """
w_retrieved = Duval_inverse(fluidity.iloc[:], 78, 5.8/2.47)
w = Duval_inverse(retrieved_A_values['A'], 78, 5.8/2.47)
twA.set_xlim( Duval_inverse(0, 78, 5.8/2.47), Duval_inverse(700, 78, 5.8/2.47))
twA.set_xticks([0, np.around(Duval_inverse(200, 78, 5.8/2.47), decimals=2), np.around(Duval_inverse(400, 78, 5.8/2.47), decimals=2), np.around(Duval_inverse(600, 78, 5.8/2.47), decimals=2)])
twA.set_xlabel('$W~\%$')
#
filevall = '/home/juanpelvis/Documents/Inclino/WaterContVallon1976.dat'
vallon = pd.read_csv(filevall, sep = ';')
# In[]: Format and save plots
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
axwater.set_ylim([-260, 0])
axwater.set_xlim([0,700])

axwater.legend(loc = 'upper right', facecolor='white', framealpha=1)
axstress.set_ylim([-260, 0])
axstress.set_xlim([-0.03, 0.17])
axstress.set_xticks([0, 0.05, 0.10, 0.15])
axstress.set_yticklabels([])
axstress.set_xlabel('Stress~MPa')
axstress.legend(facecolor='white', framealpha=1)
for a in [axzero, ax, axwater, axstress]:
    a.grid()
  
ext = '.png'
""" Annotate """
def locX(x0,xf,prop):
    return x0 + prop*(xf-x0)
propx = 0.05
ax.annotate('(a)', (locX(-0.05, 1.8, propx), -236))
axstress.annotate('(b)', (locX(-0.03, 0.20, propx), -236))
t = axwater.annotate('(c)', (locX(0, 700, propx), -236))
t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

fig.savefig('/home/juanpelvis/Documents/Inclino/'+'dudz_stress_A_wfSIA'+ext, dpi = 300, bbox_inches = 'tight')
