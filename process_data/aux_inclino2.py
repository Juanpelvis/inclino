#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:22:06 2020

@author: juanpelvis

Auxiliar set of functions to help compute the internal velocity field of Argentiere

Main functions:
    
   + ADD_TILT_AZ: Read *_raw.dat file, computes tilt and azimuth and adds it
    to the dataframe
   + PLOTS: calls matplotlib to generate the plots
    
Subroutines:
   + SET_PLOT: Sets plots conditions such as color range, sizze, etc
   + COMPUTE_TILT_AZ: Called by ADD_TILT_AD 
"""
import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm # import statsmodels 
from scipy import integrate
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from scipy.stats import t as tstud
from lmfit import Model


#from pandas.stats.api import ols # for OSL

import lmfit
register_matplotlib_converters()

def computetiltalt(borehole,startdate,enddate,correct=False,azimuth=True):
    if 'Tilt' not in borehole.file:
        file = borehole.file[:-7]+'tilt.dat'
    else:
        file = borehole.file
    dfax = pd.read_csv(file,skiprows=[0,2,3], na_values = "NAN",keep_default_na=True,converters = {'TIMESTAMP' : str}, dtype = np.float64) #float_precision='round_trip'
    dfax['TIMESTAMP']= pd.to_datetime(dfax['TIMESTAMP'], format = '%Y-%m-%d %H:%M:%S', yearfirst = True)
    #dfax = dfax.dropna() # Option B
    dfax = dfax[(dfax['TIMESTAMP'] >= startdate) & (dfax['TIMESTAMP'] <= enddate)]
    
    result = pd.DataFrame()
    result['TIMESTAMP'] = dfax['TIMESTAMP']
    for i in range(1,1+borehole.get_noi()):
        if correct:
            result['Tilt('+str(i)+')'] = dfax['tilt('+str(i)+')'] - min(dfax['tilt('+str(i)+')'])
        else:
            result['Tilt('+str(i)+')'] = dfax['tilt('+str(i)+')']
        #result['Tiltdir('+str(i)+')'] = dfax['tiltdir('+str(i)+')']
        if azimuth:
            result['Azimuth('+str(i)+')'] = -1*dfax['tiltdir('+str(i)+')']
            result['Yaw('+str(i)+')'] = -1*dfax['yaw('+str(i)+')']
        #result['Azimuth('+str(i)+')'] = result['Tiltdir('+str(i)+')']-2*result['Yaw('+str(i)+')']
        # tiltdir is the operation already (incorrect!)!
    return result

def add_tilt_az(file,noi,startdate,enddate,deg='True', azimuth=True):
    cols = ['TIMESTAMP']
    for i in range(noi):
        namesg = ['Xg('+str(i+1)+')','Yg('+str(i+1)+')','Zg('+str(i+1)+')']
        namesm = ['Xm('+str(i+1)+')','Ym('+str(i+1)+')','Zm('+str(i+1)+')']
        [cols.append(x) for x in namesg]
        if azimuth:
            [cols.append(x) for x in namesm]
    dfax = pd.read_csv(file,skiprows=[0,2,3], na_values = "NAN",keep_default_na=True,usecols = cols,converters = {'TIMESTAMP' : str}, dtype = np.float64) #float_precision='round_trip'
    dfax['TIMESTAMP']= pd.to_datetime(dfax['TIMESTAMP'], format = '%Y-%m-%d %H:%M:%S', yearfirst = True)
    #dfax = dfax.dropna() # Option B
    dfax = dfax[(dfax['TIMESTAMP'] >= startdate) & (dfax['TIMESTAMP'] <= enddate)]
    if azimuth:
        for i in range(noi):
            namesg = ['Xg('+str(i+1)+')','Yg('+str(i+1)+')','Zg('+str(i+1)+')']
            namesm = ['Xm('+str(i+1)+')','Ym('+str(i+1)+')','Zm('+str(i+1)+')']
            if deg:
                yaw = (180/np.pi)*np.arctan2(dfax[namesm[1]],dfax[namesm[0]])
                azzi = (180/np.pi)*np.arctan2(dfax[namesg[1]],dfax[namesg[0]])
                dfax['Yaw('+str(i+1)+')'] = yaw
                dfax['Azimuth('+str(i+1)+')'] = 0*90 - 1*azzi + 1*yaw#+(D*(azzi + yaw< D-360)) - D*(azzi + yaw>D)
                """
                Idea: Azimuth as angle between local X and N
                Yaw is angle between sensor and X
                +90° is to put the North in the vertical
                """
                dfax['tanTilt('+str(i+1)+')'] = np.sqrt(np.square(dfax[namesg[0]])+np.square(dfax[namesg[1]]))/dfax[namesg[2]]
                dfax['Tilt('+str(i+1)+')'] = (180/np.pi)*np.arctan2(np.sqrt(np.square(dfax[namesg[0]])+np.square(dfax[namesg[1]])),dfax[namesg[2]])
            else:
                dfax['Tilt('+str(i+1)+')'] = np.arctan(np.sqrt(np.square(dfax[namesg[0]])+np.square(dfax[namesg[1]]))/dfax[namesg[2]])
                dfax['Azimuth('+str(i+1)+')'] = np.arctan((dfax[namesm[1]]/dfax[namesm[0]]) + (dfax[namesg[1]]/dfax[namesg[0]]))
                dfax['Yaw('+str(i+1)+')'] = np.arctan2(dfax[namesm[1]],dfax[namesm[0]])   
    return dfax

def read_max_prof(file,i,max_theory):
    #print(float(str(pd.read_excel(file,header = None,nrows = 1)[1+(i-1)*5]).split(' ')[-6]))
    MAX = str(pd.read_excel(file,header = None,nrows = 1)[1+(i-1)*5]).split('trou = ')[1]
    MAX = max(float(MAX.split(' m')[0]),max(-1*max_theory))
    return MAX

def plotpd(plot_bool,filter_data,resume,dataf,NoB,noi,noimax,startdate,enddate,DT = 1):
    N = int(48*DT)
    axes,figs = {},{}
    arrow = 1
    for i in range(NoB[0],NoB[1]):
        data = dataf[i+1]
        if plot_bool['polar']:
            figs[i+1],axes[i+1] = plt.subplots(subplot_kw=dict(polar=True), figsize = (9,9))
            col = [cm.viridis(C/noimax[i+1]) for C in range(noimax[i+1])]
            for j in range(noimax[i+1]):
                if filter_data:
                    data2plot = data[['Azimuthf('+str(j+1)+')','Tilt('+str(j+1)+')']].rolling(N).mean()
                else:
                    data2plot = data[['Azimuth('+str(j+1)+')','Tilt('+str(j+1)+')']].rolling(N).mean()
                if resume[i+1]['mask'].iloc[noi[i+1] - j -1] == 1:
                    if arrow:
                        if filter_data:
                            axes[i+1].plot((np.pi/180)*data2plot['Azimuthf('+str(j+1)+')'],data2plot['Tilt('+str(j+1)+')'],linestyle = '',marker = 'o',color = col[j], label = j+1,zorder = 1)
                            axes[i+1].add_patch(draw_arrow(data2plot['Azimuthf('+str(j+1)+')'],data2plot['Tilt('+str(j+1)+')'],'red'))
                        else:
                            axes[i+1].add_patch(draw_arrow(data2plot['Azimuth('+str(j+1)+')'],data2plot['Tilt('+str(j+1)+')'],col[j]))
                    else:
                        if filter_data:
                            axes[i+1].plot((np.pi/180)*data2plot['Azimuthf('+str(j+1)+')'],data2plot['Tilt('+str(j+1)+')'],linestyle = '',marker = '.',color = col[j], label = j+1)
                        else:
                            axes[i+1].plot((np.pi/180)*data2plot['Azimuth('+str(j+1)+')'],data2plot['Tilt('+str(j+1)+')'],linestyle = '',marker = '.',color = col[j], label = j+1)
                else:
                    axes[i+1].plot([],[],linestyle = '',marker = 'X',color = 'r', label = j+1)            
            if DT < 1:
                figs[i+1].suptitle('Borehole '+str(i+1)+'. Data averaged every '+str(24*DT)+' hours')
            else:
                figs[i+1].suptitle('Borehole '+str(i+1)+'. Data averaged every '+str(DT)+' days')
            axes[i+1].set_title('Start date: '+datetime.strftime(startdate,format='%Y-%m-%d %H:%M:%S')+'\n'
                + 'End date: '+datetime.strftime(enddate,format='%Y-%m-%d %H:%M:%S'))
            axes[i+1].legend()
            
        if plot_bool['angles']:
            nc = 2
            nr = int(np.ceil(noimax[i+1]/nc))
            figs['angles'+str(i+1)],axes['angles'+str(i+1)] = plt.subplots(nr,nc,figsize = (6*nc,8*nc))
            figs['angles'+str(i+1)].tight_layout(pad = 5)#, w_pad = 1, h_pad = 0.5)
            for j in range(noimax[i+1]):
                k,l = j//nc, j - nc*int(j/nc)
                #x = np.arange(0,data.shape[0])
                x = data['TIMESTAMP']
                #print('k is ',k,' l is ',l)
                #print(axes['angles'+str(i+1)][0][0])
                axes['angles'+str(i+1)][k,l].plot(x,data['Azimuth('+str(j+1)+')'],'b-',label = 'Azimuth')
                axes['angles'+str(i+1)][k,l].plot(x,data['Yaw('+str(j+1)+')'],'g-',label = 'Yaw')
                axes['angles'+str(i+1)][k,l].plot([],[],'k-',label = 'Tilt')
                ax2 = axes['angles'+str(i+1)][k,l].twinx()
                #axes['angles'+str(i+1)][k,l].set_xlabel('Sample')
                ax2.plot(x,data['Tilt('+str(j+1)+')'],'k-',label = 'Tilt')
                if filter_data:
                    axes['angles'+str(i+1)][k,l].plot(x,data['Azimuthf('+str(j+1)+')'],'y:',label = 'Azimuth filtered')
                    axes['angles'+str(i+1)][k,l].plot(x,data['Yawf('+str(j+1)+')'],'r:',label = 'Yaw filtered')
                axes['angles'+str(i+1)][k,l].set_title('Inclinometer '+str(j+1))
            # Plot every month
                axes['angles'+str(i+1)][k,l].xaxis.set_major_locator(mdates.MonthLocator(interval = 1))
                axes['angles'+str(i+1)][k,l].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                if l == 1:
                    ax2.set_ylabel('Tilt')
                else:
                    axes['angles'+str(i+1)][k,l].set_ylabel('Azimuth and Yaw')
                    
            figs['angles'+str(i+1)].suptitle('Borehole '+str(i+1)+'. Data averaged every '+str(int(np.ceil(N/48)))+' days')
            axes['angles'+str(i+1)][0,0].legend()
    return axes, figs
        
def plot_uddz(plot_bool,resumepd,ud,udfit,NoB,startdate,enddate,dt,MAX_prof,noi):
    FIG,ax,ax2={},{},{}
    for i in range(NoB[0],NoB[1]):#noi[i-1]):
        Z = resumepd[i+1]['prof'].loc[resumepd[i+1]['mask'] == 1].sort_values(ascending=True)
        Zf = pd.Series([-1*MAX_prof[i+1]]).append(Z,ignore_index = True).append(pd.Series([0]),ignore_index = True)
        #print(Zf)
        FIG[i+1],[ax[i+1],ax2[i+1]] = plt.subplots(1,2, figsize = (10,10))
        HALF = int(0.5*ud[i+1].shape[1])
        if plot_bool['ud']:
            for j in range(0,HALF):
                ud0 = udfit[i+1][ud[i+1].columns[j+HALF]].params['ud0'].value
                ax[i+1].plot(ud0+ud[i+1][ud[i+1].columns[j+HALF]],Z,'-*',label = ud[i+1].columns[j+HALF][2:-9], color = cm.plasma(j/HALF),linewidth = 4,markersize = 10)
        if plot_bool['dudz']:
            for j in range(0,HALF):
                ax2[i+1].plot(ud[i+1][ud[i+1].columns[j]],Z,'-*',label = ud[i+1].columns[j][4:-9], color = cm.plasma(j/HALF),linewidth = 4,markersize = 10)
        
        if (type(udfit[i+1]) is dict):
            k = 0
            for F in udfit[i+1]:
                H,B,n = udfit[i+1][F].params['H'].value,udfit[i+1][F].params['B'].value,udfit[i+1][F].params['n'].value
                ud0 = udfit[i+1][F].params['ud0'].value
                ax[i+1].plot(ud0+expo_fit2(Zf,H,B,n,ud0),Zf,':x',label = 'n='+"{0:.1f}".format(n)+',B='+"{0:.1f}".format(B),color=cm.plasma(k/HALF),linewidth = 4,markersize = 10)
                k = k+1
        else:
            H,B,n = udfit[i+1].params['H'].value,udfit[i+1].params['B'].value,udfit[i+1].params['n'].value
            ud0 = udfit[i+1].params['ud0'].value
            ax[i+1].plot(ud0+expo_fit2(Zf,H,B,n,ud0),Zf,'k:',label = 'Fit, n='+"{0:.2f}".format(n)+', B='+"{0:.2f}".format(B),linewidth = 4,markersize = 10)
        
        ax[i+1].set_xticks(np.arange(0, 31, step=5))
        ax[i+1].set_yticks(np.arange(0, -1*MAX_prof[i+1], step=-25))  
        ax2[i+1].set_yticks([])#np.arange(0, -1*MAX_prof[i+1], step=-25))   
        ax[i+1].set_xlabel('m/y')
        ax2[i+1].set_xlabel('y^-1')
        ax[i+1].set_ylabel('Depth')
        ax[i+1].set_title('ud, Borehole '+str(i+1))
        ax2[i+1].set_title('du/dz, Borehole '+str(i+1))
        
        FIG[i+1].suptitle('Start date: '+datetime.strftime(startdate,format='%Y-%m-%d %H:%M:%S')+'\n'
                + 'End date: '+datetime.strftime(enddate,format='%Y-%m-%d %H:%M:%S'))
        
        ax[i+1].legend()
        ax2[i+1].legend()
        dh = 10
        ax2[i+1].set_ylim([-dh-1*MAX_prof[i+1],dh])
        ax[i+1].set_ylim([-dh-1*MAX_prof[i+1],dh])
    return ax,ax2,FIG

#def cluster_fit(udfit):

def plot_boxplot(udfit,NoB):
    M = {1:'*',2:'o',3:'^',4:'x',5:'+'}
    for i in range(NoB[0],NoB[1]):
        var = {}
        err = {}
        for time in udfit[i+1]:
            for k in udfit[i+1][time].params:
                if udfit[i+1][time].params[k].vary:
                    var[k] = {}
                    err[k] = {}
    fig,ax = plt.subplots(2,len(var))
    """ PRINT WRT BOREHOLE """
    for i in range(NoB[0],NoB[1]):
        #fig[i+1],ax[i+1] = plt.subplots()
        for time in udfit[i+1]:
            for k in var:
                var[k][time] = udfit[i+1][time].params[k].value
                err[k][time] = udfit[i+1][time].params[k].stderr
        K = 0
        for variable in var:
            C = 0
            for time in udfit[i+1]:
                #print(err[variable][time])
                yerr = err[variable][time],
                ax[0][K].errorbar(i+1,var[variable][time],yerr, color = cm.plasma(C/len(udfit[i+1])),label = time[2:-9], marker = M[i+1])
                ax[1][K].errorbar([time],[var[variable][time]],yerr,color = cm.plasma(C/len(udfit[i+1])),marker = M[i+1])
                #ax[K].legend()
                C = C+1
            ax[0][K].set_title(variable)
            ax[1][K].set_xlabel('Borehole')
            ax[1][K].set_xlabel('Time')
            ax[1][K].xaxis.set_major_locator(mdates.MonthLocator(interval = 1))
            ax[1][K].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            K = K+1
    """ PLOT WRT TIME """
    
    for i in range(0):#NoB[0],NoB[1]):
        #fig[i+1],ax[i+1] = plt.subplots()
        for time in udfit[i+1]:
            for k in var:
                var[k][time] = udfit[i+1][time].params[k].value
        K = 0
        for variable in var:
            for time in udfit[i+1]:
                ax[1][K].plot(time,var[variable][time],'o',color = cm.nipy_spectral(C/(1+NoB[1]-NoB[0])))
                     
            #ax[0][K].set_title(variable)
            
            K = K+1
        ax[1][0].plot([],[],'o',color = cm.nipy_spectral(C/(1+NoB[1]-NoB[0])),label = i+1)
        ax[1][0].legend()  
        C = C+1
        #ax[i+1].xaxis.set_major_locator(mdates.MonthLocator(interval = 1))
        #ax[i+1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    return ax,fig

def filter_fuzzy_angles(df,noi):
    # Function to filter data
    # Very crude filter, should be better
    N = 48 # 1 day rolling window
    D = 360 # angle jump
    Dc = 180
    d = 100 # safety angle
    cascade = False
    for i in range(noi):
        #yaw=df['Yaw('+str(i+1)+')'].rolling(N).median()
        yaw=df['Yaw('+str(i+1)+')'].median()
        azimuth=df['Azimuth('+str(i+1)+')'].median()
        #azimuth=df['Azimuth('+str(i+1)+')'].rolling(N).median()
        #df['median_yaw('+str(i+1)+')']=yaw
        df['Yawf('+str(i+1)+')']=df['Yaw('+str(i+1)+')'] +D*(df['Yaw('+str(i+1)+')']+d<(yaw)) -D*(df['Yaw('+str(i+1)+')']>yaw+d)
        df['Azimuthf('+str(i+1)+')']=df['Azimuth('+str(i+1)+')'] +D*(df['Azimuth('+str(i+1)+')']+d<(azimuth)) -D*(df['Azimuth('+str(i+1)+')']>azimuth+d)
        if cascade:
            df['Yawf('+str(i+1)+')']=df['Yawf('+str(i+1)+')'] +Dc*(df['Yawf('+str(i+1)+')']+d<(yaw)) -Dc*(df['Yawf('+str(i+1)+')']>yaw+d)
            df['Azimuthf('+str(i+1)+')']=df['Azimuthf('+str(i+1)+')'] +Dc*(df['Azimuthf('+str(i+1)+')']+d<(azimuth)) -Dc*(df['Azimuthf('+str(i+1)+')']>azimuth+d)
        
    return df

def draw_arrowold(theta,r,ax,fig):
    # Funny function to draw arrows along the way
    Nd = 1 # number days per of arrow
    Nd = Nd*48 # points per day
    jx, jy = -2.5, -0.035*r.max()
    x,y = [],[]
    for i in range(0,theta.shape[0],Nd):
        x.append(theta.iloc[i])
        y.append(r.iloc[i])
    x.append(theta.iloc[-1])
    y.append(r.iloc[-1])
    x = [l + jx for l in x]
    y = [l + jy for l in y]
    for i in range(len(x)-1):
        ax.annotate('',
            xy=((np.pi/180)*x[i],y[i]),  # theta, radius
            xycoords='data',
            xytext=((np.pi/180)*x[i+1],y[i+1]),    # fraction, fraction
            textcoords='data',
            arrowprops=dict(facecolor='red',edgecolor='red',width = 1,headwidth = 2, headlength = 4,arrowstyle = '<-'),
            horizontalalignment='left',
            verticalalignment='bottom',
            )
    #return
def draw_arrow(theta,r,color):
    # Funny function to draw arrows along the way
    Nd = 1 # number days per of arrow
    Nd = Nd*48 # points per day
    if Nd > 0.5*len(theta):
        Nd = 1
    jx, jy = 0,0#-2.5, -0.035*r.max()
    x,y = [],[]
    for i in range(0,theta.shape[0],Nd):
        if (np.isnan(theta.iloc[i]) == False):
            x.append(theta.iloc[i])
            y.append(r.iloc[i])
    #x.append(theta.iloc[theta.shape[0]-1])
    #y.append(r.iloc[theta.shape[0]-1])
    x = [(np.pi/180)*(l + jx) for l in x]
    y = [l + jy for l in y]
    PY = [1]
    [PY.append(2) for i in range(len(x)-1)]
    PATH = mpath.Path([(x[i],y[i]) for i in range(len(x))],PY)
    ARROW = mpatches.ArrowStyle("->", head_length=6, head_width=4)
    #print(PATH)
    arrow = mpatches.FancyArrowPatch(path = PATH, arrowstyle = ARROW, ec = color, fc = color, linewidth = 2,zorder = 3)
    return arrow

def integrate_ud(mode,f,dx,noi):
    """
        General function to integrate a given field along a direction
        Mode 1: Cumulative trapezoidal
        Mode 2: Cumulative Simpson
    """
    if mode == 1:
        times = f.columns
        for col in times:
            f['ud'+col[4:]] = integrate.cumtrapz(f[col],dx.sort_values(ascending=True),initial = 0)
    elif mode == 2:
        times = f.columns
        for col in times:
            f['ud'+col[4:]] = integrate.cumtrapz(f[col],dx.sort_values(ascending=False),initial = 0)
    else:
        raise ValueError('No integration mode was given')
    return f

def linreg_total(data):
    """ Apparently I cannot apply a window to return a tuple
    So I gotta do the samr thing twice :(
    
    This one is for the slope"""
    x = np.arange(len(data))
    res = stats.linregress(x, data)
    #tinv = lambda p, df: abs(tstud.ppf(p/2, df))
    #ts = tinv(0.05, len(x)-2)
    #
    return res.slope#ts*res.stderr)

def linreg_total_std(data):
    """ And this one for the stdev. I want to die :( """
    x = np.arange(len(data))
    res = stats.linregress(x, data)
    #
    return res.stderr#ts*res.stderr)

def linreg(data):
    """
        A priori, the data should be already inside...
    """
    x = np.arange(len(data))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, data, rcond=None)[0]
    return m

def func(x,Deltat,phiz,tiltz,l,L33,tolchi=1e-10):
    """
        Theta_0 is NOT the minimum, but the tilt at time 0
    """
    t = x+Deltat
    tancos = np.tan(tiltz)*np.cos(phiz)
    tansin2 =(np.tan(tiltz))**2*(np.sin(phiz))**2
    chi =  tolchi+np.exp(-2*t*L33)*(l*tancos - 0.5*l**2 + tansin2)+np.exp(-4*t*L33)*(tancos-0.5*l)**2 + 0.25*l**2
    theta = np.arctan(np.sqrt(chi))
    return theta


def loop4tilt(keller,params,t,ydata):
    deltat = 0
    tol = 0.001
    result = keller.fit(ydata,params,x=t)
    tiltz = params['tiltz'].value
    phiz = params['phiz'].value
    Deltat = params['Deltat'].value
    #
    l = result.params['l'].value
    L33 = result.params['L33'].value
    L13 = l*L33
    #
    theta = func(t,Deltat,phiz,tiltz,l,L33)
    error = min(theta)-tiltz
    k = 0
    while (abs(error) > tol) and (k<0):
        deltat = t[np.argmin(theta)] - t[(ydata.idxmin()-ydata.index[0])]
        # Index difference between minima
        newtiltz = 0.1+params['tiltz'].value#ydata.loc[ydata.idxmin()+0.1]#deltaix]
        params['tiltz'].value = newtiltz
        result = keller.fit(ydata,params,x=t)
        l = result.params['l'].value
        L33 = result.params['L33'].value
        theta = func(t,deltat,phiz,newtiltz,l,L33)#newtiltz
        error = min(theta)-tiltz
        #
        if k%10!=0:
            print('Iteration ',k)
            print('Delta t = ',deltat)
            print('error ',error)
        k=k+1
        if abs(error) < tol:
            break
    deltat = t[np.argmin(theta)] - t[(ydata.idxmin()-ydata.index[0])]
    return result,deltat

def keller_fit(tilt, az,l,L33):
    t = 0.5*np.arange(len(tilt))/(24*365.25) # check the original script
    keller = Model(func)
    params = keller.make_params()
    params.add('tiltz', value=tilt, min =0,vary=False)
    params.add('phiz', value=az,vary=True)# min=-np.pi/2., max=np.pi/2.)
    params.add('Deltat', value=0,vary=True)# min=-np.pi/2., max=np.pi/2.)
    params.add('l', value=l,vary=False)#,min=1,max=500)
    params.add('L33', value=L33,vary=False)#,min=1e-4,max=1)
    ydata = np.deg2rad(tilt)#chi(1e-6*3,1e-6*1,t,0.010978,1.51)#np.pi*tilt['Tilt(1)']/180
    result,deltat = loop4tilt(keller,params,t,ydata)
    phiz = result.params['phiz'].value
    tiltz = result.params['tiltz'].value
    Deltat = result.params['Deltat'].value
    results = np.rad2deg(func(t,Deltat,phiz,tiltz,l,L33))
    #numerical[i]['L33'].append(k)
    #numerical[i]['L13'].append(j)
    #numerical[i]['error'].append(np.linalg.norm(tilt-res))
    return results
def leastsq_dudz(borehole,result,dt,dx,mode=1):
    """
        New version of du/dz, now using linear regression in each segment
        IDEA: tan(theta(t)) ~ m*t + T'(t)
        So the change in tangent is only due to m, since T'(t) is the same
        for one interval
        this assumes that at short intervals of time tan(theta) evolves linearly (daily scale)
        
        Works data by data
    """
    #aux = pd.DataFrame()
    dudz = pd.DataFrame()
    stddudz = pd.DataFrame()
    for i in range(1,1+borehole.get_noi()):
        #aux = result['Tilt('+str(i)+')'].rolling(dx, center=True).apply(linreg,raw=True)
        dudz['dudz('+str(i)+')'] = (48*365.25*np.tan((np.pi/180)*result['Tilt('+str(i)+')'])).rolling(dx, center=True).apply(linreg_total,raw=True)
        stddudz['std_dudz('+str(i)+')']= (48*365.25*np.tan((np.pi/180)*result['Tilt('+str(i)+')'])).rolling(dx, center=True).apply(linreg_total_std,raw=True)
        #print(result['Tilt('+str(i)+')'].rolling(dx, center=True))
    #
    
    #dt=1
    #
    jumpy = 1+dx//(2)#*dt)
    dudz = dudz.iloc[jumpy:-jumpy:dt] # First and last members are Nan due to rolling window.2nd due to diff
    stddudz = stddudz.iloc[jumpy:-jumpy:dt]
    dudz = dudz.abs()
    TS = result.index[jumpy:-jumpy:dt]#['TIMESTAMP'].loc[::dt]#dt]
    #TS = TS[jumpy:-jumpy]
    dudz['TIMESTAMP'] = TS
    stddudz['TIMESTAMP'] = TS
    return dudz,stddudz,TS # in year-1

def leastsq_dudz_along(borehole,result,dt,dx,mode=1):
    """
        New version of du/dz, now using linear regression in each segment AND the along tilt
        IDEA: tan(theta(t)) ~ m*t + T'(t)
        So the change in tangent is only due to m, since T'(t) is the same
        for one interval
        this assumes that at short intervals of time tan(theta) evolves linearly (daily scale)
        
        Works data by data
    """
    #aux = pd.DataFrame()
    dudz = pd.DataFrame()
    stddudz = pd.DataFrame()
    for i in range(1,1+borehole.get_noi()):
        #aux = result['Tilt('+str(i)+')'].rolling(dx, center=True).apply(linreg,raw=True)
        dudz['dudz('+str(i)+')'] = (48*365.25*np.tan(np.deg2rad(result['along tilt('+str(i)+')']))).rolling(dx, center=True).apply(linreg_total,raw=True)
        stddudz['std_dudz('+str(i)+')']= (48*365.25*np.tan(np.deg2rad(result['along tilt('+str(i)+')']))).rolling(dx, center=True).apply(linreg_total_std,raw=True)
        #print(result['Tilt('+str(i)+')'].rolling(dx, center=True))
    #
    #
    jumpy = 1+dx//(2*dt)
    dudz = dudz.iloc[jumpy:-jumpy] # First and last members are Nan due to rolling window.2nd due to diff
    stddudz = stddudz.iloc[jumpy:-jumpy]
    #dudz = dudz.abs()
    TS = result.iloc[::dt].index#index[TIMESTAMP'].loc#dt]
    TS = TS[jumpy:-jumpy]
    dudz['TIMESTAMP'] = TS
    stddudz['TIMESTAMP'] = TS
    return dudz,stddudz,TS # in year-1

def leastsq_dudz_across(borehole,result,dt,dx,mode=1):
    """
        New version of du/dz, now using linear regression in each segment AND the across tilt
        IDEA: tan(theta(t)) ~ m*t + T'(t)
        So the change in tangent is only due to m, since T'(t) is the same
        for one interval
        this assumes that at short intervals of time tan(theta) evolves linearly (daily scale)
        
        Works data by data
    """
    #aux = pd.DataFrame()
    dudz = pd.DataFrame()
    stddudz = pd.DataFrame()
    for i in range(1,1+borehole.get_noi()):
        #aux = result['Tilt('+str(i)+')'].rolling(dx, center=True).apply(linreg,raw=True)
        dudz['dudz('+str(i)+')'] = (48*365.25*np.tan(np.deg2rad(result['across tilt('+str(i)+')']))).rolling(dx, center=True).apply(linreg_total,raw=True)
        stddudz['std_dudz('+str(i)+')']= (48*365.25*np.tan(np.deg2rad(result['across tilt('+str(i)+')']))).rolling(dx, center=True).apply(linreg_total_std,raw=True)
        #print(result['Tilt('+str(i)+')'].rolling(dx, center=True))
    #
    #
    jumpy = 1+dx//(2*dt)
    dudz = dudz.iloc[jumpy:-jumpy] # First and last members are Nan due to rolling window.2nd due to diff
    stddudz = stddudz.iloc[jumpy:-jumpy]
    #dudz = dudz.abs()
    TS = result.iloc[::dt].index#index[TIMESTAMP'].loc#dt]
    TS = TS[jumpy:-jumpy]
    dudz['TIMESTAMP'] = TS
    stddudz['TIMESTAMP'] = TS
    return dudz,stddudz,TS # in year-1

def compute_dudz(borehole,result,dt,dx,mode=2):
    """        
        du/dz = Delta tan(tilt) / Delta t
        
        Computes per borehole
        TP = TIME PERIOD in YEARS
    """
    dudz = pd.DataFrame()
    stddudz = pd.DataFrame()
    if mode == 1:
        for i in range(1,1+borehole.get_noi()):
            aux = 365*result['tanTilt('+str(i)+')'].loc[::].diff()
            #aux = aux.rolling(2,center=True).mean()
            aux = aux.rolling(dx,center=True).median()#median()
            dudz['dudz('+str(i)+')'] = aux#/dt
    elif mode == 2:
        for i in range(1,1+borehole.get_noi()):
            # Tan, then diff and mean, same as Tan, mean and diff
            #"""
            aux = 48*365*np.tan((np.pi/180)*result['Tilt('+str(i)+')']).diff()
            # The 48*365 helps put it in years.
            aux2 = aux.rolling(dx,center=True).std()
            aux = aux.rolling(dx,center=True).mean()#median()
            #aux2 = aux.rolling(dx,center=True).std()
            """
            # Mean of the tilt, then tan, then diff
            aux = 48*365*np.tan((np.pi/180)*result['Tilt('+str(i)+')'].rolling(dx,center=True).mean())#.diff()
            """
            dudz['dudz('+str(i)+')'] = aux.loc[::dt]#/dt
            stddudz['std_dudz('+str(i)+')'] = aux2.loc[::dt]
            # IDEA: USE OSL??
            
        # B #dudz['dudz('+str(i)+')'] = 365*result['tanTilt('+str(i)+')'].rolling(dx,center=True).mean()/dt # (A)
    # Now we integrate step by step
    # B #dudz = dudz.loc[::dx].diff()
    jumpy = 1+dx//(2*dt)
    dudz = dudz.iloc[jumpy:-jumpy] # First and last members are Nan due to rolling window.2nd due to diff
    stddudz = stddudz.iloc[jumpy:-jumpy]
    dudz = dudz.abs()
    TS = result['TIMESTAMP'].loc[::dt]
    TS = TS.iloc[jumpy:-jumpy]
    dudz['TIMESTAMP'] = TS
    stddudz['TIMESTAMP'] = TS
    return dudz,stddudz,TS

def computetiltrate(borehole,result,dx,dt):
    """        
        \dot{\theta} = Delta tilt / Delta t
        
        Computes per borehole
        TP = TIME PERIOD in YEARS
    """
    tiltrate = pd.DataFrame()
    for i in range(1,1+borehole.get_noi()):
        aux = 48*365*result['Tilt('+str(i)+')'].rolling(dx, center=True).apply(linreg_total,raw=True)
        tiltrate['tiltrate('+str(i)+')'] = aux.iloc[dx//2:-dx//2:dt]
    return tiltrate

def compute_ud(borehole,modeUD = 1):
    ud = {}
    dudz_alt = borehole.dudz.fillna(0)
    if modeUD == 1: # Bottom to top
        for time in borehole.ud_TS.index:
            ud[borehole.ud_TS.loc[time]] = integrate.cumtrapz(
                    # inclinos[i].dudz_alt.loc[time] 
                    [dudz_alt.loc[borehole.ud_TS.loc[time]][i-1] for i in borehole.inclinos if borehole.inclinos[i].mask][::-1],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask][::-1], initial = 0)
        ud['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask][::-1]
    elif modeUD == 0: # Top to bottom
        for time in borehole.ud_TS.index:
            ud[borehole.ud_TS.loc[time]] = integrate.cumtrapz(
                    [dudz_alt.loc[borehole.ud_TS.loc[time]][i-1] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    #[borehole.inclinos[i].dudz_alt.loc[time] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask], initial = 0)
        ud['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask]
    return pd.DataFrame.from_dict(ud)

def compute_ud_lsq(borehole,modeUD = 1):
    ud = {}
    dudz_alt = borehole.dudz_lsq.fillna(0)
    if modeUD == 1: # Bottom to top
        for time in borehole.ud_lsq_TS:#.index:
            #print(time)
            ud[time] = integrate.cumtrapz(
                    # inclinos[i].dudz_alt.loc[time] 
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask][::-1],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask][::-1], initial = 0)
        ud['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask][::-1]
    elif modeUD == 0: # Top to bottom
        for time in borehole.ud_TS:#.index:
            ud[time] = integrate.cumtrapz(
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    #[borehole.inclinos[i].dudz_alt.loc[time] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask], initial = 0)
        ud['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask]
    return pd.DataFrame.from_dict(ud)

def compute_udconf(borehole,DX,modeUD=1, lsq = False):
    """
    For standard deviation: Take lower and upper bounds 
    x +- 1.96*std/sqrt(n); for n=#samples, dt
    at 0.95 confidence interval
    """
    #Zstar = 1.96/np.sqrt(DX)
    tinv  = lambda p, df: abs(tstud.ppf(p/2, df))
    Zstar = tinv(0.05, DX-2) #95% with t-Student
    udmin, udmax = {},{}
    if lsq:
        dudz_alt = borehole.dudz_lsq.fillna(0)
        std = borehole.stddudz_lsq.fillna(0)
    else:
        dudz_alt = borehole.dudz.fillna(0)
        std = borehole.stddudz.fillna(0)
    if modeUD == 1: # Bottom to top
        for time in borehole.ud_TS:
            udmin[time] = integrate.cumtrapz(
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')']-Zstar*std.loc[time].loc['std_dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask][::-1],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask][::-1], initial = 0)
            #
            udmax[time] = integrate.cumtrapz(
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')']+Zstar*std.loc[time].loc['std_dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask][::-1],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask][::-1], initial = 0)
        udmin['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask][::-1]
        udmax['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask][::-1]
    if modeUD == 0: # Top to bottom
        for time in borehole.ud_TS.index:
            udmin[time] = integrate.cumtrapz(
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')']-Zstar*std.loc[time].loc['std_dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask], initial = 0)
            #
            udmax[time] = integrate.cumtrapz(
                    [dudz_alt.loc[time].loc['dudz('+str(i)+')']+Zstar*std.loc[time].loc['std_dudz('+str(i)+')'] for i in borehole.inclinos if borehole.inclinos[i].mask],
                    [borehole.inclinos[i].depth for i in borehole.inclinos if borehole.inclinos[i].mask], initial = 0)
        udmin['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask]
        udmax['number'] = [borehole.inclinos[i].number for i in borehole.inclinos if borehole.inclinos[i].mask]
    return pd.DataFrame.from_dict(udmin), pd.DataFrame.from_dict(udmax)

def compute_udconf_from_error(borehole,dx,modeUD=1):
    """ Assumes my std is stderr
    remember: stderr = stddev / sqrt(n)
    Z = X + Y, X = N(m1,s1²), Y = N(m2,s2²)
    Z = N(m1 + m2, s1² + s2²)
    s² = variance, square of stddev
    
    So: stderr_sum = sqrt(a²stderr²_1 + b²stderr²_2 + c²stderr²_3)
    
    Coefficients: 
        Dz * 0.5
    """
    coeff = {}
    std_ud = pd.DataFrame()
    for i in range(2,1+borehole.get_noi()):
        coeff[i] = 0.5*(borehole.inclinos[i].depth - borehole.inclinos[i-1].depth)
    for i in range(2,1+borehole.get_noi()):
        stdaux = np.power(coeff[2]*borehole.stddudz_lsq['std_dudz('+str(1)+')'],2) + np.power(coeff[2]*borehole.stddudz_lsq['std_dudz('+str(2)+')'],2)
        for j in range(3,i+1):
            stdaux = stdaux + np.power(coeff[j]*borehole.stddudz_lsq['std_dudz('+str(j)+')'],2) + np.power(coeff[j]*borehole.stddudz_lsq['std_dudz('+str(j-1)+')'],2)
        std_ud['conf_int('+str(i)+')'] = np.sqrt(stdaux)
    std_ud['conf_int(1)'] = 0*stdaux
    return 1.96*std_ud

def compute_cumstd(borehole,DX,mode):
    
    return
def expo_fit2(x,H,B,n,UD_0):
    alpha = 10
    yis2 = (365*24*3600)**2
    rho = 900*1e-6
    return -UD_0 + B*0.5*np.power(9.81*rho*np.sin(alpha*np.pi/180),n)*(np.power(H,n+1) - np.power(x,n+1))

def expo_fit3(x,H,A,n,UD_0):
    alpha = 10
    yis = 365*24*3600
    rho = 900*1e-6
    tau = rho*9.81*0.075 #np.sin(alpha*np.pi/180)
    """
    # Slope is 7.5% +- 1%
    # A is given in a^-1 MPa^-n
    """
    # My column is depth, and to reset the origin I substract H^n+1
    return -UD_0 + (2*A/(n-1))*np.power(tau,n)*(np.power(H,n+1)-np.power(x,n+1))

def expo_fit(x,A,C,n):
    #return np.power((x-C)/A,np.divide(1,n+1))   
    return C + A*np.power(-1*x,n+1)

def expo_flor(x,A,C):
    return C + A*np.power(-1*x,4.38)

def fit_udold(ud,z,mode,MAX_prof):
    #mode = 2
    imodel = lmfit.Model(expo_fit2)
    Params = lmfit.Parameters()
    imodel.independent_vars = ['x']
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    Params.add_many(('H', MAX_prof, False, None, None, None, None),
                    ('B', 7, False, 0, 20, None, None),
                ('n', 3.6, False, 3.30, 3.7, None, None),
                ('ud0', 0, True, 0.0, 20, None, None))
    if mode == 2:
        Params['ud0'] = lmfit.Parameter('ud0', 0, False, None, None, None, None)
    if mode == 0:
        Params['n'] = lmfit.Parameter('n', 3.38, False,3.38, 3.38, None, None)
    try :
        result = imodel.fit(ud,x=z,params= Params, method='least_squares')
        print(result.fit_report())
    except:
        # to be improved
        #result = imodel.fit(0*ud,x=z,params= Params, method='least_squares')
        print('No fitting possible: Maybe Data is zero? (NAN)')
    return result

def fit_ud(ud,z,fixed_vars):
    """
    New version of fitting variables
    fixed_vars is a list of lists: [0] Variable, [1] Value, [2] Vary and so on
    """
    imodel = lmfit.Model(expo_fit3)
    Params = lmfit.Parameters()
    imodel.independent_vars = ['x']
    #               (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    Params.add_many(('H', 235, False, None, None, None, None),
                    ('A', 1, True, 0, None, None, None),
                ('n', 3, False, None, None, None, None),
                ('UD_0', 0, None, None, None, None, None))
    for var in fixed_vars:
        if var[3] == var[4]:
            var[2] = False
            var[4] = var[3]+1.0

        Params[var[0]] = lmfit.Parameter(var[0],var[1],var[2],var[3],var[4],None,None)
    try :
        if len(ud.shape) > 1:
            cols = ud.columns
            result = imodel.fit(ud[cols[0]],x=z,params= Params, method='least_squares')
        else:
            result = imodel.fit(ud,x=z,params= Params, method='least_squares')
        print(result.fit_report())
    except:
        result = False
        print('No fitting possible: Maybe Data is zero? (NAN)')
    return result