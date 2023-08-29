#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:08:27 2020

@author: juanpelvis

The goal is to wrap the inclinometry data into a ready-to-use class
and all methods inside as well
"""
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
import aux_inclino2 as AUX
import pandas as pd# Dataframe handling package
import numpy as np
import inclinoclass as IC
from datetime import datetime
import warnings
from scipy import interpolate
import matplotlib.pyplot as plt

class borehole:
    """
    Class borehole contains instances of class inclinometer
    Boreholes are active or not
    Loading inclinos is the same as instancing them
    """
    def __init__(self,path="",number=1,active=1):
        self.number = number
        self.active = active
        self.fileresume = path + 'fichier_profondeur/Prof_G'+str(number)+'.csv'
        self.has_azimuth = True
        if number in [2,3,4]:
            self.file = path + 'ARG_Mar_2021/inclino_2019_'+str(number)+'_Halfh_raw.dat'#'SAUSSURE_living/Saussure_Inclino_'+str(number)+'_Halfh_raw.dat'
            self.filepres = path + 'ARG_Mar_2021/inclino_2019_'+str(number)+'_Halfh_press.dat'
        elif number in [1,5]:
            self.file = path + 'ARG_Nov_2020/inclino_2019_'+str(number)+'_Halfh_raw.dat'#'SAUSSURE_living/Saussure_Inclino_'+str(number)+'_Halfh_raw.dat'
            self.filepres = path + 'SAUSSURE_living/Saussure_Inclino_'+str(number)+'_Halfh_press.dat'
        elif number in [10]: # Taconnaz
            self.file = path + 'TAC_living/GlacioClim_fr_Taco_Inclino_Halfh_raw.dat'
            self.fileresume = path + 'fichier_profondeur/Prof_FR03.csv'
            self.filetemp = path + 'TAC_living/GlacioClim_fr_Taco_Inclino_Halfh_tc.dat'
            self.filepres = ''#path + 'ARG_Nov_2020/inclino_2019_'+str(number)+'_Halfh_press.dat'
            self.has_azimuth = False
        elif number in [11, 13, 14]: # living SAUSSURE
            self.file = path + 'SAUSSURE_living/Saussure_Inclino_'+str(number)+'_Halfh_Tilt.dat'
            self.fileresume = path + 'fichier_profondeur/ARG21_'+str(number - 10)+'.csv'
            self.has_azimuth = False
        elif number == 12: # living SAUSSURE
            self.file = path + 'SAUSSURE_living/Saussure_Inclino_'+str(number)+'_Halfh_Tilt_edited.dat'
            self.fileresume = path + 'fichier_profondeur/ARG21_2_Full.csv'
            self.has_azimuth = False
        self.flowdir_wrtnorth = 55 #55 #direction of flow wrt north direction in deg, according to the 3D flow data (55 to the left)
        self.inclinos = {}
        self.dt = 1
        self.dx = 48
        self.data_TS = pd.DataFrame()
        self.ud_TS = pd.DataFrame()
        self.ud_lsq_TS = pd.DataFrame()
        self.ud = pd.DataFrame()
        self.ud_lsq = pd.DataFrame()
        self.dudz = pd.DataFrame()
        self.stddudz = pd.DataFrame()
        self.stddudz_lsq = pd.DataFrame()
        self.tiltrate = pd.DataFrame()
        self.plotsize = (7,8)
        self.pressure = pd.DataFrame()
        self.tilt = pd.DataFrame()
        self.along_tilt = pd.DataFrame()
        self.across_tilt = pd.DataFrame()
        self.yaw = pd.DataFrame()
        self.azimuth = pd.DataFrame()
        self.depth = pd.DataFrame()
        self.temperature = pd.DataFrame()
    def get_noi(self):
        return len(self.inclinos)
    def get_working_noi(self):
        return sum([self.inclinos[i].mask >0 for i in self.inclinos])
    def read_pressure(self):
        """ Pressure values have different names in each borehole! >:( """
        presname = {1:'P50', 2:'Pup',3:'P50',4:'P',5:'P50'}
        dfax  = pd.read_csv(self.filepres,skiprows=[0,2,3], na_values = "NAN",
        usecols=['TIMESTAMP',presname[self.number]],keep_default_na=True,converters = {'TIMESTAMP' : str})
        dfax.rename(columns={presname[self.number]: 'P'},inplace = True)
        self.pressure = dfax
    def read_temp(self):
        if self.number == 10:
            dfax = pd.read_csv(self.filetemp, skiprows = [0,2,3], na_values = "NAN")
            dfax['TIMESTAMP']= pd.to_datetime(dfax['TIMESTAMP'], format = '%Y-%m-%d %H:%M:%S')
            dfax.set_index('TIMESTAMP', inplace = True)
            self.temperature = dfax
        else:
            print('This borehole does not have temperature data')
    def load_inclino(self):
        self.inclinos = {}
        aux = pd.read_csv(self.fileresume)
        for i in range(aux.shape[0]):
            capt = aux['capt'].loc[i]
            depth = aux['prof'].iloc[i]
            mask = aux['mask'].iloc[i]
            self.inclinos[capt] = IC.inclino(capt,depth,mask)
    def check_load(self):
        """ Automatically reads file profondeur """
        if self.get_noi() == 0:
            print('Inclinometer data not loaded. Loading')
            self.load_inclino()
    def check_dataload(self,result,startdate,enddate):
        """ Automatically reads datalogger entries """
        if result.empty:
            if isinstance(startdate,str):
                if (len(startdate) < 11) or (len(enddate) < 11):
                    startdate = startdate[:10]+' 00:00:00'
                    enddate = enddate[:10]+' 00:00:00'
                startdate = datetime.strptime(startdate,'%Y-%m-%d %H:%M:%S')
                enddate = datetime.strptime(enddate,'%Y-%m-%d %H:%M:%S')
            result = AUX.add_tilt_az(self.file,self.get_noi(),startdate,enddate,azimuth=(self.number<10))
        return result
            
    def compute_tilt_az(self,startdate = '2020-01-01',enddate = '2020-01-31',dt = 1, dx = 1, compud = 0):
        self.check_load()
        result = self.check_dataload(pd.DataFrame(),startdate,enddate)
        self.data_TS = result['TIMESTAMP']
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].tilt = result['Tilt('+str(i)+')'].rolling(dx, center = True).mean()
            self.inclinos[i].tilt.index = result['TIMESTAMP']
            if self.has_azimuth:
                self.inclinos[i].azimuth = result['Azimuth('+str(i)+')'].rolling(dx, center = True).mean()
                self.inclinos[i].azimuth.index = result['TIMESTAMP']
    
    def correct_tilt(self):
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].tilt = self.inclinos[i].tilt - min(self.inclinos[i].tilt)
            
    def project_tilt(self):
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].along_tilt = self.inclinos[i].tilt*(np.cos(np.deg2rad(self.inclinos[i].azimuth - self.flowdir_wrtnorth)))
            self.inclinos[i].across_tilt = self.inclinos[i].tilt*(np.sin(np.deg2rad(self.inclinos[i].azimuth - self.flowdir_wrtnorth)))
    
    def compute_tilt_az_alt(self,startdate = '2020-01-01',enddate = '2020-01-31',dt = 1, dx_azimuth = 1,compud = 0,correct=False):
        self.check_load()
        resultalt = AUX.computetiltalt(self,startdate,enddate,correct,azimuth=self.has_azimuth)
        self.data_TS = resultalt['TIMESTAMP'].iloc[dx_azimuth//2:-dx_azimuth//2]
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].tilt = resultalt['Tilt('+str(i)+')'].iloc[dx_azimuth//2:-dx_azimuth//2]
            self.inclinos[i].tilt.index = self.data_TS
            if self.has_azimuth:
                result = self.check_dataload(pd.DataFrame(),startdate,enddate)
                self.inclinos[i].azimuth = resultalt['Azimuth('+str(i)+')'].rolling(dx_azimuth, center = True).mean().iloc[dx_azimuth//2:-dx_azimuth//2]#result
                self.inclinos[i].yaw = result['Yaw('+str(i)+')'].iloc[dx_azimuth//2:-dx_azimuth//2]
                self.inclinos[i].azimuth.index = self.data_TS
                #self.inclinos[i].yaw.index = self.data_TS
    
    def compute_tiltrate(self,result = pd.DataFrame(),dt = 1,dx = 48,correct=False):
        resultalt = self.tilt
        self.tiltrate = AUX.computetiltrate(self,resultalt,dx,dt)
    
    def compute_dudz(self,result = pd.DataFrame(),startdate = '2020-01-01',enddate = '2020-01-31',dt = 1,dx = 48,correct=False):
        """" dt is time between data, dx is averaging timestep """
        #if dx < dt:
        #    warnings.warn('Averaging time for du/dz '+str(dx)+' is not greater than data timestep, changing to '+str(dt))
        #    dx = dt# + 1
        #result = self.check_dataload(pd.DataFrame(),startdate,enddate)
        resultalt = AUX.computetiltalt(self,startdate,enddate,correct)
        aux,aux2,self.ud_TS = AUX.compute_dudz(self,resultalt,dt,dx,mode = 2)
        #aux,self.ud_TS = AUX.compute_dudz(self,result,dt,dx,mode = 2)
        self.dudz = aux.set_index('TIMESTAMP')
        self.stddudz = aux2.set_index('TIMESTAMP')
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].dudz = aux['dudz('+str(i)+')']   

    def compute_dudz_lsq(self,startdate = '2020-01-01',enddate = '2020-01-31',dt = 1,dx = 48,correct=False):
        #resultalt = AUX.computetiltalt(self,startdate,enddate,correct)
        if len(self.tilt) == 0:
            self.return_tilt()
        resultalt = self.tilt
        aux,aux2,self.ud_lsq_TS = AUX.leastsq_dudz(self, resultalt, self.dt, self.dx,1)
        self.dudz_lsq = aux.set_index('TIMESTAMP')
        self.stddudz_lsq = aux2.set_index('TIMESTAMP')
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].dudz = aux['dudz('+str(i)+')']
    
    def compute_dudz_lsq_along(self,startdate = '2020-01-01',enddate = '2020-01-31',dt = 1,dx = 48,correct=False):
        #if self.inclinos[1].along_tilt != 'empty':
        #    self.return_projected_tilt()# AUX.computetiltalt(self,startdate,enddate,correct)
        resultalt = self.along_tilt
        aux,aux2,self.ud_lsq_TS = AUX.leastsq_dudz_along(self, resultalt, self.dt, self.dx,1)
        self.dudz_lsq_along = aux.set_index('TIMESTAMP')
        #self.stddudz_lsq = aux2.set_index('TIMESTAMP')
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].dudz_along = aux['dudz('+str(i)+')']
            
    def compute_dudz_lsq_across(self,startdate = '2020-01-01',enddate = '2020-01-31',dt = 1,dx = 48,correct=False):
        #self.return_projected_tilt()# AUX.computetiltalt(self,startdate,enddate,correct)
        resultalt = self.across_tilt
        aux,aux2,self.ud_lsq_TS = AUX.leastsq_dudz_projected(self, resultalt, self.dt, self.dx,1)
        self.dudz_lsq_across = aux.set_index('TIMESTAMP')
        #self.stddudz_lsq = aux2.set_index('TIMESTAMP')
        for i in range(1,1+self.get_noi()):
            self.inclinos[i].dudz_across = aux['dudz('+str(i)+')']
            
    def compute_ud(self,mode = 1, DX = 48, minmax = True):
        ud = AUX.compute_ud(self,mode)
        if minmax:
            udmin, udmax = AUX.compute_udconf(self,DX,lsq=False)
            self.udmin = udmin.set_index('number')
            self.udmax = udmax.set_index('number')
        self.ud = ud.set_index('number')

    def compute_ud_lsq(self,mode = 1, DX = 48, minmax = True):
        ud = AUX.compute_ud_lsq(self,mode)
        self.ud_TS = ud.columns[:-1]#last one is 'number'
        if minmax:
            udmin, udmax = AUX.compute_udconf(self,DX=DX,lsq=True)
            self.udmin_lsq = udmin.set_index('number')
            self.udmax_lsq = udmax.set_index('number')
        self.ud_lsq = ud.set_index('number')
        #for i in self.ud.index:
       #     self.inclinos[i].ud = self.ud.loc[i]
    def fit_ud(self):
        result = AUX.fit_ud()
        return result
    def report(self):
        print('Data about borehole number '+str(self.number)+'\r\n')
        print('Total number of inclinometers: '+str(self.get_noi())+'\r\n')
        print('Total number of inclinometers that work '+str(self.get_working_noi())+' \r\n')
    def return_tilt(self, interpol = True):
        df = pd.DataFrame()
        if interpol:
            self.interpol_tilt()
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df['Tilt('+str(i)+')'] = self.inclinos[i].tilt
        #df.set_index(self.data_TS, inplace=True)
        self.tilt = df
    def return_projected_tilt(self):
        self.project_tilt()
        df,df2 = pd.DataFrame(), pd.DataFrame()
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df['along tilt('+str(i)+')'] = self.inclinos[i].along_tilt
            df2['across tilt('+str(i)+')'] = self.inclinos[i].across_tilt
        df.set_index(self.data_TS, inplace=True)
        self.along_tilt = df
        #df2.set_index(self.data_TS, inplace=True)
        self.across_tilt = df2
    def return_across_tilt(self):
        self.project_tilt()
        df = pd.DataFrame()
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df['across tilt('+str(i)+')'] = self.inclinos[i].across_tilt
        #df.set_index(self.data_TS, inplace=True)
        self.across_tilt = df
    def return_azimuth(self):
        df = pd.DataFrame()
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df['azimuth('+str(i)+')'] = self.inclinos[i].azimuth
        #df.set_index(self.data_TS.loc[self.inclinos[1].azimuth.index], inplace=True)
        self.azimuth = df
    def return_yaw(self):
        df = pd.DataFrame()
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df['yaw('+str(i)+')'] = self.inclinos[i].yaw
        #df.set_index(self.data_TS, inplace=True)
        self.yaw = df
    def return_depth(self):
        df = pd.DataFrame(columns = ['depth'])
        for i in range(1,self.get_noi()+1):#self.inclinos:
            df.loc[i] = self.inclinos[i].depth
        self.depth = df
    def interpol_tilt(self):
        """
        Let's start with the easy linear interpolation.
        """
        for i in self.inclinos:
            self.inclinos[i].tilt.interpolate(method ='linear', limit_direction ='both',inplace=True)
    def compute_dudz_keller(self):
        """ Fit the inclinometry data to find the best Keller fit of L13,L33 """
        mindic ={}
        for i in borehole.inclinos:
            index = np.argmin(borehole.inclinos[i].tilt)
            mindic[i] = {'tilt': borehole.inclinos[i].tilt.iloc[index], 'azimuth':borehole.inclinos[i].azimuth.iloc[index], 'date':borehole.data_TS.iloc[index]}
        # Do it first without the minimum tilt
        # self.dudz_keller = AUX.keller_fit(tilt, az)
        self.results = AUX.keller_fit(self.tilt, self.azimuth)
    def smooth_tilt(self, dxspan = 2*48, shift_filtered = True):
        """ Applies an exponential smoothing over a span of dx timesteps"""
        for i in self.inclinos:
            self.inclinos[i].tilt = self.inclinos[i].tilt.ewm(span=dxspan).mean(center=True)
            if shift_filtered:
                self.inclinos[i].tilt = self.inclinos[i].tilt.shift(periods=-1*dxspan//4)[0*dxspan//2:-dxspan//2]
                # Half the half, because I am centering!
        if shift_filtered:
            self.data_TS = self.data_TS[0*dxspan//2:-dxspan//2]
    def smooth_azimuth(self, dxspan = 48, shift_filtered = True):
        """ Applies an exponential smoothing over a span of dx timesteps"""
        for i in self.inclinos:
            self.inclinos[i].azimuth = self.inclinos[i].azimuth.ewm(span=dxspan).mean(center=True)
            if shift_filtered:
                self.inclinos[i].azimuth = self.inclinos[i].azimuth.shift(periods=-1*dxspan//4)[0*dxspan//2:-dxspan//2]
                # Half the half, because I am centering!
    def cut_time_period(self, time):
        """
        Remove signal when data is bad
        """
        for t in time:
            start = t[0]
            end = t[1]
            if t[2] == 'all':
                for i in self.inclinos:
                    self.inclinos[i].tilt[start : end] = np.nan
                    self.inclinos[i].azimuth[start : end] = np.nan
            else:
                for i in t[2]:
                    self.inclinos[i].tilt[start : end] = np.nan
                    self.inclinos[i].azimuth[start : end] = np.nan
    def plot_dudz_lsq(self, imax = 0):
        fig,ax = plt.subplots(figsize = (3,5))
        if imax:
            ax.plot(self.dudz_lsq.mean().iloc[:imax],self.depth.iloc[:imax])
        else:
            index_to_sort = np.argsort( self.depth, axis = 0)
            
            ax.plot(self.dudz_lsq.mean().iloc[index_to_sort['depth']],self.depth.iloc[index_to_sort['depth']])
        ax.set_ylabel('Depth m')
        ax.set_xlabel('du/dz')
        return fig, ax