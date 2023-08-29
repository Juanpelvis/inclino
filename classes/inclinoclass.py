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
import pandas as pd# Dataframe handling package
import aux_inclino as AUX
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates

from datetime import datetime


class inclino:
    """
    Class inclino is created when instancing borehole
    Inclinos have:
        - number
        - Depth
        - Mask
        - du/dz empty by default
        - ud empty by default
        - Tilt empty by default
        - Azimuth empty by default
    """
    inc = pd.DataFrame()
    def __init__(self,number = np.nan,depth = np.nan,mask = 1):
        self.number = number
        self.depth = depth
        self.mask = mask
        self.azimuth = 'empty'
        self.tilt = 'empty'
        self.yaw = 'empty'
        self.ud = 'empty'
        self.dudz = 'empty'
        self.along_tilt = 'empty'
        self.across_tilt = 'empty'
