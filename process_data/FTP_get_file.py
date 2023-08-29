# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 07:26:07 2021

@author: piardl


REMEMBER TO CONNECT TO CISCO
"""

from ftplib import FTP
from datetime import datetime
import os
#
#dataset = 'tac'
dataset = 'sau'
#
ftp_path = {}
ftp_files = {}
local_data_path = {}
# data path on the remote FTP serveur
 # for Taconnaz data
ftp_path['tac']='/France'          
ftp_files['tac']=['GlacioClim_fr_Taco_Inclino_Halfh_tilt.dat','GlacioClim_fr_Taco_Inclino_Halfh_raw.dat','GlacioClim_fr_Taco_Inclino_Halfh_tc.dat', 'ARG_Halfh_2021.dat']
local_data_path['tac']='/home/juanpelvis/Documents/Inclino/TAC_living/'

# for Argentière data 
ftp_path['sau']='/Saussure'          
ftp_files['sau']=['Saussure_Inclino_1_Halfh_Tilt.dat','Saussure_Inclino_2_Halfh_Tilt.dat','Saussure_Inclino_3_Halfh_Tilt.dat','Saussure_Inclino_4_Halfh_Tilt.dat','Saussure_Inclino_2_Halfh_TiltC.dat','Saussure_Inclino_2_Halfh_press.dat','Saussure_Inclino_4_Halfh_press.dat',]#,'Saussure_Inclino_2_Halfh_Raw.dat','Saussure_Inclino_3_Halfh_raw.dat','Saussure_Inclino_4_Halfh_raw.dat']
local_data_path['sau']='/home/juanpelvis/Documents/Inclino/SAUSSURE_living/'

print('login in ')
ftp = FTP("loggernet.ige-grenoble.fr")
ftp.login("glacioclim", "glacio@clim!")
ftp.cwd(ftp_path[dataset])

# Print out the files
for file in ftp_files[dataset]:
	print("Downloading..." + file)
	ftp.retrbinary("RETR " + file ,open( local_data_path[dataset] + file, 'wb').write)
print("Voili voila c'est fini")
ftp.close()

if dataset == 'sau':
    for file in ftp_files[dataset]:
        os.rename(local_data_path[dataset] + file, local_data_path[dataset] + file[:17]+'1'+file[17:])