#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:22:38 2019

@author: dvdgmf
"""
import numpy as np
import pandas as pd
import copy
import os

path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/TAG/yearly/'
for file in os.listdir(path):
       print('reading file: ' + path + file)
       df_rain = pd.read_csv(os.path.join(path, file), sep=',', decimal='.', encoding="utf8", header = 0)

       #rain_pixels = np.where((df['sfcprcp'] >= 0.1))
       #df_reg_copy = copy.deepcopy(df)
       #df_rain = df_reg_copy.iloc[rain_pixels]

       labels = ['C1','C2','C3','C4']

       df_rain['CLASSE'] = pd.cut(df_rain['sfcprcp'], bins=[0,1,10,20,400], labels=labels, include_lowest=True)

       df_rain['CLASSE'].value_counts()

       # SUBSET BY SPECIFIC CLASS
       # n = 0.865
       # to_remove = np.random.choice(df_rain[df_rain['CLASSE']=='C1'].index,size=int(df_rain.shape[0]*n),replace=False)
       # df_rain = df_rain.drop(to_remove)


       # Saving the new output DB's (rain and no rain):
       file_name = os.path.splitext(file)[0] + "_CLASSE.csv"
       df_rain.to_csv(os.path.join(path, file_name), index=False, sep=",", decimal='.')
       print("The file ", file_name, " was genetared!")

