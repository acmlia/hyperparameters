#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:22:38 2019

@author: dvdgmf
"""
import numpy as np
import pandas as pd
import copy

df = pd.read_csv('/media/dvdgmf/DATA2_RPK/mozao/meteoro_regions_rosante/'
              'CSV/yearly_clip_R1_OK.csv', sep = ',', header = 0)

rain_pixels = np.where((df['sfcprcp'] >= 0.1))
df_reg_copy = copy.deepcopy(df)
df_rain = df_reg_copy.iloc[rain_pixels]

labels = ['C1','C2','C3','C4','C5','C6']

df_rain['CLASSE'] = pd.cut(df_rain['sfcprcp'],
       bins=[0,1,5,10,15,20,60], labels=labels, include_lowest=True)

df_rain['CLASSE'].value_counts()

file_name = 'yearly_R1_rain_classes.csv'
df_rain.to_csv(file_name, index=False, sep=",", decimal='.')