#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:22:38 2019

@author: dvdgmf
"""
import seaborn as sns
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from collections import Counter
#import copy

df_rain = pd.read_csv('/media/dvdgmf/DATA2_RPK/mozao/datasets/meteo_regions/'
              'yearly_clip_R1_OK_TAG.csv', sep = ',', header = 0)

#rain_pixels = np.where((df['sfcprcp'] >= 0.1))
#df_reg_copy = copy.deepcopy(df)
#df_rain = df_reg_copy.iloc[rain_pixels]

labels = ['C1','C2','C3','C4']

df_rain['CLASSE'] = pd.cut(df_rain['sfcprcp'],
       bins=[0,1,10,20,400], labels=labels, include_lowest=True)

df_rain['CLASSE'].value_counts()
#colunas = list(df_rain.columns.values)
#colunas = [e for e in colunas if e not in ('TagRain', 'CLASSE')]

# SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
n = 0.9
to_remove = np.random.choice(df_rain[df_rain['CLASSE']=='C1'].index,
                             size=int(df_rain[df_rain['CLASSE']=='C1'].shape[0]*n),replace=False)
df_rain = df_rain.drop(to_remove)

# EXPLORATORY ANALISYS
sns.set(style="ticks")
sns.pairplot(df_rain, hue="CLASSE")

# Prototype generation: under-sampling by generating new samples
df = df_rain
x, y = df.loc[:,colunas], df.loc[:,['CLASSE']]

x_arr = np.asanyarray(x)
y_arr = np.asanyarray(y)
y_arr = np.ravel(y_arr)

# Applying the Imabalanced Learn Solution: SMOTEENN
print('Original dataset shape %s' % Counter(y_arr))
sm = SMOTEENN(random_state=42)
x_res, y_res = sm.fit_resample(x_arr, y_arr)
print('Resampled dataset shape %s' % Counter(y_res))

resultado = pd.DataFrame(data=x_res[:],    # values
#                         index=x_res[1:,0],    # 1st column as index
                         columns=colunas)

file_name = 'YRLY_R1_OK_SMOTEEN.csv'
resultado.to_csv(file_name, index=False, sep=",", decimal='.')