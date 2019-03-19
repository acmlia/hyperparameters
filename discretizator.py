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
from sklearn.preprocessing import Normalizer

original_df = pd.read_csv('/media/dvdgmf/DATA2_RPK/mozao/datasets/meteo_regions/'
              'yearly_clip_R1_OK_TAG.csv', sep = ',', header = 0)

# ---------------------
# DROP ZERO RAIN PIXELS
# ---------------------
rain_pixels = np.where((original_df['sfcprcp'] >= 0.1))
df_rain = original_df.iloc[rain_pixels]

# -----------------
# CREATE CLASS BINS 
# -----------------
labels = ['C1','C2','C3','C4']
df_rain['CLASSE'] = pd.cut(df_rain['sfcprcp'],
       bins=[0,1,10,20,400], labels=labels, include_lowest=True)
df_rain['CLASSE'].value_counts()

# ---------------------
# DROP SPECIFIC COLUMNS 
# ---------------------
colunas = list(df_rain.columns.values)
colunas = [e for e in colunas if e not in ('lat',
 'lon', 'sfccode', 'T2m', 'tcwv', 'skint', 'cnvprcp', '10V', '10H',
 '18V', '18H', '23V', '36H', '89H', '166H', 'emis10V', 'emis10H',
 'emis18V', 'emis18H', 'emis23V', 'emis36V', 'emis36H', 'emis89V',
 'emis89H', 'emis166V', 'emis166H', 'emis186V', 'emis190V', '10VH',
 '18VH', 'SSI', 'delta_neg', 'delta_pos', 'MPDI', 'MPDI_scaled',
 'PCT10', 'PCT18', 'TagRain')]

df_rain = df_rain.loc[:,colunas]

# ----------------------------------------
# SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
# ----------------------------------------
n = 0.9
to_remove = np.random.choice(
        df_rain[df_rain['CLASSE']=='C1'].index,
        size=int(df_rain[df_rain['CLASSE']=='C1'].shape[0]*n),
        replace=False)

df_rain = df_rain.drop(to_remove)

# ----------------------------------
# NORMALIZE the WHOLE DATASET
# ----------------------------------
#sknorm = Normalizer()
#array_df_rain_norm = sknorm.fit_transform(df_rain.iloc[:,0:12])
#norm_cols = list(df_rain.columns.values)
#colunas = [e for e in colunas if e not in ('CLASSE')]
#resultado = pd.DataFrame(data=array_df_rain_norm[:],columns=colunas)

# ----------------------------------
# EXPLORATORY ANALISYS: SEABORN PLOT
# ----------------------------------
sns.set(style="ticks")
sns.pairplot(df_rain, hue="CLASSE")

# -----------------------------
# Prototype generation: SMOTEEN
# -----------------------------
df = df_rain
x, y = df.loc[:,colunas], df.loc[:,['CLASSE']]
x_arr = np.asanyarray(x)
y_arr = np.asanyarray(y)
y_arr = np.ravel(y_arr)
df_rain['CLASSE'].value_counts()
print('Original dataset shape %s' % Counter(y_arr))
sm = SMOTEENN(random_state=42)
x_res, y_res = sm.fit_resample(x_arr, y_arr)
print('Resampled dataset shape %s' % Counter(y_res))
df_rain['CLASSE'].value_counts()

# ----------------------------------------------------
# Converting Ndarray back to pd.DataFrame to save CSV
# ----------------------------------------------------
resultado = pd.DataFrame(data=x_res[:],columns=colunas)
file_name = 'YRLY_R1_OK_SMOTEEN.csv'
resultado.to_csv(file_name, index=False, sep=",", decimal='.')