#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:50:53 2019

@author: rainfall
"""
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

 # Load dataset:
path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/TAG/yearly/'
file = 'yearly_clip_R1_OK_TAG.csv'
df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')

# Split into input (X) and output (Y) variables:
df2 = df[['36V', '89V', '166V', '190V']]
#x = df2.reindex(columns=cols)
x = df2[['36V', '89V', '166V', '190V']]
y = df[['TagRain']]

# Scaling the input paramaters:

scaler_min_max = MinMaxScaler()
x_minmax = scaler_min_max.fit_transform(x)

scaler_abs_max = MaxAbsScaler()
x_abs_max = scaler_abs_max.fit_transform(x)

stand_sc = StandardScaler()
x_stand_sc = stand_sc.fit_transform(x)

norm_sc = Normalizer()
x_norm = norm_sc.fit_transform(x)

x_power_box = PowerTransformer(method='box-cox').fit_transform(x)
x_power_yeo = PowerTransformer(method='yeo-johnson').fit_transform(x)

x_quantil = QuantileTransformer(output_distribution = 'uniform').fit_transform(x)
