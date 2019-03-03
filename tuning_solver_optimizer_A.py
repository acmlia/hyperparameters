#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:26:23 2019

@author: rainfall
"""

# ------------------------------------------------------------------------------
# Loading the libraries to be used: import numpy
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    

class TuningSolverOptimizer:
    
    def create_model(self, optimizer = 'Adam'):

        # Function to create model, required for KerasRegressor:
        model = Sequential()
        model.add(Dense(5, input_dim=9, activation='tanh'))
        model.add(Dense(5, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        # Compile model:
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model 
    
    def run_TuningSolverOptimizer(self):
        '''
		Fucntion to create the instance and configuration of the KerasRegressor
		model(keras.wrappers.scikit_learn).
		'''
        # Fix random seed for reproducibility:
        seed = 7
        np.random.seed(seed)
        
        # Load dataset:
        path = '/media/DATA/tmp/datasets/regional/qgis/rain/'
        file = 'yearly_br_rain_var2d_OK.csv'
        df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')

        # Split into input (X) and output (Y) variables:
        df2 = df[['36V', '36H', '89V', '89H', '166V', '166H', '190V','PCT36','PCT89']]
        #x = df2.reindex(columns=cols)
        x = df2[['36V', '36H', '89V', '89H', '166V', '166H', '190V','PCT36','PCT89']]
        y = df[['sfcprcp']]

        # Scaling the input paramaters:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=101)

        # Create the instance for KerasRegressor:
        model = KerasRegressor(build_fn=self.create_model, epochs=1000, batch_size=10, verbose=0)
        
        # Define the grid search parameters:
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(optimizer = optimizer)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(x_train, y_train)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Saving a model
if __name__ == '__main__':
    _start_time = time.time()

    tic()

    training_model = TuningSolverOptimizer()
    grid_result = training_model.run_TuningSolverOptimizer()
    joblib.dump(grid_result, 'model_trained_solver_optimizer_A.pkl')

    tac() 

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------      