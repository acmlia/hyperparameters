# ------------------------------------------------------------------------------
# Loading the libraries to be used: 
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


#------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class LoadTrainedModels:
    
    def ScoresFromTrainedModels(self, grid_result):
        '''
		 Fucntion to verify the scores from GridSearch CV on the trained
         model configurations.
	    '''
        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            return()
# -----------------------------------------------------------------------------
# Saving a model
# joblib.dump(model, 'model_trained_batch_epoch.pkl')

# Loading a model
# loaded_model = joblib.load('picles_do_satanas.pkl')

# pickle.dump(model, open('model_trained.sav', 'wb'))

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    _start_time = time.time()

    tic()

    training_model = LoadTrainedModels()
    #model_trained = training_model.run_TuningBatchEpoch()
    try:
        grid_result = joblib.load('modelomozao.pkl')
    except AttributeError as e:
        print(e)
    tac()


