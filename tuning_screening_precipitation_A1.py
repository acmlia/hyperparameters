# ------------------------------------------------------------------------------
# Loading the libraries to be used:
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from collections import Counter
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from imblearn.combine import SMOTEENN, SMOTETomek

# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


class TuningScreeningPrecipitation:
    

    def  create_model(self, neurons=1):
        '''
		 Fucntion to create the instance and configuration of the keras
		 model(Sequential and Dense).
		'''
        # Create the Keras model:
        model = Sequential()
        model.add(Dense(neurons, input_dim=4, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(neurons, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='SGD', learning_rate=0.2, momentum=0.2, batch_size=10, epochs=200, metrics=['accuracy'],)
        return model

    def run_TuningScreeningPrecipitation(self):
        '''
		Fucntion to create the instance and configuration of the KerasRegressor
		model(keras.wrappers.scikit_learn).
		'''

        # Fix random seed for reproducibility:
        seed = 7
        np.random.seed(seed)

        # Load dataset:
        path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/TAG/yearly/'
        file = 'yearly_clip_R1_OK_TAG.csv'
        df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
        x, y= df.loc[:,['36V', '89V', '166V', '190V']], df.loc[:,['TagRain']]
        
        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        y_arr = np.ravel(y_arr)
        
        # Applying the Imabalanced Learn Solution: SMOTEENN
        print('Original dataset shape %s' % Counter(y_arr))
        sm = SMOTEENN(random_state=42)
        x_res, y_res = sm.fit_resample(x_arr, y_arr)
        print('Resampled dataset shape %s' % Counter(y_res))

        # Scaling the input paramaters:
#       scaler_min_max = MinMaxScaler()
#       x_scaled = scaler_min_max.fit_transform(x)
        norm_sc = Normalizer()
        x_normalized= norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_arr, test_size=0.10, random_state=101)

#        # Inserting the modelcheckpoint:
#        checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
#        callbacks_list = [checkpoint]

        # Create the instance for KerasRegressor:
        model = KerasClassifier(build_fn=self.create_model, verbose=0)

        # Define the grid search parameters:
        neurons = [4, 7, 12, 16, 20, 24]
        param_grid = dict(neurons=neurons)
        grid_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
        grid_result = grid_model.fit(x_train, y_train)
        return grid_result

        # Summarize results:
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        # list all data in history
        #history = model.fit(...)
        #print(grid_result.callbacks.History())

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Saving a model
if __name__ == '__main__':
    _start_time = time.time()

    tic()

    training_model = TuningScreeningPrecipitation()
    grid_result = training_model.run_TuningScreeningPrecipitation()
    joblib.dump(grid_result, 'model_trained_screening_precipitation_A1.pkl')

    tac()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------