from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import regularizers
from keras.activations import relu, elu, linear, softmax, tanh, softplus
from keras.callbacks import EarlyStopping, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam, Nadam, sgd,Adadelta, RMSprop
from keras.losses import mean_squared_error, categorical_crossentropy, logcosh
from keras.utils.np_utils import to_categorical
from keras import metrics


from keras import regularizers
#keras Load Model
from keras.models import load_model

import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos.model.layers import hidden_layers
from talos import live
from talos.model import lr_normalizer, early_stopper, hidden_layers
import os

from CNN.cnn import acc_pearson_r  as acc_pearson_r

def mlp_main(x, y, x_val, y_val, params):

    model_mlp = Sequential()
    nSNP = x.shape[1]
    try:
        out_c= y.shape[1]
    except IndexError:
        out_c=1


    model_mlp.add(Dense(params['first_neuron'], input_dim=nSNP,
                        activation=params['activation'],
                        kernel_initializer='normal', kernel_regularizer=regularizers.l2(params['reg1'])))

    model_mlp.add(Dropout(params['dropout_1']))
    if (params['hidden_layers'] != 0):
        # if we want to also test for number of layers and shapes, that's possible
        for _ in range(params['hidden_layers']):
            model_mlp.add(Dense(params['hidden_neurons'], activation=params['activation'],
                                kernel_regularizer=regularizers.l2(params['reg1'])))

            # hidden_layers(model, params, 1)
            model_mlp.add(Dropout(params['dropout_2']))

    model_mlp.add(Dense(out_c,activation=params['last_activation'],
    kernel_regularizer=regularizers.l2(params['reg2'])))
    if params['optimizer']=='Adam':
        params['optimizer']= Adam
    if params['optimizer']=='Nadam':
        params['optimizer']= Nadam
    if params['optimizer']=='sgd':
        params['optimizer']= sgd

    model_mlp.compile(loss=mean_squared_error,
                     optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                     metrics=[acc_pearson_r])
    #es = EarlyStopping(monitor=mean_squared_error, mode='min', verbose=1)

    # callbacks=[live()] see the output
    # callbacks= es to EarlyStopping

    out_mlp = model_mlp.fit(x, y, validation_split=0.2,
                            verbose=0, batch_size=params['batch_size'],
                            epochs=params['epochs'])

    return out_mlp, model_mlp

def mlp_main_cat(x, y, x_val, y_val, params):

    model_mlp = Sequential()
    nSNP = x.shape[1]
    last_layer= y.shape[1]

    model_mlp.add(Dense(params['first_neuron'], input_dim=nSNP,
                        activation=params['activation'],
                        kernel_initializer='normal', activity_regularizer=regularizers.l1(params['reg1'])))

    model_mlp.add(Dropout(params['dropout_1']))
    if (params['hidden_layers'] != 0):
        # if we want to also test for number of layers and shapes, that's possible
        for _ in range(params['hidden_layers']):
            model_mlp.add(Dense(params['hidden_neurons'], activation=params['activation'],
                                activity_regularizer=regularizers.l2(params['reg1'])))

            # hidden_layers(model, params, 1)
            model_mlp.add(Dropout(params['dropout_2']))
    model_mlp.add(Dense(last_layer,activation='softmax'))
    if params['optimizer']=='Adam':
        params['optimizer']= Adam
    if params['optimizer']=='Nadam':
        params['optimizer']= Nadam
    if params['optimizer']=='sgd':
        params['optimizer']= sgd

    model_mlp.compile(loss='categorical_crossentropy',optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],
    params['optimizer'])), metrics=['accuracy'])


    #acc or mean_squared_error in metrics
    # simple early stopping
    # if you monitor is an accuracy parameter (pearson here, you should chose mode="max"), otherwise it would be "min"
    #es = EarlyStopping(monitor=mean_squared_error, mode='min', verbose=1)

    # callbacks=[live()] see the output
    # callbacks= es to EarlyStopping

    out_mlp = model_mlp.fit(x, y, validation_split=0.2,
                            verbose=0, batch_size=params['batch_size'],
                            epochs=params['epochs'])

    return out_mlp, model_mlp
