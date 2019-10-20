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

#keras to CNN
from keras.layers import Flatten, Conv1D, MaxPooling1D
# defining network
from keras.layers import Flatten, Conv1D, MaxPooling1D
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


#custom metric
def acc_pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

# nStride=3  # stride between convolutions
# nFilter=32 # no. of convolutions


def cnn_main(x, y, x_val, y_val, params):
    # next we can build the model exactly like we would normally do it
    # Instantiate
    model_cnn = Sequential()
    nSNP = x.shape[1]
    try:
        out_c = y.shape[1]
    except IndexError:
        out_c = 1
    x = np.expand_dims(x, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    # add convolutional layer

    if (params['nconv']==1):
        model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                             strides=params['nStride'], input_shape=(nSNP, 1),
                             kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                             activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

        model_cnn.add(MaxPooling1D(pool_size=params['pool']))
        # Solutions above are linearized to accommodate a standard layer

    else:
        for _ in range(params['nconv']):
            if (_==0):
                model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                                     strides=params['nStride'], input_shape=(nSNP, 1),
                                     kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                                     activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

                model_cnn.add(MaxPooling1D(pool_size=params['pool']))
                # Solutions above are linearized to accommodate a standard layer
            else:
                model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                                         strides=params['nStride'],
                                         kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                                         activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

                model_cnn.add(MaxPooling1D(pool_size=params['pool']))

    model_cnn.add(Flatten())

    if (params['hidden_layers'] != 0):
        # if we want to also test for number of layers and shapes, that's possible
        for _ in range(params['hidden_layers']):
            model_cnn.add(Dense(params['hidden_neurons'], activation=params['activation_2'],
                                kernel_regularizer=regularizers.l2(params['reg2'])))
            model_cnn.add(Dropout(params['dropout_2']))

    model_cnn.add(Dense(out_c, activation=params['last_activation'], kernel_regularizer=regularizers.l2(params['reg3'])
                        ))
    if params['optimizer']=='Adam':
        params['optimizer']= Adam
    if params['optimizer']=='Nadam':
        params['optimizer']= Nadam
    if params['optimizer']=='sgd':
        params['optimizer']= sgd
    model_cnn.compile(loss=mean_squared_error,
                      optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                      metrics=[acc_pearson_r])

    # simple early stopping
    # if you monitor is an accuracy parameter (pearson here, you should chose mode="max"), otherwise it would be "min"
    # 7/08/2019 change mean_squared_error here by acc_pearson_r and mode='min' by mode='max'
    #es = EarlyStopping(monitor=acc_pearson_r, mode='max', verbose=1)

    # callbacks=[live()] see the output
    # callbacks= es to EarlyStopping

    out_cnn = model_cnn.fit(x, y, validation_split=0.2,
                            verbose=0, batch_size=params['batch_size'],
                            epochs=params['epochs'], callbacks=[live()])

    return out_cnn, model_cnn


#CNN main categories
def cnn_main_cat(x, y, x_val, y_val, params):
    # next we can build the model exactly like we would normally do it
    # Instantiate
    model_cnn = Sequential()
    nSNP = x.shape[1]
    out_c= y.shape[1]
    x = np.expand_dims(x, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    # add convolutional layer
    if (params['nconv']==1):
        model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                             strides=params['nStride'], input_shape=(nSNP, 1),
                             kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                             activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

        model_cnn.add(MaxPooling1D(pool_size=params['pool']))
        # Solutions above are linearized to accommodate a standard layer

    else:
        for _ in range(params['nconv']):
            if (_==0):
                model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                                     strides=params['nStride'], input_shape=(nSNP, 1),
                                     kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                                     activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

                model_cnn.add(MaxPooling1D(pool_size=params['pool']))
                # Solutions above are linearized to accommodate a standard layer
            else:
                model_cnn.add(Conv1D(params['nFilter'], kernel_size=params['kernel_size'],
                                         strides=params['nStride'],
                                         kernel_regularizer=regularizers.l2(params['reg2']), kernel_initializer='normal',
                                         activity_regularizer=regularizers.l1(params['reg1']),activation=params['activation_1']))

                model_cnn.add(MaxPooling1D(pool_size=params['pool']))

    model_cnn.add(Flatten())
    if (params['hidden_layers'] != 0):
        # if we want to also test for number of layers and shapes, that's possible
        for _ in range(params['hidden_layers']):
            model_cnn.add(Dense(params['hidden_neurons'], activation=params['activation_1'],
                                activity_regularizer=regularizers.l1(params['reg1'])))
            model_cnn.add(Dropout(params['dropout']))

    model_cnn.add(Dense(out_c, activation='softmax'))

    if params['optimizer']=='Adam':
        params['optimizer']= Adam
    if params['optimizer']=='Nadam':
        params['optimizer']= Nadam
    if params['optimizer']=='sgd':
        params['optimizer']= sgd

    model_cnn.compile(loss='categorical_crossentropy', optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],
    params['optimizer'])),metrics=['accuracy'])

    # simple early stopping
    # if you monitor is an accuracy parameter (pearson here, you should chose mode="max"), otherwise it would be "min"
    #es = EarlyStopping(monitor=mean_squared_error, mode='min', verbose=1)

    # callbacks=[live()] see the output
    # callbacks= es to EarlyStopping

    out_cnn = model_cnn.fit(x, y, validation_split=0.2,
                            verbose=0, batch_size=params['batch_size'],
                            epochs=params['epochs'])

    return out_cnn, model_cnn
