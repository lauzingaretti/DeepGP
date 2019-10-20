from CNN.cnn import acc_pearson_r  as acc_pearson_r
from MLP.mlp import mlp_main as mlp_main
from MLP.mlp import mlp_main_cat as mlp_main_cat
import os
import sys
import pandas as pd
import numpy as np
import talos as ta

from matplotlib import pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import regularizers
from keras.activations import relu, elu, linear, softmax, tanh, softplus
from keras.callbacks import EarlyStopping, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam, Nadam, sgd, Adadelta, RMSprop
from keras.losses import mean_squared_error, categorical_crossentropy, logcosh
from keras.utils.np_utils import to_categorical
from keras import metrics

# keras to CNN
from keras.layers import Flatten, Conv1D, MaxPooling1D
# defining network
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras import regularizers

import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos.model.layers import hidden_layers
from talos import live
from talos.model import lr_normalizer, early_stopper, hidden_layers
import os
from talos import Deploy


def run_mlp_main(X_tr, y_tr, X_vl, y_val, output, main, prop,trait,cat,
lr,dr_1,dr_2,reg_1,reg_2,act_1,hn,hl,epochs,op,bs,N_neurons_FL):
    # population: Number of networks/genomes in each generation.
    # we only need to train the new ones....
    # generations: Number of times to evolve the population.
    # hyperparameters
    p = {'lr':  lr,
         'reg1': reg_1,
         'reg2': reg_2,
         'first_neuron': N_neurons_FL,
         'hidden_neurons': hn,
         'hidden_layers': hl,
         'batch_size': bs,
         'epochs': epochs,
         'dropout_1': dr_1,
         'dropout_2': dr_2,
         'optimizer': op,
         'activation': act_1,
         'last_activation': [linear]}
    if cat is False:
        talos_mlp = ta.Scan(x=X_tr,
                            y=y_tr,
                            model=mlp_main, params=p,
                            grid_downsample=prop,
                            print_params=True,
                            dataset_name='mlp_model')
        best_model = talos_mlp.best_model(metric='val_acc_pearson_r', asc=False)
    else:
        talos_mlp = ta.Scan(x=X_tr,
                            y=y_tr,
                            model=mlp_main_cat, params=p,
                            grid_downsample=prop,
                            print_params=True,
                            dataset_name='mlp_model')
        best_model = talos_mlp.best_model(metric='val_acc')

    x_val = X_vl
    yy_hat = best_model.predict(x_val, batch_size=128)


    # save model

    talos_Data = pd.DataFrame(talos_mlp.data)
    talos_Data["val_loss"] = pd.to_numeric(talos_Data["val_loss"], errors="coerce")
    if cat is False:
        talos_Data["acc_pearson_r"] = pd.to_numeric(talos_Data["acc_pearson_r"], errors="coerce")
        talos_Data["val_acc_pearson_r"] = pd.to_numeric(talos_Data["val_acc_pearson_r"], errors="coerce")


    os.chdir(output)

    if not os.path.exists("mlp"):
        os.makedirs("mlp")
        dir = output + "mlp/"

    os.chdir(os.path.join(output, 'mlp/'))

    # write output
    talos_Data.to_csv("mlp_prediction_talos.csv", index=False)

    r = ta.Reporting("mlp_prediction_talos.csv")

    if not os.path.exists("figures"):
        os.makedirs("figures")
        dir = output + "mlp/figures/"

    os.chdir(os.path.join(output, 'mlp/figures/'))
    if cat is False:
        try:
            number= y_tr.shape[1]
            for i in range(0, number):
                corr = np.corrcoef(y_val[:, i], yy_hat[:, i])[0, 1]
                # correlation btw predicted and observed
                fig = plt.figure()
                # plot observed vs. predicted targets
                plt.title('mlp: Observed vs Predicted Y trait_' + str(i) + 'cor:' + str(corr))
                plt.ylabel('Predicted')
                plt.xlabel('Observed')
                plt.scatter(y_val[:, i], yy_hat[:, i], marker='o')
                fig.savefig("mlp_Tunne_trait_" + str(i) + '.png',
                            dpi=300)
                plt.close(fig)
        except IndexError:
            yy_hat=np.reshape(yy_hat,-1)

            corr = np.corrcoef(y_val, yy_hat)[0, 1]
            # correlation btw predicted and observed
            fig = plt.figure()
            # plot observed vs. predicted targets
            plt.title('mlp: Observed vs Predicted Y trait_' + str(trait) + 'cor:' + str(corr))
            plt.ylabel('Predicted')
            plt.xlabel('Observed')
            plt.scatter(y_val, yy_hat, marker='o')
            fig.savefig("mlp_Tunne_trait_" + str(trait) + '.png',
                        dpi=300)
            plt.close(fig)


    os.chdir(output)
    if os.path.exists("best_model"):
        os.rmdir("best_model")
        os.makedirs("best_model")
    if not os.path.exists("best_model"):
        os.makedirs("best_model")

    os.chdir(os.path.join(output, 'best_model/'))

    # os.chdir(main)
    if cat is False:
        Deploy(talos_mlp, 'mlp_optimize', metric='val_acc_pearson_r', asc=False)
    else:
        Deploy(talos_mlp,'mlp_optimize')
