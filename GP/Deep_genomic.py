"""
 Deep_genomic.py allows predicting complex traits by using Deep Learning and Penalized Liner models
 Authors: Laura M Zingaretti (m.lau.zingaretti@gmail.com) and iguel Perez-Enciso (miguel.perez@uab.es)
"""
'''
    Copyright (C) 2019  Laura Zingaretti
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import matplotlib as mpl
mpl.use('Agg')
from CNN.cnn import acc_pearson_r  as acc_pearson_r
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pandas as pd
import numpy as np
from run_cnn import run_cnn_main
from run_mlp import run_mlp_main
from run_ridge import run_ridge_main
from run_lasso import run_lasso_main
from run_rnn import run_rnn_main
import seaborn as sns
from sklearn.model_selection import train_test_split


from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


import os

import sys
import argparse
import logging


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--Xtr", required=True, help="train Predictor matrix")
    parser.add_argument("--ytr", required=True,help= "train response matrix")
    parser.add_argument("--cat", required=False, help="Should be y_train matrix normalized by categories? If yes, category should be provided as cat argument, e.g. --cat T ")
    # if you only want to train the model, you don't need a validation set
    parser.add_argument("--Xval", default=None,required=False, help="validation Predictor")
    parser.add_argument("--yval",default=None, required=False,help= "validation response matrix")
    parser.add_argument("--scale",help="Boolean indicating if y should be scaled", action="store_true",default=False)
    parser.add_argument("--dummy", help="Convert SNPs into OneHot encoding", action="store_true",default=True)
    parser.add_argument("--categorical", help="Are outputs categorical variables?", action="store_true", default=False)
    parser.add_argument("--gridProp", help="proportion of random search", default=0.01, type=float)
    parser.add_argument("--trait", help="name of trait to be used in the analysis", default=None,required=False)
    parser.add_argument("--output", help="path to output dir",required=True)
    Method = parser.add_mutually_exclusive_group()
    Method.add_argument('--mlp', action='store_true')
    Method.add_argument('--cnn', action='store_true')
    Method.add_argument('--lasso', action='store_true')
    Method.add_argument('--ridge', action='store_true')
    parser.add_argument('--lr',type=float, nargs='+', help='the learning rate parameter to be used in NN configuration. It can be a list to tune', required=False,default=0.025)
    parser.add_argument('--dr1',type=float, nargs='+', help='the dropout to be used in  the first layer of  NN configuration. It can be a list to tune', required=False, default=0)
    parser.add_argument('--dr2',type=float, nargs='+', help='the dropout to be used in  the hidden_layers of the NN configuration. It can be a list to tune', required=False, default=0)
    parser.add_argument('--reg1',type=float, nargs='+', help='the L2 regularization to be used in  the first layer of  NN configuration. It can be a list to tune', required=False,default=0)
    parser.add_argument('--nconv',type=int, nargs='+',help='number of convolutions layers to be considered in convolutional operation. It only works if Method is CNN and it can be a list to tune', default=1)
    parser.add_argument('--act1', action='store',type=str, nargs='+', default=['relu', 'elu', 'linear', 'softmax', 'tanh', 'softplus'],help="Examples: --act1 relu elu, --act1 linear")
    parser.add_argument('--act2', action='store',type=str, nargs='+', default=['relu', 'elu', 'linear', 'softmax', 'tanh', 'softplus'],help="Examples: --act2 relu elu, --act2 linear")
    parser.add_argument('--reg2',type=float, nargs='+', help='the L2 regularization to be used in  the hidden_layers of  NN configuration. It can be a list to tune', required=False,default=0)
    parser.add_argument('--reg3',type=float, nargs='+', help='the L2 regularization to be used in  the output layers of the  NN configuration. It can be a list to tune', required=False,default=0)
    parser.add_argument('--hn',type=int, nargs='+', help='Number of hidden neurons to be used on the NN configuration. It can be a list to tune', required=False, default=8)
    parser.add_argument('--ps',type=int, nargs='+', help='Pool size to be used on the CNN configuration. It can be a list to tune', required=False,default=3)
    parser.add_argument('--hl',type=int, nargs='+', help='Number of hidden layers to be used in the NN configuration. It can be a list to tune', required=False,default=1)
    parser.add_argument('--ks',type=int, nargs='+', help='kernel_size to be used in CNN configuration. It can be a list to tune', required=False,default=3)
    parser.add_argument('--ns',type=int, nargs='+', help='stride to be used in CNN configuration. It can be a list to tune', required=False,default=1)
    parser.add_argument('--epochs',type=int, nargs='+', help='nepochs to be used for training. It can be a list to tune', required=False,default=50)
    parser.add_argument('--nfilters',type=int, nargs='+', help='number of filter (#convolutions in each convolutional layer). It can be a list to tune', required=False,default=8)
    parser.add_argument('--optimizer', action='store',type=str, nargs='+', default=['Adam', 'Nadam', 'sgd'],help="Examples: -optimizer Adam")
    parser.add_argument('--bs',type=int, nargs='+', help='batch size to be used in NN configuration. It can be a list to tune', required=False,default=16)
    parser.add_argument('--N_neurons_FL',type=int, nargs='+', help='NNeurons to be used at the first layer of MLP configuration. It can be a list to tune', required=False,default=8)

    args = parser.parse_args()

    if not isinstance(args.hn, list):
	       args.hn = [args.hn]
    if not isinstance(args.nconv, list):
	       args.nconv = [args.nconv]
    if not isinstance(args.ps, list):
	       args.ps = [args.ps]
    if not isinstance(args.hl, list):
	       args.hl = [args.hl]
    if not isinstance(args.ks, list):
	       args.ks = [args.ks]
    if not isinstance(args.ns, list):
	       args.ns = [args.ns]
    if not isinstance(args.epochs, list):
	       args.epochs = [args.epochs]
    if not isinstance(args.nfilters, list):
    	   args.nfilters = [args.nfilters]
    if not isinstance(args.bs, list):
	       args.bs = [args.bs]
    if not isinstance(args.N_neurons_FL, list):
        args.N_neurons_FL = [args.N_neurons_FL]
    if not isinstance(args.reg1, list):
	       args.reg1 = [args.reg1]
    if not isinstance(args.reg2, list):
	       args.reg2 = [args.reg2]
    if not isinstance(args.reg3, list):
	       args.reg3 = [args.reg3]
    if not isinstance(args.dr1, list):
	       args.dr1 = [args.dr1]
    if not isinstance(args.dr2, list):
	       args.dr2 = [args.dr2]
    if not isinstance(args.lr, list):
	       args.lr = [args.lr]

    if args.dummy:
        if args.Xval is not None:
            Xtr=pd.read_csv(args.Xtr,sep='\s+')
            Xval=pd.read_csv(args.Xval,sep='\s+')
            All_X = pd.concat([Xtr,Xval])
            All_X = All_X.round(decimals=0)
            #All_X = All_X.apply(pd.to_numeric)
            le = OneHotEncoder(sparse=False)
            label_encoder = LabelEncoder()
            X_enco = All_X.apply(label_encoder.fit_transform)
            onehot_encoder = OneHotEncoder(sparse=False)
            All_X_oh = onehot_encoder.fit_transform(X_enco)
            X_tr = All_X_oh[0:Xtr.shape[0], :]
            X_vl = All_X_oh[Xtr.shape[0]:All_X_oh.shape[0], ]
        else:
            X_tr = pd.read_csv(args.Xtr,sep='\s+').round(decimals=0)
            #X_tr = X_tr.apply(pd.to_numeric)
            le = OneHotEncoder(sparse=False)
            label_encoder = LabelEncoder()
            X_enco = X_tr.apply(label_encoder.fit_transform)
            onehot_encoder = OneHotEncoder(sparse=False)
            X_tr = onehot_encoder.fit_transform(X_enco)
            y_tr = pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)
            X_tr, X_vl, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2)
    else:
        if args.Xval is not None:
            X_tr= pd.read_csv(args.Xtr,sep='\s+').apply(pd.to_numeric)
            X_vl= pd.read_csv(args.Xval,sep='\s+').apply(pd.to_numeric)
        else:
            X_tr=pd.read_csv(args.Xtr,sep='\s+').apply(pd.to_numeric)
            y_tr=pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)
            X_tr, X_vl, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2)
    if args.categorical is False:
        if args.scale is True:
            if args.cat is not None:
                if args.yval is not None:
                    y_tr = pd.read_csv(args.ytr,sep='\s+')
                    y_val = pd.read_csv(args.yval,sep='\s+').apply(pd.to_numeric)
                    cat = args.cat
                    loc = y_tr.columns.get_loc(cat)
                    Factors=np.unique(y_tr.iloc[:, loc])
                    scaled=pd.DataFrame([])
                    for i in Factors:
                        M = y_tr.iloc[np.where(y_tr.iloc[:, loc] == i)]
                        M = M.iloc[:, :-1].apply(scale)
                        scaled=scaled.append(M)

                    y_val = y_val.apply(scale)
                    scaled.columns=y_tr.columns[:-1]
                    y_val.columns=y_tr.columns[:-1]
                    y_tr = scaled

                else:
                    y_tr = pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)
                    y_tr = y_tr.apply(scale)
                    y_val = y_val.apply(scale)
            else:
                if args.yval is not None:
                    y_tr = pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)
                    y_val = pd.read_csv(args.yval,sep='\s+').apply(pd.to_numeric)
                    y_val = y_val.apply(scale)
                    y_tr = y_tr.apply(scale)
                else:
                    y_tr = pd.read_csv(args.ytr).apply(pd.to_numeric)
                    y_tr = y_tr.apply(scale)
                    y_val = y_val.apply(scale)
        else:
            if args.yval is not None:
                y_tr = pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)
                y_val = pd.read_csv(args.yval,sep='\s+').apply(pd.to_numeric)
            else:
                y_tr = pd.read_csv(args.ytr,sep='\s+').apply(pd.to_numeric)

        if args.trait is not None:
            trait= args.trait
            loc=y_tr.columns.get_loc(trait)
            y_tr=y_tr.iloc[:,loc]
            y_val=y_val.iloc[:,loc]

        else:
            trait=None
        y_tr = np.asarray(y_tr)
        y_val = np.asarray(y_val)
    else:
        if args.yval is not None:
            y_tr = pd.read_csv(args.ytr,sep='\s+')
            y_val = pd.read_csv(args.yval,sep='\s+')
        else:
            y_tr = pd.read_csv(args.ytr,sep='\s+')
        if args.trait is not None:
            trait = args.trait
            loc = y_tr.columns.get_loc(trait)
            y_tr = y_tr.iloc[:, loc]
            y_val = y_val.iloc[:, loc]
            y_tr = np.asarray(y_tr)
            y_val = np.asarray(y_val)
            encoder = LabelEncoder()
            encoder.fit(y_tr)
            y_tr = encoder.transform(y_tr)
            y_tr = to_categorical(y_tr)
            encoder = LabelEncoder()
            encoder.fit(y_val)
            y_val = encoder.transform(y_val)
            y_val=to_categorical(y_val)
        else:
            if len(y_tr.columns) > 1:
                sys.exit("To categorical Values,Only one trait is allowed ")
            else:
                y_tr = np.asarray(y_tr)
                y_val = np.asarray(y_val)
                encoder = LabelEncoder()
                encoder.fit(y_tr)
                y_tr = encoder.transform(y_tr)
                y_tr = to_categorical(y_tr)
                encoder = LabelEncoder()
                encoder.fit(y_val)
                y_val = encoder.transform(y_val)
                y_val = to_categorical(y_val)

    X_vl=np.asarray(X_vl)
    X_tr=np.asarray(X_tr)

    main=os.getcwd()
    diro=args.output

###to cat


    if args.cnn:
        if trait is not None:
            run_cnn_main(X_tr,y_tr,X_vl,y_val,output=diro,main=main,prop=args.gridProp,trait=trait,cat=args.categorical,lr=args.lr,dr_1=args.dr1,dr_2=args.dr2,reg_1=args.reg1,reg_2=args.reg2,reg_3=args.reg3,nconv=args.nconv,act_1=args.act1,act_2=args.act2,hn=args.hn,ps=args.ps,hl=args.hl,ks=args.ks,ns=args.ns,epochs=args.epochs,nf=args.nfilters,op=args.optimizer,bs=args.bs)
        else:
            run_cnn_main(X_tr, y_tr, X_vl, y_val, output=diro, main=main, prop=args.gridProp,trait=None,cat=args.categorical,lr=args.lr,dr_1=args.dr1,dr_2=args.dr2,reg_1=args.reg1,reg_2=args.reg2,reg_3=args.reg3,nconv=args.nconv,act_1=args.act1,act_2=args.act2,hn=args.hn,ps=args.ps,hl=args.hl,ks=args.ks,ns=args.ns,epochs=args.epochs,nf=args.nfilters,op=args.optimizer,bs=args.bs)
    if args.mlp:
        if trait is not None:
            run_mlp_main(X_tr,y_tr,X_vl,y_val,output=diro,main=main,prop=args.gridProp,trait=trait,cat=args.categorical,lr=args.lr,dr_1=args.dr1,dr_2=args.dr2,reg_1=args.reg1,reg_2=args.reg2,act_1=args.act1,hn=args.hn,hl=args.hl,epochs=args.epochs,op=args.optimizer,bs=args.bs,N_neurons_FL=args.N_neurons_FL)
        else:
            run_mlp_main(X_tr, y_tr, X_vl, y_val, output=diro, main=main, prop=args.gridProp,trait=None,cat=args.categorical,lr=args.lr,dr_1=args.dr1,dr_2=args.dr2,reg_1=args.reg1,reg_2=args.reg2,act_1=args.act1,hn=args.hn,hl=args.hl,epochs=args.epochs,op=args.optimizer,bs=args.bs,N_neurons_FL=args.N_Neurons_FL)
    if args.ridge:
        if args.categorical is True:
            sys.exit("ridge only allowing to continous outputs")
        else:
            run_ridge_main(X_tr,y_tr,X_vl,y_val,output=diro,main=main)
    if args.lasso:

        if args.categorical is True:
            sys.exit("ridge only allowing to continous outputs")
        else:
            run_lasso_main(X_tr,y_tr,X_vl,y_val,output=diro,main=main)
