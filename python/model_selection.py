#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:15:09 2025

@author: kyle
""" 

from models import Mars, ParameterModel, DNN, RNNParam
import numpy as np
from itertools import chain
import sys
from multiprocessing import Process

yfile = 'data/ExplorerRun.0_lna_good.csv'
xfolder = 'data/lna_x'

if __name__ == "__main__":
    x, y = ParameterModel.parse_data(xfolder, yfile)
    n = x.shape[0]
    input_d = x.shape[1]
    output_d = y.shape[1]
    quarter_d = int(input_d/4)
    eighth_d = int(input_d/8)
    sixteenth_d = int(input_d/16)
    thirtysec_d = int(input_d/32)

    def Mars_model_gen():
        for max_degree in range(1, 4):
            for max_terms in np.logspace(2, 4, 3):
                yield Mars(max_terms=max_terms, max_degree=max_degree)

    def DNN_model_gen():
        for alpha in np.logspace(-3, -1, 3):
            for lr in np.logspace(-3, -1, 3):
                yield DNN(output_d, (quarter_d, eighth_d, sixteenth_d, thirtysec_d), alpha=alpha, learning_rate=lr, max_iter=10000)
                yield DNN(output_d, (sixteenth_d, thirtysec_d), lr, max_iter=10000)
            
    def RNN_model_gen():
        for hidden in [quarter_d, eighth_d, sixteenth_d, thirtysec_d]:
            for lr in np.logspace(-3, -1, 6):
                yield RNNParam(output_d, hidden, input_d, learning_rate=lr, epochs=10000)
            

    if 'mars' in str(sys.argv[1]).lower():
        models = Mars_model_gen()
    elif 'dnn' in str(sys.argv[1]).lower():
        models = DNN_model_gen()
    elif 'rnn' in str(sys.argv[1]).lower():
        models = RNN_model_gen() 
    else:
        models = chain(Mars_model_gen(), DNN_model_gen(), RNN_model_gen())                  

    frac_train = 0.8
    num_train = int(len(x) * 0.8)
    num_val = len(x) - num_train
    xtrain = x[0:num_train]
    ytrain = y[0:num_train]
    xval = x[num_train:]
    yval = y[num_train:]

    best_error = 10000
    best_model = None
    #errors = np.zeros(len(models), 1)
    for model in models:
        model.train_data(xtrain, ytrain)
        train_error = model.test_data(xtrain, ytrain)*100
        error = model.test_data(xval, yval)*100
        print('================')
        print(f'Model {model}\nTrain Error: {train_error:0.2f}% Test Error: {error:0.2f}%')
        print('================')
        if error < best_error:
            best_error = error
            best_model = model

    print('*****************')
    print('Best model is: ' + str(best_model))
    print(f'Error of: {best_error:0.2f}%')
    print('*****************')

