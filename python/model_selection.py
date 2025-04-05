#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:15:09 2025

@author: kyle
"""

from models import Mars, ParameterModel, DNN, RNNParam
import numpy as np
from itertools import chain

datafile = ''
x, y = ParameterModel.parse_data(datafile)
n = x.shape[0]
input_d = x.shape[1]
output_d = y.shape[1]
quarter_d = int(input_d/4)
eighth_d = int(input_d/8)
sixteenth_d = int(input_d/16)
thirtysec_d = int(input_d/32)

def Mars_model_gen():
    for max_degree in range(2, 6):
        for max_terms in np.logspace(2, 4, 3):
            yield Mars(max_terms=max_terms, max_degree=max_degree)

def DNN_model_gen():
    for lr in np.logspace(-3, -1, 3):
        yield DNN(output_d, (quarter_d, eighth_d, sixteenth_d, thirtysec_d), lr)
        yield DNN(output_d, (sixteenth_d, thirtysec_d), lr)
        
def RNN_model_gen():
    for lr in np.logspace(-3, -1, 3):
        yield RNNParam(output_d, (quarter_d, eighth_d, sixteenth_d, thirtysec_d), input_d)
        yield RNNParam(output_d, (sixteenth_d, thirtysec_d), input_d)
        

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
errors = np.zeros(len(models), 1)
for model in models:
    model.train_data(xtrain, ytrain)
    error = model.test_data(xval, yval)
    if error < best_error:
        best_error = error
        best_model = model

print('Best model is: ' + str(best_model))
print('Error of: ' + str(best_error))

