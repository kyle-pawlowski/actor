#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:15:09 2025

@author: kyle
"""

from models import Mars, ParameterModel
import numpy as np

models = [Mars()]

datafile = ''

x, y = ParameterModel.parse_data(datafile)
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

