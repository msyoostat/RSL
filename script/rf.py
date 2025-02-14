#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import pyreadr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import random

## seed number is stated in "RandomForestRegressor"
#%% load data

result = pyreadr.read_r('rsl_full_data.RData') 
print(result.keys(),flush=True)
test_data_x= result['test.data.lithk']
train_data_x= result['train.data.lithk']
val_data_x= result['val.data.lithk']

train_data_y= result['train.data.y']
test_data_y= result['test.data.y']
val_data_y= result['val.data.y']
grid_subset_output_cities= result['grid.subset.output.cities']

train_name= result['train_name']
test_name= result['test_name']
val_name= result['val_name']
del(result)

#%% change pandas to numpy for speed 

test_data_x=test_data_x.to_numpy()
test_data_y=test_data_y.to_numpy()

train_data_x=train_data_x.to_numpy()
train_data_y=train_data_y.to_numpy()

val_data_x=val_data_x.to_numpy()
val_data_y=val_data_y.to_numpy()


#%% normalization of x

scaler_x = StandardScaler()
scaler_x.fit(train_data_x)

train_data_x= scaler_x.fit_transform(train_data_x)
val_data_x= scaler_x.fit_transform(val_data_x)
test_data_x= scaler_x.fit_transform(test_data_x)

#%% pca of x

pca = PCA(n_components=250)
 
train_data_x = pca.fit_transform(train_data_x) 

val_data_x = pca.transform(val_data_x)

test_data_x = pca.transform(test_data_x)

# np.savetxt('train_data_x.txt', train_data_x, delimiter=',')
# np.savetxt('val_data_x.txt', val_data_x, delimiter=',')
# np.savetxt('test_data_x.txt', test_data_x, delimiter=',')

explained_variance = pca.explained_variance_ratio_
sum(explained_variance)


train_data_x = torch.from_numpy(train_data_x)
val_data_x = torch.from_numpy(val_data_x)
test_data_x = torch.from_numpy(test_data_x)

train_data_y = torch.from_numpy(train_data_y)
val_data_y = torch.from_numpy(val_data_y)

train_data_x=train_data_x.type('torch.FloatTensor')
test_data_x = test_data_x.type('torch.FloatTensor')
val_data_x = val_data_x.type('torch.FloatTensor')

train_data_y=train_data_y.type('torch.FloatTensor')
val_data_y=val_data_y.type('torch.FloatTensor')

#%% random forest fitting

rf = RandomForestRegressor(random_state=42)

# Train the model on training data
rf.fit(train_data_x, train_data_y[:,0])
rf.predict(train_data_x)
model_list ={0:rf}
q= train_data_y.shape[1]
for i in range(1,q,1):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(train_data_x, train_data_y[:,i]);
    model_list[i]=rf
    print(i)


#%% validation

predictions_val=np.zeros((val_data_y.shape[0],val_data_y.shape[1]))

for i in range(0,q,1):
    predictions_val[:,i]=model_list[i].predict(val_data_x)
    print(i)

#%% calibration
S_square_val=np.zeros((predictions_val.shape[1]))
intercept_val=np.zeros((predictions_val.shape[1]))
slope_val=np.zeros((predictions_val.shape[1]))

for idx in range(0,predictions_val.shape[1],1):
    regr = LinearRegression()
    regr.fit(predictions_val[:,idx].reshape(-1,1), val_data_y[:,idx] )
    prediction_from_linear = regr.predict(predictions_val[:,idx].reshape(-1,1),)
    # Calculate the prediction interval
    SSE = np.sum( (val_data_y[:,idx].numpy()- prediction_from_linear)**2 )
    S_square = SSE/ ( prediction_from_linear.shape[0]-2 )
    S_square_val[idx]=S_square
    intercept= regr.intercept_
    slope= regr.coef_
    intercept_val[idx]=intercept
    slope_val[idx]=slope[0]



#%% test data 

predictions_test=np.zeros((test_data_y.shape[0],test_data_y.shape[1]))

for i in range(0,q,1):
    predictions_test[:,i]=model_list[i].predict(test_data_x)
    print(i)


print("test stat",flush=True)


#%% delete variable except for..
import pickle
result = {'predictions_val':predictions_val,'S_square_val':S_square_val, 'intercept_val':intercept_val,'slope_val':slope_val,
          'predictions_test':predictions_test, 'test_data_y':test_data_y, 'test_name':test_name,'train_name':train_name,'grid_subset_output_cities':grid_subset_output_cities}


with open('rf.pickle', 'wb') as f:
    pickle.dump(result, f)
