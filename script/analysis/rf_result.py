#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import pyreadr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from scipy import stats
import pickle



with open("/Users/myungsooyoo/Desktop/My_stuff/research/emulator/python/darwin/final/previous/previous_final/result/rf.pickle", 'rb') as f:
    result = pickle.load(f)

print(result.keys(),flush=True)
S_square_val= result['S_square_val']
predictions_test= result['predictions_test']
test_data_y= result['test_data_y']
test_name= result['test_name']
train_name= result['train_name']
grid_subset_output_cities= result['grid_subset_output_cities']
slope_val = result['slope_val']
intercept_val = result['intercept_val']
predictions_val=result['predictions_val']


#%% stats... citywise average length
length_all= np.zeros( (predictions_test.shape[0],predictions_test.shape[1]) )
for i in range(0,length_all.shape[1],1):
    for j in range(0, length_all.shape[0],1): 
        print(i)
        ssx= np.sum( np.square(predictions_val[:,i] - np.mean(predictions_val[:,i]) ) )
        nominator= np.square( predictions_test[j,i] - np.mean(predictions_val[:,i]) )
        temp= S_square_val[i] * (1+ (1/predictions_test.shape[0]) + ( nominator/ssx ) )
        temp= np.sqrt(temp)
        t_upper=stats.t(df=predictions_test.shape[0]-2).ppf((0.925))
        # length
        length_all[j,i]= (t_upper * temp) *2

length_ave = np.mean(length_all,0)
for i in range(0, length_ave.shape[0],1):
    #print(i)
    print( np.round(length_ave[i],6) )


#%% citiwise average coverage rate
prediction_test_point =np.zeros( (predictions_test.shape[0],predictions_test.shape[1]) ) 
for i in range(0,prediction_test_point.shape[1],1):
    print(i)
    for j in range(0,prediction_test_point.shape[0],1):
        prediction_test_point[j,i] = intercept_val[i]+predictions_test[j,i] *slope_val[i]

prediictions_ensemble_upper=prediction_test_point + length_all/2
prediictions_ensemble_lower=prediction_test_point - length_all/2

for i in range(0,length_ave.shape[0],1):
    #print(i)
    coverage = (test_data_y[:,i] <=prediictions_ensemble_upper[:,i]) & (test_data_y[:,i] >=prediictions_ensemble_lower[:,i])

    print(    np.round( np.count_nonzero(coverage)/ test_data_y.shape[0] ,6)  )

#%% stats...citiwise.. average R squared... 
for i in range(0,length_ave.shape[0],1):
    regr = LinearRegression()
    regr.fit(prediction_test_point[:,i].reshape(-1,1), test_data_y[:,i] )
    r_squared = regr.score(prediction_test_point[:,i].reshape(-1,1),  test_data_y[:,i])
   # print(i)
    print( np.round( r_squared,6 ) )
#%% plot

grid_subset_output_cities=grid_subset_output_cities.to_numpy()
idx= 3
xmin= np.min(test_data_y[:,idx])
xmax= np.max(test_data_y[:,idx])

ymin= np.min(prediction_test_point[:,idx])
ymax= np.max(prediction_test_point[:,idx])

y_lower_min= np.min( prediictions_ensemble_lower[:,idx] )
y_upper_max= np.max( prediictions_ensemble_upper[:,idx] )

lim_min=np.min([xmin,ymin,y_lower_min])
lim_max=np.max([xmax,ymax,y_upper_max])
regr = LinearRegression()

regr.fit(prediction_test_point[:,idx].reshape(-1,1), test_data_y[:,idx] )
r_squared = regr.score(prediction_test_point[:,idx].reshape(-1,1),  test_data_y[:,idx])
print(r_squared)
prediction_from_linear = regr.predict(prediction_test_point[:,idx].reshape(-1,1),)

plt.figure()
plt.scatter(prediction_test_point[:,idx], test_data_y[:,idx] ,color="black")  
plt.xlim([lim_min-0.01, lim_max+0.01])
plt.ylim([lim_min-0.01, lim_max+0.01])
plt.plot(prediction_test_point[:,idx],prediction_from_linear,color="blue")
plt.axline([lim_min-0.01, lim_min-0.01], [lim_max+0.01, lim_max+0.01],color="red")
#plt.plot(prediictions_ensemble_lower[:,idx],prediction_from_linear,color="brown")
plt.title("cities= "+ str(grid_subset_output_cities[idx,0])+",R square="+str(round(r_squared,3)),fontsize=20)

plt.xlabel('predicted')
plt.ylabel('Actual')
plt.plot(prediction_test_point[:,idx],prediictions_ensemble_lower[:,idx],color="green")
plt.plot(prediction_test_point[:,idx],prediictions_ensemble_upper[:,idx],color="green")



