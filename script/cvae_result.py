#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor
import pyreadr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from scipy import stats
import pickle
import torch.nn.functional as F
import time

with open('/Users/myungsooyoo/Desktop/My_stuff/research/emulator/python/darwin/final/previous/previous_final/result/cvae.pickle', 'rb') as f:
    result = pickle.load(f)

print(result.keys(),flush=True)
test_data_x=result['test_data_x']
costval_training = result['costval_training']
costval_validation = result['costval_validation']
S_square_val= result['S_square_val']
predictions_ensemble= result['predictions_ensemble']
test_data_y= result['test_data_y']
test_name= result['test_name']
train_name= result['train_name']
grid_subset_output_cities= result['grid_subset_output_cities']
slope_val = result['slope_val']
intercept_val = result['intercept_val']
prediction_ensemble_val_mean=result['prediictions_ensemble_val_mean']

#%% check the test loss


plt.plot(torch.tensor(costval_training).numpy())
plt.plot(torch.tensor(costval_validation).numpy())

### check the test loss: ensemble mean
predictions_ensemble.shape
prediictions_ensemble_mean= np.mean( predictions_ensemble,axis=0 )


#%% stats... citywise average length

length_all= np.zeros( (prediictions_ensemble_mean.shape[0],prediictions_ensemble_mean.shape[1]) )
for i in range(0,length_all.shape[1],1):
    for j in range(0, length_all.shape[0],1): 
        print(i)
        ssx= np.sum( np.square(prediction_ensemble_val_mean[:,i] - np.mean(prediction_ensemble_val_mean[:,i]) ) )
        nominator= np.square( prediictions_ensemble_mean[j,i] - np.mean(prediction_ensemble_val_mean[:,i]) )
        temp= S_square_val[i] * (1+ (1/prediction_ensemble_val_mean.shape[0]) + ( nominator/ssx ) )
        temp= np.sqrt(temp)
        t_upper=stats.t(df=prediictions_ensemble_mean.shape[0]-2).ppf((0.925))
        # length
        length_all[j,i]= (t_upper * temp) *2

length_ave = np.mean(length_all,0)
for i in range(0, length_ave.shape[0],1):
    #print(i)
    print( np.round(length_ave[i],6) )

#%% stats...citiwise average coverage rate
prediction_test_point =np.zeros( (prediictions_ensemble_mean.shape[0],prediictions_ensemble_mean.shape[1]) ) ## adjusted by linear regression
for i in range(0,prediction_test_point.shape[1],1):
    print(i)
    for j in range(0,prediction_test_point.shape[0],1):
        prediction_test_point[j,i] = intercept_val[i]+prediictions_ensemble_mean[j,i] *slope_val[i]

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
    #print(i)
    print( np.round( r_squared,6 ) )

#%% plot

### my plot similar to qq plot

grid_subset_output_cities=grid_subset_output_cities.to_numpy()

idx= 3
xmin= np.min(test_data_y[:,idx])
xmax= np.max(test_data_y[:,idx])

ymin= np.min(prediictions_ensemble_mean[:,idx])
ymax= np.max(prediictions_ensemble_mean[:,idx])

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
plt.title("cities= "+ str(grid_subset_output_cities[idx,0])+",R square="+str(round(r_squared,5)),fontsize=20)

plt.xlabel('predicted')
plt.ylabel('Actual')
plt.plot(prediictions_ensemble_mean[:,idx],prediictions_ensemble_lower[:,idx],color="green")
plt.plot(prediictions_ensemble_mean[:,idx],prediictions_ensemble_upper[:,idx],color="green")



#%% computational time

input_size_x= test_data_x.shape[1]
output_size_y=test_data_y.shape[1]
hidden_dim1 = 3200
hidden_dim2 = 1600
hidden_dim3 = 800
hidden_dim4 = 400
z_dim= 200

class CVAE(nn.Module):
    def __init__(self, y_dim, h_dim1, h_dim2, h_dim3, h_dim4, z_dim, x_dim,dropout_prob): # y_dim= output dimension, x_dim= input dimension
        super(CVAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(y_dim + x_dim, h_dim1)
        self.fc12 = nn.Linear(h_dim1, h_dim2)
        self.fc13 = nn.Linear(h_dim2, h_dim3)
        self.fc14 = nn.Linear(h_dim3, h_dim4)
        self.fc21 = nn.Linear(h_dim4, z_dim)
        self.fc22 = nn.Linear(h_dim4, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + x_dim, h_dim1)
        self.fc42 = nn.Linear(h_dim1, h_dim2)
        self.fc43 = nn.Linear(h_dim2, h_dim3)
        self.fc44 = nn.Linear(h_dim3, h_dim4)
        self.fc5 = nn.Linear(h_dim4, y_dim)
        #dropout
        self.dropout = nn.Dropout(p=dropout_prob)
    def encoder(self, Y, X):
        concat_input = torch.cat([Y, X], 1)
        h = F.relu(self.fc1(concat_input))
        h = self.dropout(h)
        h = F.relu(self.fc12(h))
        h = self.dropout(h)
        h = F.relu(self.fc13(h))
        h = self.dropout(h)
        h = F.relu(self.fc14(h))
        h = self.dropout(h)
        return self.fc21(h), self.fc22(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, X):
        concat_input = torch.cat([z, X], 1)
        h = F.relu(self.fc4(concat_input))
        h = self.dropout(h)
        h = F.relu(self.fc42(h))
        h = self.dropout(h)
        h = F.relu(self.fc43(h))
        h = self.dropout(h)
        h = F.relu(self.fc44(h))
        h = self.dropout(h)
        h = self.fc5(h)
        return h
    
    def forward(self, Y, X):
        mu, log_var = self.encoder(Y, X)
        z = self.sampling(mu, log_var)
        return self.decoder(z, X), mu, log_var


# build model
cvae = torch.load("/Users/myungsooyoo/Desktop/My_stuff/research/emulator/python/darwin/final/previous/previous_final/result/cvae_time.pt",map_location="cpu")
cvae.eval()
ensemble_N=1
predictions_ensemble=np.zeros((ensemble_N,test_data_y.shape[0],test_data_y.shape[1]))

execution_times=[]

for j in range(1000):
    start_time= time.time()
    for i in range(0,ensemble_N,1):
        Z= torch.randn(test_data_x.shape[0],z_dim)
        predictions_test = cvae.decoder(Z,test_data_x)
        predictions_test=torch.Tensor.detach(predictions_test).cpu()
        predictions_test=predictions_test.numpy()
        predictions_ensemble[i,:,:] = predictions_test
    end_time=time.time()
    elapsed_time= end_time-start_time
    execution_times.append(elapsed_time)
    
execution_times_with_cali=[]

prediction_test_point =np.zeros( (119,27) ) ## adjusted by linear regression

for j in range(1000):
    start_time= time.time()
    for i in range(0,ensemble_N,1):
        Z= torch.randn(test_data_x.shape[0],z_dim)
        predictions_test = cvae.decoder(Z,test_data_x)
        predictions_test=torch.Tensor.detach(predictions_test).cpu()
        predictions_test=predictions_test.numpy()
        predictions_ensemble[i,:,:] = predictions_test
    prediictions_ensemble_mean= np.mean( predictions_ensemble,axis=0 )
    
    for i in range(0,prediction_test_point.shape[1],1):
        for j in range(0,prediction_test_point.shape[0],1):
            prediction_test_point[j,i] = intercept_val[i]+prediictions_ensemble_mean[j,i] *slope_val[i]
    end_time=time.time()
    elapsed_time= end_time-start_time
    execution_times_with_cali.append(elapsed_time)

