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
import time
import torch.nn.functional as F




with open("/Users/myungsooyoo/Desktop/My_stuff/research/emulator/python/darwin/final/previous/previous_final/result/fnn.pickle", 'rb') as f:
    result = pickle.load(f)

print(result.keys(),flush=True)
test_data_x=result['test_data_x']
costval_training = result['costval_training']
costval_validation = result['costval_validation']
S_square_val= result['S_square_val']
predictions_test= result['predictions_test']
test_data_y= result['test_data_y']
test_name= result['test_name']
train_name= result['train_name']
grid_subset_output_cities= result['grid_subset_output_cities']
slope_val = result['slope_val']
intercept_val = result['intercept_val']
predictions_val=result['predictions_val']

#%% check the test loss
plt.plot(torch.tensor(costval_training).numpy())
plt.plot(torch.tensor(costval_validation).numpy())


#%% citywise average length

length_all= np.zeros( (predictions_test.shape[0],predictions_test.shape[1]) )
for i in range(0,length_all.shape[1],1):
    for j in range(0, length_all.shape[0],1): 
        print(i)
        ssx= np.sum( np.square(predictions_val[:,i] - np.mean(predictions_val[:,i]) ) )
        nominator= np.square( predictions_test[j,i] - np.mean(predictions_val[:,i]) )
        temp= S_square_val[i] * (1+ (1/predictions_test.shape[0]) + ( nominator/ssx ) )
        temp= np.sqrt(temp)
        t_upper=stats.t(df=predictions_test.shape[0]-2).ppf((0.975)) 
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

plt.figure(dpi=200)
#plt.figure()
plt.scatter(prediction_test_point[:,idx], test_data_y[:,idx] ,color="black")  
plt.xlim([lim_min-0.01, lim_max+0.01])
plt.ylim([lim_min-0.01, lim_max+0.01])
plt.plot(prediction_test_point[:,idx],prediction_from_linear,color="blue")
plt.axline([lim_min-0.01, lim_min-0.01], [lim_max+0.01, lim_max+0.01],color="red")
#plt.plot(prediictions_ensemble_lower[:,idx],prediction_from_linear,color="brown")
plt.title("Balboa RSL",fontsize=20)

plt.xlabel('predicted')
plt.ylabel('actual')
plt.plot(prediction_test_point[:,idx],prediictions_ensemble_lower[:,idx],color="green")
plt.plot(prediction_test_point[:,idx],prediictions_ensemble_upper[:,idx],color="green")



#%% computational time
class net(nn.Module):
  def __init__(self,input_size,output_size,h_dim1,h_dim2,h_dim3,h_dim4,dropout_prob):
    super(net,self).__init__() ## inheritence
    self.l1 = torch.nn.Linear(input_size,h_dim1)
    self.l2 = torch.nn.Linear(h_dim1,h_dim2)
    self.l3 = torch.nn.Linear(h_dim2,h_dim3)
    self.l4 = torch.nn.Linear(h_dim3,h_dim4)
    self.l5 = torch.nn.Linear(h_dim4,output_size)
    #dropout
    self.dropout = nn.Dropout(p=dropout_prob)

  def forward(self,x):
    output = F.relu(self.l1(x))
    output = self.dropout(output)
    output = F.relu(self.l2(output))
    output = self.dropout(output)
    output = F.relu(self.l3(output))
    output = self.dropout(output)
    output = F.relu(self.l4(output))
    output = self.dropout(output)
    output = self.l5(output)
    return output


cvae = torch.load("/Users/myungsooyoo/Desktop/My_stuff/research/emulator/python/darwin/final/previous/previous_final/result/fnn_time.pt",map_location="cpu")
cvae.eval()

execution_times=[]

for j in range(1000):
    start_time= time.time()
    predictions_test3 = cvae(test_data_x)
    end_time=time.time()
    elapsed_time= end_time-start_time
    execution_times.append(elapsed_time)


prediction_test_point =np.zeros( (predictions_test.shape[0],predictions_test.shape[1]) ) ## adjusted by linear regression    
execution_times_with_cali=[]

for k in range(1000):
    start_time= time.time()
    predictions_test3 = cvae(test_data_x)
    for i in range(0,prediction_test_point.shape[1],1):
        for j in range(0,prediction_test_point.shape[0],1):
            prediction_test_point[j,i] = intercept_val[i]+predictions_test[j,i] *slope_val[i]
    end_time=time.time()
    elapsed_time= end_time-start_time
    execution_times_with_cali.append(elapsed_time)


