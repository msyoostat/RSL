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


torch.manual_seed(42)
print(torch.cuda.is_available(),flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


#%% load data

result = pyreadr.read_r('data_all_cities_py_15june.RData') 
#result = pyreadr.read_r("/Users/myungsooyoo/Desktop/My_stuff/research/emulator/AML_project_important-20230821T155429Z-001/AML_project_important/zero_out/data_all_cities_py_15june.RData") 
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

#%% FNN

input_size_x= train_data_x.shape[1]
output_size_y=train_data_y.shape[1]
hidden_dim1 = 3200
hidden_dim2 = 1600
hidden_dim3 = 800
hidden_dim4 = 400


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



# build model
cvae = net(input_size= input_size_x, output_size=output_size_y, h_dim1=hidden_dim1, h_dim2=hidden_dim2, h_dim3= hidden_dim3, h_dim4=hidden_dim4,dropout_prob=0.3)
### move model to gpu 
print("move model to gpu",flush=True)
cvae= cvae.to(device) # send weights to GPU. Do this BEFORE defining Optimizer

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

batch_size=8

train_dataset = TensorDataset(train_data_y,train_data_x)
val_dataset = TensorDataset(val_data_y,val_data_x)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

train_dataset_size = len(train_dataloader.dataset)
val_dataset_size = len(val_dataloader.dataset)

# epochs
epochs = 750

optimizer = torch.optim.SGD(cvae.parameters(),lr=0.001)

# costvalue
costval_training = []
costval_validation= []

for j in range(epochs):
    print(f"Epoch {j + 1}\n-------------------------------")
    training_loss=[]
    validation_loss=[]
    # Loop over batches in an epoch using DataLoader
    cvae.train()
    for id_batch, (y_batch,x_batch) in enumerate(train_dataloader):    
        #prediction
        x_batch= x_batch.to(device)
        y_batch= y_batch.to(device)
        Y_hat = cvae(x=x_batch)
        #calculating loss
        cost = RMSELoss(yhat=Y_hat, y=y_batch )
   
        #backprop
        optimizer.zero_grad()
        cost.backward() 
        optimizer.step() 
        loss, current = cost.item(), (id_batch + 1)* len(x_batch)
        training_loss.append(loss*len(x_batch))
    training_loss=np.sum(training_loss)/ train_dataset_size
    costval_training.append(training_loss)
    print(f"training loss: {training_loss:>7f} ",flush=True)
    # validation loop
    cvae.eval()
    with torch.no_grad():
        for id_batch, (y_batch_val,x_batch_val) in enumerate(val_dataloader):
            x_batch_val= x_batch_val.to(device)
            y_batch_val= y_batch_val.to(device)
            Y_hat_val = cvae(x=x_batch_val)
            # calculate loss 
            cost = RMSELoss(yhat=Y_hat_val, y=y_batch_val )
            loss, current = cost.item(), (id_batch + 1)* len(x_batch_val)
            validation_loss.append(loss* len(y_batch_val))
    validation_loss=np.sum(validation_loss) / val_dataset_size
    costval_validation.append(validation_loss)
    print(f"validation loss: {validation_loss:>7f} ",flush=True)    


#%% save the model


PATH = "fnn_time.pt"
# Save
torch.save(cvae, PATH)
 
print("training stat done",flush=True)

#%% get the calibration parameters
cvae.eval()
val_data_x= val_data_x.to(device)
predictions_val = cvae(val_data_x)
predictions_val=torch.Tensor.detach(predictions_val).cpu()
predictions_val=predictions_val.numpy()

print("validation done",flush=True)
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

test_data_x= test_data_x.to(device)
predictions_test = cvae(test_data_x)
predictions_test=torch.Tensor.detach(predictions_test).cpu()
predictions_test=predictions_test.numpy()


print("test stat",flush=True)


test_data_x=torch.Tensor.detach(test_data_x).cpu()

#%% delete variable except for..
costval_training = np.array(costval_training)

costval_validation = np.array(costval_validation)
import pickle
result = {'test_data_x':test_data_x,'predictions_val':predictions_val,'costval_training':costval_training, 'costval_validation':costval_validation, 'S_square_val':S_square_val, 'intercept_val':intercept_val,'slope_val':slope_val,
          'predictions_test':predictions_test, 'test_data_y':test_data_y, 'test_name':test_name,'train_name':train_name,'grid_subset_output_cities':grid_subset_output_cities}


with open('fnn.pickle', 'wb') as f:
    pickle.dump(result, f)
