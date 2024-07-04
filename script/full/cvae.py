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

#%% cvae

# This is the size of our encoded representations
input_size_x= train_data_x.shape[1]
output_size_y=train_data_y.shape[1]
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
cvae = CVAE(y_dim=output_size_y, h_dim1=hidden_dim1, h_dim2=hidden_dim2, h_dim3= hidden_dim3, h_dim4=hidden_dim4,  z_dim=z_dim, x_dim=input_size_x,dropout_prob=0.3)
### move model to gpu 
print("move model to gpu",flush=True)
cvae= cvae.to(device) # send weights to GPU. Do this BEFORE defining Optimizer

# return reconstruction error + KL divergence losses
def loss_function(predicted, target, mu, log_var):
    BCE = torch.sum(torch.square(predicted-target))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss= torch.mean(BCE+KLD)
    return total_loss


batch_size=8

train_dataset = TensorDataset(train_data_y,train_data_x)
val_dataset = TensorDataset(val_data_y,val_data_x)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

train_dataset_size = len(train_dataloader.dataset)
val_dataset_size = len(val_dataloader.dataset)

# epochs
epochs = 800

optimizer = torch.optim.SGD(cvae.parameters(),lr=0.0001)

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
        recon_batch, mu, log_var  = cvae(Y=y_batch,X=x_batch)
        #calculating loss
        cost = loss_function(predicted=recon_batch, target=y_batch, mu=mu, log_var=log_var )
   
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
            recon_batch_val, mu_val, log_var_val  = cvae(Y=y_batch_val,X=x_batch_val)
            # calculate loss 
            cost = loss_function(predicted=recon_batch_val, target=y_batch_val, mu=mu_val, log_var=log_var_val )
            loss, current = cost.item(), (id_batch + 1)* len(x_batch_val)
            validation_loss.append(loss* len(y_batch_val))
    validation_loss=np.sum(validation_loss) / val_dataset_size
    costval_validation.append(validation_loss)
    print(f"validation loss: {validation_loss:>7f} ",flush=True)    


 
#%% save the model


PATH = "cvae_time.pt"
# Save
torch.save(cvae, PATH)

print("training stat done",flush=True)

#%% get the calibration parameters
cvae.eval()
ensemble_N=500
predictions_ensemble_validation=np.zeros((ensemble_N,val_data_y.shape[0],val_data_y.shape[1]))

for i in range(0,ensemble_N,1):
    Z= torch.randn(val_data_x.shape[0],z_dim)
    val_data_x= val_data_x.to(device)
    Z= Z.to(device)
    predictions_val = cvae.decoder(Z,val_data_x)
    predictions_val=torch.Tensor.detach(predictions_val).cpu()
    predictions_val=predictions_val.numpy()
    predictions_ensemble_validation[i,:,:] = predictions_val
    print(i)

print("val ensemble done",flush=True)
prediictions_ensemble_val_mean= np.mean( predictions_ensemble_validation,axis=0 )


S_square_val=np.zeros((prediictions_ensemble_val_mean.shape[1]))
intercept_val=np.zeros((prediictions_ensemble_val_mean.shape[1]))
slope_val=np.zeros((prediictions_ensemble_val_mean.shape[1]))

for idx in range(0,prediictions_ensemble_val_mean.shape[1],1):
    regr = LinearRegression()
    regr.fit(prediictions_ensemble_val_mean[:,idx].reshape(-1,1), val_data_y[:,idx] )
    prediction_from_linear = regr.predict(prediictions_ensemble_val_mean[:,idx].reshape(-1,1),)
    # Calculate the prediction interval
    SSE = np.sum( (val_data_y[:,idx].numpy()- prediction_from_linear)**2 )
    S_square = SSE/ ( prediction_from_linear.shape[0]-2 )
    S_square_val[idx]=S_square
    intercept= regr.intercept_
    slope= regr.coef_
    intercept_val[idx]=intercept
    slope_val[idx]=slope[0]


#%% test data 

ensemble_N=500
predictions_ensemble=np.zeros((ensemble_N,test_data_y.shape[0],test_data_y.shape[1]))


for i in range(0,ensemble_N,1):
    Z= torch.randn(test_data_x.shape[0],z_dim)
    test_data_x= test_data_x.to(device)
    Z= Z.to(device)
    predictions_test = cvae.decoder(Z,test_data_x)
    predictions_test=torch.Tensor.detach(predictions_test).cpu()
    predictions_test=predictions_test.numpy()
    predictions_ensemble[i,:,:] = predictions_test
    print(i)

print("test stat",flush=True)


test_data_x=torch.Tensor.detach(test_data_x).cpu()


#%% delete variable except for..
costval_training = np.array(costval_training)

costval_validation = np.array(costval_validation)
import pickle
result = {'test_data_x':test_data_x,'prediictions_ensemble_val_mean':prediictions_ensemble_val_mean,'costval_training':costval_training, 'costval_validation':costval_validation, 'S_square_val':S_square_val, 'intercept_val':intercept_val,'slope_val':slope_val,
          'predictions_ensemble':predictions_ensemble, 'test_data_y':test_data_y, 'test_name':test_name,'train_name':train_name,'grid_subset_output_cities':grid_subset_output_cities}


with open('cvae.pickle', 'wb') as f:
    pickle.dump(result, f)
