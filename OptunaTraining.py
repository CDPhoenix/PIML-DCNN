# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:59:12 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY

"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import numpy as np
from model import Model
from OptunaModel import optunaModel


class Container():
    
    def __init__(self,network,train_data_X,train_data_Y,train_backup,
                 test_data_X,test_data_Y,test_backup,batch_size,sizes,epoch,clip=5.0,optModel=False):
        
        self.model = network
        self.train_X = train_data_X
        self.train_Y = train_data_Y
        self.train_backup = train_backup
        self.test_X = test_data_X
        self.test_Y = test_data_Y
        self.test_backup = test_backup
        self.batch_size = batch_size
        self.wholeDataSize = sizes
        self.clip = clip
        self.n_epochs = epoch
        self.optModel = optModel
        
        if self.optModel == True:
            self.width = self.model.width
            self.depth = self.model.depth
            self.width3d = self.model.width3d
            self.depth3d = self.model.depth3d
            self.height3d = self.model.height3d
            self.batch_size = self.model.channels
            self.kernel_size = self.model.kernel_size
            self.padding = self.model.padding
            self.stride = self.model.stride
        


    def train(self,optimizer,criterion):
        
        self.model.train()
        
        final_loss = 0
        
        clip = 5.0
        
        for i in range(int(self.wholeDataSize[1]/self.batch_size)):
            data = self.train_X[:,i*self.batch_size:(i+1)*self.batch_size,:,:]
            Temp_array = data.unsqueeze(1)
            optimizer.zero_grad() 
            output,_ = self.model(Temp_array,data,self.train_backup)#,PowerInput)
            loss = torch.sqrt(criterion(output,self.train_Y))
            final_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
        
        return final_loss
    
    
    def test(self,criterion):
        
        self.model.eval()
        
        Error = []
        
        with torch.no_grad():
            
            for i in range(int(self.wholeDataSize[1]/self.batch_size)):
                data = self.test_X[:,i*self.batch_size:(i+1)*self.batch_size,:,:]
                Temp_array = data.unsqueeze(1)
                output,_ = self.model(Temp_array,data,self.test_backup)#,PowerInput
                loss = criterion(output,self.test_Y)/torch.mean(self.test_Y)
                Error.append(loss.cpu().detach().numpy())
            
            #Error_avg = sum(Error)/len(Error)
            
            Error_numpyMean = np.mean(Error)
        
        #accuracy = 1-Error_numpyMean
        
        return Error_numpyMean#accuracy
    
    def objective(self,trial):
        
        if self.optModel==True:

            
            num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
            num_conv_layers_3d = trial.suggest_int("num_conv_layers_3d",1,3)
            num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16))
                           for i in range(num_conv_layers)]
            num_filters_3d = [int(trial.suggest_discrete_uniform("num_filter_3d_"+str(i), 1,3,1))
                           for i in range(num_conv_layers_3d)]                           
            num_neurous = trial.suggest_int("num_neurons", 16, 96, 16)
            
            newModel = optunaModel(trial,self.width,self.depth,self.width3d,self.height3d,self.depth3d,
                                   self.batch_size,num_filters,num_filters_3d,num_conv_layers, 
                                   num_conv_layers_3d,self.kernel_size,self.padding,
                                   self.stride,num_neurous).cuda()
            
            self.model = newModel
            
        criterion = nn.MSELoss()
        MSE = criterion.cuda()
        MAE = nn.L1Loss() #Absolute Error
        MAE = MAE.cuda()
        #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-6, 2*1e-3, log=True)
        #optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
        # Training of the model
        for epoch in range(self.n_epochs):
            
            self.train(optimizer,MSE)  # Train the model
            
        accuracy = self.test(MAE)   # Evaluate the model
        print(accuracy)
        
        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
                
        
        return accuracy
    
    


