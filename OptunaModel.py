# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:59:43 2023

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
import pandas as pd


class optunaModel(nn.Module):
    
    def __init__(self,trial,width,depth,width3d,height3d,depth3d,batch_size,filters,
                  filters_3d,num_conv_layers, num_conv_layer_3d,kernel_size,padding,stride,num_neurous):
        
        super(optunaModel,self).__init__()
        
        self.width = width
        self.depth = depth
        self.width3d = width3d
        self.depth3d = depth3d
        self.height3d = height3d
        self.batch_size = batch_size
        self.filters = filters
        self.filters_3d = filters_3d
        self.kernel_size = kernel_size
        self.padding  = padding
        self.stride = stride
        
        self.convs = nn.ModuleList([nn.Conv2d(self.batch_size, filters[0],self.kernel_size,
                                              self.stride,self.padding)])
        
        self.convs_3d = nn.ModuleList([nn.Conv3d(1, filters_3d[0], self.kernel_size,
                                                self.stride,self.padding)])
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        
        for i in range(1,num_conv_layers):
            
            self.convs.append(nn.Conv2d(filters[i-1], filters[i],self.kernel_size,self.stride,self.padding))
            self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
            self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.convs.append(nn.ReLU())
        
        for i in range(1,num_conv_layer_3d):
            
            self.convs_3d.append(nn.Conv3d(filters_3d[i-1], filters_3d[i],self.kernel_size,self.stride,self.padding))
            self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
            self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
            self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.convs_3d.append(nn.ReLU())
        
        self.convs2d = nn.Sequential(*self.convs)
        self.convs3d = nn.Sequential(*self.convs_3d)
            
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(int(self.width*self.depth*filters[-1]),num_neurous)
        self.fc1_3d = nn.Linear(int(self.width3d*self.depth3d*self.height3d*filters_3d[-1]),num_neurous)
        
        self.fc2 = nn.Linear(num_neurous,32)
        self.fc2_3d = nn.Linear(num_neurous,32)
        
        self.fc3 = nn.Linear(32,16)
        self.fc3_3d = nn.Linear(32,16)
        
        self.fc4 = nn.Linear(16,1)
        self.fc4_3d = nn.Linear(16,1)
    
        
    def forward(self,Array,Input,U):
        
        x1 = self.flatten(self.convs2d(Input))
        x2 = self.convs3d(Array)
        
        x3 = torch.zeros(x2.size()).cuda()
        
        if list(U.size()) == []:
            
            x3 = x2*U
            
        else:
            
            for i in range(len(x3)):
                
                x3[i] = x2[i]*U[i]/1.57e-5
        
        x3 = self.flatten(x3)
        
        x4 = self.fc1(x1)
        x5 = self.fc2(x4)
        x6 = self.fc3(x5)
        output1 = self.fc4(x6)
        
        x4_3d = self.fc1_3d(x3)
        x5_3d = self.fc2_3d(x4_3d)
        x6_3d = self.fc3_3d(x5_3d)
        output2 = self.fc4_3d(x6_3d)
        
        output = output2*output1*0.028
        
        return output
        
        
        
        
        
        
        
        
        
            
            
        
        
        


