# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:56:48 2023

@author: 86130
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd

class Model(nn.Module):
    def __init__(self,width,depth,width3d,depth3d,batch_size,channels_3d,kernel_size,stride,padding,use_Unet = False,epoch_rate = 0.2):
        super(Model, self).__init__()
        self.width = width
        self.depth = depth
        self.width3d = width3d
        self.depth3d = depth3d
        self.height3d = batch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = batch_size
        self.channels_3d = channels_3d
        self.use_Unet = use_Unet
        self.epoch_rate = epoch_rate
        #Setting 2D convolution     
        self.conv1 = nn.Conv2d(self.channels,32,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv2 = nn.Conv2d(32,64,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv3 = nn.Conv2d(64,3,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        #Setting 3D convolution
        self.conv1_3d = nn.Conv3d(self.channels_3d,3,self.kernel_size,self.stride,self.padding)
        self.conv_weights1 = nn.Conv3d(1,3,self.kernel_size,self.stride,self.padding)

        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv2_3d = nn.Conv3d(3,16,self.kernel_size,self.stride,self.padding)
        self.conv_weights2 = nn.Conv3d(3,1,self.kernel_size,self.stride,self.padding)
        self.lin_weights = nn.Linear(int(self.width3d*self.depth3d*self.height3d*1),1)
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        
        self.conv3_3d = nn.Conv3d(16,3,self.kernel_size,self.stride,self.padding)
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        

        
        self.bn_2d = nn.BatchNorm2d(3)
        self.bn_3d = nn.BatchNorm3d(3)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12700,64)
        self.fc1_3d = nn.Linear(12700,64)
        
        self.fc2 = nn.Linear(64,16)
        self.fc2_3d = nn.Linear(64,16)
        
        self.fc3 = nn.Linear(16,1)
        self.fc3_3d = nn.Linear(16,1)
        
        self.fc1_U = nn.Linear(1,16)
        self.fc2_U = nn.Linear(16,1)
        self.act = nn.PReLU()
        
    def forward(self,Array,Input,back_up,epoch_now):
        
        U = back_up[:,0].unsqueeze(1)
        rows = back_up[:,1].unsqueeze(1)
        
        # Deep Convolution Neural Network
        #x1 = F.relu(F.max_pool2d(self.conv1(Input),2))
        x1 = F.max_pool2d(self.conv1(Input),2)
        #x2 = F.relu(F.max_pool2d(self.conv2(x1),1))
        x2 = F.max_pool2d(self.conv2(x1),1)
        #x3 = F.relu(F.max_pool2d(self.conv3(x2),1))
        x3 = F.max_pool2d(self.conv3(x2),1)
        #x3 = self.bn_2d(x3)#引入batch normalization
        #x3 = self.act(x3)
        
        #x1_3d = F.relu(F.max_pool3d(self.conv1_3d(Array),2))
        x1_3d = F.max_pool3d(self.conv1_3d(Array),2)
        #x2_3d = F.relu(F.max_pool3d(self.conv2_3d(x1_3d),1))
        x2_3d = F.max_pool3d(self.conv2_3d(x1_3d),1)
        #x2_3d = F.relu(F.max_pool3d(self.conv2_3d(x1_3d),2))
        #x3_3d = F.relu(F.max_pool3d(self.conv3_3d(x2_3d),1))
        x3_3d = F.max_pool3d(self.conv3_3d(x2_3d),1)
        #x3_3d = self.act(x3_3d)
        x3_3d = self.flatten(x3_3d)
        x3 = self.flatten(x3)
        x3_3d = torch.cat((x3_3d,rows),dim=1)
        #x3_3d = torch.cat((x3_3d,H),dim=1)
        x3 = torch.cat((x3,rows),dim=1)
        #x3 = torch.cat((x3,H),dim=1)
        
        #Early drop
        if (epoch_now < self.epoch_rate)|(epoch_now>1-self.epoch_rate):
            x3_3d = F.dropout(x3_3d,training=self.training)
            x5 = F.dropout(x3,training=self.training)
        else:
            x3_3d = x3_3d
            x5 = x3
        
        #Parts of Physics-Informed Machine Learning
        
        #Get predicted lacunarity by the model
        fc1_3d = self.fc1_3d(x3_3d.float())
        fc2_3d = self.fc2_3d(fc1_3d)
        fc4_3d = self.act(self.fc3_3d(fc2_3d))        
        x4_3d = fc4_3d*U/1.57e-5#Generate Reynold number
            
        ratio = fc4_3d.cpu().detach().numpy()/min(fc4_3d.cpu().detach().numpy())

        #Get predicted characteristic height by the model   
        fc1 = self.fc1(x5.float())
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        output1 = self.act(fc3)
        
        # Conduct non-linear regression  
        Re1 = self.fc1_U(x4_3d.float())
        output2 = self.act(self.fc2_U(Re1)) #Predict Nusselt number by the model

        output = output2/output1*0.028# Predict coefficient of convective heat transfer by the model
        A = fc4_3d.cpu().detach().numpy()
        B = output1.cpu().detach().numpy()
        C = x4_3d.cpu().detach().numpy()
        D = output2.cpu().detach().numpy()
        #Best Boundary: 3-8
        return output, output2/output1,ratio,x4_3d#,torch.mean(fc4_3d)/9 + 4/torch.mean(fc4_3d),torch.mean(output1)/6 + 3.17/torch.mean(output1)