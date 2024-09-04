# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:58:40 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY

Reference Source: https://blog.csdn.net/kobayashi_/article/details/108951993

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd

class DownSampleLayer(nn.Module):
    
    
    def __init__(self,input_channel,output_channel):
        
        super(DownSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            
            nn.Conv2d(input_channel, output_channel, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            
            )
        self.downsample = nn.Sequential(
            
            nn.Conv2d(output_channel, output_channel, kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            
            )
    def forward(self,x):
        
        output = self.Conv_BN_ReLU_2(x)
        output_downSample = self.downsample(output)
        
        return output, output_downSample

class UpSampleLayer(nn.Module):
    
    
    def __init__(self,input_channel,output_channel):
        
        super(UpSampleLayer,self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            
            nn.Conv2d(input_channel,output_channel*2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(output_channel*2),
            nn.ReLU(),
            nn.Conv2d(output_channel*2, output_channel*2, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(output_channel*2),
            nn.ReLU()
            
            )
        self.upsample = nn.Sequential(
            
            nn.ConvTranspose2d(output_channel*2, output_channel, kernel_size=3,stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
            
            )
        
    def forward(self,x,out):
        x_output = self.Conv_BN_ReLU_2(x)
        x_output = self.upsample(x_output)
        cat_out = torch.cat((x_output,out),dim=1)
        
        return cat_out


class UNet(nn.Module):
    def __init__(self,batch_size):
        
        
        super(UNet, self).__init__()
        out_channels=[2**(i+4) for i in range(5)] 
        
        #Down Sampling
        
        self.d1=DownSampleLayer(batch_size,out_channels[0])
        self.d2=DownSampleLayer(out_channels[0],out_channels[1])
        self.d3=DownSampleLayer(out_channels[1],out_channels[2])
        self.d4=DownSampleLayer(out_channels[2],out_channels[3])
        
        #Up Sampling
        
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])
        
        #Output
        
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            #nn.Sigmoid(),
            nn.ReLU(),
            # BCELoss
        )
    def forward(self,x):
        
        
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

        
        
        
        
        
        