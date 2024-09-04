# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:24:44 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY

Contact: dapengw@umich.edu, 20074734d@connect.polyu.hk
"""

import scipy.io as scio
import torch
import numpy as np


class MatRead():
    
    def __init__(self,Tensorcontainer,Arraycontainer,path,cases,batch_size,CUDA = False):
        
        
        self.tensor = Tensorcontainer
        self.Array = Arraycontainer
        self.path = path
        self.cases = cases
        #self.rowspace = rowspace
        self.Heights = self.cases[:,0:4]
        #self.rowspace = self.rowspace[:,4]
        self.Heights = np.unique(self.Heights,axis = 0)
        """
        COMOSL Reading Parameters information:
            'NU': Reading Nusselt number
            'RE': Reading Reynold number
            'LSC': Reading lacunairty number
            'hc': Reading coefficient of convective heat transfer
            'PRs': Reading Power Ratio
            'V': Reading inflow air velocity
            'h_EXPECT': Reading coefficient of convective heat transfer calculated by empirical method
        
        """
        
        self.parameters = {'NU':'NU',
                           'RE':'RE',
                           'LSC':'LSC',
                           'hc':'hc',
                           'PRs':'PRs',
                           'V':'V',
                           'h_EXPECT':'h_EXPECT'}
                           #'Q':'Q',
                           #'Q_EXPECT':'Q_EXPECT'}
        self.batch_size = batch_size
        self.CUDA = CUDA
        
    def datasetGenerate(self,Flatten):
        
        for case in self.cases:
            List = []
            for j in range(len(case)):
                
                if int(case[j]) == 3:
                    
                    List.append(3)
                    
                elif case[j] < 1:
                    
                    List.append(0.912)
                    
                else:
                    
                    List.append(case[j])
            
            
            #if int(case[2]) == 3:
            case_index = str(List[0]) + '_' + str(List[1]) + '_' + str(List[2]) + '_' + str(List[3]) + '_' + str(case[4]) + '_'
            #else:
            #    case_index = str(case[0]) + '_' + str(case[1]) + '_' + str(case[2]) + '_' + str(case[3]) + '_' + str(case[4]) + '_'
            
            datafile = self.path + case_index + 'Array.mat'
            
            data = scio.loadmat(datafile)
            self.Array.append(data['ArrayLog'])
            
            if self.CUDA:
                
                
                self.tensor.append(torch.tensor(data['ArrayLog']).cuda())
            
            else:
                
                self.tensor.append(torch.tensor(data['ArrayLog']))
        
        #张量补零->不仅在单一维度补零
        #Padding at all dimensions
        
        Height_size,Width_size = [],[]

        for i in range(len(self.tensor)):
            
            Width_size.append(list(self.tensor[i].size())[0])
            Height_size.append(list(self.tensor[i].size())[1])
        
        Space_height = max(Height_size)
        Space_width = max(Width_size)
        
        
        
        for i in range(len(self.tensor)):
            
            sizes = list(self.tensor[i].size())
            
            delta = Space_height - sizes[1]
            delta_width = Space_width - sizes[0]
            
            if delta != 0:
                
                if self.CUDA:
                    zeros = torch.zeros(sizes[0],delta,sizes[2]).cuda()
                else:
                    zeros = torch.zeros(sizes[0],delta,sizes[2])
                    
                self.tensor[i] = torch.cat((self.tensor[i],zeros),dim = 1)
                
            if delta_width != 0:
                
                if self.CUDA:
                    zeros = torch.zeros(delta_width,sizes[1]+delta,sizes[2]).cuda()
                    
                else:
                    
                    zeros = torch.zeros(delta_width,sizes[1]+delta,sizes[2])
                
                self.tensor[i] = torch.cat((zeros,self.tensor[i]),dim = 0)
            
            self.tensor[i] = self.tensor[i].permute(2,0,1)
        
        
        #Flatten the data
        if Flatten == 1:
            
            if self.CUDA:
                
                dataset = torch.Tensor(sizes[0]*len(self.tensor),Space_height*sizes[2]).cuda()
                
            else:
                
                dataset = torch.Tensor(sizes[0]*len(self.tensor),Space_height*sizes[2])
        
            for i in range(len(self.tensor)):
                self.tensor[i] = torch.reshape(self.tensor[i],(Space_width,Space_height*sizes[2]))
                dataset[i*sizes[0]:(i+1)*sizes[0],:] = self.tensor[i]
                
        else:
            
            if self.CUDA:
                
                dataset = torch.Tensor(len(self.tensor),sizes[2],Space_width,Space_height).cuda()
                
            else:
                
                dataset = torch.Tensor(len(self.tensor),sizes[2],Space_width,Space_height)
                
            for i in range(len(self.tensor)):
                
                dataset[i,:,:,:] = self.tensor[i]
        
        return dataset
    
    def paramsRead(self,param,target=1):
        
        params_temp = 0
        
        counting = 0
        #for i in range(len(self.rowspace)):
        for i in range(len(self.Heights)):
            List = []
            for j in range(len(self.Heights[i])):
                
                if int(self.Heights[i][j]) == 3:
                    
                    #self.Heights[i][j] = int(self.Heights[i][j])
                    List.append(3)
                elif self.Heights[i][j]<1:
                    List.append(0.912)
                    
                else:
                    List.append(self.Heights[i][j])
                
            #elif int(self.Heights[]):
                
            #else:
                
            params_index = str(List[0]) + '_' + str(List[1]) + '_' + str(List[2]) + '_' + str(self.Heights[i][3]) + '_'
                
            param_path = self.path + params_index + self.parameters[param] + '.mat'
            #param_path = self.path + self.parameters[param] + self.rowspace[i]+ '.mat'
            params = scio.loadmat(param_path)[param][0]
              
            if target == 1:
                
                params = np.log10(params)
                
            params = params.astype(np.float64)
            params = torch.tensor(params)
            
            if counting == 0:
                params_temp = params
            else:
                params_temp = torch.cat((params_temp,params),dim = 0)
            
            counting = counting + 1
                    
        
        if self.CUDA == True:
            params_temp = params_temp.cuda()
        
        params_temp = params_temp.to(torch.float32)
        params_temp = torch.reshape(params_temp,(list(params_temp.size())[0],1))
        
        return params_temp










