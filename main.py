# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:35:24 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY

Contact: dapengw@umich.edu, 20074734d@connect.polyu.hk

"""
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matRead import MatRead
from OriginalModel1 import Model # Model without Pocket Loss
#from ModelBoundary import Model # Model with Pocket Loss
from SHUFFLE import Shuffle
from torchmetrics.functional import r2_score
import sys
import time

# Define Process Bar:
# From: https://zhuanlan.zhihu.com/p/360444190
def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum, ' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


SEED = 6  # Initialize Random Seed, proposed by Alice ZHAO
use_CUDA = False #Whether automatically use CUDA during data preprocessing 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

path = 'D:/PolyU/EnergyAI/PIML-DCNN/'# File 'data' detailed position, unzip data.zip to get 'data' file

casepath = path + 'dataset_new.mat'
datapath = path + 'data/'

casepath1 = path + 'dataset2.mat'
"""
Read '.mat' files,

'.mat' files containing data of information matrix, and physics field calculation results from COMSOL

"""
#cases = scio.loadmat(casepath)
#cases = cases['dataset']
Flatten = 0  # Whether flatten the data during the training and testing
#rows = torch.tensor(cases[:, 3]).unsqueeze(1)
batch_size = 3
#batch_size_data = 110

#Initialize two containers for transfering read '.mat' files data to tensor for PyTorch
Tensorcontainer, Arraycontainer = [], []
Tensorcontainer_valid, Arraycontainer_valid = [], []

def Set_generation(casepath,datapath,batch_size,Flatten,Tensorcontainer,Arraycontainer):
    cases = scio.loadmat(casepath)
    cases = cases['dataset']
    cases = cases[:,0:5]
    #Flatten = 0  # Whether flatten the data during the training and testing
    rows = torch.tensor(cases[:, 3]).unsqueeze(1)
    
    #Tranferring read '.mat' data to Tensor for PyTorch
    dataRead = MatRead(Tensorcontainer, Arraycontainer,
                       datapath, cases, batch_size, use_CUDA)
    dataset = dataRead.datasetGenerate(Flatten)

    # See more details of parameters in file 'matRead.py'
    NU = dataRead.paramsRead('hc', target=0)
    U = dataRead.paramsRead('V', target=0)
    #Nu = dataRead.paramsRead('NU', target=0)
    LSC = dataRead.paramsRead('LSC', target=0)
    #LSC_min = min(LSC.detach().numpy())
    hc_expect = dataRead.paramsRead('h_EXPECT',target = 0)

    #Combine all reading data for shuffling
    NU = torch.cat((NU,hc_expect),dim=1)
    NU = torch.cat((NU, LSC), dim=1)
    NU = torch.cat((NU, U), dim=1)
    cases = torch.tensor(cases)
    NU = torch.cat((cases, NU), dim=1)

    
    return dataset,NU
dataset,NU= Set_generation(casepath,datapath,batch_size,Flatten,Tensorcontainer, Arraycontainer)
#valid_data_X,valid_data_Y,valid_backup = Set_generation(casepath1, datapath,batch_size,Flatten,
#                                                         Tensorcontainer_valid, Arraycontainer_valid)
dataset = dataset[20:,:,:,:]
NU = NU[20:,:]
#valid_data_X1 = dataset[40:60,:]
#valid_data_Y1 = NU[40:60,:]
#cases1 = valid_data_Y1[:,0:5]

valid_data_X = dataset[80:100,:]
valid_data_Y = NU[80:100,:]
cases1 = valid_data_Y[:,0:5]

#valid_data_X = torch.cat((valid_data_X1,valid_data_X2),dim=0)
#valid_data_Y = torch.cat((valid_data_Y1,valid_data_Y2),dim=0)

dataset1 = dataset[0:80,:]
NU1 = NU[0:80,:]
dataset2 = dataset[100:,:]
NU2 = NU[100:,:]
#dataset3 = dataset[100:,:]
#NU3 = NU[100:,:]

dataset = torch.cat((dataset1,dataset2),dim=0)
#dataset = torch.cat((dataset,dataset3),dim=0)
NU = torch.cat((NU1,NU2),dim=0)
#NU = torch.cat((NU,NU3),dim=0)
#dataset = dataset[20:,:]
#NU = NU[20:,:]
#back_up = back_up[0:160,:]
#Divide reading data to dataset and input physical information
dataset, NU = Shuffle(dataset, NU, SEED)#Shuffling dataset
cases = NU[:, 0:5]#.cpu().detach().numpy()# Height configuration
rows = NU[:, 3].unsqueeze(1)# row spacing
LSC = NU[:, 7].unsqueeze(1)# lacunarity
hc_expect = NU[:,6].unsqueeze(1)
back_up = NU[:, -1]# air velocity

rows = rows*back_up.unsqueeze(1) # Generate Gamma for the model
back_up = torch.cat((back_up.unsqueeze(1), rows), dim=1)# Combine Gamma & air velocity；Update: Combine Lacunarity & air velocity

cases = torch.cat((cases,cases1),dim=0)
#cases = torch.cat((cases,cases2),dim=0)
torch.save(cases,'cases.pt')
# Get values of hc, coefficient of convective heat transfer
NU = NU[:, 5]
NU = NU.unsqueeze(1)

#------
cases_valid = valid_data_Y[:, 0:5].cpu().detach().numpy()# Height configuration
rows_valid = valid_data_Y[:, 3].unsqueeze(1)# row spacing
LSC_valid = valid_data_Y[:, 7].unsqueeze(1)# lacunarity
valid_backup = valid_data_Y[:, -1]# air velocity
hc_valid = valid_data_Y[:,6].unsqueeze(1)

rows_valid = rows_valid*valid_backup.unsqueeze(1) # Generate Gamma for the model
valid_backup = torch.cat((valid_backup.unsqueeze(1), rows_valid), dim=1)# Combine Gamma & air velocity；Update: Combine Lacunarity & air velocity

# Get values of hc, coefficient of convective heat transfer
valid_data_Y = valid_data_Y[:, 5]
valid_data_Y = valid_data_Y.unsqueeze(1)
#valid_backup = back_up[160,:,:]
"""
#Tranferring read '.mat' data to Tensor for PyTorch
dataRead = MatRead(Tensorcontainer, Arraycontainer,
                   datapath, cases, batch_size, use_CUDA)
dataset = dataRead.datasetGenerate(Flatten)

# See more details of parameters in file 'matRead.py'
NU = dataRead.paramsRead('hc', target=0)
U = dataRead.paramsRead('V', target=0)
Nu = dataRead.paramsRead('NU', target=0)
LSC = dataRead.paramsRead('LSC', target=0)
LSC_min = min(LSC.detach().numpy())
hc_expect = dataRead.paramsRead('h_EXPECT',target = 0)

#Combine all reading data for shuffling
NU = torch.cat((NU,hc_expect),dim=1)
NU = torch.cat((NU, LSC), dim=1)
NU = torch.cat((NU, U), dim=1)
cases = torch.tensor(cases)
NU = torch.cat((cases, NU), dim=1)
dataset, NU = Shuffle(dataset, NU, SEED)#Shuffling dataset

#Divide reading data to dataset and input physical information

cases = NU[:, 0:5].cpu().detach().numpy()# Height configuration
rows = NU[:, 3].unsqueeze(1)# row spacing
LSC = NU[:, 7].unsqueeze(1)# lacunarity
back_up = NU[:, -1]# air velocity

rows = LSC*back_up.unsqueeze(1) # Generate Gamma for the model
back_up = torch.cat((back_up.unsqueeze(1), rows), dim=1)# Combine Gamma & air velocity；Update: Combine Lacunarity & air velocity

# Get values of hc, coefficient of convective heat transfer
NU = NU[:, 5]
NU = NU.unsqueeze(1)
"""

"""
#Reading validation dataset data
casepath1 = path + 'dataset2.mat'
cases_valid = scio.loadmat(casepath1)
cases_valid = cases_valid['dataset']
Flatten = 0  # Whether flatten the data during the training and testing
rows_valid = torch.tensor(cases_valid[:, 3]).unsqueeze(1)
batch_size = 3
Tensorcontainer_valid, Arraycontainer_valid = [], []
dataRead_valid = MatRead(Tensorcontainer_valid, Arraycontainer_valid,
                   datapath, cases_valid, batch_size, use_CUDA)
dataset_valid = dataRead_valid.datasetGenerate(Flatten)


NU = dataRead.paramsRead('hc', target=0)
U = dataRead.paramsRead('V', target=0)
Nu = dataRead.paramsRead('NU', target=0)
LSC = dataRead.paramsRead('LSC', target=0)
LSC_min = min(LSC.detach().numpy())
hc_expect = dataRead.paramsRead('h_EXPECT',target = 0)
"""



# Divide datasets into training & testing dataset
train_data_X = dataset[0:100, :, :, :]
train_data_Y = NU[0:100]
train_backup = back_up[0:100]

test_data_X = dataset[100:, :, :, :]
test_data_Y = NU[100:]
test_backup = back_up[100:]

# Define Model and Hyperparameters
batch_size_data = 100
#epoch_early = 0.1
epoch_rate = 0.25
model = Model(498, 35, 498, 35, batch_size, 1, 3, 1, 1,epoch_rate).cuda()
count = 0
criterion = nn.MSELoss()
criterion = criterion.cuda()
MAE = nn.L1Loss()  # Absolute Error

sizes = list(train_data_X.size())
sizes1 = list(test_data_X.size())

epoch = 110
#Loading checkpoint weights of model if it exists
#model.load_state_dict(torch.load('./checkpoint22.pth'))
model.load_state_dict(torch.load('./checkpoint1_com.pth')) #Only for model without pocket loss


learning_rate = 1e-4
clip = 5.0
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,weight_decay=1e-4)

#Define Loss container for recording loss value plot
Loss = []

# Training
loss_record = 1000

for j in range(epoch):
    final_loss = 0
    for z in range(int(sizes[0]/batch_size_data)):
        batch_data_X = train_data_X[z *batch_size_data:(z+1)*batch_size_data, :, :, :]
        batch_data_Y = train_data_Y[z *batch_size_data:(z+1)*batch_size_data, :]
        batch_data_backup = train_backup[z *batch_size_data:(z+1)*batch_size_data]
        
        for i in range(int(sizes[1]/batch_size)):
            epoch_now = j/epoch
            model.train()
            data = batch_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cpu()
            data = data.cuda()
            Temp_array = data.unsqueeze(1).cpu()
            Temp_array = Temp_array.cuda()
            test_data = test_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
            test_Temp_array = test_data.unsqueeze(1)
            optimizer.zero_grad()
            #output, x1, _, loss2, loss3 = model.forward(Temp_array, data, batch_data_backup.cuda(),epoch_now)
           
            output,x1,_,_ = model.forward(Temp_array,data,batch_data_backup.cuda(),epoch_now) #Only for model without pocket loss
            data = data.cpu()
            Temp_array = Temp_array.cpu()

            #PocketLoss = 0.1*torch.sqrt(loss2**2 + loss3**2)
            loss = torch.sqrt(criterion(output.cpu(), batch_data_Y))#+0.1*torch.sqrt(loss2**2 + loss3**2)
            #loss = torch.sqrt(criterion(output.cpu(), batch_data_Y)) #Only for model without pocket loss
            loss.backward()
            """
            model.eval()
            with torch.no_grad():
                output2, x2, _,_ = model.forward(test_Temp_array, test_data, test_backup.cuda(),0.5)
                loss1 = MAE(output2, test_data_Y.cuda())/torch.mean(test_data_Y.cuda())
                if loss1.cpu().detach().numpy() < loss_record:
                    loss_record = loss.cpu().detach().numpy()
                    torch.save(model.state_dict(),"checkpoint1_com.pth")
            """
            final_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            time.sleep(0.01)

            process_bar(i+1, int(sizes[1]/batch_size))

    if j % 1 == 0:
        loss_print = final_loss
        train_data_Y_mean = torch.mean(train_data_Y)
        rel_error = loss_print/train_data_Y_mean.item()
        Loss.append(loss_print)
        print('\n Training Loss of iteration {} is: '.format(j) + str(loss_print) + '\n')
        
#model.load_state_dict(torch.load('./checkpoint22.pth'))
#Plot Loss
plt.figure()
plt.plot(Loss)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


#Plot model overall performance on training set
data = train_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
Temp_array = data.unsqueeze(1)
#output1, x1, _, _,_ = model.forward(Temp_array, data, train_backup.cuda(), 0.5)
output1, x1, _,_ = model.forward(Temp_array, data, train_backup.cuda(), 0.5) #Only for model without pocket loss

fig1, ax1 = plt.subplots()
ax1.plot(x1.cpu().detach().numpy(), train_data_Y.cpu().detach().numpy(), 'o', label='train_data')
ax1.plot(x1.cpu().detach().numpy(),output1.cpu().detach().numpy(), label='Prediction')
ax1.legend()
ax1.set_title('Performance on train data')
ax1.set_xlabel('Re$^m$/D')
ax1.set_ylabel('Coefficient of heat transfer')
plt.show()

# Testing

#Define Error container for record test error
Error = []
model.eval()
with torch.no_grad():

    for i in range(int(sizes1[1]/batch_size)):
        data = test_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
        Temp_array = data.unsqueeze(1)
        #output2, x2, _, loss2, loss3 = model.forward(Temp_array, data, test_backup.cuda(),0.5)
        output2,x2,_,_ = model.forward(Temp_array,data,test_backup.cuda(),1) #Only for model without pocket loss
        loss1 = MAE(output2, test_data_Y.cuda())/torch.mean(test_data_Y.cuda())
        Error.append(loss1.cpu().detach().numpy())
    
    #Get average error on testing dataset
    Error_avg = sum(Error)/len(Error)

    Error_numpyMean = np.mean(Error)
    print(Error_numpyMean)
    
    #Calculate R2 for linear regression
    Rsq = r2_score(output2, test_data_Y.cuda()).item()

# Plotting the test dataset performance

fig, ax = plt.subplots()
ax.plot(x2.cpu().detach().numpy(), test_data_Y.cpu().detach().numpy(), 'o', label='test_data')
ax.plot(x2.cpu().detach().numpy(),output2.cpu().detach().numpy(), label='Prediction')
ax.legend()
ax.set_title('Performance on test data')
ax.set_xlabel('Re$^m$/D')
ax.set_ylabel('Coefficient of heat transfer')
plt.show()

#Validation
ErrorV = []
model.eval()

with torch.no_grad():

    for i in range(int(sizes1[1]/batch_size)):
        data = valid_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
        Temp_array = data.unsqueeze(1)
        #output3, x3, _, loss2, loss3 = model.forward(Temp_array, data, valid_backup.cuda(),0.5)
        output3,x3,_,_ = model.forward(Temp_array,data,valid_backup.cuda(),1) #Only for model without pocket loss
        loss = MAE(output3, valid_data_Y.cuda())/torch.mean(valid_data_Y.cuda())
        ErrorV.append(loss.cpu().detach().numpy())
    
    #Get average error on testing dataset
    ErrorV_avg = sum(ErrorV)/len(ErrorV)

    ErrorV_numpyMean = np.mean(ErrorV)
    print(ErrorV_numpyMean)
    
    #Calculate R2 for linear regression
    RsqV = r2_score(output3, valid_data_Y.cuda()).item()
    
# Plotting the test dataset performance

fig2, ax2 = plt.subplots()
ax2.plot(x3.cpu().detach().numpy(), valid_data_Y.cpu().detach().numpy(), 'o', label='test_data')
ax2.plot(x3.cpu().detach().numpy(),output3.cpu().detach().numpy(), label='Prediction')
ax2.legend()
ax2.set_title('Performance on valid data')
ax2.set_xlabel('Re$^m$/D')
ax2.set_ylabel('Coefficient of heat transfer')
plt.show()

data = dataset[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
Temp_array = data.unsqueeze(1)

#Get overall prediction on whole dataset.
#output3, x3, ratio, _, _ = model.forward(Temp_array, data, back_up.cuda(), 1)
output3, x3, ratio, _ = model.forward(Temp_array, data, back_up.cuda(), 1) #Only for model without pocket loss
