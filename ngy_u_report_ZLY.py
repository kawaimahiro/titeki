# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:09:18 2018

@author: DHU_Z
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.datasets import load_iris
import torch.optim as optim
import random
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as py

learning_rate = 1e-3
iris = load_iris()  
criterion = nn.MSELoss()
trainingLoss = []
trainingAcc = []
testLoss = []
testAcc = []
'''
irisClassification = nn.Sequential(                  #ANN
        nn.Linear(3,8),#nn.Linear(3,1)
        nn.Linear(8,10),
        nn.Linear(10,20),
        nn.Linear(20,30),
        nn.Linear(30,10),
        nn.Linear(10,5),
        nn.Linear(5,3),
        nn.Linear(3,1)    
        )

'''
irisClassification = nn.Sequential(                    #線形回帰
        nn.Linear(3,1)
        )

training_data = torch.Tensor( iris['data'][:,0:3]).cuda()
label_data = torch.Tensor(iris['data'][:,3]).cuda()
p = np.array(range(0,150))
random.shuffle(p)
train_data = p[0:120]
test_data = p[120:150] 
ann = irisClassification.cuda()
optimizer = optim.SGD(ann.parameters(), lr=learning_rate)

for epoch in range(1,1000):
    lossTrain = 0
    lossTest = 0
    ann.train()
    a1 = 0
    for train in train_data:
        inputData = training_data[train,:]
        target = label_data[train]
        out = ann(inputData)
        loss = criterion(out, target)
        
        #lossTrain += loss
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        
    ann.eval()
    for test in test_data:
        inputData = training_data[test,:]
        target = label_data[test]
        out = ann(inputData)
        loss = criterion(out, target)
        lossTest += loss
    for test in train_data:
        inputData = training_data[test,:]
        target = label_data[test]
        out = ann(inputData)
        loss = criterion(out, target)
        a1 += loss
    trainingLoss.append(np.float(a1/120))

    testLoss.append(np.float(lossTest/30))
    if epoch%10 == 0:
        print('epoch: '+str(epoch)+' TestLoss:'+str(np.float(lossTest/30))+' TrainLoss'+str(np.float(a1/120)))

epochNum = 100                    #横軸の範囲          
emin = 0
xEpoch = np.array( range(emin,epochNum))
trace2 = go.Scatter(
    x = xEpoch,
    y = testLoss[emin:epochNum],
    mode = 'lines',
    name = '汎化誤差'
)
trace3 = go.Scatter(
    x = xEpoch,
    y = trainingLoss[emin:epochNum],
    mode = 'lines',
    name = '訓練誤差'
)
'''
trace4 = go.Scatter(
    x = [83,83],
    y = [0.035,0.05],
   # y = trainingLoss[emin:epochNum],
    mode = 'lines',
    name = '最適容量'
)
'''
layout = dict(title = 'モデルの誤差',
              
      xaxis = dict(title = 'Epoch'),
      yaxis = dict(title = '誤差')
      
      )
data = [trace2, trace3]#,trace4]
fig = go.Figure(data=data,layout = layout)
name = 'test.html'

py.plot(fig,filename = name)