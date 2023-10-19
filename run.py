
from read_data import scan_data

X1,y1 = scan_data('./data/data1.csv')
X2,y2 = scan_data('./data/data2.csv')
X3,y3 = scan_data('./data/data3.csv')
X4,y4 = scan_data('./data/data4.csv')

import numpy as np
import torch


X = np.array([X1,X2,X3,X4])
X_tensor = torch.from_numpy(X).float().reshape(-1,24)

y = np.array([y1,y2,y3,y4])
y_tensor = torch.from_numpy(y).reshape(-1)

from torch.utils.data import TensorDataset,DataLoader,random_split

dataset = TensorDataset(X_tensor,y_tensor)
train_size = 1600
test_size = 400

train_set,test_set = random_split(dataset,[train_size,test_size])
train_load = DataLoader(train_set,batch_size=64,shuffle=True)
test_load = DataLoader(test_set,batch_size=64,shuffle=True)

from class_nn import Net
import torch.nn as nn

model = Net()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 200

from train import train_model

train_model(model,num_epochs,loss_fn,optimizer,train_load)

from Accuracy import accuracy

print('Train Accuracy: ',accuracy(train_load,model))
print('Test Accuracy: ',accuracy(test_load,model))

torch.save(model.state_dict(),'./200epochs.pth')
