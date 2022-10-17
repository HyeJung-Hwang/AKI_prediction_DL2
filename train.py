import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import time
import copy

from torch import Tensor
import glob
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import os
from data import load_data
from models import Net,Net2

def train(model, Loss, optimizer, num_epochs):
  train_loss_arr = []
  test_loss_arr = []

  best_test_loss = 99999999
  early_stop, early_stop_max = 0., 3.
   
  for epoch in range(num_epochs):

    epoch_loss = 0.
    for batch_X, _ in train_loader:
      
      batch_X = batch_X.to(device)
      optimizer.zero_grad()

      # Forward Pass
      model.train()
      outputs = model(batch_X)
      train_loss = Loss(outputs, batch_X)
      epoch_loss += train_loss.data

      # Backward and optimize
      train_loss.backward()
      optimizer.step()

    train_loss_arr.append(epoch_loss / len(train_loader.dataset))

    if epoch % 10 == 0:
      model.eval()

      test_loss = 0.

      for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)

        # Forward Pass
        outputs = model(batch_X)
        batch_loss = Loss(outputs, batch_X)
        test_loss += batch_loss.data

      test_loss = test_loss
      test_loss_arr.append(test_loss)

      if best_test_loss > test_loss:
          best_test_loss = test_loss
          early_stop = 0
          print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss, test_loss))
      else:
          early_stop += 1
          print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss, test_loss))   
          
    if early_stop >= early_stop_max:
        break
  torch.save(model.state_dict(), "/home/cse_urp_dl2/Documents/hhj/ECG/saved_models/check_FC01.pt")
  best_model_wts = copy.deepcopy(model.state_dict())
  model.load_state_dict(best_model_wts)
  return model
#data
npy_path = "/home/cse_urp_dl2/Documents/hhj/ECG/400/"
csv_path = "/home/cse_urp_dl2/Documents/hhj/ECG/train400_100.csv"
train_loader, test_loader = load_data(npy_path,csv_path)
#model & loss
AE = Net2()
AE_loss = nn.MSELoss()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
AE = AE.to(device)
learning_rate = 0.01
num_epochs = 50
AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
model = train(AE, AE_loss, AE_optimizer, num_epochs = 50)
torch.save(model.state_dict(), "/home/cse_urp_dl2/Documents/hhj/ECG/saved_models/check_FC01.pt")


    
