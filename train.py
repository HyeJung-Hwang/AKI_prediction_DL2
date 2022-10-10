from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import time
import copy
import torch
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
from models import Net

def train(model, train_loader,DEVICE, optimizer):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion =  nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.to(torch.float32), target.to(torch.float32))
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader, DEVICE):

    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += float(criterion(output.to(torch.float32), target.to(torch.float32)))

    test_loss /= len(test_loader.dataset)

    return test_loss


def train_model(train_loader, val_loader, DEVICE, num_epochs=30):
    model = Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = 100000000
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, num_epochs + 1):
        since = time.time()
        train(model, train_loader, DEVICE, optimizer)
        train_loss = evaluate(model, train_loader, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}'.format(train_loss))
        print('val Loss: {:.4f}'.format(val_loss))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


def main():
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    data_path = "/home/cse_urp_dl2/Documents/hhj/BP/data/"
    train_loader, valid_loader = load_data(data_path)

    model = train_model(train_loader, valid_loader, DEVICE)
    torch.save(model.state_dict(), "/home/cse_urp_dl2/Documents/hhj/BP/saved_models/check_FC00.pt")


if __name__ == "__main__":
    main()