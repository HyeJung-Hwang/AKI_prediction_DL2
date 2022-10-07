import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
from torchsummary import summary

device = 'cuda:'+str(0)
data_path = 'C:/Users/changminhyun/Dropbox/workspace/CVS_Check/DATA/Normalization_data'
trX = np.load(data_path+'/train_X.npy')
trY = np.load(data_path+'/train_Y.npy')
tsX = np.load(data_path+'/test_X.npy')
tsY = np.load(data_path+'/test_Y.npy')
tsXg = np.load(data_path+'/test_X_good.npy')
tsXb = np.load(data_path+'/test_X_bad.npy')
tsXa = np.load(data_path+'/test_x_ambiguous.npy')

x_train = torch.FloatTensor(trX).to(device)
y_train = torch.FloatTensor(trY).to(device)
x_test = torch.FloatTensor(tsX).to(device)
y_test = torch.FloatTensor(tsY).to(device)
x_testg = torch.FloatTensor(tsXg).to(device)
x_testb = torch.FloatTensor(tsXb).to(device)
x_testa = torch.FloatTensor(tsXa).to(device)

ds = TensorDataset(x_train, y_train)
batch_size = 50
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=None)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq0 = torch.nn.Sequential(nn.Conv1d(1,4,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(4),nn.Conv1d(4,4,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(4),nn.MaxPool1d(2, stride=2))
        self.seq1 = torch.nn.Sequential(nn.Conv1d(4,8,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(8),nn.Conv1d(8,8,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(8),nn.MaxPool1d(2, stride=2))
        self.seq2 = torch.nn.Sequential(nn.Conv1d(8,16,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(16),nn.Conv1d(16,16,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(16),nn.MaxPool1d(2, stride=2))
        self.seq3 = torch.nn.Sequential(nn.Conv1d(16,32,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(32),nn.Conv1d(32,32,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(32),nn.MaxPool1d(2, stride=2))
        self.seq4 = torch.nn.Sequential(nn.Conv1d(32,64,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(64),nn.Conv1d(64,64,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(64),nn.MaxPool1d(2, stride=2))
        self.seq5 = torch.nn.Sequential(nn.Conv1d(64,128,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(128),nn.Conv1d(128,128,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(128),nn.MaxPool1d(2, stride=2))
        self.seq6 = torch.nn.Sequential(nn.Conv1d(128,256,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(256),nn.Conv1d(256,256,3,stride=1,padding=1),nn.ReLU(),nn.BatchNorm1d(256))
        self.FC1 = torch.nn.Sequential(nn.Linear(512,512),nn.ReLU())
        self.FC2 = torch.nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.seq0(x)
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = self.seq6(x)
        x = torch.flatten(x,1)
        x = self.FC1(x)
        output = self.FC2(x)
        return output

device = torch.device("cuda") # PyTorch v0.4.0
model = VGG().to(device)
summary(model, (150,))

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss

total_epoch = 100
learning_rate = 10**-4
n, m = len(np.where(trY == 0)[1]), len(np.where(trY == 0)[0])
BCE = BCELoss_class_weighted(torch.Tensor([n/(n+m), m/(n+m)]))
confmat = ConfusionMatrix(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(),learning_rate)

for epoch in range(0,total_epoch):
    loss_print = []
    for X, Y in train_loader:
        fX = model(X)
        loss = BCE(fX,Y)
        loss_print.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch == 0 or (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            print('Epoch : '+str(epoch+1)+' Loss : '+ str(sum(loss_print)/(trX.shape[0]/batch_size)))
            fXtrain = model(x_train)
            C = confmat(torch.round(fXtrain).cpu(), torch.round(y_train).type('torch.IntTensor').cpu())
            Precision = C[1,1] / (C[1,1]+C[0,1])
            Sensitivity = C[1,1] / (C[1,1]+C[1,0])
            Specificity = C[0,0] / (C[0,0]+C[0,1])
            Accuracy = (C[0,0]+C[1,1]) / (C[0,0]+C[1,0]+C[0,1]+C[1,1])
            print('Training - Accuracy : '+str((Accuracy*100).numpy())+' Precision : '+str((Precision*100).numpy())+' Sensitivity : '+str((Sensitivity*100).numpy())+' Specificity : '+str((Specificity*100).numpy()))
            fXtest = model(x_test)
            C = confmat(torch.round(model(x_test)).cpu(), torch.round(y_test).type('torch.IntTensor').cpu())
            Precision = C[1,1] / (C[1,1]+C[0,1])
            Sensitivity = C[1,1] / (C[1,1]+C[1,0])
            Specificity = C[0,0] / (C[0,0]+C[0,1])
            Accuracy = (C[0,0]+C[1,1]) / (C[0,0]+C[1,0]+C[0,1]+C[1,1])
            print('Test Accuracy : '+str((Accuracy*100).numpy())+' Precision : '+str((Precision*100).numpy())+' Sensitivity : '+str((Sensitivity*100).numpy())+' Specificity : '+str((Specificity*100).numpy()))
            print(C)
            C = confmat(torch.round(model(x_testg)).cpu(), torch.ones(x_testg.shape[0],1,dtype = torch.int32).cpu())
            Precision = C[1,1] / (C[1,1]+C[0,1])
            Sensitivity = C[1,1] / (C[1,1]+C[1,0])
            Specificity = C[0,0] / (C[0,0]+C[0,1])
            Accuracy = (C[0,0]+C[1,1]) / (C[0,0]+C[1,0]+C[0,1]+C[1,1])
            print('Good - Accuracy : '+str((Accuracy*100).numpy())+' Precision : '+str((Precision*100).numpy())+' Sensitivity : '+str((Sensitivity*100).numpy())+' Specificity : '+str((Specificity*100).numpy()))
            print(C)
            C = confmat(torch.round(model(x_testb)).cpu(),torch.zeros(x_testb.shape[0],1,dtype = torch.int32).cpu())
            Precision = C[1,1] / (C[1,1]+C[0,1])
            Sensitivity = C[1,1] / (C[1,1]+C[1,0])
            Specificity = C[0,0] / (C[0,0]+C[0,1])
            Accuracy = (C[0,0]+C[1,1]) / (C[0,0]+C[1,0]+C[0,1]+C[1,1])
            print('Bad - Accuracy : '+str((Accuracy*100).numpy())+' Precision : '+str((Precision*100).numpy())+' Sensitivity : '+str((Sensitivity*100).numpy())+' Specificity : '+str((Specificity*100).numpy()))
            print(C)
            fXtesta = torch.round(model(x_testa)).cpu().detach().numpy().reshape([-1])
            print('Ambiguous - pos_num : ' +str(np.size(np.where(fXtesta==1)))+ '  neg_num : ' + str(np.size(np.where(fXtesta==0))))
        model.train()

training_output = fXtrain.cpu()
training_label = torch.round(y_train).type('torch.IntTensor').cpu()
test_output = fXtest.cpu()
test_label = torch.round(y_test).type('torch.IntTensor').cpu()

X_tr, Y_tr, A_tr = [], [], []
X_ts, Y_ts, A_ts = [], [], []

for k in range(0,10001):
    t = k * 0.00011
    out = training_output >= t
    C = confmat(out, training_label)
    Y_tr.append((C[1,1]/(C[1,1]+C[1,0])).detach().numpy())
    X_tr.append((C[0,1]/(C[0,0]+C[0,1])).detach().numpy())
    A_tr.append(((C[0,0]+C[1,1])/(C[0,0]+C[1,0]+C[0,1]+C[1,1])).detach().numpy())
    out = test_output >= t
    C = confmat(out, test_label)
    Y_ts.append((C[1,1] / (C[1,1]+C[1,0])).detach().numpy())
    X_ts.append((C[0,1] / (C[0,0]+C[0,1])).detach().numpy())
    A_ts.append(((C[0,0]+C[1,1]) / (C[0,0]+C[1,0]+C[0,1]+C[1,1])).detach().numpy())

max_idx = np.argmax(np.array(Y_tr)-np.array(X_tr))
print(max_idx)
plt.plot(np.array(Y_tr)-np.array(X_tr))
plt.plot(max_idx,np.array(Y_tr)[max_idx]-np.array(X_tr)[max_idx],'or')
plt.xlabel('Idx Number')
plt.ylabel('J')
plt.grid('on')
plt.show()

plt.plot(X_tr,Y_tr,'-b',linewidth=4)
plt.plot(X_ts,Y_ts,'-r',linewidth=4)
plt.plot([0,1],[0,1],'--ok',linewidth=0.5)
plt.plot(X_tr[max_idx],Y_tr[max_idx],'oc')
plt.plot(X_ts[max_idx],Y_ts[max_idx],'oc')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid('on')
plt.show()

specificity_tr = 1-X_tr[max_idx]
sensitivity_tr = Y_tr[max_idx]
ac_tr = A_tr[max_idx]
specificity_ts = 1-X_ts[max_idx]
sensitivity_ts = Y_ts[max_idx]
ac_ts = A_ts[max_idx]

print(max_idx,X_tr[max_idx],Y_tr[max_idx])
print(specificity_tr*100,sensitivity_tr*100,ac_tr*100,specificity_ts*100,sensitivity_ts*100,ac_ts*100)

from scipy import integrate
print(-integrate.trapezoid(Y_tr,x=X_tr))
print(-integrate.trapezoid(Y_ts,x=X_ts))
