import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.FC_1 = nn.Linear(150, 128)
        self.FC_2 = nn.Linear(128, 64)
        self.FC_3 = nn.Linear(64, 32)
        self.FC_4 = nn.Linear(32, 64)
        self.FC_5 = nn.Linear(64, 128)
        self.FC_6 = nn.Linear(128, 150)

    def forward(self, x):
        x = x.view(-1, 1 * 150)
        out = self.FC_1(x)
        out = F.relu(out)

        out = self.FC_2(out)
        out = F.relu(out)

        out = self.FC_3(out)
        out = F.relu(out)

        out = self.FC_4(out)
        out = F.relu(out)

        out = self.FC_5(out)
        out = F.relu(out)

        out = self.FC_6(out)


        return out