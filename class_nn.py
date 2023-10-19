import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=24,out_features=64)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=64,out_features=128)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=128,out_features=4)

    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
