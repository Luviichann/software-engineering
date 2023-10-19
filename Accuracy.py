
import torch

def accuracy(dataset,model):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X,y in dataset:
            outputs = model(X)
            _,y_pre = torch.max(outputs,dim=1)
            total += y.shape[0]
            correct += int((y_pre == y).sum())

    return correct/total




