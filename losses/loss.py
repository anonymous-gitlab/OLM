import torch.nn as nn
import torch.nn.functional as F

# class CELoss(nn.Module):
#     def __init__(self):
#         super(CELoss, self).__init__()
#         self.loss = nn.NLLLoss(reduction='sum')

#     def forward(self, pred, target):
#         pred = F.log_softmax(pred, 1)
#         target = target.squeeze().long()
#         loss = self.loss(pred, target) / len(pred)
#         return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss