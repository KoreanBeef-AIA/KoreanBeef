from torch.nn import functional as F
import torch
import torch.nn as nn

def get_loss(loss_name: str):
    
    if loss_name == 'crossentropy':
        return F.cross_entropy
    
    elif loss_name == 'focal':
        return FocalLoss
    
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        
    def forward(self, inputs, targets, alpha, gamma):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        
        return focal_loss