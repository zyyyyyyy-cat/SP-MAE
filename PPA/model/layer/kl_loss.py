import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from config import cfg


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = target_data.data.to(cfg.MODEL.DEVICE)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss