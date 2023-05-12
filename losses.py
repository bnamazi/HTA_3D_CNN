import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()



class EMD(nn.Module):
    def __init__(self):
        super(EMD, self).__init__()
    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.cumsum(predictions, dim=1)
        targets_onehot = F.one_hot(targets, predictions.shape[-1])
        targets_onehot = torch.cumsum(targets_onehot, dim=1)
        lossvalue = torch.norm(predictions - targets_onehot, p=2).mean()
        return lossvalue
