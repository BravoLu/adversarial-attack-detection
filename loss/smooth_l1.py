import torch
import torch.nn as nn 

from .utils import weighted_loss 

'''
    rpn: beta = 1 / 9.0
    head: beta = 1.0
'''
'''
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    return loss 

class SmoothL1Loss(nn.Module):
    
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta 
        self.reduction = reduction 
        self.loss_weight = loss_weight 

    def forward(self, 
                        pred,
                        target,
                        weight=None,
                        avg_factor=None,
                        reduction_override=None,
                        **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_bbox 
'''

class SmoothL1Loss(nn.Module):
    def  __init__(self, sigma):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma 

    def forward(self, pred, target, label):
        in_weight = torch.zeros(target.shape).cuda()
        in_weight[(label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
        loss = self.smooth_l1_loss(pred, target, in_weight.detach())
        loss /= ((target >=0).sum().float())
        
    def smooth_l1_loss(self, x, in_weight):
        sigma2 = self.sigma ** 2
        diff = in_weight * (x-t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1./sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff**2) + (1-flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()