import functools
import torch.nn.functional as F

def reduce_loss(loss, reduction):
    ''' Reduce loss as specified 

    Args:
        loss (Tensor): 
        reduction: 'none', 'mean', 'sum'
    '''

    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    '''Apply element-wise weight and reduce loss.

    Args:
        loss(Tensor): Element-wise loss.
        weight(Tensor): Element-wise weights.
        reduction (str)
        avg_factor(float): Average factor when computing the mean of losses 
    
    Returns:
        Tensor: Processed loss values
    '''
    # if 
    if weight is not None:
        loss = loss * weight 
    
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum" ')
    return loss

def weighted_loss(loss_func):
    '''Create a weighted version of a given loss function 

    loss_func(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs)

    '''
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):

        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    
    return wrapper 
