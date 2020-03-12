import torch 
import numpy as np 

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    # torch.isinstance(torch.FloatTensor, True)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor 

def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()
        