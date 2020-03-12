import torch 
import torchvision.transforms as T

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A 

def tensor2img(x, mean , std):
    '''
        Args:
            x: Tensor 
            mean: Tensor
            std: Tensor
        Return:
            Image 
    '''
    clip = lambda x: clip_tensor(x, 0, 255)
    
    tf = T.Compose([T.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1/x, std))),
        T.Normalize(mean=list(map(lambda x: -x, mean)), std=[1,1,1]),
        T.ToPILImage(),
        #T.Lambda(clip),
    ])
    return tf(x[0].cpu())
