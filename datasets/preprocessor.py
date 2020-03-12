import random
import numpy as np 
from datasets.voc import VOC2007
from skimage import transform
import torchvision.transforms as T 
import torch

def preprocess(img, min_size=600, max_size=1000, no_resize=False):
    if not no_resize:
        C, H, W = img.shape 
        scale1 = min_size/min(H,W)
        scale2 = max_size/max(H, W)
        scale = min(scale1, scale2)
        img = img / 255. 
        img = transform.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    else:
        img = img/255
        
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.astype(np.float32)
    img = normalizer(torch.from_numpy(img))
    img = img[None].cuda()
    return img 

class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size 
        self.max_size = max_size 

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape 
        img = self._preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H 
        #print(H, W, o_H, o_W)
        bbox = self.resize_bbox(bbox, (H, W), (o_H, o_W))

        img, params = self.random_flip(img, x_random=True, return_param=True)
        bbox = self.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        
        return img, bbox, label, scale
    
    def _preprocess(self, img, min_size, max_size):
        C, H, W = img.shape 
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = transform.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing= False)
        
        return self._normalize(img)

    def _normalize(self, img):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        img = img.astype(np.float32)
        img = normalizer(torch.from_numpy(img))

        return img.numpy()

    def resize_bbox(self, bbox, in_size, out_size):
        bbox = bbox.copy()
        #print(out_size[0].size(),  in_size[0].size())
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = y_scale * bbox[:, 0]
        bbox[:, 2] = y_scale * bbox[:, 2]
        bbox[:, 1] = x_scale * bbox[:, 1]
        bbox[:, 3] = x_scale * bbox[:, 3]
        return bbox
    
    def flip_bbox(self, bbox, size, y_flip=False, x_flip=False):
        H, W = size
        bbox = bbox.copy()
        if y_flip:
            y_max = H - bbox[:, 0]
            y_min = H - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max
        if x_flip:
            x_max = W - bbox[:, 1]
            x_min  = W - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max
        
        return bbox 

    def random_flip(self, img, y_random=False, x_random=False,
            return_param=False, copy=False):
        y_flip, x_flip = False, False
        if y_random:
            y_flip = random.choice([True, False])
        if x_random:
            x_flip = random.choice([True, False])
            
        if y_flip:
            img = img[:, ::-1, :]
        if x_flip:
            img = img[:, :, ::-1]
            
        if copy:
            img = img.copy()
            
        if return_param:
            return img, {'y_flip':y_flip, 'x_flip':x_flip}
        else:
            return img



class Preprocessor(object):
    def __init__(self,  config=None):
        '''
        self.configs = configs 
        self.dataset = globals()[configs.DATASET]()
        '''
        self.dataset = VOC2007()
        #self.transformer = Transform(configs.MIN_SIZE, configs.MAX_SIZE)
        self.transformer = Transform(600, 1000)
    def __getitem__(self, idx):
        img, bbox, label, _ = self.dataset.get_example(idx)

        img, bbox, label, scale = self.transformer((img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.dataset)

class TestDataset:
    def __init__(self):
        self.db = VOC2007(split='test')
        self.transformer = Transform(600, 1000)
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = self.transformer._preprocess(ori_img, 600, 1000)
        return img, np.array(ori_img.shape[1:]), bbox, label, difficult

    def __len__(self):
        return len(self.db)