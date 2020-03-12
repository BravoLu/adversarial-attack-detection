from __future__ import absolute_import
from __future__ import division
import torch
import numpy as np 
import cupy
from torch.nn import functional as F 
from torch import nn 


from utils import *
from models.utils.bbox_tools import loc2bbox 
from models.utils.nms import non_maximum_suppression 
from datasets.preprocessor import Transform 

def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, loc_normalize_mean = (0., 0., 0., 0.), loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor 
        self.rpn = rpn 
        self.head = head 

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std 
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class 

    def forward(self, x, scale=1.):
        '''
            Args:
                x:
                scale: orignal size / target size
        '''
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices 

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3 
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3 
            self.score_thresh = 0.05 
        else:
            raise ValueError('preset must be visualize or evaluate' )

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh 
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(cupy.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cupy.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # label range from [0, self.n_class - 2]
            label.append((l-1) * np.ones((len(keep), )))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score 
    '''
    @nograd 
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval() 
        prepared_imgs = imgs 
        
        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            # img[None] for add one dimension.
            img = to_tensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            roi_score = roi_scores.data 
            roi_cls_loc = roi_cls_loc.data 
            roi = to_tensor(rois) / scale 

            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(to_numpy(roi).reshape((-1, 4)), to_numpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = to_tensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)

            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = to_numpy(F.softmax(to_tensor(roi_score), dim=1))

            raw_cls_bbox = to_numpy(cls_bbox)
            raw_prob = to_numpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox) 
            labels.append(label)
            scores.append(score)
        
        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores 
    '''

    @nograd
    def predict(self, x, size):
        self.use_preset('evaluate')
        bboxes = list()
        labels = list()
        scores = list()
        
        #print(x.shape, size)
        scale = x.shape[3] / size[1]
        roi_cls_loc, roi_scores, rois, _ = self(x, scale=scale)
        roi_score = roi_scores.data 
        roi_cls_loc = roi_cls_loc.data 
        roi = to_tensor(rois) / scale 

        mean = torch.Tensor((0., 0., 0., 0.)).cuda().repeat(21)[None]
        std = torch.Tensor((0.1, 0.1, 0.2, 0.2)).cuda().repeat(21)[None]
        
        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, 21, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(to_numpy(roi).reshape((-1, 4)), to_numpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = to_tensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, 21 * 4)

        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:,1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

        prob = to_numpy(F.softmax(to_tensor(roi_score), dim=1))
        raw_cls_bbox = to_numpy(cls_bbox)
        raw_prob = to_numpy(prob)

        bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
        bboxes.append(bbox)
        labels.append(label)
        scores.append(score)

        return bboxes, labels, scores 

    def get_optimizer(self):
        use_adam = True 
        lr = 1e-3 
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value], 'lr':lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params':[value], 'lr':lr, 'weight_decay': 0.1}]
        if use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        
        return self.optimizer
    
    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

