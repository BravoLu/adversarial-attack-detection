import numpy as np 
import torch
import torch.nn as nn 
from utils import to_tensor 
from models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj
from torch.nn import functional as F
from advertorch.attacks.base import Attack
from advertorch.attacks.utils import rand_init_delta 
import pdb 

def perturb_iterative(xvars, bboxes, labels, size, scale, model, nb_iter, eps, eps_iter, delta_init=None,    
                    minimize=False, ord=1, clip_min=0.0, clip_max=1.0, l1_sparsity=None):
    # 
    if delta_init is not None:
        delta = delta_init 
    else:
        delta = torch.zeros_like(xvars)
    
    delta.requires_grad_() 
    for i in range(nb_iter):
        # roi_locs, roi_scores
        N, C, H, W = xvars.shape 
        img_size  = (H, W)
        features = model.extractor(xvars + delta)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = model.rpn(features, img_size, scale)

        proposal_target_creator = ProposalTargetCreator()
        sample_roi, gt_roi_loc, gt_roi_label = proposal_target_creator(rois, bboxes, labels, model.loc_normalize_mean, model.loc_normalize_std)
        sample_roi_index = torch.zeros(len(sample_roi))

        roi_cls_loc, roi_scores = model.head(features, sample_roi, sample_roi_index)

        # RPN loss 
        anchor_target_creator = AnchorTargetCreator()
        gt_rpn_loc, gt_rpn_label = anchor_target_creator(
            bboxes, 
            anchor, 
            img_size
        )
        gt_rpn_label = to_tensor(gt_rpn_label).long()
        gt_rpn_loc = to_tensor(gt_rpn_loc)

        ce_loss = nn.CrossEntropyLoss().cuda()        

        # RPN classification loss
        #rpn_cls_loss = ce_loss(rpn_scores, gt_rpn_label.cuda())
        rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_rpn_label.cuda(), ignore_index=-1)

        gt_roi_label = to_tensor(gt_roi_label).long()
        loss = ce_loss(roi_scores, gt_roi_label.cuda())

        loss += rpn_cls_loss 

        if minimize:
            loss = -loss 
        
        loss.backward()
        if ord == np.inf:
            #pdb.set_trace()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            #delta.data = clamp(xvars.data + delta.data, clip_min, clip_max) - xvars.data

        elif ord == 2:
            grad = delta.grad.data 
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvars.data + delta.data, clip_min, clip_max) - xvars.data 
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        
        elif ord == 1:
            grad = delta.grad.data 
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(int(np.round((1-l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            if xvars.is_cuda:
                delta.data = delta.data.cuda()
            delta.data = clamp(xvars.data + delta.data, clip_min, clip_max) - xvars.data
        
        else:
            error = 'Only ord = inf, ord = 1 and ord = 2 have been implemented'
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    #delta = torch.zeros_like(xvars)
    #x_adv = clamp(xvars + delta, clip_min, clip_max)
    x_adv = xvars + delta
    #pdb.set_trace()
    return x_adv 
    #return xvars
            


class LabelMixin(object):
    def _get_predicted_label(self, x, size):
        with torch.no_grad():
            bboxes, labels, scores = self.model.predict(x, size)
        
        return bboxes[0],  labels[0],  scores[0]  
    
    def _verify_and_process_inputs(self, x, y, size):
        if self.targeted:
            assert y is not None
        
        if not self.targeted:
            if y is None:
                _, y,  _ = self._get_predicted_label(x, size)
        
        x = x.detach().clone()
        #y = y.detach().clone()

        return x, y 

class PGDetAttack(Attack, LabelMixin):
    def __init__(self,
                        model,
                        eps=0.5,
                        nb_iter=40,
                        eps_iter=0.01,
                        rand_init=False,
                        clip_min=0.,
                        clip_max=1.,
                        ord=np.inf,
                        l1_sparsity=None,
                        targeted=False):
        super(PGDetAttack, self).__init__(model, None, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter 
        self.rand_init = rand_init 
        self.ord = ord 
        self.targeted = targeted 
        self.model = model 

        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)
    
    def perturb(self, x, y=None, size=None):
        bboxes,  labels,  scores = self._get_predicted_label(x, size)
        x, y = self._verify_and_process_inputs(x, labels, size)
        scale = size[0] / x.size(2)
        #print(labels)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(x+delta.data, min=self.clip_min, max=self.clip_max) - x 
        # def perturb_iterative(xvars, bboxes, labels, size, scale, model, nb_iter, eps, eps_iter, delta_init=None, minimize=False, ord=np.inf, clip_min=0.0, clip_max=1.0, l1_sparsity=None):
        rval = perturb_iterative(x, bboxes, labels, size, scale, self.model, nb_iter=self.nb_iter, 
                        eps=self.eps, eps_iter=self.eps_iter,  delta_init=delta, minimize=self.targeted, ord=self.ord, clip_min=self.clip_min, clip_max=self.clip_max,  l1_sparsity=self.l1_sparsity)
                    
        return rval.data 

    



#attack = PGDetAttack(faster_rcnn)
