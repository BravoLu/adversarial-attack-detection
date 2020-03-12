from __future__ import absolute_import 
import cupy  as cp
import os 

from tqdm import tqdm 
import argparse

from datasets import * 
from models.faster_rcnn_vgg16 import FasterRCNN_vgg16
from torch.utils import data
from trainer import FasterRCNNTrainer 
from utils import *
from PGD_attack import PGDetAttack

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = sizes[0].numpy()
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs.cuda(), sizes)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def attack_eval(dataloader, faster_rcnn):
    attack = PGDetAttack(faster_rcnn)
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for _, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        #sizes = [sizes[0][0].item(), sizes[1][0].item()]
        sizes = sizes[0]
        #print(imgs.size(),  sizes)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs.cuda(), sizes)
        if len(pred_bboxes_[0]) != 0:
            imgs = attack.perturb(imgs.cuda(), y=None, size=sizes)
            pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs.cuda(), sizes)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


if __name__ == "__main__":
    testset = TestDataset()
    opt = Option()
    args = opt.parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataloader = data.DataLoader(
        testset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    faster_rcnn = FasterRCNN_vgg16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('checkpoints/threat_model.pth')
    eval_result = eval(dataloader, faster_rcnn, test_num=10000)
    #eval_result = attack_eval(dataloader, faster_rcnn)
    print(eval_result['map'])


