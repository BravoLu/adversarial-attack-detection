from __future__ import absolute_import 
import cupy  as cp
import os 

from tqdm import tqdm 
import argparse

from datasets import * 
from models.faster_rcnn_vgg16 import FasterRCNN_vgg16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer 
from utils import *

def train():
    
    dataset = Preprocessor()
    dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    faster_rcnn = FasterRCNN_vgg16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    epochs = 12
    best_map  = 0
    lr_ = 1e-3
    for epoch in range(epochs):
        trainer.reset_meters()
        total_loss = 0
        for ii, (img, bbox_, label, scale) in enumerate(dataloader):
            scale = scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label.cuda()
            loss = trainer.train_step(img, bbox, label, scale)
            total_loss += loss[-1].item()
        trainer.save(save_path='checkpoints/%d.pth'%epoch)
        print('Epoch [%d]  total loss: %.4f'%(epoch+1, total_loss / len(dataloader)))

if __name__ == '__main__':

    opt = Option()
    args = opt.parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train()