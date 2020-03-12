import os 
import torch 
from models.faster_rcnn_vgg16 import FasterRCNN_vgg16
import cv2 
from datasets import preprocess 
from tqdm import tqdm 
from PIL import Image 
import numpy as np 
from utils import visualize
from PGD_attack import PGDetAttack
from utils.tensor_tool import tensor2img

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

if __name__ == "__main__":
    faster_rcnn = FasterRCNN_vgg16()
    params = torch.load('checkpoints/threat_model_faster_rcnn.pth')
    faster_rcnn.load_state_dict(params)
    faster_rcnn = faster_rcnn.cuda()
    attack = PGDetAttack(faster_rcnn)

    with open('test.txt', 'r') as f:
        test_ids = [l.strip() for l in f.readlines()]
    
    test_imgs = [os.path.join('/data/VOCdevkit/VOC2007/JPEGImages/%s.jpg'%id_) for id_ in test_ids]
    #test_imgs = ['demo/after.jpg']
    

    dir_ = '2020-2-15'
    for i,img in tqdm(enumerate(test_imgs)):
        #dir_ = str(i)
        if not os.path.exists('demo/%s'%dir_):
            os.makedirs('demo/%s'%dir_)
        path = img
        fname = os.path.basename(path)
        #out_file = '/home/shaohao/adversarial-attack-detection/visualization/%s'%fname
        out_file = 'demo/%s/%s_0.jpg'%(dir_, str(i))
        img = Image.open(img)
        img = img.convert('RGB')
        img = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))
        raw_size = img.shape[1:]
        H, W = raw_size
        img = preprocess(img)
        bboxes, labels, scores = faster_rcnn.predict(img, raw_size)
        visualize((bboxes, labels, scores), path, out_file)
        if len(bboxes[0]) == 0:
            continue
        '''
        check_img = tensor2img(img, mean, std)
        check_img = check_img.resize((W, H))
        check_img.save('demo/%s/check.png'%dir_)
        check_img = Image.open('demo/%s/check.png'%dir_).convert('RGB')
        check_img = np.asarray(check_img, dtype=np.float32).transpose((2, 0, 1))
        check_img = preprocess(check_img)
        bboxes, labels, scores = faster_rcnn.predict(check_img, raw_size)
        # for comparsion
        visualize((bboxes, labels, scores), 'demo/%s/check.png'%dir_, 'demo/%s/check_visualization.jpg'%dir_)
        #print(img.size())
        #print(bboxes)
        '''

        img_tensor = attack.perturb(img, y=None, size=raw_size)
        #print(img_tensor.size())
        img = tensor2img(img_tensor, mean, std)
        img = img.resize((W, H))
        img.save('demo/%s/tmp.png'%dir_)
        '''
        bboxes, labels, scores = faster_rcnn.predict(img_tensor, raw_size)
        visualize((bboxes, labels, scores), 'demo/%s/tmp.png'%dir_, 'demo/%s/visualization_direct_perturbed.jpg'%dir_)
        '''

        img = Image.open('demo/%s/tmp.png'%dir_).convert('RGB')
        img = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))
        raw_size = img.shape[1:]
        img_tensor = preprocess(img)
        bboxes, labels, scores = faster_rcnn.predict(img_tensor, raw_size)
        visualize((bboxes, labels, scores), 'demo/%s/tmp.png'%dir_, 'demo/%s/%s_1.jpg'%(dir_, str(i)))        


