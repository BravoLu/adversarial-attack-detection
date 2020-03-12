import os 
import xml.etree.ElementTree as ET 

import numpy as np 
from PIL import Image

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

class VOC2007(object):
    def __init__(self, root='/home/shaohao/data/VOCdevkit/VOC2007', split='trainval', use_difficult=False, return_difficult=False):
        
        id_list_file = os.path.join(root, 'ImageSets/Main/%s.txt'%split)
        with open(id_list_file, 'r') as f:
            self.ids = [id_.strip() for id_ in open(id_list_file)]
        
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult 
        self.root = root 
        self.label_names = VOC_BBOX_LABEL_NAMES

    def get_example(self, i):
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.root, 'Annotations/%s.xml'%id_))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text == 1):
                continue
            
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.float32)

        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        
        img_file = os.path.join(self.root, 'JPEGImages/%s.jpg'%id_)
        img = self.read_image(img_file)

        return img, bbox, label, difficult
    
    def __len__(self):
        return len(self.ids)
        
    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')

        img = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))
        return img
    
if __name__ == "__main__":
    d = VOC2007()
    sample = d.get_example(0)
    print(sample)