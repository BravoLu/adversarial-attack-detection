import cv2 
from utils.color import COLOR  
from datasets.voc import VOC_BBOX_LABEL_NAMES

def visualize(result, img_path, out_file):
    '''
        Args:
            result: [bboxes, label, scores]
            img_path: str 
            out_file: str
    ''' 
    bboxes, labels, scores = result 
    bboxes, labels, scores = bboxes[0], labels[0], scores[0]

    img = cv2.imread(img_path)
    for i, bbox in enumerate(bboxes):
        ymin, xmin, ymax, xmax = bbox 
        label = labels[i]
        color = list(COLOR.values())[label]
        score = scores[i]
        text = VOC_BBOX_LABEL_NAMES[label] + '|:{:02f}'.format(score)
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=3)
        cv2.putText(img, text, (xmin, ymin-2), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
    
    cv2.imwrite(out_file, img)



