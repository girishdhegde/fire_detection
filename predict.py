import numpy as np
# import matplotlib.pyplot as plt
import cv2

import torch
from utils import nms

def viz(img, pred, classes=None, colors=None):
    size = img.shape[1]
    for clas in pred:
        for boxes in clas:
            for box in boxes:
                box = box.squeeze()
                pt1 = (int(box[0] * size), int(box[1] * size))
                pt2 = (int(box[2] * size), int(box[3] * size))
                x, y = pt1
                if (x > 0) and (x < size) and (y > 0) and (y < size):
                    x, y = pt2
                    if (x > 0) and (x < size) and (y > 0) and (y < size):
                        if (pt1[0] != pt2[0]) and (pt1[1] != pt2[1]):
                            img = cv2.rectangle(img, tuple(pt1), tuple(pt2), color=(0, 0, 255), thickness=3)
                        # img = cv2.putText(img, str(predi[row, col, 5*b])+','+str(pred[row, col, 5*b+1]), 
                        #                     tuple(pt1), fontFace=0, fontScale=.5, color=(0, 255, 255))
    return img

def detect(img, net, ):
    m, n, *_ = img.shape
    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
    with torch.no_grad():
        out = net(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float())
        out = nms(out, conf_th=0.0, iou_th=0.2, S=7, B=2, C=2)
        img = viz(img, out[0])
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv2.resize(img, (n, m), interpolation=cv2.INTER_CUBIC)

# if __name__ == '__main__':
#     from yolo_resnet import yolo
#     load = './weights.pt'
#     net = yolo()
#     net.load_state_dict(torch.load(load))
#     net.eval()
#     img = cv2.imread('./trainset/images/0.png')
#     out = detect(img, net)



