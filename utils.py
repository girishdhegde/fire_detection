import torch
import torch.nn as nn
import torch.nn.functional as F

def convert(predictions, s=7, B=2):
    '''
    paramas:
    predictions: batch_size x s x s x (C + 5B) tensor
    s x s: total grid cells
    B: bbox per grid cell
    '''

    out    = predictions.clone()
    centre = out.clone()
    # print('prediction:\n', out*100)
    # (center x wrt grid, center y wrt grid, height wrt image, width wrt image) to
    # (center x wrt image, center y wrt image, height wrt image, width wrt image) conversion
    cell_size_wrt_image = 1 / s
    for row in range(s):
        for col in range(s):
            for b in range(0, 5*B, 5):
                # centres from grid cell coord. to image coord. conversion
                centre[:, row, col, b+0] = (out[:, row, col, b+0] + col)  * cell_size_wrt_image
                centre[:, row, col, b+1] = (out[:, row, col, b+1] + row)  * cell_size_wrt_image
    # print('centre:\n', centre*100)
    
    # (center x, center y, height, width) attributes of bboxes, to 
    # (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    for b in range(0, 5*B, 5):
        out[:, :, :, b+0] = (centre[:, :, :, b+0] - centre[:, :, :, b+2] / 2)
        out[:, :, :, b+1] = (centre[:, :, :, b+1] - centre[:, :, :, b+3] / 2)
        out[:, :, :, b+2] = (centre[:, :, :, b+0] + centre[:, :, :, b+2] / 2) 
        out[:, :, :, b+3] = (centre[:, :, :, b+1] + centre[:, :, :, b+3] / 2)

    return out

def iou(box1, box2):
    '''
    Params:
    Intersection Over Union = Area of intersection / Area of union
    box: bounding box
          batch_size x 5 tensor or batch_size x 4 tensor
          [x1, y1, x2, y2, confidence P(optional)]
          (x1, y1): top left corner
          (x2, y2): bottom right corner

    Returns: Batch_size x IOU

    '''

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1, min=0) * torch.clamp(inter_rect_y2-inter_rect_y1, min=0)
 
    #Union Area
    b1_area = torch.abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = torch.abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-8)
    
    return iou



def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



def nms_(prediction, conf_th=0.5, iou_th=0.5, S=7, B=2, C=20):
    '''
    Params:
    prediction: S x S x (C + 5B) tensor
    conf_th: Confidence probability threshold
    iou_th: IOU threshold
    S x S: Total grid cells
    B: Bounding boxes per grid cell  
    C: Classes
    
    Returns:
    non-maximal-suppression(prediction)
    '''
    classes = [[] for _ in range(C)]

    # Confidence probability thresholding
    for row in range(S):
        for col in range(S):
            pred = prediction[row, col]
            # print(pred.shape)
            idx = torch.argmax(pred[5*B:])
            # idx = 0
            for b in range(1, B+1):
                if pred[5*b-1] > conf_th:
                    classes[idx].append(pred[5*(b-1): 5*b])

    # IOU thresholding
    out = [[] for _ in range(C)]
    for idx, clas in enumerate(classes):
        if len(clas) > 0:
            temp = sorted(clas, key=lambda x: x[-1], reverse=True)
            box1 = temp[0].unsqueeze(0)
            bboxes = [box1, ]
            # print(box1.shape)
            if len(temp) > 1:
                for box2 in temp[1:]:
                    iu = iou(box1, box2.unsqueeze(0)) 
                    # print(iu)
                    if iu[0] < iou_th:
                        bboxes.append(box2)
            out[idx].append(bboxes)

    return out

def nms(prediction, conf_th=0.5, iou_th=0.5, S=7, B=2, C=20):
    '''
    Params:
    prediction: Batch_size x S x S x (C + 5B) tensor
    conf_th: Confidence probability threshold
    iou_th: IOU threshold
    S x S: Total grid cells
    B: Bounding boxes per grid cell  
    C: Classes
    
    Returns:
    non-maximal-suppression(prediction)
    '''
    prediction = convert(prediction.cpu(), s=S, B=B)

    out = []
    for pred in prediction:
        out.append(nms_(pred, conf_th, iou_th, S, B, C))

    return out

def render(img, prediction, classes, ):
    pass

def MAP():
    pass


class yoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, coord=5, noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = coord
        self.lambda_noobj = noobj

    def forward(self, prediction, target):
        '''
        Params:
        prediction: Batch_size x S x S X (5B + C)
        target    : Batch_size x S x S x ( 5 + C)
        (Don't use inplace operators)
        '''

        with torch.no_grad():
            bs = prediction.shape[0]
            # (center x, center y, height, width) attributes of bboxes, to 
            # (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
            pred_converted   = convert(prediction, self.S, self.B)
            target_converted = convert(target, self.S, B=1)

        centre_loss = torch.tensor(0.0)
        dim_loss = torch.tensor(0.0)
        conf_loss = torch.tensor(0.0)
        no_conf_loss = torch.tensor(0.0)
        class_loss = torch.tensor(0.0)

        # for all image in batch
        for pred_idx in range(bs):
            # for all grid cells in image
            for row in range(self.S):
                for col in range(self.S):
                    with torch.no_grad():
                        bbox_idx  = 0
                        iou_score = 0
                        # Finding i.e. has the highest IOU of any predictor in that grid cell
                        box1 = pred_converted[pred_idx, row, col, 0: 5*self.B].view(self.B, 5)
                        box2 = target_converted[pred_idx, row, col, 0: 5].repeat(self.B, 1)
                        # Calculating IOU b/n prediction and target
                        iou_score = iou(box1, box2)
                        bbox_idx  = torch.argmax(iou_score)
                        # print(iou_score, bbox_idx)

                    obj_present = True if target[pred_idx, row, col, 4] > 0 else False

                    if obj_present:
                        pred_x, pred_y, pred_w, pred_h, pred_p = prediction[pred_idx, row, col, 5*bbox_idx: 5*bbox_idx+5]
                        targ_x, targ_y, targ_w, targ_h, targ_p = target[pred_idx, row, col, 0: 5]
                        # centers: (x, y) loss where there is a object
                        centre_loss = ((pred_x - targ_x) ** 2 + (pred_y - targ_y) ** 2) + centre_loss

                        # width and height loss where there is a object
                        dim_loss = ((torch.sign(pred_w) * torch.sqrt(torch.abs(pred_w) + 1e-6) - torch.sqrt(targ_w)) ** 2 +
                                   (torch.sign(pred_h) * torch.sqrt(torch.abs(pred_h) + 1e-6) - torch.sqrt(targ_h)) ** 2) + dim_loss

                        # Confidence loss where there is a object
                        # for box_idx in range(self.B):
                        #     p = prediction[pred_idx, row, col, 5*box_idx+4]
                        conf_loss = (pred_p - targ_p) ** 2 + conf_loss

                    else:
                        # Confidence loss where there is no object
                        targ_p = target[pred_idx, row, col, 4]
                        for box_idx in range(self.B):
                            p = prediction[pred_idx, row, col, 5*box_idx+4]
                            no_conf_loss = (p - targ_p) ** 2 + no_conf_loss

        # Class probability loss
        obj = target[:, :, :, 4: 5]
        class_loss = torch.sum(
                                 obj.repeat(1, 1, 1, self.C) * 
                                (prediction[:, :, :, 5*self.B:] - target[:, :, :, 5:]) ** 2
                               ) + class_loss

        loss = (
                self.lambda_coord * centre_loss + 
                self.lambda_coord * dim_loss + 
                conf_loss + 
                self.lambda_noobj * no_conf_loss + 
                class_loss
               ) 

        return loss



class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        # predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        pred_converted   = convert(predictions, self.S, self.B)
        target_converted = convert(target, self.S, B=1)

        # predictions = torch.cat([predictions[:, :, :, 10:], predictions[:, :, :, :10]], dim=3)
        # target      = torch.cat([target[:, :, :, 5:], target[:, :, :, :5]], dim=3)
        # pred_converted = torch.cat([pred_converted[:, :, :, 10:], pred_converted[:, :, :, :10]], dim=3)
        # target_converted = torch.cat([target_converted[:, :, :, 5:], target_converted[:, :, :, :5]], dim=3)
        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(pred_converted[..., 0:4], target_converted[..., 0:4], 'corners')
        iou_b2 = intersection_over_union(pred_converted[..., 5:9], target_converted[..., 0:4], 'corners')
        # print(iou_b1, iou_b2)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 4].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 5:9]
                + (1 - bestbox) * predictions[..., 0:4]
            )
        )

        box_targets = exists_box * target[..., 0:4]

        # Take sqrt of width, height of boxes to ensure that
        # box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        # box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss_xy = self.mse(
            torch.flatten(box_predictions[:, :, :, :2], end_dim=-2),
            torch.flatten(box_targets[:, :, :, :2], end_dim=-2),
        )
        whp = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        wht = torch.sqrt(box_targets[..., 2:4])
        box_loss_wh = self.mse(
            torch.flatten(whp, end_dim=-2),
            torch.flatten(wht, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 9:10] + (1 - bestbox) * predictions[..., 4:5]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 4:5]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 4:5], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 9:10], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 10:], end_dim=-2,),
            torch.flatten(exists_box * target[..., 5:], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss_xy  # first two rows in paper
            + self.lambda_coord * box_loss_wh
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        return loss

if __name__ == '__main__':
    pass

    # criterion =  yoloLoss(7, 2, 20, 5, 0.5)
    # pred = torch.randn((1, 7, 7, 30), requires_grad=True)
    # target = torch.ones((1, 7, 7, 25))

    # loss = criterion((pred + 10).cuda(), target.cuda())
    # loss.backward()
    # print(loss)
    # print(pred.grad.shape)

    # # testing
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # import numpy as np
    # import cv2

    # img = np.zeros((1000, 1000, 3), np.uint8)
    # s = 2
    # for i in range(0, s*500, 500):
    #     img[:, i] = 255
    #     img[i, :] = 255

    # # pred = [[
    # #         [[0.5, 0.6, .2, .6, 1.0], [0.5, 0.7, .2, .6, 1.0]],
    # #         [[0.2, 0.3, .2, .2, 1.0], [0.1, 0.1, .5, .35, 1.0]]
    # #        ],]

    # pred = [
    #         [
    #         [[0.5, 0.6, .2, .6, 1.0, 0.5, 0.6, .2, .6, 1.0], [0.5, 0.7, .2, .6, 1.0, 0.5, 0.7, .2, .6, 1.0]],
    #         [[0.2, 0.3, .2, .2, 1.0, 0.2, 0.3, .2, .2, 1.0], [0.1, 0.1, .5, .35, 1.0, 0.1, 0.1, .5, .35, 1.0]]
    #         ],
    #         [
    #         [[0.5, 0.6, .2, .6, 1.0, 0.5, 0.6, .2, .6, 1.0], [0.5, 0.7, .2, .6, 1.0, 0.5, 0.7, .2, .6, 1.0]],
    #         [[0.2, 0.3, .2, .2, 1.0, 0.2, 0.3, .2, .2, 1.0], [0.1, 0.1, .5, .35, 1.0, 0.1, 0.1, .5, .35, 1.0]]
    #         ],
    #        ]

    # # print(np.array(pred).shape)

    # outo = convert(torch.tensor(pred), s=s, B=2)
    # out = outo.numpy() * 1000
    # out = out.astype(np.int32)
    # print('out:\n', out)

    # iu1 = iou(outo[:, 0, 0], outo[:, 1, 0])
    # iu2 = iou(outo[:, 0, 0], outo[:, 1, 1])
    # iu3 = iou(outo[:, 0, 1], outo[:, 1, 1])
    
    # print('\niou:', iu1, iu2, iu3)

    # # draw boxes
    # for row in range(s):
    #     for col in range(s):
    #         box = out[0, row, col]
    #         # print(box)
    #         cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), color=(0, 255, 255))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pred = torch.ones((2, 7, 7, 30))
    # out = nms(pred, C=20)
    # print(out.shape)

