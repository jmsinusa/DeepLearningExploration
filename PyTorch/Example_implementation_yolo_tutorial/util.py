"""
To hold helper functions for the yolo network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2  # Although I'm going to try not to use this.


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    Transform predictions
    :param prediction: output from forward pass of model
    :param inp_dim: image input dimensions
    :param anchors:
    :param num_classes:
    :param CUDA: True if GPU
    :return:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Apply sigmoid activation to class score.
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # The last thing we want to do here, is to resize the detections map to the size of the input image.
    # The bounding box attributes here are sized according to the feature map (say, 13 x 13).
    # If the input image was 416 x 416, we multiply the attributes by 32, or the stride variable.
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """

    :param prediction:
    :param confidence: Supress all with confidence below this
    :param num_classes: Number of classes
    :param nms_conf: NMS IOU threshold
    :return:
    """
    output = None  # Tutorial doesn't do this, but I think it's better.

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask  # Those without high confidence are set to zeros

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]  # Replace the first four predictions with the new box_corner values

    batch_size = prediction.size(0)

    write = False  # flag is used to indicate that we haven't initialized output,
    # a tensor we will use to collect true detections across the entire batch.

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # confidence threshholding
        # NMS

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # This makes a much simpler tensor, with the five positional arguments,
        # then the confidence (of the most confident class) and the arg of the most
        # confidence class
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        if image_pred_.shape[0] == 0:
            continue  # If no detections, stop considering this image.

        # NB: The tutorial uses the following here:
        #         try:
        #             image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        #         except:
        #             continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        for cls in img_classes:
            # perform NMS
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    # Note this takes the current box, and all subsequent boxes
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output

# NB: Tutorial uses:
#     try:
#         return output
#     except:
#         return 0

