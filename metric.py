# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:27:12 2022

@author: CY
"""
import numpy as np

def metric_iou_dice(prediction, target):
    # [tile_counts, tile_size, nb_classes]
    # prediction = np.zeros(logits.shape)
    # prediction[np.where(np.greater_equal(logits, threshold))] = 1
    inte = np.sum(target * prediction)
    union = np.sum(target) + np.sum(prediction) - inte

    iou = float(inte) / (float(union)+ 0.001)

    l = np.sum(target * target)
    r = np.sum(prediction * prediction)
    dice = (2 * inte) / (l + r+0.0001)
    return iou, dice