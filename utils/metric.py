import torch
import torch.nn as nn


def f1_score(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))


# create f1 score for binary classification
def f1_score_binary(pred, target):
    pred = torch.round(torch.sigmoid(pred))
    return f1_score(pred, target)
