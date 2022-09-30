import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

class MaskedBinaryCrossEntropyLoss(nn.Module):
    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        print(pred.shape)
        print(label.shape)
        mask = label[:, -1, ...].float()
        mask = torch.stack([mask] * (label.shape[1] - 1), dim=1)

        value = label[:, :-1, ...].float()

        total_entropy = F.binary_cross_entropy_with_logits(
            pred, value.float(), reduction="none"
        )

        masked_entropy = total_entropy * (1 - mask)
        count = torch.numel(masked_entropy) - torch.sum(mask)
        return torch.sum(total_entropy) / count
#create loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous()
        #y_pred = y_pred > 0.5
        y_true = y_true.contiguous()

        intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + self.smooth)))

        return loss.mean()

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


"""class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))"""


class ClassifyLoss(nn.Module):
    def __init__(self):
        super(ClassifyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class ComputeLoss(nn.Module):
    def __init__(self, cfg):
        super(ComputeLoss, self).__init__()
        self.cfg = cfg
        self.criterion = {
            "dice": DiceLoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "dice": DiceLoss(),
            "focal": FocalLoss(),
            "smooth": LabelSmoothingLoss(),
            "classify": ClassifyLoss()
        }

    def forward(self, pred, target):
        loss = 0
        for key in self.cfg.LOSS.LOSS_TYPE:
            loss += self.cfg.LOSS.LOSS_WEIGHT[key] * self.criterion[key](pred, target)
        return loss


class ComputeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seg_loss = DiceLoss()
        self.cls_loss = LabelSmoothingLoss()

    def forward(self, pred_cls, pred_seg, target_cls, target_seg):
        loss_cls =  self.cls_loss(pred_cls, target_cls)
        loss_seg = self.seg_loss(pred_seg, target_seg)
        print("LOSS: cls {}, seg {}".format(loss_cls, loss_seg))

        loss = loss_cls + loss_seg*0.5
        return loss


def build_loss(cfg):
    loss = ComputeLoss()
    return loss


if __name__ == '__main__':

    a = torch.rand(1, 3, 3)
    b = torch.rand(1, 3, 3)

    a = SegLoss()(a, b)
