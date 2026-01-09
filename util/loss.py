# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py

import logging

import torch
# import torch.nn.functional as F
from torch import nn

# from util.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes



# from detectron2.utils.comm import get_world_size
# from point_features import (
#     get_uncertain_point_coords_with_randomness,
#     point_sample,
# )

#detectron2.projects.point_rend.

# def dice_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         num_masks: float,
#     ):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * (inputs * targets).sum(-1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss.sum() / num_masks


# dice_loss_jit = torch.jit.script(
#     dice_loss
# )  # type: torch.jit.ScriptModule


# def sigmoid_ce_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         num_masks: float,
#     ):
#     """
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     Returns:
#         Loss tensor
#     """
#     loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

#     return loss.mean(1).sum() / num_masks


# sigmoid_ce_loss_jit = torch.jit.script(
#     sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule


# def calculate_uncertainty(logits):
#     """
#     We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
#         foreground class in `classes`.
#     Args:
#         logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
#             class-agnostic, where R is the total number of predicted masks in all images and C is
#             the number of foreground classes. The values are logits.
#     Returns:
#         scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
#             the most uncertain locations having the highest uncertainty score.
#     """
#     assert logits.shape[1] == 1
#     gt_class_logits = logits.clone()
#     return -(torch.abs(gt_class_logits))
