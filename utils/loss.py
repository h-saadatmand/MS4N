#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:17:48 2025

@author: hassan
"""


import torch
import torch.nn as nn

from torch.nn import functional as F

'''
def get_loss_module(class_weights=None):
    return NoFussCrossEntropyLoss(weight=class_weights, reduction='none')

class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, reduction='none'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

'''



def get_loss_module():
    
    
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
# def get_loss_module():
#         return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


# def l2_reg_loss(model):
#     """Returns the squared L2 norm of output layer of given model"""

#     for name, param in model.named_parameters():
#         if name == 'output_layer.weight':
#             return torch.sum(torch.square(param))


# class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
#     """
#     pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
#     This function satisfies these requirements
#     """

#     def forward(self, inp, target):
#         return F.cross_entropy(inp, target.long(), weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)