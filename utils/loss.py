#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:17:48 2025

@author: hassan
"""


import torch
import torch.nn as nn

from torch.nn import functional as F


def get_loss_module(class_weights=None):
    return NoFussCrossEntropyLoss(weight=class_weights, reduction='none')

class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, reduction='none'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)




