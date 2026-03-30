import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import bisect

class MultiStepCosineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        """MultiStepLR combined with CosineAnnealingLR.

        LR remains constant before at first, then decays by cosine curves 
        between milestones, and stays constant after the last one.
        """
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # initial lr before the first milestone
        if self.last_epoch < self.milestones[0]:
            return self.base_lrs
        # constant lr after the last milestone
        if self.last_epoch >= self.milestones[-1]:
            scale = self.gamma ** (len(self.milestones) - 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        # cosine decay between milestones
        idx = bisect.bisect_right(self.milestones, self.last_epoch)
        start_epoch = self.milestones[idx - 1]
        end_epoch = self.milestones[idx]
        start_scale = self.gamma ** (idx - 1)   # last lr scale
        end_scale = self.gamma ** idx           # next lr scale
        t = self.last_epoch - start_epoch
        T = end_epoch - start_epoch
        scale = 0.5 * (start_scale + end_scale) + \
                0.5 * (start_scale - end_scale) * math.cos(math.pi * t / T)
        return [base_lr * scale for base_lr in self.base_lrs]
