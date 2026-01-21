from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class QARepConv(nn.Module):
    """Quantization-aware re-parameterizable Conv2d + BN2d block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
        stride=1, padding=0, dilation=1, groups=1, inference_mode=False, num_conv_branches=1):
        super().__init__()
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            # conv branch
            self.rbr_conv = nn.ModuleList(
                self.make_conv(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation, groups=groups)
                for _ in range(num_conv_branches))
            # scale branch
            self.rbr_scale = nn.Conv2d(in_channels, out_channels, 1,
                stride=stride, padding=0, dilation=1, groups=groups, bias=False) \
                if kernel_size > 1 else None
            # skip conn branch
            self.rbr_skip = nn.Identity() \
                if in_channels == out_channels and stride == 1 else None
            # avg pool branch
            self.rbr_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) \
                if in_channels == out_channels and stride == 1 else None
        # final bn
        self.bn = nn.BatchNorm2d(out_channels)

    @staticmethod
    def make_conv(in_channels: int, out_channels: int, kernel_size: int,
        stride=1, padding=0, dilation=1, groups=1):
        conv = nn.Sequential()
        conv.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        conv.bn = nn.BatchNorm2d(out_channels)
        return conv

    @staticmethod
    def fuse_conv(branch: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv2d + BN2d branch."""
        std = (branch.bn.running_var + branch.bn.eps).sqrt()
        t = (branch.bn.weight / std).reshape(-1, 1, 1, 1)
        w = branch.conv.weight * t
        b = branch.bn.bias - branch.bn.weight * branch.bn.running_mean / std
        return w, b

    @staticmethod
    def fuse_skip(in_channels: int, kernel_size: int, groups: int) -> torch.Tensor:
        """Convert Identity to Conv2d."""
        center = (kernel_size - 1) // 2
        if groups == in_channels:  # depthwise
            w = torch.zeros(in_channels, 1, kernel_size, kernel_size)
            for i in range(in_channels):
                w[i, 0, center, center] = 1
        elif groups == 1:  # standard conv
            w = torch.zeros(in_channels, in_channels, kernel_size, kernel_size)
            for i in range(in_channels):
                w[i, i, center, center] = 1
        else:
            raise NotImplementedError("Only depthwise & standard convs are supported")
        return w

    @staticmethod
    def fuse_avg(in_channels: int, kernel_size: int, groups: int):
        """Convert AvgPool to Conv2d."""
        input_dim = in_channels // groups
        w = torch.zeros((in_channels, input_dim, kernel_size, kernel_size))
        for i in range(in_channels):
            w[i, i % input_dim, :, :] = 1.0 / (kernel_size * kernel_size)
        return w

    def reparameterize(self):
        """Re-parameterization for inference."""
        if self.inference_mode:
            return
        in_channels = self.rbr_conv[0].conv.in_channels
        kernel_size = self.rbr_conv[0].conv.kernel_size[0]
        groups = self.rbr_conv[0].conv.groups
        # fuse conv branch
        w_conv, b_conv = 0, 0
        for conv in self.rbr_conv:
            _w_conv, _b_conv = self.fuse_conv(conv)
            w_conv, b_conv = w_conv + _w_conv, b_conv + _b_conv
        # fuse scale branch
        w_scale = F.pad(self.rbr_scale.weight, [(kernel_size - 1) // 2] * 4) \
            if self.rbr_scale is not None else 0
        # fuse skip conn branch
        w_skip = self.fuse_skip(in_channels, kernel_size, groups).to(w_conv.device) \
            if self.rbr_skip is not None else 0
        # fuse avg pool branch
        w_avg = self.fuse_avg(in_channels, kernel_size, groups).to(w_conv.device) \
            if self.rbr_avg is not None else 0
        # merge kernels
        self.reparam_conv = nn.Conv2d(
            self.rbr_conv[0].conv.in_channels, 
            self.rbr_conv[0].conv.out_channels,
            self.rbr_conv[0].conv.kernel_size[0],
            stride=self.rbr_conv[0].conv.stride, 
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation, 
            groups=self.rbr_conv[0].conv.groups, 
            bias=True)
        self.reparam_conv.weight.data = w_conv + w_scale + w_skip + w_avg
        self.reparam_conv.bias.data = b_conv
        # remove unused params
        for p in self.parameters():
            p.detach_()
        self.inference_mode = True
        del self.rbr_conv, self.rbr_scale, self.rbr_skip

    def forward(self, x):
        if self.inference_mode:
            y = self.reparam_conv(x)
        else:
            y = 0
            for conv in self.rbr_conv:
                y += conv(x)
            if self.rbr_scale is not None:
                y += self.rbr_scale(x)
            if self.rbr_skip is not None:
                y += self.rbr_skip(x)
            if self.rbr_avg is not None:
                y += self.rbr_avg(x)
        return self.bn(y)
