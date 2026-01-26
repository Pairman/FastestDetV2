import torch
import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 =  nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(output_channels),
                                      nn.ReLU(inplace=True)
                                     )
    
    def forward(self, x):
        return self.conv1x1(x)

class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups = input_channels, bias = False),
                                     nn.BatchNorm2d(input_channels),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_channels)
                                    ) 
    
    def forward(self, x):
        return self.conv5x5(x)

class SPP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)

        self.S1 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.S2 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.S3 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.output = nn.Sequential(nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                   )
                                   
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):    
        x = self.Conv1x1(x)

        y1 = self.S1(x)
        y2 = self.S2(x)
        y3 = self.S3(x)

        y = torch.cat((y1, y2, y3), dim=1)
        y = self.relu(x + self.output(y))

        return y

class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num):
        super(DetectHead, self).__init__()
        self.conv1x1 =  Conv1x1(input_channels, input_channels)

        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, 4)
        self.cls_layers = Head(input_channels, category_num)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1x1(x)
        
        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))

        return torch.cat((obj, reg, cls), dim =1)



from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.shufflenetv2.shufflenetv2 import ShuffleNetV2

class FastestDetV1(nn.Module):
    def __init__(self, num_classes: int, load_weights: bool=False):
        super(FastestDetV1, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, not load_weights)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spp = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
        self.det = DetectHead(self.stage_out_channels[-2], num_classes)

    def forward(self, x):
        p1, p2, p3 = self.backbone(x)
        p1 = self.avg_pool(p1)
        p3 = self.upsample(p3)
        p = torch.cat((p1, p2, p3), dim=1)
        y = self.spp(p)
        return self.det(y)