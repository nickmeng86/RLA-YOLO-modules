from ..modules.conv import Conv
import torch.nn.functional as F
import torch.nn as nn
import torch

class PSD_Convs(nn.Module):
    """
    PSD-Convs can be viewed as three consecutive "dilated convolution" nodes, 
    which in our diagrams are represented as three PSD-Conv nodes with different 
    dilations for clarity. However, these nodes do not correspond to independent convolution layers.
    
    This is because the parameter-sharing strategy applies the same convolution kernel
    multiple times with different dilations, keeping the weights identical. 
    During forward propagation, the convolutions are applied sequentially, 
    with each output serving as the input to the next convolution. 
    During backpropagation, gradients flow through all dilated convolutions, 
    and because the weights are shared, the gradients from all three convolutions 
    are accumulated to update the same kernel. This enables the model to learn 
    multi-scale features simultaneously during training.
    """
    def __init__(self, channels, kernel_size=3, dilations=[1, 2, 3]):
        super().__init__()
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.paddings = [(kernel_size - 1) * d // 2 for d in dilations]

    def forward(self, x):
        outputs = []
        current = x
        for dilation, pad in zip(self.dilations, self.paddings):
            y = F.conv2d(current, weight=self.conv.weight, bias=None,
                         stride=1, padding=pad, dilation=dilation)
            outputs.append(y)
            current = y
        return outputs  # [y1, y2, y3]


class PSDP(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 2, 3]):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (1 + len(dilations)), c2, 1, 1)
        self.psd_convs = PSD_Convs(c_, dilations=dilations)

    def forward(self, x):
        cv1_out = self.cv1(x)
        psd_out = self.psd_convs(cv1_out)
        cat_out = torch.cat([cv1_out] + psd_out, dim=1)
        out = self.cv2(cat_out)
        return out

