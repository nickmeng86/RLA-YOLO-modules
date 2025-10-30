from ..modules.conv import Conv
import torch.nn.functional as F
import torch.nn as nn
import torch

class PSDP(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 2, 3]) -> None:
    # def __init__(self, c1, c2, dilations=[1, 3, 5]) -> None:
        super().__init__()

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (1 + len(dilations)), c2, 1, 1)
        self.PSD_Conv = nn.Conv2d(in_channels=c_, out_channels=c_, kernel_size=3, stride=1, padding=1, bias=False)
        self.dilations = dilations

    def forward(self, x):
        y = [self.cv1(x)]
        for dilation in self.dilations:
            y.append(F.conv2d(y[-1], weight=self.PSD_Conv.weight, bias=None, dilation=dilation, padding=(dilation * (3 - 1) + 1) // 2))
        return self.cv2(torch.cat(y, 1))
