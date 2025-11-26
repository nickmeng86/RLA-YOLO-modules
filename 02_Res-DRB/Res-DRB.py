from other.UniRepLKNet import get_bn, get_conv2d, fuse_bn, merge_dilated_into_large_kernel
from ..modules.conv import Conv
from ..modules.block import C2f
import torch.nn as nn
import torch

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        """
        The commented configurations show the original DRB settings from UniRepLKNet.
        Users may experiment with other settings for better performance.
        For Light_DRB, we adopt a simplified version, as detailed below.
        """
        # if kernel_size == 17:
        #     self.kernel_sizes = [5, 9, 3, 3, 3]
        #     self.dilates = [1, 2, 4, 5, 7]
        # elif kernel_size == 15:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 5, 7]
        # elif kernel_size == 13:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 11:
        #     self.kernel_sizes = [5, 5, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 9:
        #     self.kernel_sizes = [5, 5, 3, 3]
        #     self.dilates = [1, 2, 3, 4]
        # elif kernel_size == 7:
        #     self.kernel_sizes = [5, 3, 3]
        #     self.dilates = [1, 2, 3]
        # elif kernel_size == 5:
        #     self.kernel_sizes = [3, 3]
        #     self.dilates = [1, 2]

        # Configurations for Light_DRB
        if kernel_size == 7:
            self.kernel_sizes = [7, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class Light_DRB(nn.Module):
    def __init__(self, dim_in, dim_out, act=True):
        super().__init__()
        self.branch_1 = Conv(dim_in, dim_out, 3, d=1, act=act)
        self.branch_2 = DilatedReparamBlock(dim_in, 5)
        self.branch_3 = DilatedReparamBlock(dim_in, 7)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return out
        
class Res_DRB(nn.Module):
    def __init__(self, dim, act=True):
        super().__init__()      
        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)
        self.light_drb = Light_DRB(dim_in=dim // 2, dim_out=dim, act=act)
        self.conv_1x1 = Conv(dim * 3, dim, k=1, act=act)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        out = self.light_drb(conv_3x3)
        out = self.conv_1x1(out) + x
        return out

# class C2f_Res_DRB(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Res_DRB(self.c) for _ in range(n))


