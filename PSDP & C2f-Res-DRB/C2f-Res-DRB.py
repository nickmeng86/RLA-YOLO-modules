from other.UniRepLKNet import get_bn, get_conv2d, fuse_bn, merge_dilated_into_large_kernel
from ..modules.conv import Conv
from ..modules.block import C2f
import torch.nn as nn
import torch

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
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

class Res_DRB(nn.Module):
    def __init__(self, dim, act=True) -> None:
        super().__init__()

        # Step 1: Initialize a convolutional layer with 3x3 kernel size
        # 步骤1：初始化一个3x3卷积层
        # The actual convolutional operation has been removed for protection
        # 实际的卷积操作已删除以保护代码

        # Step 2: Initialize other convolutional layers with different dilation rates
        # 步骤2：初始化其他具有不同膨胀率的卷积层
        # These layers are removed to prevent exposing the core logic
        # 这些层已删除以防止泄露核心逻辑

        # Step 3: Initialize 1x1 convolutional layer
        # 步骤3：初始化一个1x1卷积层
        # The actual 1x1 convolution layer has been removed to protect the design
        # 实际的1x1卷积层已删除以保护设计

    def forward(self, x):
        # Step 4: Apply initial 3x3 convolution
        # 步骤4：应用初始的3x3卷积
        # The actual convolution is removed
        # 实际的卷积操作已删除

        # Step 5: Apply dilated convolution blocks with different dilation rates
        # 步骤5：应用具有不同膨胀率的膨胀卷积块
        # The dilated convolution operations have been removed
        # 膨胀卷积操作已删除

        # Step 6: Concatenate outputs of dilated convolutions
        # 步骤6：拼接膨胀卷积的输出
        # Concatenation logic has been removed for protection
        # 拼接逻辑已删除以保护代码

        # Step 7: Apply 1x1 convolution and add residual connection
        # 步骤7：应用1x1卷积并加上残差连接
        # This operation has been removed
        # 此操作已删除
        pass


class C2f_Res_DRB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        # Step 8: Initialize a list of Res_DRB blocks
        # 步骤8：初始化Res_DRB块的列表
        # The list initialization has been removed
        # 列表初始化已删除
        pass

