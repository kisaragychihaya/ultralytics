from torch import nn
import torch
from ultralytics.nn.modules.conv import Conv
class RFBconv(nn.Module):
    def __init__(self, c1):
        super(RFBconv, self).__init__()
        # 第一条分支
        c2 = c1
        self.branch1 = nn.Sequential(
            Conv(c1,c2,1,1),
            nn.Conv2d(c2, c2, kernel_size=3, dilation=1, padding=1)
        )

        # 第二条分支
        self.branch2 = nn.Sequential(
            Conv(c1, c2, 1,1),
            nn.Conv2d(c2, c2, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(c2, c2, kernel_size=3, dilation=3, padding=3)
        )

        # 第三条分支
        self.branch3 = nn.Sequential(
            Conv(c1, c2, 1,1),
            nn.Conv2d(c2, c2, kernel_size=5, dilation=1, padding=2),
            nn.Conv2d(c2, c2, kernel_size=3, dilation=5, padding=5)
        )

        # 1x1卷积用于通道数变换
        self.conv1x1 = Conv(3 * c2, c2, 1)

    def forward(self, x):
        # 分别对三条分支进行前向传播
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # 将三个分支的结果连接在一起
        concatenated = torch.cat((out1, out2, out3), dim=1)

        # 使用1x1卷积进行通道数变换
        result = self.conv1x1(concatenated)
        result = result + x

        return result


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

