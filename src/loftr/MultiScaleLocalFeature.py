import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalityChannelAttention(nn.Module):
    def __init__(self, dim=256, winsize=8):
        super(LocalityChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        self.win_pool = nn.AvgPool2d(kernel_size=winsize, stride=winsize//2)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.win_pool(x)
        y = self.mlp(y)
        # hard-sigmoid
        y = F.relu6(y + 3., inplace=True) / 6.
        y = F.interpolate(y, size=(h, w), mode='nearest')
        return x * y


class GroupEncoderBlock(nn.Module):
    def __init__(self, indim=256, outdim=256, bias=True):
        super(GroupEncoderBlock, self).__init__()
        self.conv_d1 = nn.Conv2d(indim, indim, kernel_size=3, stride=1, padding=1, groups=indim, bias=bias)
        self.conv_d2 = nn.Conv2d(indim, indim, kernel_size=5, stride=1, padding=2, groups=indim, bias=bias)
        self.conv_d3 = nn.Conv2d(indim, indim, kernel_size=7, stride=1, padding=3, groups=indim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Conv2d(indim * 4, indim * 4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(indim * 4, outdim//2, kernel_size=1, stride=1, padding=0, bias=bias)
        )
        self.semantic = nn.Sequential(  # LCAM + 3*3conv
            LocalityChannelAttention(dim=indim, winsize=16),
            nn.Conv2d(indim, outdim//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x0 = self.conv_d1(x)  # 3*3
        x1 = self.conv_d2(x)  # 5*5
        x2 = self.conv_d3(x)  # 7*7
        x0 = torch.cat((x, x0, x1, x2), dim=1)
        del x1, x2
        x0 = self.mlp(x0)
        x = self.semantic(x)  # LCAM + 3*3conv
        return torch.cat((x0, x), dim=1)


class MultiScaleLocalFeature(nn.Module):
    def __init__(self, dim=256):
        super(MultiScaleLocalFeature, self).__init__()
        self.local1 = nn.Sequential(
            GroupEncoderBlock(indim=dim, outdim=dim, bias=True),  # LCAM + 3*3conv + cat
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)  # 3*3conv
        )
        self.local2 = nn.Sequential(
            GroupEncoderBlock(indim=dim, outdim=dim, bias=True),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )
        self.local3 = nn.Sequential(
            GroupEncoderBlock(indim=dim, outdim=dim, bias=True),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )
        self.local4 = nn.Sequential(
            GroupEncoderBlock(indim=dim, outdim=dim, bias=True),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.local1(x)  # HFM
        x = self.local2(x)
        x = self.local3(x)
        x = self.local4(x)
        return x

# Share
class MultiScaleLocalFeature_Share(nn.Module):
    def __init__(self, dim=256):
        super(MultiScaleLocalFeature, self).__init__()
        self.local = nn.Sequential(
            GroupEncoderBlock(indim=dim, outdim=dim, bias=True),  # LCAM + 3*3conv + cat
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)  # 3*3conv
        )

    def forward(self, x):
        x = self.local(x)  # HFM
        x = self.local(x)
        x = self.local(x)
        x = self.local(x)
        return x
