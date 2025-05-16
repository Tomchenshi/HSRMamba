import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from loss import cal_sam
import torch


def sorted_hyperspectral(image):
    B, C, H, W = image.shape
    # 展平空间维度
    batch_flattened = image.view(B, C, -1)  # (B, C, H*W)

    # 对 batch 维度计算光谱维度的平均相关性
    mean_flattened = batch_flattened.mean(dim=0)  # (C, H*W)
    correlation_matrix = torch.corrcoef(mean_flattened)  # (C, C)

    # 计算每个通道的平均相关性并统一排序
    mean_correlation = correlation_matrix.mean(dim=1)  # 每个通道的平均相关性
    sorted_indices = torch.argsort(mean_correlation, descending=True)  # 从高到低排序

    # 使用统一的排序索引对每个样本的通道排序
    sorted_images = []
    for b in range(B):
        sorted_single_image = image[b][sorted_indices, :, :]  # (C, H, W)
        sorted_images.append(sorted_single_image)

    sorted_batch = torch.stack(sorted_images, dim=0)  # (B, C, H, W)

    return sorted_batch

def process_image_transpose(x, channel_size=8, block_size=8):
    B, C, H, W = x.shape

    # 确保图像的高度、宽度和通道数可以被 block_size 整除
    assert H % block_size == 0 and W % block_size == 0 and C % channel_size == 0, \
        "Height, Width and Channels must be divisible by block_size"

    # 计算每个维度上的小块数
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    n_blocks_c = C // channel_size

    # 使用 unfold 将图像划分为多个小块
    unfold = torch.nn.Unfold(kernel_size=block_size, stride=block_size)
    x_unfold = unfold(x)  # 结果形状为 (B, C * block_size * block_size, L)
    L = x_unfold.shape[2]
    x_unfold = x_unfold.view(B, n_blocks_c, channel_size, block_size, block_size, L)
    x_unfold = x_unfold.permute(0, 1, 5, 2, 3, 4).contiguous()  # B, n_blocks_c, L, channel_size, size, size
    x_chw = x_unfold.flatten(1)
    x_cwh = x_unfold.transpose(4, 5).contiguous().flatten(1)
    x_hcw = x_unfold.transpose(3, 4).contiguous().flatten(1)
    x_whc = x_unfold.transpose(3, 5).contiguous().flatten(1)
    x_hwc = x_unfold.permute(0, 1, 2, 4, 5, 3).contiguous().flatten(1)
    x_wch = x_unfold.permute(0, 1, 2, 5, 3, 4).contiguous().flatten(1)

    return x_chw, x_cwh, x_hcw, x_whc, x_hwc, x_wch


def unprocess_transpose(x, x_size, channel_size=8, block_size=8):

    B, D, L = x.shape
    C, H, W = x_size[0], x_size[1], x_size[2]
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    n_blocks_c = C // channel_size
    n_hw = n_blocks_h * n_blocks_w
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=block_size, stride=block_size)  # B C block_size block_size
    chw = x[:, 0].view(B, n_blocks_c, n_blocks_h * n_blocks_w, channel_size, block_size,
                       block_size).permute(0, 1, 3, 4, 5, 2).contiguous().view(B, -1, n_hw)
    cwh = x[:, 1].view(B, n_blocks_c, n_blocks_h * n_blocks_w, channel_size, block_size,
                       block_size).permute(0, 1, 3, 5, 4, 2).contiguous().view(B, -1, n_hw)
    hcw = x[:, 2].view(B, n_blocks_c, n_blocks_h * n_blocks_w, block_size, channel_size,
                       block_size).permute(0, 1, 4, 3, 5, 2).contiguous().view(B, -1, n_hw)
    whc = x[:, 3].view(B, n_blocks_c, n_blocks_h * n_blocks_w, block_size, block_size,
                       channel_size).permute(0, 1, 5, 4, 3, 2).contiguous().view(B, -1, n_hw)
    hwc = x[:, 4].view(B, n_blocks_c, n_blocks_h * n_blocks_w, block_size, block_size,
                       channel_size).permute(0, 1, 5, 3, 4, 2).contiguous().view(B, -1, n_hw)
    wch = x[:, 5].view(B, n_blocks_c, n_blocks_h * n_blocks_w, block_size, block_size,
                       channel_size).permute(0, 1, 5, 3, 4, 2).contiguous().view(B, -1, n_hw)
    chw = fold(chw).view(B, C, -1)
    cwh = fold(cwh).view(B, C, -1)
    hcw = fold(hcw).view(B, C, -1)
    whc = fold(whc).view(B, C, -1)
    hwc = fold(hwc).view(B, C, -1)
    wch = fold(wch).view(B, C, -1)
    return chw, cwh, hcw, whc, hwc, wch



def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1, groups=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=groups)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation, groups=groups)

    else:
       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation, groups=groups)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

