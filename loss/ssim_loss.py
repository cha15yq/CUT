from torch.autograd import Variable
import torch
from math import exp
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2) ** 2 / float(2*sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, dilation=1):
    kernel_size = window_size + (dilation - 1) * (window_size - 1) - 1
    mu1 = F.conv2d(img1, window, padding=kernel_size//2, dilation=dilation, groups=channel)
    mu2 = F.conv2d(img2, window, padding=kernel_size//2, dilation=dilation, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1* img2, window, padding=kernel_size//2, dilation=dilation, groups=channel) - mu1_mu2

    C1 = (0.01*1) ** 2
    C2 = (0.03*1) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_avg_ms_ssim(img1, img2, level, weights=[1], window_size=5):
    if len(img1.size()) != 4:
        channel = 1
    else:
        (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if len(weights) != level:
        weights = [1] * level
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum()
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        weights = weights.cuda(img1.get_device())
    for i in range(level):
        ssim_value = _ssim(img1, img2, window, window_size, channel, True, 1)
        if i == 0:
            avg_loss = weights[i] * (1.0 - ssim_value)
        else:
            avg_loss += weights[i] * (1.0 - ssim_value)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    return avg_loss
