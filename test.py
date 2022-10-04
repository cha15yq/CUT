import argparse
import torch
import os
from torch.utils.data import DataLoader
from dataset.dataset import Crowd
from model.model import Count
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='new_QNRF/val',
                        help='the directory of the data')

    parser.add_argument('--pretrained', default='pretrained/pcpvt_large.pth',
                        help='the path to the pretrained model')
    parser.add_argument('--model-path', default='history/lr_1e-4_gamma_2_15(1)/best_model.pth',
                        help='the path to the model')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.45, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--device', default='0',
                        help="assign device")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    dataset = Crowd(args.data_dir, 512, args.downsample_ratio, method='val')
    dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = Count(args)
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    model.eval()
    res = []
    step = 0
    for im, gt, size in dataloader:
        im = im.to(device)
        with torch.set_grad_enabled(False):
            result, _, _, _, _, _ = model(im)
            res1 = gt.item() - torch.sum(result).item()
            res.append(res1)
            print('{}/{}: GT:{}, predict:{:.2f}, diff:{:.2f}'.format(step, len(dataset), gt.item(), torch.sum(result).item(),
                                                             res1), size[0])
            step = step + 1
    print('MAE: {:.2f}, MSE:{:.2f}'.format(np.mean(np.abs(res)), np.sqrt(np.mean(np.square(res)))))

