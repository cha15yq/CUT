from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import random
import torchvision.transforms.functional as F
import glob
from torchvision import transforms
import numpy as np
import math
import h5py
import cv2


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(Dataset):
    def __init__(self, root, crop_size, d_ratio, method='train'):
        self.imlist = sorted(glob.glob(os.path.join(root, 'images/*.jpg')))
        self.c_size = crop_size
        self.d_ratio = d_ratio
        self.root = root
        assert self.c_size % self.d_ratio == 0, f"crop size {crop_size} should be divided by downsampling ratio {d_ratio}. "
        if method not in ['train', 'val']:
            raise Exception('Method is not implemented!')
        self.method = method
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.452016860247, 0.447249650955, 0.431981861591],
                                 [0.23242045939, 0.224925786257, 0.221840232611])
        ])

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, item):
        im_path = self.imlist[item]
        name = os.path.basename(im_path).split('.')[0]
        gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))
        img = Image.open(im_path).convert('RGB')
        keypoints = np.load(gd_path)
        if self.method == 'train':
            den_path = os.path.join(self.root, 'gt_den', '{}.h5'.format(name))
            den_map = h5py.File(den_path, 'r')['density_map']
            return self.train_transform(img, den_map)

        elif self.method == 'val':
            w, h = img.size
            new_w = math.ceil(w / 32) * 32
            new_h = math.ceil(h / 32) * 32
            img = img.resize((new_w, new_h), Image.BICUBIC)
            return self.transform(img), len(keypoints), name

    def train_transform(self, img, d_map):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        d_map = d_map[i: (i + h), j: (j + w)]
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        d_map = d_map.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))

        if random.random() > 0.5:
            img = F.hflip(img)
            d_map = np.fliplr(d_map)
        b_map = (d_map > 1e-3).astype(np.float32)

        return self.transform(img), torch.from_numpy(d_map.copy()).float().unsqueeze(0), \
               torch.from_numpy(b_map.copy()).float().unsqueeze(0)
