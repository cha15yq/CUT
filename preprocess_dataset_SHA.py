from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import scipy.spatial
import tqdm
import scipy.ndimage as ndimage
import scipy
import h5py


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #print('density map count: ', density.sum())
    return density


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images', 'ground-truth').replace('IMG', "GT_IMG").replace('.jpg', '.mat')
    points = loadmat(mat_path)["image_info"][0][0][0][0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        print('Size change!', im_w, im_h)
        im = cv2.resize(im, (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--origin-dir', default='part_A',
                        help='orginal data directory')
    parser.add_argument('--data-dir', default='test_A',
                        help='processed data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 256
    max_size = 1920

    for phase in ['train', 'test']:
        sub_dir = os.path.join(args.data_dir, phase)
        sub_dir_img = os.path.join(sub_dir, 'images')
        sub_dir_pts = os.path.join(sub_dir, 'gt_points')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            os.makedirs(sub_dir_img)
            os.makedirs(sub_dir_pts)
        if phase == 'train':
            im_list = sorted(glob(os.path.join(args.origin_dir, 'train_data/images', "*.jpg")))
            sub_dir_den = os.path.join(sub_dir, 'gt_den')
            if not os.path.exists(sub_dir_den):
                os.makedirs(sub_dir_den)
        else:
            im_list = sorted(glob(os.path.join(args.origin_dir, 'test_data/images', "*.jpg")))
        for im_path in tqdm.tqdm(im_list):
            im, points = generate_data(im_path)
            name = os.path.basename(im_path)
            if phase == 'train':
                w, h = im.size
                d = np.zeros((h, w))
                for j in range(len(points)):
                    point_x, point_y = points[j][0: 2].astype('int')
                    if point_y >= h or point_x >= w:
                        continue
                    d[point_y, point_x] = 1
                d = gaussian_filter_density(d)
                with h5py.File(os.path.join(sub_dir_den, '{}.h5'.format(name.replace('.jpg', ''))), 'w') as hf:
                    hf['density_map'] = d
                print(name, 'GT_num:', len(points), 'Density_sum: {:.2f}'.format(d.sum()))
            im_save_path = os.path.join(sub_dir_img, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy').replace('images', 'gt_points')
            np.save(gd_save_path, points)
