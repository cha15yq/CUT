import scipy.ndimage as ndimage
import numpy as np
import glob
import os
from PIL import Image
import h5py
from tqdm import tqdm
import scipy
import scipy.spatial

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
            if sigma > 15:
                sigma = 15
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


if __name__ == '__main__':
    p_list = glob.glob(os.path.join("processed_qnrf/Train/gt_points", "*.npy"))

    for i in tqdm(p_list):
        points = np.load(i)[:, :2].astype('int')
        im_name = i.replace('gt_points', 'images').replace('npy', 'jpg')
        img = Image.open(im_name)
        w, h = img.size
        density_map = np.zeros((h, w))
        d = np.zeros((h, w))
        for j in range(len(points)):
            point_x, point_y = points[j][0:2]
            if point_y >= h or point_x >= w:
                continue
            d[point_y, point_x] = 1
        density_map += gaussian_filter_density(d)
        print(len(points), density_map.sum())
        with h5py.File(i.replace('points', 'den').replace('img', "GT_img").replace('npy', 'h5'), 'w') as hf:
            hf['density_map'] = density_map

