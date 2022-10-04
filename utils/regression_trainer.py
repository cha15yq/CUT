from dataset.dataset import Crowd
from torch.utils.data import DataLoader
import torch
import logging
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from model.model import Count
import numpy as np
import os
import time
from timm.optim import create_optimizer
import random
from loss.ssim_loss import cal_avg_ms_ssim
import torch.nn.functional as F
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=4, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        setup_seed(args.seed)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(0)
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  crop_size=args.crop_size,
                                  d_ratio=self.d_ratio,
                                  method=x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=(args.num_workers * self.device_count),
                                          pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}

        self.model = Count(args)
        self.model.to(self.device)
        self.optimizer = create_optimizer(args, self.model)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)
        self.cls = FocalLoss(gamma=args.gamma)

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 40 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 40)
            self.epoch = epoch
            self.train_epoch()
            if epoch >= args.start_val and epoch % args.val_epoch == 0:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_ssim_l2 = AverageMeter()
        epoch_ssim_l3 = AverageMeter()
        epoch_ssim_l4 = AverageMeter()

        epoch_tv_l2 = AverageMeter()
        epoch_tv_l3 = AverageMeter()
        epoch_tv_l4 = AverageMeter()

        epoch_seg_l2 = AverageMeter()
        epoch_seg_l3 = AverageMeter()
        epoch_seg_l4 = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        for step, (img, den_map, b_map) in enumerate(
                self.dataloaders['train']):
            inputs = img.to(self.device)
            gt_den_map = den_map.to(self.device)
            gt_bg_map = b_map.to(self.device)

            with torch.set_grad_enabled(True):
                N = inputs.shape[0]
                pred_den_map_l2, pred_den_map_l3, pred_den_map_l4, seg_map_l2, seg_map_l3, seg_map_l4 = self.model(
                    inputs)

                loss_ssim_l2 = cal_avg_ms_ssim(pred_den_map_l2 * gt_bg_map, gt_den_map * gt_bg_map, level=3,
                                               window_size=3)
                loss_ssim_l3 = cal_avg_ms_ssim(pred_den_map_l3 * gt_bg_map, gt_den_map * gt_bg_map, level=3,
                                               window_size=3)
                loss_ssim_l4 = cal_avg_ms_ssim(pred_den_map_l4 * gt_bg_map, gt_den_map * gt_bg_map, level=3,
                                               window_size=3)

                mu_normed_l2 = get_normalized_map(pred_den_map_l2)
                mu_normed_l3 = get_normalized_map(pred_den_map_l3)
                mu_normed_l4 = get_normalized_map(pred_den_map_l4)
                gt_mu_normed = get_normalized_map(gt_den_map)

                loss_tv_l2 = (nn.L1Loss(reduction='none')(mu_normed_l2, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
                loss_tv_l3 = (nn.L1Loss(reduction='none')(mu_normed_l3, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
                loss_tv_l4 = (nn.L1Loss(reduction='none')(mu_normed_l4, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)

                loss_seg_l2 = self.cls(seg_map_l2, gt_bg_map)
                loss_seg_l3 = self.cls(seg_map_l3, gt_bg_map)
                loss_seg_l4 = self.cls(seg_map_l4, gt_bg_map)

                loss = (loss_ssim_l4 + loss_ssim_l3) * 0.5 + loss_seg_l2 + loss_seg_l3 + loss_seg_l4 + loss_ssim_l2 \
                       + 0.01 * loss_tv_l2 + 0.005 * (loss_tv_l3 + loss_seg_l3)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                gt_counts = torch.sum(gt_den_map.view(inputs.shape[0], -1), dim=1).detach().cpu().numpy()
                pre_count = torch.sum(pred_den_map_l2.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gt_counts
                epoch_loss.update(loss.item(), N)
                epoch_mae.update(np.mean(np.abs(res)), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_ssim_l2.update(loss_ssim_l2.item(), N)
                epoch_tv_l2.update(loss_tv_l2.item(), N)
                epoch_ssim_l3.update(loss_ssim_l3.item(), N)
                epoch_tv_l3.update(loss_tv_l3.item(), N)
                epoch_ssim_l4.update(loss_ssim_l4.item(), N)
                epoch_tv_l4.update(loss_tv_l4.item(), N)

                epoch_seg_l2.update(loss_seg_l2.item(), N)
                epoch_seg_l3.update(loss_seg_l3.item(), N)
                epoch_seg_l4.update(loss_seg_l4.item(), N)

        logging.info(
            'Epoch {} Train, Loss: {:.2f}, MSE: {:.2f}, MAE: {:.2f}, \nlevel2: ssim: {:.4f} seg: {:.4f} tv:{:.4f};\nlevel3: ssim: {:.4f} seg: {:.4f} tv:{:.4f};\nlevel4: ssim: {:.4f} seg: {:.4f} tv:{:.4f};  Cost: {:.1f} sec '
                .format(self.epoch, epoch_loss.getAvg(), np.sqrt(epoch_mse.getAvg()), epoch_mae.getAvg(),
                        epoch_ssim_l2.getAvg(), epoch_seg_l2.getAvg(), epoch_tv_l2.getAvg(), epoch_ssim_l3.getAvg(),
                        epoch_seg_l3.getAvg(), epoch_tv_l3.getAvg(), epoch_ssim_l4.getAvg(), epoch_seg_l4.getAvg(),
                        epoch_tv_l4.getAvg(),
                        (time.time() - epoch_start)))

        if self.epoch % 2 == 0:
            model_state_dict = self.model.state_dict()
            save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dict,
            }, save_path)
            self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, gt_counts, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.shape[0] == 1
            with torch.set_grad_enabled(False):
                den_map, _, _, _, _, _ = self.model(inputs)

                res = gt_counts[0].item() - torch.sum(den_map).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f},  Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        model_state_dict = self.model.state_dict()

        if (mae + mse) < (self.best_mae + self.best_mse):
            self.best_mae = mae
            self.best_mse = mse
            torch.save(model_state_dict, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.epoch)))
            logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))

        print("Best Result: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))


def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = density_map / (mu_sum + 1e-6)
    return mu_normed

