from model.pvt import *
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class Count(nn.Module):
    def __init__(self, args):
        super(Count, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ch1 = nn.Sequential(nn.Conv2d(128, 256, 1, 1),
                                 nn.ReLU(True))
        self.ch2 = nn.Sequential(nn.Conv2d(320, 256, 1, 1),
                                 nn.ReLU(True))
        self.ch3 = nn.Sequential(nn.Conv2d(512, 256, 1, 1),
                                 nn.ReLU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                   nn.ReLU(True))

        self.den = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 64, 3, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 1, 1, 1),
                                 nn.ReLU())

        self.cls = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                 nn.ReLU(True),
                                 nn.Conv2d(128, 64, 3, 1, 1),
                                 nn.ReLU(True),
                                 nn.Conv2d(64, 1, 1, 1),
                                 nn.Sigmoid())

        self.LA_end1 = SAAM(256, 4, 4, 1)

        self.apply(self._init_weights)
        self.encoder = pcpvt_large_v0(args.pretrained, drop_rate=args.drop, attn_drop_rate=args.drop,
                                      drop_path_rate=args.drop_path)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.encoder(x)
        out_l2 = self.ch1(out[1])
        out_l3 = self.ch2(out[2])
        out_l4 = self.ch3(out[3])

        seg_l4 = self.cls(self.upsample(self.upsample(out_l4)))
        refined_l4 = self.LA_end1(self.upsample(self.upsample(out_l4)) * (seg_l4 >= 0.5)) + self.upsample(self.upsample(out_l4))
        den_l4 = self.den(self.conv3(refined_l4))

        f_l4_l3 = self.conv1(torch.cat([self.upsample(out_l4), out_l3], dim=1)) + out_l3
        seg_l3 = self.cls(self.upsample(f_l4_l3))
        refined_f_l4_l3 = self.LA_end1(self.upsample(f_l4_l3) * (seg_l3 >= 0.5)) + self.upsample(f_l4_l3)
        den_l3 = self.den(self.conv3(refined_f_l4_l3))

        f_l4_l3_l2 = self.conv2(torch.cat([self.upsample(f_l4_l3), out_l2], dim=1)) + out_l2
        seg_l2 = self.cls(f_l4_l3_l2)
        refined_f_l4_l3_l2 = self.LA_end1(f_l4_l3_l2 * (seg_l2 >= 0.5)) + f_l4_l3_l2
        den_l2 = self.den(self.conv3(refined_f_l4_l3_l2))

        return den_l2, den_l3, den_l4, seg_l2, seg_l3, seg_l4

