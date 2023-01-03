from network.basic_block import weights_init
from network.basic_block import GeometryFeature, SparseDownSampleClose
import torch.nn as nn
import torch
import pytorch_lightning as pl


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.geoplanes = 3
        self.geofeature = GeometryFeature()

        self.hidden_layer = self.args.hidden_layer
        # *mid_branch network encoder
        self.mid_branch_conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False), nn.BatchNorm2d(32),
                                             nn.LeakyReLU(0.1, inplace=True),
                                             nn.Conv2d(32, self.hidden_layer // 2, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_layer // 2), nn.LeakyReLU(0.1, inplace=True),
                                             nn.Conv2d(self.hidden_layer // 2, self.hidden_layer, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_layer), nn.LeakyReLU(0.1, inplace=True),
                                             nn.Conv2d(self.hidden_layer, self.hidden_layer, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_layer), nn.LeakyReLU(0.1, inplace=True),
                                             nn.Conv2d(self.hidden_layer, self.hidden_layer // 2, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_layer // 2), nn.LeakyReLU(0.1, inplace=True),
                                             nn.Conv2d(self.hidden_layer // 2, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32),
                                             nn.LeakyReLU(0.1, inplace=True))
        self.mid_branch_output = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(2),
                                               nn.LeakyReLU(0.1, inplace=True))
        # *others
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        position = input['position']
        K = input['K']

        # *生成几何特征通道
        unorm = position[:, 0:1, :, :]  # 沿着x方向 -1 -> 1
        vnorm = position[:, 1:2, :, :]  # 沿着y方向 1 -> -1
        new_shape = (unorm.shape[0], 1, 1, 1)
        f352 = K[:, 1, 1]
        f352 = f352.view(new_shape)
        c352 = K[:, 1, 2]
        c352 = c352.view(new_shape)
        f1216 = K[:, 0, 0]
        f1216 = f1216.view(new_shape)
        c1216 = K[:, 0, 2]
        c1216 = c1216.view(new_shape)
        # 坐标降采样
        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)
        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)
        # sparse depth 降采样保留近处点
        valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352, 1216, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352, 1216, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352, 1216, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352, 1216, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352, 1216, c352, c1216, f352, f1216)

        mid_feature = self.mid_branch_conv(d)
        mid_output = self.mid_branch_output(mid_feature)

        input['geo_s1'] = geo_s1
        input['geo_s2'] = geo_s2
        input['geo_s3'] = geo_s3
        input['geo_s4'] = geo_s4
        input['geo_s5'] = geo_s5
        input['geo_s6'] = geo_s6

        input['mid_branch_output'] = mid_output[:, 0:1, ...]
        input['mid_branch_confidence'] = mid_output[:, 1:2, ...]
        input['mid_branch_feature'] = mid_feature

        return input
