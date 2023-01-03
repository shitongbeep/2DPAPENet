import torch.nn as nn
import pytorch_lightning as pl
from utils.criteria import FeatureDistance, Distance
from network.basic_block import weights_init
from torch.nn.parameter import Parameter
import torch


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.feat_layer = 32

        self.distance = FeatureDistance()
        self.loss = Distance()
        self.mid_output = nn.Sequential(nn.Conv2d(self.feat_layer, 1, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(1),
                                        nn.LeakyReLU(0.1, inplace=True))
        self.edgeGx = nn.Conv2d(1, 1, 3, padding=1, padding_mode='replicate')
        self.edgeGy = nn.Conv2d(1, 1, 3, padding=1, padding_mode='replicate')
        edgeweightGx = torch.tensor([-1, 0, 1, -2, 0, 2, -1, 0, 1]).view((3, 3)).unsqueeze(0).unsqueeze(0).float()
        edgeweightGy = torch.tensor([1, 2, 1, 0, 0, 0, -1, -2, -1]).view((3, 3)).unsqueeze(0).unsqueeze(0).float()
        self.edgeGx.weight = Parameter(edgeweightGx, requires_grad=False)
        self.edgeGy.weight = Parameter(edgeweightGy, requires_grad=False)
        self.edgeGx.weight.requires_grad = False
        self.edgeGy.weight.requires_grad = False
        weights_init(self)

    def forward(self, input):
        gt = input['gt']
        cd_branch_output = input['cd_branch_output'].detach()
        # mid_output = input['mid_branch_output']
        mid_feature = input['mid_branch_feature']
        mid_output = self.mid_output(mid_feature)

        cd_edge_x = self.edgeGx(cd_branch_output)
        cd_edge_y = self.edgeGy(cd_branch_output)
        mid_edge_x = self.edgeGx(mid_output)
        mid_edge_y = self.edgeGy(mid_output)

        cd_edge = torch.sqrt(cd_edge_x**2 + cd_edge_y**2)
        mid_edge = torch.sqrt(mid_edge_x**2 + mid_edge_y**2)
        input['cd_edge'] = cd_edge
        input['mid_edge'] = mid_edge

        distance = self.distance(cd_edge, mid_edge)
        loss = self.loss(mid_output, cd_branch_output, gt) + distance

        # input['distance'] = distance
        input['distillation_loss'] = loss

        return input
