from network._2dpaenet import get_model as _2dpaenet_backbone
from network.base_model import LightningBaseModel
from network.basic_block import weights_init, DySPN
import torch


class get_model(LightningBaseModel):

    def __init__(self, args):
        super(get_model, self).__init__(args)
        assert args.dilation_rate == 2, "only support ditation_rate = 2"
        self.args = args
        # with torch.no_grad():
        self.backbone = _2dpaenet_backbone(args)
        self.dd_dyspn = DySPN(32, 7, 6)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        # with torch.no_grad():
        input = self.backbone(input)

        coarse_depth = input['fuse_output']
        rd_feature = input['dd_feature']
        rd_refined_depth = self.dd_dyspn(rd_feature, coarse_depth, d)
        input['refined_depth'] = rd_refined_depth

        return input
