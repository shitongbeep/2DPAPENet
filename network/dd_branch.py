from network.basic_block import convbnlrelui, deconvbnlrelui, weights_init
from network.basic_block import BasicBlockGeo, GeometryFeature, SparseDownSampleClose
import torch.nn as nn
import torch
import pytorch_lightning as pl


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.geoplanes = 3
        self.geofeature = GeometryFeature()

        # *dd_branch network encoder
        self.dd_branch_conv_init = convbnlrelui(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.dd_branch_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer11 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer22 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer33 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer4 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer44 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer5 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer55 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)
        # dd_branch network decoder
        self.dd_branch_decoder_layer4 = deconvbnlrelui(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer3 = deconvbnlrelui(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer2 = deconvbnlrelui(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer1 = deconvbnlrelui(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_conv_uninit = deconvbnlrelui(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_output = convbnlrelui(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        # *others
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        mid_branch_output = input['mid_branch_output']
        mid_branch_confidence = input['mid_branch_confidence']
        dd = torch.cat([d, mid_branch_output], dim=1)

        geo_s1 = input['geo_s1']
        geo_s2 = input['geo_s2']
        geo_s3 = input['geo_s3']
        geo_s4 = input['geo_s4']
        geo_s5 = input['geo_s5']
        geo_s6 = input['geo_s6']

        # *Encoder
        # 2 --> 32
        dd_branch_feature = self.dd_branch_conv_init(dd)
        # iv  64 --> 64
        dd_branch_feature1 = self.dd_branch_encoder_layer1(dd_branch_feature, geo_s1, geo_s2)
        dd_branch_feature1 = self.dd_branch_encoder_layer11(dd_branch_feature1, geo_s2, geo_s2)
        # iii  128 --> 128
        dd_branch_feature2 = self.dd_branch_encoder_layer2(dd_branch_feature1, geo_s2, geo_s3)
        dd_branch_feature2 = self.dd_branch_encoder_layer22(dd_branch_feature2, geo_s3, geo_s3)
        # ii  256 --> 256
        dd_branch_feature3 = self.dd_branch_encoder_layer3(dd_branch_feature2, geo_s3, geo_s4)
        dd_branch_feature3 = self.dd_branch_encoder_layer33(dd_branch_feature3, geo_s4, geo_s4)
        # i  512 --> 512
        dd_branch_feature4 = self.dd_branch_encoder_layer4(dd_branch_feature3, geo_s4, geo_s5)
        dd_branch_feature4 = self.dd_branch_encoder_layer44(dd_branch_feature4, geo_s5, geo_s5)

        dd_branch_feature5 = self.dd_branch_encoder_layer5(dd_branch_feature4, geo_s5, geo_s6)
        dd_branch_feature5 = self.dd_branch_encoder_layer55(dd_branch_feature5, geo_s6, geo_s6)
        # *Decoder
        # 1  1024 --> 512
        dd_branch_feature_decoder4 = self.dd_branch_decoder_layer4(dd_branch_feature5)
        dd_branch_feature_decoder4 = dd_branch_feature_decoder4 + dd_branch_feature4
        # 2  512 --> 256
        dd_branch_feature_decoder3 = self.dd_branch_decoder_layer3(dd_branch_feature_decoder4)
        dd_branch_feature_decoder3 = dd_branch_feature_decoder3 + dd_branch_feature3
        # 3  256 --> 128
        dd_branch_feature_decoder2 = self.dd_branch_decoder_layer2(dd_branch_feature_decoder3)
        dd_branch_feature_decoder2 = dd_branch_feature_decoder2 + dd_branch_feature2
        # 4  128 --> 64
        dd_branch_feature_decoder1 = self.dd_branch_decoder_layer1(dd_branch_feature_decoder2)
        dd_branch_feature_decoder1 = dd_branch_feature_decoder1 + dd_branch_feature1
        # 64 --> 32
        dd_branch_feature_decoder = self.dd_branch_conv_uninit(dd_branch_feature_decoder1)
        dd_branch_feature_decoder = dd_branch_feature_decoder + dd_branch_feature
        # 32 --> 1
        dd_output = self.dd_branch_output(dd_branch_feature_decoder)
        mid_conf, dd_conf = torch.chunk(self.softmax(torch.cat((mid_branch_confidence, dd_output[:, 1:2, ...]), dim=1)), 2, dim=1)

        input['dd_feature'] = dd_branch_feature_decoder
        input['dd_branch_output'] = dd_output[:, 0:1, ...]
        input['dd_branch_confidence'] = dd_output[:, 1:2, ...]
        # input['mid_branch_confidence'] = mid_conf
        input['fuse_output'] = dd_output[:, 0:1, ...] * dd_conf + mid_branch_output * mid_conf

        return input
