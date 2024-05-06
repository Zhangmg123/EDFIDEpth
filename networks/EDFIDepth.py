import torch
import torch.nn as nn
import torch.nn.functional as F

from .edfi_decoder import EDFI
from .PQI import PSP
from .mpvit import mpvit_tiny, mpvit_xsmall, mpvit_small, mpvit_base
########################################################################################################################


class EDFIDepth(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None,
                 frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.max_depth = max_depth
        if version == "tiny":
            self.backbone = mpvit_tiny()
        elif version == "xsmall":
            self.backbone = mpvit_xsmall()
        elif version == "small":
            self.backbone = mpvit_small()
        elif version == "base":
            self.backbone = mpvit_base()

        in_channels = [224, 368, 480, 480]
        out_channels = 128

        self.decoder = Decoder(in_channels, out_channels)

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, imgs):

        conv0, conv1, conv2, conv3, conv4 = self.backbone(imgs)

        out = self.decoder(conv1, conv2, conv3, conv4)

        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return out_depth


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=512, kernel_size=1)

        self.edfi1 = EDFI(out_channels, num_heads=4)
        self.edfi2 = EDFI(out_channels, num_heads=8)
        self.edfi3 = EDFI(out_channels, num_heads=16)
        self.edfi4 = EDFI(out_channels, num_heads=32)

        embed_dim = 512

        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)
        self.skip_conv4 = nn.Conv2d(
            in_channels=in_channels[3], out_channels=out_channels, kernel_size=1)
        self.skip_convq = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        norm_cfg = dict(type='BN', requires_grad=True)
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.psp = PSP(**decoder_cfg)

    def forward(self, x_1, x_2, x_3, x_4):
        q4 = self.se(x_4)
        #####
        q4 = self.skip_conv4(q4)
        x_4 = self.skip_conv4(x_4)
        q3 = self.edfi4(x_4, q4)


        x_3 = self.skip_conv3(x_3)
        q2 = self.edfi3(x_3, q3)

        x_2 = self.skip_conv2(x_2)
        q1 = self.edfi2(x_2, q2)


        x_1 = self.skip_conv1(x_1)
        q0 = self.edfi1(x_1, q1)

        q0 = self.up(q0)

        return q0
