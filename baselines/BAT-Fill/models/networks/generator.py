"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock_GC as ResnetBlock_GC
from models.networks.architecture import GC_SPADEResnetBlock as GC_SPADEResnetBlock
from models.networks.architecture import GatedConv2dWithActivation as GatedConv2d
from models.networks.architecture import Attention as Attention

class UpsamplerGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadeposition3x3')
        parser.add_argument('--resnet_n_blocks', type=int, default=6, help='number of residual blocks in the global generator network')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_cuda = (len(self.opt.gpu_ids) > 0) and (-1 not in self.opt.gpu_ids)
        nf = opt.ngf
        # final_nc = nf
        activation = nn.ReLU(False)
        norm_layer = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        self.fc = GatedConv2d(3, 4 * nf, 3, padding=1, norm_layer=norm_layer, activation=activation)
        self.fc2 = GatedConv2d(4 * nf, 4 * nf, 3, padding=1, norm_layer=norm_layer, activation=activation)
        self.encoder = nn.Sequential(nn.ReflectionPad2d(2),
                    GatedConv2d(self.opt.input_nc, nf, kernel_size=5, padding=0, norm_layer=norm_layer, activation=activation),
                    GatedConv2d(nf, 2*nf, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation=activation),
                    GatedConv2d(2*nf, 4*nf, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation=activation))
        self.fuse_conv = GatedConv2d(8 * nf, 4 * nf, 3, padding=1, norm_layer=norm_layer, activation=activation)
        res_blocks = []
        dilations = [1,1,2,4,8,16]
        for i in range(opt.resnet_n_blocks):
            res_blocks += [ResnetBlock_GC(4 * nf,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  dilation=dilations[i],
                                  kernel_size=3)]
        self.res_blocks = nn.Sequential(*res_blocks)
        if self.opt.use_attention:
            self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)

        self.sp0 = GC_SPADEResnetBlock(4 * nf, 4 * nf, opt)
        self.sp1 = GC_SPADEResnetBlock(4 * nf, 2 * nf, opt)


        self.up = nn.Upsample(scale_factor=2)
        self.conv_img = nn.Conv2d(nf*2, 3, 3, padding=1)

    def forward(self, input, z=None):
        masked_img = input[0]
        tran_sample = input[1]
        image, mask = masked_img[:,:3], masked_img[:,3:4]
        seg = masked_img
        b, c, h, w = masked_img.shape
        # print(x.shape)
        x = self.fc(tran_sample)
        x = self.up(x)
        x = self.fc2(x)
        x_enc = self.encoder(masked_img)
        x = self.fuse_conv(torch.cat([x, x_enc], 1))
        x = self.res_blocks(x)
        if self.opt.use_attention:
            x = self.attn(x)
        x = self.up(x)
        x = self.sp0(x, seg)
        x = self.up(x)
        x = self.sp1(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return tran_sample, x