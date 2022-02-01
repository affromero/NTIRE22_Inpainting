"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE


class Attention(nn.Module):
    def __init__(self, ch, use_sn, with_attn=False):
        super(Attention, self).__init__()
        # Channel multiplier
        self.with_attn = with_attn 
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        out = self.gamma * o + x
        if self.with_attn:
            return out, beta
        else:
            return out

class Attn_Aggr(nn.Module):
    def __init__(self, ch_in, ch_att, use_sn, rate=1):
        super(Attn_Aggr, self).__init__()
        # Channel multiplier
        self.ch = ch_att
        self.ch_in = ch_in
        self.down = nn.Conv2d(self.ch_in, self.ch, kernel_size=rate, padding=rate//4, stride=rate)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        self.up = nn.ConvTranspose2d(self.ch, self.ch_in, kernel_size=rate, padding=0, stride=rate)
        if use_sn:
            self.down = spectral_norm(self.down)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
            self.up = spectral_norm(self.up)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, att):
        x_down = self.down(x)
        g = F.max_pool2d(self.g(x_down), [2,2])
        g = g.view(-1, self. ch // 2, x_down.shape[2] * x_down.shape[3] // 4)
        o = self.o(torch.bmm(g, att.transpose(1,2)).view(-1, self.ch // 2, x_down.shape[2], x_down.shape[3]))
        return self.gamma * self.up(o) + x
        

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out,attention
        else:
            return out

class DMFB(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), dilation=1, kernel_size=3):
        super().__init__()
        pw_d = dilation * (kernel_size - 1) // 2
        pw = (kernel_size - 1) // 2
        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, 64, kernel_size=kernel_size, dilation=1)),
            activation)
        self.conv_d1 = nn.Sequential(
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=1))
        self.conv_d2 = nn.Sequential(
                nn.ReflectionPad2d(2*(kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=2))
        self.conv_d4 = nn.Sequential(
                nn.ReflectionPad2d(4*(kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=4))
        self.conv_d8 = nn.Sequential(
                nn.ReflectionPad2d(8*(kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=8))
        self.conv_k2 = nn.Sequential(
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=1))
        self.conv_k3 = nn.Sequential(
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=1))
        self.conv_k4 = nn.Sequential(
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(64, 64, kernel_size=kernel_size, dilation=1))
        self.conv_tail = nn.Sequential(
                norm_layer(nn.Conv2d(256, 256, kernel_size=1, dilation=1)))

    def forward(self, input):
        x = self.conv_head(input)
        x1 = self.conv_d1(x)
        x2 = self.conv_d2(x)
        x2 = self.conv_k2(x2 + x1) 
        x3 = self.conv_d4(x)
        x3 = self.conv_k3(x3 + x2)
        x4 = self.conv_d8(x)
        x4 = self.conv_k4(x4 + x3)
        x = torch.cat([x1,x2,x3,x4], 1)
        x = self.conv_tail(x)
        out = x + input
        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, semantic_nc=None, groups=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        if semantic_nc is None:
            semantic_nc = opt.semantic_nc

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, groups=groups)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, groups=groups)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, groups=groups)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class GC_SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, semantic_nc=None, groups=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        if semantic_nc is None:
            semantic_nc = opt.semantic_nc

        # create conv layers
        self.conv_0 = GatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, activation=None)
        self.conv_1 = GatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, activation=None)
        if self.learned_shortcut:
            self.conv_s = GatedConv2dWithActivation(fin, fout, kernel_size=1, bias=False, activation=None)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = GatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, norm_layer=spectral_norm, activation=None)
            self.conv_1 = GatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, norm_layer=spectral_norm, activation=None)
            if self.learned_shortcut:
                self.conv_s = GatedConv2dWithActivation(fin, fout, kernel_size=1, bias=False, norm_layer=spectral_norm, activation=None)


        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), dilation=1, kernel_size=3):
        super().__init__()

        pw_d = dilation * (kernel_size - 1) // 2
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw_d),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, dilation=dilation)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, dilation=1))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out

class ResnetBlock_GC(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), dilation=1, kernel_size=3, groups=1):
        super().__init__()

        pw_d = dilation * (kernel_size - 1) // 2
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw_d),
            GatedConv2dWithActivation(dim, dim, kernel_size=kernel_size, dilation=dilation,
                                    groups=groups, norm_layer=norm_layer, activation=activation),
            nn.ReflectionPad2d(pw),
            GatedConv2dWithActivation(dim, dim, kernel_size=kernel_size, dilation=1, 
                                    groups=groups, norm_layer=norm_layer, activation=None)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                bias=True, norm_layer=None, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        if norm_layer is not None:
            self.conv2d = norm_layer(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.mask_conv2d = norm_layer(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        else:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        
        return x


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
