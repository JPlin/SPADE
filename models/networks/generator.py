"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {'num_upsampling_layers': 'normal'}

    @staticmethod
    def modify_commandline_options(opt):
        default_o = SPADEGenerator.default_opt()
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        SPADEGenerator.modify_commandline_options(self.opt)
        nf = opt['ngf']

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt['use_vae']:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt['z_dim'], 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt['semantic_nc'], 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt['num_upsampling_layers'] == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt['num_upsampling_layers'] == 'normal':
            num_up_layers = 5
        elif opt['num_upsampling_layers'] == 'more':
            num_up_layers = 6
        elif opt['num_upsampling_layers'] == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt['num_upsampling_layers'])

        sw = opt['crop_size'] // (2**num_up_layers)
        sh = round(sw / opt['aspect_ratio'])

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt['use_vae']:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0),
                                self.opt['z_dim'],
                                dtype=torch.float32,
                                device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt['ngf'], self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt['num_upsampling_layers'] == 'more' or \
           self.opt['num_upsampling_layers'] == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt['num_upsampling_layers'] == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {
            'resnet_n_downsample': 4,
            'resnet_n_upsample': 4,
            'resnet_n_blocks': 9,
            'resnet_kernel_size': 3,
            'resnet_initial_kernel_size': 7,
            'norm_G': 'instance'
        }

    @staticmethod
    def modify_commandline_options(opt):
        default_o = Pix2PixHDGenerator.default_opt()
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v

    def __init__(self, opt):
        super().__init__()
        Pix2PixHDGenerator.modify_commandline_options(opt)

        input_nc = opt['input_nc']
        norm_layer = get_nonspade_norm_layer(opt, opt['norm_G'])
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [
            nn.ReflectionPad2d(opt['resnet_initial_kernel_size'] // 2),
            norm_layer(
                nn.Conv2d(input_nc,
                          opt['ngf'],
                          kernel_size=opt['resnet_initial_kernel_size'],
                          padding=0)), activation
        ]

        # downsample
        mult = 1
        for i in range(opt['resnet_n_downsample']):
            model += [
                norm_layer(
                    nn.Conv2d(opt['ngf'] * mult,
                              opt['ngf'] * mult * 2,
                              kernel_size=3,
                              stride=2,
                              padding=1)), activation
            ]
            mult *= 2

        # resnet blocks
        for i in range(opt['resnet_n_blocks']):
            model += [
                ResnetBlock(opt['ngf'] * mult,
                            norm_layer=norm_layer,
                            activation=activation,
                            kernel_size=opt['resnet_kernel_size'])
            ]

        # upsample
        for i in range(opt['resnet_n_upsample']):
            nc_in = int(opt['ngf'] * mult)
            nc_out = int((opt['ngf'] * mult) / 2)
            model += [
                norm_layer(
                    nn.ConvTranspose2d(nc_in,
                                       nc_out,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)), activation
            ]
            # model += [
            #     nn.Upsample(scale_factor=2, mode='bilinear'),
            #     nn.ReflectionPad2d(1),
            #     norm_layer(
            #         nn.Conv2d(nc_in,
            #                   nc_out,
            #                   kernel_size=3,
            #                   stride=1,
            #                   padding=0))
            # ]
            mult = mult // 2

        # final output conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc_out, opt['output_nc'], kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class FeatGenerator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {
            'norm_FG': 'spectralinstance',
            'FG_resnet_kernel_size': 3,
            'FG_keep': 3,
            'FG_c': 256
        }

    @staticmethod
    def modify_commandline_options(opt):
        default_o = FeatGenerator.default_opt()
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v

    def __init__(self, opt):
        super().__init__()
        FeatGenerator.modify_commandline_options(opt)

        input_nc = opt['FG_c']
        keep_conv = opt['FG_keep']
        norm_layer = get_nonspade_norm_layer(opt, opt['norm_FG'])
        activation = nn.ReLU(True)

        model = []
        for i in range(keep_conv):
            model += [
                ResnetBlock(input_nc,
                            norm_layer=norm_layer,
                            activation=activation,
                            kernel_size=opt['FG_resnet_kernel_size'])
            ]
        model += [nn.Conv2d(input_nc, input_nc, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class HairGenerator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {
            'HG_resnet_kernel_size': 3,
            'HG_neck_shape': [64, 4, 4],
            'HG_up_c': [128, 128, 128],
            'HG_keep_c': [128, 64, 32],
            'HG_out_c': 3,
            'HG_drop_out': False
        }

    @staticmethod
    def modify_commandline_options(opt):
        default_o = HairGenerator.default_opt()
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v

    def __init__(self, opt):
        super().__init__()
        HairGenerator.modify_commandline_options(opt)

        self.neck_shape = opt['HG_neck_shape']
        self.up_c = opt['HG_up_c']
        self.keep_c = opt['HG_keep_c']
        self.out_c = opt['HG_out_c']
        self.drop_out = opt['HG_drop_out']

        last_c = self.neck_shape[0]

        self.up_convs = []
        for up_c_ in self.up_c:
            self.up_convs += [
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(last_c, up_c_, 3, 1, 1, bias=False),
                nn.BatchNorm2d(up_c_),
                nn.ReLU(inplace=True)
            ]
            last_c = up_c_
        self.up_convs = nn.Sequential(*self.up_convs)

        self.keep_convs = []
        for keep_c_ in self.keep_c:
            self.keep_convs += [
                nn.Conv2d(last_c, keep_c_, 1, 1, 0, bias=False),
                nn.BatchNorm2d(keep_c_),
                nn.ReLU(inplace=True)
            ]
            last = keep_c_
        if self.drop_out:
            self.keep_convs.append(nn.Dropout2d())
        self.keep_convs = nn.Sequential(*self.keep_convs)

        self.last_conv = nn.Conv2d(last_c, self.out_c, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.fc(input)
        x = x.view([x.size(0)] + self.neck_shape)
        x = self.up_convs(x)
        x = self.keep_convs(x)
        x = self.last_conv(x)
        return x