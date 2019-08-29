"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.train_utils as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {'netD_subarch': 'n_layer', 'num_D': 2}

    @staticmethod
    def modify_commandline_options(opt):
        default_o = MultiscaleDiscriminator.default_opt()
        # define properties of each discriminator of the multiscale discriminator
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v
        subnetD = util.find_class_in_module(
            opt['netD_subarch'] + 'discriminator',
            'models.networks.discriminator')
        subnetD.modify_commandline_options(opt)

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        MultiscaleDiscriminator.modify_commandline_options(self.opt)

        for i in range(opt['num_D']):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt['netD_subarch']
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' %
                             subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input,
                            kernel_size=3,
                            stride=2,
                            padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def default_opt():
        return {'n_layers_D': 4}

    @staticmethod
    def modify_commandline_options(opt):
        default_o = NLayerDiscriminator.default_opt()
        for k, v in default_o.items():
            if k not in opt.keys():
                opt[k] = v

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        NLayerDiscriminator.modify_commandline_options(self.opt)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[
            nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, False)
        ]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[
                norm_layer(
                    nn.Conv2d(nf_prev,
                              nf,
                              kernel_size=kw,
                              stride=stride,
                              padding=padw)),
                nn.LeakyReLU(0.2, False)
            ]]

        sequence += [[
            nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
