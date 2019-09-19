import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *
import util.train_utils as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt['gpu_ids']) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt['init_type'], opt['init_variance'])
    return net


def define_G(opt, net_name=None):
    netG_cls = find_network_using_name(
        opt['netG'] if net_name is None else net_name, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt, net_name=None):
    netD_cls = find_network_using_name(
        opt['netD'] if net_name is None else net_name, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, opt)


def define_RES(opt, in_conv=3, net_name=None):
    from models.networks import resnet
    from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
    from torch import nn

    resnet_table = {
        'resnet18': resnet.resnet18,
        'resnet34': resnet.resnet34,
        'resnet50': resnet.resnet50
    }
    norm_layer = SynchronizedBatchNorm2d if opt[
        'bn_type'] == 'sync_batch' else nn.BatchNorm2d
    nete, netd = resnet_table[
        'resnet34' if net_name is not None else net_name](
            num_classes=opt['encoder_channel'],
            in_conv=in_conv,
            norm_layer=norm_layer)
    print('initialize resnet', nete.__class__.__name__,
          netd.__class__.__name__)

    if len(opt['gpu_ids']) > 0:
        assert (torch.cuda.is_available())
        nete.cuda()
        netd.cuda()
    return nete, netd
