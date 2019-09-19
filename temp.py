import torch
import argparse
import models.networks as networks
from options import create_options

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--opt_name', required=True, help='options file name')
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--gpu_ids',
                    type=str,
                    default='0,1,2,3',
                    help='gpu id used')
parser.add_argument('--continue',
                    action='store_true',
                    help='continue to train or train from start')
parser.add_argument('--mode', default='train', help='train mode')
args = parser.parse_args()

# get options [opt_name + '_options.py']
options = create_options(args.opt_name)
options.update_with_args(args)
expr_dir = options.print_options()
opt = options.parse()

netG = networks.define_G(opt)
input = torch.randn((1, 4, 512, 512)).cuda()
output = netG(input)
print(input.shape, output.shape)