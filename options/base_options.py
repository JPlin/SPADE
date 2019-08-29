import os
import torch
import models
from util import data_utils


class BaseOptions():
    def __init__(self):
        self.opt = {}
        # training related
        self.opt['mode'] = 'train'
        self.opt['workers'] = 0
        self.opt['nThreads'] = 0
        self.opt['start_epoch'] = 0
        self.opt['im_size'] = 256
        # normal|xavier|kaiming|orthogonal
        self.opt['init_type'] = 'xavier'
        self.opt['init_variance'] = 0.02
        # lambda | step | plateau | cosine
        self.opt['lr_policy'] = 'step'
        # no dropout for the generator
        self.opt['no_dropout'] = False
        # number of iter at starting learning rate
        self.opt['niter'] = 50
        # number of iter to linearly decay learning rate to zero
        self.opt['niter_decay'] = 0
        # multiply by a gamma every lr_decay_iters iterations
        self.opt['lr_decay_iters'] = 50
        self.opt['optimizer'] = 'adam'
        self.opt['beta1'] = 0.5
        self.opt['beta2'] = 0.999
        self.opt['lr'] = 0.0002
        # ls | original | hinge
        self.opt['gan_mode'] = 'hinge'
        # log frequency (default: 100)
        self.opt['print_freq'] = 100
        # display frequenry
        self.opt['log_freq'] = 200
        # frequency of saving checkpoints at the end of epochs
        self.opt['save_epoch_freq'] = 5
        # models are saved here
        self.opt['logs_dir'] = './logs'
        # if specified, print more debugging information
        self.opt['verbose'] = False
        # manual seed for rnd
        self.opt['manual_seed'] = 999
        # mean
        self.opt['mean'] = [0.5, 0.5, 0.5]
        # std
        self.opt['std'] = [0.5, 0.5, 0.5]
        # scaling and cropping of images at load time
        self.opt['resize_or_crop'] = 'resize_and_crop'

    def print_options(self):
        message = ''
        message += '-------------------- Options ------------------\n'
        for k, v in sorted(self.opt.items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------- Options -----------------\n'
        print(message)

        # save to disk
        expr_dir = os.path.join(self.opt['logs_dir'], self.opt['model'],
                                self.opt['opt_name'])
        data_utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        return expr_dir

    def parse(self):
        # set gpu ids
        str_ids = self.opt['gpu_ids'].split(',')
        self.opt['gpu_ids'] = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt['gpu_ids'].append(id)
        if len(self.opt['gpu_ids']) > 0:
            torch.cuda.set_device(self.opt['gpu_ids'][0])

        return self.opt

    def update_with_args(self, args):
        for k, v in vars(args).items():
            self.opt[k] = v
        if self.opt['mode'] == 'train':
            self.opt['isTrain'] = True
        else:
            self.opt['isTrain'] = False


if __name__ == '__main__':
    base_option = BaseOptions()
    opt = base_option.parse()
    base_option.print_options()