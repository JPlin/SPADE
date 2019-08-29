import os
import shutil
from collections import OrderedDict

import torch
from util import train_utils


class BaseModel():
    # modify parser to add command line options
    # and change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_trian):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt['gpu_ids']
        self.use_gpu = True if len(self.gpu_ids) > 0 else False
        self.isTrain = True if opt['mode'] == 'train' else False
        self.device = torch.device(
            f'cuda:{self.gpu_ids[0]}' if self.gpu_ids else torch.device('cpu'))
        print('using cuda ', self.gpu_ids)
        self.save_dir = os.path.join(opt['logs_dir'], opt['name'])
        if opt['resize_or_crop'] != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu else torch.ByteTensor

        # attributes that needed to be registered
        # log names
        self.loss_dict = OrderedDict()
        self.vis_dict = OrderedDict()
        self.hist_dict = OrderedDict()
        self.fig_dict = OrderedDict()
        # train names
        self.train_nets = []
        self.image_paths = []
        self.optimizers = []
        # net names
        self.model_dict = {}
        self.fix_net_dict = {}

    def set_input(self, input):
        pass

    def forward(self):
        pass

    # load and print networks
    # create sheduler
    def setup(self, opt, expr_dir):
        # set fix net eval()
        for _, v in self.fix_net_dict.items():
            self.load_nets_by_path(v['net'], v['path'])
            v['net'].eval()

        # initialize with exist model
        info = {'epoch': opt['start_epoch'], 'step': 0}
        if opt['mode'] == 'test' or opt['continue']:
            info = self.load_nets(self.model_dict,
                                  self.find_last_cp_dir(expr_dir))
            info['epoch'] += 1

        # setup scheduler
        if opt['mode'] == 'train':
            self.schedulers = [
                train_utils.get_scheduler(optimizer, opt)
                for optimizer in self.optimizers
            ]

        # display parameters
        self.print_networks(opt['verbose'])
        return info

    # make models eval mode during test time
    def eval(self):
        for name in self.train_net_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, warpping 'forward' in no_grad() so we don't save intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # optimize one step
    def optimize_parameters(self):
        pass

    # get last checkpoint dir
    def find_last_cp_dir(self, expr_dir):
        cp_list = os.listdir(expr_dir)
        cp_list = [x for x in cp_list if x.startswith('epoch')]
        cp_list = sorted(cp_list, key=lambda x: int(x[5:]), reverse=True)
        return os.path.join(expr_dir, cp_list[0])

    # udpate learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheculer in self.schedulers:
            scheculer.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('new learning rate = %.7f' % lr)

    # return training losses/errors.
    def get_current_losses(self):
        return self.loss_dict

    # return visualize image
    def update_visuals(self):
        pass

    def get_current_visuals(self):
        self.update_visuals()
        return self.vis_dict

    def get_current_hist(self):
        return self.hist_dict

    def update_figs(self):
        pass

    def get_current_fig(self):
        self.update_figs()
        return self.fig_dict

    def get_current_log(self):
        log_rect = OrderedDict()
        log_rect['scalar'] = self.get_current_losses()
        self.update_visuals()
        log_rect['images'] = self.get_current_visuals()
        log_rect['histogram'] = self.get_current_hist()
        self.update_figs()
        log_rect['figure'] = self.get_current_fig()
        return log_rect

    @staticmethod
    # save networks
    def save_nets(nets, info, folder):
        """
        :param nets:  list of {'net_name':(model,optimizer)}
        :param info:  any other stats like epoch,loss,...
        :param folder: directory to save model and info
        :return:
        """
        os.makedirs(folder, exist_ok=True)
        for net_name, v in nets.items():
            if 'optimizer' in v:
                model = v['model']
                optimizer = v['optimizer']
                path = os.path.join(folder, net_name)
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
        torch.save(info, os.path.join(folder, "info"))
        print("saved checkpoint to directory {}".format(
            os.path.abspath(folder)))
        # only save last 10 epoch of models
        subfolder, now_epoch = folder.split('epoch')
        delete_path = os.path.join(subfolder,
                                   'epoch' + str(int(now_epoch) - 10))
        if os.path.isdir(delete_path):
            shutil.rmtree(delete_path)

    @staticmethod
    # load networks
    def load_nets(nets, folder):
        '''
        nets: list of model
        folder: 
        '''
        for net_name, v in nets.items():
            if 'optimizer' in v:
                print(f'load model from {os.path.join(folder, net_name)}!')
                model = v['model']
                path = os.path.join(folder, net_name)
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['state_dict'])
                v['optimizer'].load_state_dict(checkpoint['optimizer'])
        return torch.load(os.path.join(folder, "info"))

    @staticmethod
    # load exist networks
    def load_nets_by_path(net, path):
        checkpoint = torch.load(path,
                                map_location=lambda storage, loc: storage)
        weight = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            name = k[7:] if 'module.' in k else name
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    # print network information
    def print_networks(self, verbose):
        print('-------- Networks initialized --------')
        for net in self.train_nets:
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' %
                  ('net' + name, num_params / 1e6))
        print('--------------------------------------')

    # set requires_grad = False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
