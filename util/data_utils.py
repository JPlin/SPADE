import os
import shutil
import sys

import numpy as np
import torch

from torchvision.utils import make_grid


class MySampler:
    def __init__(self, data_loader, opt):
        self.loader = data_loader
        self.batch_size = opt['batch_size']
        self.iterator = iter(self.loader)

    def next(self):
        try:
            b = next(self.iterator)
            size = b[0].size(0)
            if size != self.batch_size:
                self.iterator = iter(self.loader)
                b = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            b = next(self.iterator)
        return b

    def len_samples(self):
        return len(self.loader)

    def __len__(self):
        return len(self.loader)


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
    print("saved checkpoint to directory {}".format(os.path.abspath(folder)))


def load_nets(nets, folder):
    '''
    nets: list of model
    folder: 
    '''
    for net_name, v in nets.items():
        if 'optimizer' in v:
            model = v['model']
            path = os.path.join(folder, net_name)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            v['optimizer'].load_state_dict(checkpoint['optimizer'])
    return torch.load(os.path.join(folder, "info"))


def check_paths(save_dir):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def naive_save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint', 'model_best'))


def make_grid_3(img_s, img_a, img_g, style='imagenet'):
    if style == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    imgs = torch.stack([img_s, img_a, img_g])
    imgs = torch.transpose(imgs, 0, 1)
    imgs = torch.reshape(imgs, (-1, *imgs.size()[2:]))
    imgs = make_grid(imgs, 9)
    for t, m, s in zip(imgs, mean, std):
        t.mul_(s).add_(m)
    return imgs


def make_grid_n(img_list, style='imagenet'):
    if style == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    imgs = torch.stack(img_list)
    imgs = torch.transpose(imgs, 0, 1)
    imgs = torch.reshape(imgs, (-1, *imgs.size()[2:]))
    imgs = make_grid(imgs, 9, padding=8)
    for t, m, s in zip(imgs, mean, std):
        t.mul_(s).add_(m)
    return imgs


def unmold_input(tensor,
                 keep_dims=False,
                 channel_first=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
    '''
    input: numpy or torch.Tensor
    keep_dims: output one sample , or all sample
    output: numpy
    '''
    # tensor: torch tensor
    if type(tensor) == torch.Tensor:
        if tensor.size()[1] > 3:
            tensor = tensor[:, :3]
        p = tensor.cpu().detach().numpy()
        p = np.transpose(p, (0, 2, 3, 1))
        p = p * std + mean
        p = p * 255
        p = p.astype(np.uint8)
        if channel_first:
            p = np.transpose(p, (0, 3, 1, 2))
        if keep_dims:
            return p
        else:
            return p[0]
    else:
        shapes = tensor.shape
        if shapes[1] <= 5:
            tensor = tensor[:, :3]
            tensor = tensor.transpose((0, 2, 3, 1))
        p = tensor * std + mean
        return p


def mask2im(mask):
    mask_size = mask.size()
    if mask_size[1] != 3:
        mask = mask.repeat(1, 3, 1, 1)
    mask = (mask - 0.5) / 0.5
    return mask


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            _mkdir(path)
    else:
        _mkdir(paths)


def _mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        check = input(f'Log dir : {path} exists, delete it? (Y/N)')
        if check.lower() == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
