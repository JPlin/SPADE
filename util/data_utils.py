import os
import shutil
import sys

from skimage import color
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


def make_grid_n(img_list, style='imagenet'):
    if style == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if isinstance(img_list, list):
        imgs = torch.stack(img_list)
        imgs = torch.transpose(imgs, 0, 1)
        imgs = torch.reshape(imgs, (-1, *imgs.size()[2:]))
        imgs = make_grid(imgs, 9, padding=8)
        for t, m, s in zip(imgs, mean, std):
            t.mul_(s).add_(m)
    elif img_list.__class__ == torch.Tensor:
        tensor = img_list.detach().cpu()
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        imgs = make_grid(tensor, normalize=True)
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


def vis_orient(dxdy, ret_type='tensor'):
    '''
    visualize dxdy in HSV space and translate to RGB space
    In:
    - dxdy: 
        np.array or torch.tensor
        [B,2,H,W] or [B,H,W,2] or [2,H,W] range [-1, 1]
    Out:
    - RGB: 
        np.array [B,3,H,W] 
        np.uint8
    '''
    assert len(dxdy.shape) == 3 or len(dxdy.shape) == 4
    # to numpy array
    if dxdy.__class__ == torch.Tensor:
        dxdy = dxdy.detach().cpu().numpy()
    # to batch
    if len(dxdy.shape) == 3:
        dxdy = dxdy[np.newaxis, :]
    if dxdy.shape[-1] == 2:
        dxdy = np.transpose(dxdy, (0, 3, 1, 2))
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    norm = np.sqrt(dx**2 + dy**2) + 1e-8
    dx = dx / norm
    dy = dy / norm
    dx = np.where(dy < 0, -dx, dx)
    dy = np.where(dy < 0, -dy, dy)
    angle = (dx + 1) / 2.
    hsv = np.stack([angle, np.ones_like(angle), np.ones_like(angle)], -1)
    RGB = np.stack([color.hsv2rgb(x) for x in hsv], axis=0) * 255
    RGB = np.transpose(RGB.astype(np.uint8), (0, 3, 1, 2))
    if ret_type == 'tensor':
        RGB = make_grid(torch.tensor(RGB))
    return RGB


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    '''
    image_tensor: [tensor*] or tensor , [B,3,H,W] or [B,H,W] or [H, W]
    '''
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def tile_images(imgs, picturesPerRow=4):

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate(
            [imgs,
             np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)],
            axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(
            np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)],
                           axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled