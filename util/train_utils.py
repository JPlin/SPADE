from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision.utils import make_grid

import importlib
import torch
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiStepStatisticCollector:
    def __init__(self, log_dir=None, comment='', global_step=0):
        self.count = global_step
        self.writer = SummaryWriter(logdir=log_dir, comment=comment)

    def close(self):
        self.writer.close()

    def next_step(self):
        self.count = self.count + 1

    def current_step(self):
        return self.count

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            kwargs['global_step'] = self.count
            return getattr(self.writer, name)(*args, **kwargs)

        return wrapper


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size -
                                               1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images,
                                  0)  # collect all the images and return
        return return_images


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_scheduler(optimizer, opt):
    if opt['lr_policy'] == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt['start_epoch'] -
                             opt['niter']) / float(opt['niter_decay'] + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt['lr_decay_iters'],
                                        gamma=0.1)
    elif opt['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt['niter'],
                                                   eta_min=0)
    elif opt['lr_policy'] == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lambda epoch: 1.)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt['lr_policy'])
    return scheduler


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print(
            "In %s, there should be a class whose name matches %s in lowercase without underscore(_)"
            % (module, target_cls_name))
        exit(0)

    return cls