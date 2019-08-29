from .base_options import BaseOptions


class Xoptions(BaseOptions):
    def __init__(self):
        super(Xoptions, self).__init__()
        # must needed
        self.opt['model'] = 'dxdy'
        self.opt[
            'norm_G'] = 'spectralinstance'  # [instance norm or batch norm]
        self.opt[
            'norm_D'] = 'spectralinstance'  # [instance norm or batch norm]
        self.opt['input_nc'] = 3
        self.opt['outpu_nc'] = 2
        self.opt['ndf'] = 64
        self.opt['dataset_name'] = 'dxdy'
        self.opt['im_size'] = 512

        # total epoch  = niter + niter_decay + 1
        # n_layers | multiscale | image
        self.opt['net_D'] = 'multiscale'
        self.opt['flip_label'] = False
        self.opt['soft_labels'] = False
        self.opt['expansion'] = 0.7
        self.opt['align_face'] = True
