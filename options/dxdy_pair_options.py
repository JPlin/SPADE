from .base_options import BaseOptions


class Xoptions(BaseOptions):
    def __init__(self):
        super(Xoptions, self).__init__()
        # must needed
        self.opt['model'] = 'dxdy_pair'
        self.opt[
            'norm_G'] = 'spectralinstance'  # [instance norm or batch norm]
        self.opt[
            'norm_D'] = 'spectralinstance'  # [instance norm or batch norm]
        self.opt['input_nc'] = 3
        self.opt['output_nc'] = 2
        self.opt['netD'] = 'multiscale'
        self.opt['ndf'] = 64
        self.opt['netG'] = 'pix2pixhd'
        self.opt['ngf'] = 64
        self.opt['dataset_name'] = 'hairmix'
        self.opt['im_size'] = 512

        # total epoch  = niter + niter_decay + 1
        # n_layers | multiscale | image
        self.opt['flip_label'] = False
        self.opt['soft_labels'] = False
        self.opt['expansion'] = 0.5
        self.opt['target_face_scale'] = 0.7
        self.opt['align_face'] = True
