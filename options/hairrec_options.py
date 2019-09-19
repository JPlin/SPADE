from .base_options import BaseOptions


class Xoptions(BaseOptions):
    def __init__(self):
        super(Xoptions, self).__init__()
        # must needed
        self.opt['model'] = 'hairrec'
        self.opt['pretrain'] = True

        # network
        self.opt['cos_k'] = 12
        self.opt['input_nc_A'] = 2
        self.opt['input_nc_B'] = 3
        self.opt['encoder_channel'] = 1024
        self.opt['bn_type'] = 'sync_batch'  # batch
        self.opt['netGA'] = 'feat'
        self.opt['netDA'] = 'realfake'
        self.opt['netGB'] = 'feat'
        self.opt['netDB'] = 'realfake'
        self.opt['netEDA'] = 'resnet34'
        self.opt['netEDB'] = 'resnet34'

        # dataset
        self.opt['dataset_name'] = 'orient'
        self.opt['im_size'] = 256

        # total epoch  = niter + niter_decay + 1
        self.opt['lr_G'] = 2e-4
        self.opt['lr_D'] = 2e-4
        self.opt['lr_ED'] = 1e-3

        self.opt['expansion'] = 0.5
        self.opt['target_face_scale'] = 0.7
        self.opt['align_face'] = True
        self.opt['lambda_reg'] = 1.
        self.opt['lambda_cos'] = 1.
        self.opt['lambda_coli'] = 1.
        self.opt['lambda_proj'] = 1.
        self.opt['lambda_idt'] = 1.
