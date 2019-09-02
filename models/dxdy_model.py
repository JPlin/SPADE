import torch
from collections import OrderedDict
import models.networks as networks
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.base_model import BaseModel
from util import data_utils
import tools
import haya_data


class DxdyModel(BaseModel):
    def name(self):
        return 'DxdyModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        pass

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # set networks
        self.initialize_networks(opt)

        # set loss functions
        self.initialize_loss(opt)

        # set optimizer
        self.initialize_optimizer(opt)

        self.initialize_other(opt)

        self.model_dict = {
            'netG': {
                'model': self.netG.module if self.use_gpu else self.netG,
                'optimizer': self.optimizer_G
            },
            'netD': {
                'model': self.netD.module if self.use_gpu else self.netD,
                'optimizer': self.optimizer_D
            }
        }
        self.opt = opt

    def initialize_networks(self, opt):
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(opt)
        # set require gradients
        if self.isTrain:
            self.set_requires_grad([self.netG, self.netD], True)
        else:
            self.set_requires_grad([self.netG, self.netD], False)
        if self.use_gpu:
            self.netG = DataParallelWithCallback(self.netG,
                                                 device_ids=opt['gpu_ids'])
            self.netD = DataParallelWithCallback(self.netD,
                                                 device_ids=opt['gpu_ids'])
        self.train_nets = [self.netG, self.netD]

    def initialize_optimizer(self, opt):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())
        beta1, beta2 = opt['beta1'], opt['beta2']
        G_lr, D_lr = opt['lr'], opt['lr']
        self.optimizer_G = torch.optim.Adam(G_params,
                                            lr=G_lr,
                                            betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(D_params,
                                            lr=D_lr,
                                            betas=(beta1, beta2))
        self.old_lr = opt['lr']

    def initialize_loss(self, opt):
        self.criterionGAN = networks.GANLoss(opt['gan_mode'],
                                             tensor=self.FloatTensor,
                                             opt=opt)
        # if self.use_gpu:
        #     self.criterionGAN = DataParallelWithCallback(
        #         self.criterionGAN, device_ids=opt['gpu_ids'])
        self.criterionReg = torch.nn.L1Loss()

    def initialize_other(self, opt):
        full_body_mesh_vert_pos, full_body_mesh_face_inds = tools.load_body_mesh(
        )
        self.full_body_mesh_vert_pos = full_body_mesh_vert_pos.unsqueeze(0)
        self.full_body_mesh_face_inds = full_body_mesh_face_inds.unsqueeze(0)
        sample_dataset = haya_data.Hair3D10KConvDataOnly()
        self.sample_loader = torch.utils.data.DataLoader(
            sample_dataset,
            batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=opt['workers'],
            drop_last=True)
        self.sample_iter = iter(self.sample_loader)
        assert len(sample_dataset) > 0
        print(f'{len(sample_dataset)} is loaded')

    def set_input(self, data):
        self.image = data['image'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.intensity = data['intensity'].to(self.device)
        self.gt_dxdy = data['dxdy'].to(self.device)
        try:
            sample_data = next(self.sample_iter)
        except StopIteration:
            self.sample_iter = iter(self.sample_loader)
            sample_data = next(self.sample_iter)
        convdata = sample_data['convdata'].to(self.device)
        strands = convdata.permute(
            0, 2, 3, 4,
            1)[:, :3, :, :, :].contiguous()  # b x 3 x 32 x 32 x 300
        body_mesh_vert_pos = self.full_body_mesh_vert_pos.expand(
            strands.size(0), -1, -1).to(strands.device)
        body_mesh_face_inds = self.full_body_mesh_face_inds.expand(
            strands.size(0), -1, -1).to(strands.device)
        # generate random mvps
        mvps, _, _ = tools.generate_random_mvps(strands.size(0),
                                                strands.device)

        # render the 2D information
        self.strand_dxdy, self.strand_mask, body_mask, _, strand_vis, mvps, _ = tools.render(
            mvps,
            strands,
            body_mesh_vert_pos,
            body_mesh_face_inds,
            self.opt['im_size'],
            self.opt['expansion'],
            align_face=self.opt['align_face'],
            target_face_scale=self.opt['target_face_scale'])

    def forward(self):
        mask_ = self.mask.unsqueeze(1).type(self.image.dtype)
        strand_mask_ = self.strand_mask.unsqueeze(1).type(
            self.strand_dxdy.dtype)

        # for G
        self.pred_dxdy = self.netG(torch.cat([self.image, mask_], dim=1))
        fake_sample = self.pred_dxdy * mask_.type(self.pred_dxdy.dtype)
        self.g_fake_score = self.netD(fake_sample)
        # for D
        fake_sample = self.pred_dxdy.detach() * mask_
        fake_sample.requires_grad_()
        real_sample = self.strand_dxdy * strand_mask_

        self.d_real_score, self.d_fake_score = self.netD(
            real_sample), self.netD(fake_sample)
        # for vis
        masked_pred_dxdy = torch.where(mask_ > 0., self.pred_dxdy,
                                       -torch.ones_like(self.pred_dxdy))
        masked_gt_dxdy = torch.where(mask_ > 0., self.gt_dxdy,
                                     -torch.ones_like(self.gt_dxdy))
        self.vis_dict['image'] = data_utils.make_grid_n(self.image[:6])
        self.vis_dict['gt_dxdy'] = data_utils.vis_orient(self.gt_dxdy[:6])
        self.vis_dict['pred_dxdy'] = data_utils.vis_orient(self.pred_dxdy[:6])
        self.vis_dict['masked_pred_dxdy'] = data_utils.vis_orient(masked_pred_dxdy[:6])
        self.vis_dict['masked_gt_dxdy'] = data_utils.vis_orient(masked_gt_dxdy[:6])
        self.vis_dict['strand_dxdy'] = data_utils.vis_orient(self.strand_dxdy[:6])

    def backward_G(self):
        reg_loss = self.dxdy_reg_loss(self.pred_dxdy,
                                      self.gt_dxdy) * self.intensity
        reg_loss = (self.mask.expand_as(reg_loss).float() *
                    reg_loss).mean() * self.opt.get('lambda_reg', 1.)
        g_loss = self.criterionGAN(self.g_fake_score,
                                   True,
                                   for_discriminator=False)

        sum([g_loss, reg_loss]).mean().backward()
        self.loss_dict['loss_reg'] = reg_loss.item()
        self.loss_dict['loss_g'] = g_loss.item()

    def backward_D(self):
        d_fake = self.criterionGAN(self.d_fake_score, False)
        d_real = self.criterionGAN(self.d_real_score, True)

        sum([d_fake, d_real]).mean().backward()
        self.loss_dict['loss_d_fake'] = d_fake.item()
        self.loss_dict['loss_d_real'] = d_real.item()

    def optimize_parameters(self):
        for net in self.train_nets:
            net.train()

        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    ##################################################################
    # Helper functions
    ##################################################################
    def update_learning_rate(self, epoch):
        if epoch > self.opt['niter']:
            lrd = self.opt['lr'] / self.opt['niter_decay']
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_G = new_lr / 2
            new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def dxdy_reg_loss(self, y_hat, y):
        '''
        y_hat, y: B 2 H W
        return: B H W
        '''
        y_norm = y_hat / (torch.norm(y_hat, dim=1, keepdim=True) + 0.0000001)
        cos = torch.abs(torch.sum(y_norm * y, dim=1, keepdim=False))
        norm = torch.abs(
            torch.norm(y_hat, dim=1, keepdim=False) - torch.ones_like(cos))
        return 1 - cos + norm

    def discriminate(self, fake_image, real_image):
        fake_concat = torch.cat([fake_image], dim=1)
        real_concat = torch.cat([real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real