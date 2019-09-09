import torch
from collections import OrderedDict
import models.networks as networks
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.base_model import BaseModel
from util import data_utils
import tools
import haya_data


class DxdyPairModel(BaseModel):
    def name(self):
        return 'DxdyPairModel'

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
        # full_body_mesh_vert_pos, full_body_mesh_face_inds = tools.load_body_mesh(
        # )
        # self.full_body_mesh_vert_pos = full_body_mesh_vert_pos.unsqueeze(0)
        # self.full_body_mesh_face_inds = full_body_mesh_face_inds.unsqueeze(0)
        # sample_dataset = haya_data.Hair3D10KConvDataOnly()
        # self.sample_loader = torch.utils.data.DataLoader(
        #     sample_dataset,
        #     batch_size=opt['batch_size'],
        #     shuffle=False,
        #     num_workers=opt['workers'],
        #     drop_last=True)
        # self.sample_iter = iter(self.sample_loader)
        # assert len(sample_dataset) > 0
        # print(f'{len(sample_dataset)} is loaded')
        pass

    def set_input(self, data):
        self.image = data['image'].cuda(non_blocking=True)
        self.mask = data['mask'].float().cuda()
        self.intensity = data['intensity'].float().cuda(non_blocking=True)
        self.gt_dxdy = data['dxdy'].float().cuda()
        self.render_dxdy = data['render_dxdy'].float().cuda(non_blocking=True)
        self.synthetic_label = data['synthetic_label'].float().cuda(non_blocking=True)

    def forward(self):
        mask_ = self.mask.unsqueeze(1).type(self.gt_dxdy.dtype)

        # for G
        self.pred_dxdy = self.netG(torch.cat([self.gt_dxdy, mask_], dim=1))
        fake_sample = self.pred_dxdy * mask_.type(self.pred_dxdy.dtype)
        self.g_fake_score = self.netD(fake_sample)
        # for D
        fake_sample = self.pred_dxdy.detach() * mask_
        fake_sample.requires_grad_()
        real_sample = self.render_dxdy * mask_

        self.d_real_score, self.d_fake_score = self.netD(
            real_sample), self.netD(fake_sample)
        # for vis
        self.mask_ = mask_
        
    def update_visuals(self):
        masked_pred_dxdy = torch.where(self.mask_ > 0., self.pred_dxdy,
                                       -torch.ones_like(self.pred_dxdy))
        masked_gt_dxdy = torch.where(self.mask_ > 0., self.gt_dxdy,
                                     -torch.ones_like(self.gt_dxdy))

        self.vis_dict['image'] = data_utils.make_grid_n(self.image[:6])
        self.vis_dict['gt_dxdy'] = data_utils.vis_orient(self.gt_dxdy[:6])
        self.vis_dict['pred_dxdy'] = data_utils.vis_orient(self.pred_dxdy[:6])
        self.vis_dict['masked_pred_dxdy'] = data_utils.vis_orient(
            masked_pred_dxdy[:6])
        self.vis_dict['masked_gt_dxdy'] = data_utils.vis_orient(
            masked_gt_dxdy[:6])
        self.vis_dict['render_dxdy'] = data_utils.vis_orient(
            self.render_dxdy[:6])

    def backward_G(self):
        # unpair data asparse loss
        reg_sparse_loss = self.dxdy_reg_loss(self.pred_dxdy,
                                      self.gt_dxdy) * self.intensity
        reg_sparse_loss = (self.mask.expand_as(reg_sparse_loss).float() *
                    reg_sparse_loss).view(reg_sparse_loss.size(0), -1).mean(dim = 1)
        reg_sparse_loss = (reg_sparse_loss * (1. - self.synthetic_label)).mean()
        # pair data dense loss
        reg_dense_loss = self.dxdy_reg_loss(self.pred_dxdy, self.render_dxdy)
        reg_dense_loss = (self.mask.expand_as(reg_dense_loss).float() *
                    reg_dense_loss).view(reg_dense_loss.size(0), -1).mean(dim = 1)
        reg_dense_loss = (reg_dense_loss * self.synthetic_label).mean()
        # gan loss
        g_loss = self.criterionGAN(self.g_fake_score,
                                   True,
                                   for_discriminator=False)

        sum([g_loss, reg_sparse_loss, reg_dense_loss]).mean().backward()
        self.loss_dict['loss_reg_sparse'] = reg_sparse_loss.item()
        self.loss_dict['loss_reg_dense'] = reg_dense_loss.item()
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

    def inference(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)
            mask_ = mask.unsqueeze(1).type(image.dtype)
            pred_dxdy = self.netG(torch.cat([image, mask_], dim=1))
            masked_pred_dxdy = torch.where(mask_ > 0., pred_dxdy,
                                           -torch.ones_like(pred_dxdy))
            return {'image': image, 'mask': mask, 'pred_dxdy': pred_dxdy}
