import torch
from collections import OrderedDict
import models.networks as networks
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.base_model import BaseModel
from util import data_utils
import tools
import haya_data


class HairrecModel(BaseModel):
    def name(self):
        return 'HairrecModel'

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
        self.netGA = networks.define_G(opt, opt['netGA'])
        self.netGB = networks.define_G(opt, opt['netGB'])
        self.netDA = networks.define_D(opt, opt['netDA'])
        self.netDB = networks.define_D(opt, opt['netDB'])
        self.netEA, self.netHairA = networks.define_RES(
            opt, opt['input_nc_A'], opt['netEDA'])
        self.netEB, self.netHairB = networks.define_RES(
            opt, opt['input_nc_B'], opt['netEDB'])

        if self.opt['pretrain']:
            self.train_nets = [
                self.netGA, self.netGB, self.netDA, self.netDB, self.netEA,
                self.netHairA, self.netEB, self.netHairB
            ]
        else:
            self.train_nets = [self.netEA, self.netHairA]

        # set require gradients
        if self.isTrain:
            self.set_requires_grad(self.train_nets, True)
        else:
            self.set_requires_grad(self.train_nets, False)

        if self.use_gpu:
            for i in range(len(self.train_nets)):
                self.train_nets[i] = DataParallelWithCallback(
                    self.train_nets[i], device_ids=opt['gpu_ids'])
            if self.opt['pretrain']:
                self.netGA, self.netGB, self.netDA, self.netDB, self.netEA, \
                    self.netHairA, self.netEB, self.netHairB = self.train_nets
            else:
                self.netEA, self.netHairA = self.train_nets

    def initialize_optimizer(self, opt):
        beta1, beta2 = opt['beta1'], opt['beta2']
        G_lr, D_lr, ED_lr = opt.get('lr_G', opt['lr']), opt.get(
            'lr_D', opt['lr']), opt.get('lr_ED', opt['lr'])
        ED_params = list(self.netEA.parameters()) + list(
            self.netHairA.parameters()) + list(self.netEB.parameters()) + list(
                self.netHairB.paramters())
        self.optimizer_ED = torch.optim.Adam(ED_params,
                                             lr=ED_lr,
                                             betas=(beta1, beta2))
        if not self.opt['pretrain']:
            G_params = list(self.netGA.parameters()) + list(
                self.netGB.parameters())
            D_params = list(self.netDB.parameters()) + list(
                self.netDB.parameters())
            self.optimizer_G = torch.optim.Adam(G_params,
                                                lr=G_lr,
                                                betas=(beta1, beta2))
            self.optimizer_D = torch.optim.Adam(D_params,
                                                lr=D_lr,
                                                betas=(beta1, beta2))
            self.old_lr = opt['lr']

    def initialize_loss(self, opt):
        self.criterionReg = torch.nn.L1Loss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionGAN = networks.GANLoss(opt['gan_mode'],
                                             tensor=self.FloatTensor,
                                             opt=opt)

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
            drop_last=True,
            pin_memory=True)
        self.sample_iter = iter(self.sample_loader)
        assert len(sample_dataset) > 0
        print(f'{len(sample_dataset)} is loaded')

    def set_input(self, data):
        try:
            sample_data = next(self.sample_iter)
        except StopIteration:
            self.sample_iter = iter(self.sample_loader)
            sample_data = next(self.sample_iter)
        convdata = sample_data['convdata'].to(self.device)
        strands = convdata.permute(
            0, 2, 3, 4,
            1)[:, :3, :, :, :].contiguous()  # b x 3 x 32 x 32 x 300

        # generate random mvps
        mvps, _, _ = tools.generate_random_mvps(strands.size(0),
                                                strands.device)

        # render the 2D information
        self.dxdy_gt, self.strand_mask, body_mask, _, self.strand_vis, self.mvps, _ = self.render_dxdy(
            mvps, strands)
        self.psi, self.psi_inv = tools.compute_cosine_transform(
            100, self.opt['cos_k'])
        self.cA_gt = tools.strands_to_c(strands,
                                        self.psi_inv.to(strands.device))
        self.strands_gt = strands

        if not self.opt['pretrain']:
            image = data['image'].to(self.device, non_blocking=True)
            self.mask = data['mask'].to(self.device, non_blocking=True)
            self.intensity = data['intensity'].to(self.device,
                                                  non_blocking=True)
            self.dxdy_gabor = data['dxdy'].float().to(self.device,
                                                      non_blocking=True)
            self.mvps_gabor = data['aligned_MVP'].float().to(self.device,
                                                             non_blocking=True)
            self.image = image * self.mask.unsqueeze(1)

    def forward(self):
        self.Feat_A = self.netEA(self.dxdy_gt)
        self.cA = self.netHairA(self.Feat_A)

        if not self.opt['pretrain']:
            self.Feat_B = self.netEB(self.image)
            self.cB = self.netHairB(self.Feat_B)
            # Let's start cycle
            self.fake_B = self.netGB(self.Feat_A)
            self.fake_A = self.netGA(self.Feat_B)
            self.rec_A = self.netGA(self.fake_B)
            self.rec_B = self.netGB(self.fake_A)
            self.fake_cB = self.netHairB(self.fake_B)
            self.fake_cA = self.netHairA(self.fake_A)

        # scale the size of everything
        if self.pred_dxdy.size(-1) != self.mask.size(-1):
            self.mask = torch.nn.functional.interpolate(
                self.mask.unsqueeze(1),
                size=self.pred_dxdy.shape[-2:],
                mode='nearest').squeeze()
            self.intensity = torch.nn.functional.interpolate(
                self.intensity.unsqueeze(1),
                size=self.pred_dxdy.shape[-2:],
                mode='nearest').squeeze()
            self.gt_dxdy = torch.nn.functional.interpolate(
                self.gt_dxdy, size=self.pred_dxdy.shape[-2:], mode='nearest')

    def update_visuals(self):
        self.vis_dict['dxdy_gt'] = data_utils.vis_orient(self.dxdy_gt[:6])
        self.vis_dict['dxdy_A'] = data_utils.vis_orient(self.dxdy_A[:6])

        if not self.opt['pretrain']:
            self.vis_dict['image'] = data_utils.make_grid_n(self.image[:6])
            self.vis_dict['dxdy_fake_A'] = data_utils.vis_orient(
                self.dxdy_fake_A[:6])
            self.vis_dict['dxdy_B'] = data_utils.vis_orient(self.dxdy_B[:6])
            self.vis_dict['dxdy_fake_B'] = data_utils.vis_orient(
                self.dxdy_fake_B[:6])

    ##################################################################
    # Caculate loss and Backward
    ##################################################################

    def backward_G(self):
        loss_G_A = self.criterionGAN(self.netDA(self.fake_A), True)
        loss_G_B = self.criterionGAN(self.netDB(self.fake_B), True)

        loss_cycle_A = self.criterionCycle(
            self.rec_A, self.Feat_A) * self.opt['lambda_cycle']
        loss_cycle_B = self.criterionCycle(
            self.rec_B, self.Feat_B) * self.opt['lmabda_cycle']

        lambda_idt = self.opt['lambda_idt']
        if lambda_idt > 0.:
            idt_A = self.netGA(self.Feat_A)
            idt_B = self.netGB(self.Feat_B)
            loss_idt_A = self.criterionIdt(idt_A, self.Feat_A) * lambda_idt
            loss_idt_B = self.criterionIdt(self.idt_B,
                                           self.Feat_B) * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.loss_dict['loss_g'] = loss_G.item()

    def backward_D_A(self):
        loss_D_A = self.backward_D_base(self.netDA, self.Feat_A, self.fake_A)
        self.loss_dict['loss_d_a'] = loss_D_A.item()

    def backward_D_B(self):
        loss_D_B = self.backward_D_base(self.netDB, self.Feat_B, self.fake_B)
        self.loss_dict['loss_d_b'] = loss_D_B.item()

    def backward_D_base(self, netD, real, fake):
        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * .5
        loss_D.backward()
        return loss_D

    def backward_ED(self):
        loss_rec_A, self.dxdy_A = self.backward_ED_base(
            self.cA, self.cA_gt, self.strands_gt, self.dxdy_gt, self.mvps)

        if not self.opt['pretrain']:
            loss_rec_fake_A, self.dxdy_fake_A = self.backward_ED_base(
                self.fake_cB, self.cA_gt, self.strands_gt, self.dxdy_gt,
                self.mvps)
            loss_consistency, self.dxdy_B, self.dxdy_fake_B = self.backward_ED_consis(
                self.cB, self.fake_cA, self.mvps_gabor)
        else:
            loss_rec_A = 0
            loss_consistency = 0

        loss_ED = loss_rec_A + loss_rec_fake_A + loss_consistency
        loss_ED.backward()

        self.loss_dict['loss_rec_A'] = loss_rec_A.item()
        self.loss_dict['loss_rec_fake_A'] = loss_rec_fake_A.item()
        self.loss_dict['loss_consistency'] = loss_consistency.item()

    def backward_ED_base(self, c, c_gt, strand_gt, dxdy_gt, mvps):
        pred_c = c.view(c.size(0), 3, -1, c.size(2),
                        c.size(3)).permute(0, 1, 3, 4, 2)
        pred_strands = tools.c_to_strands(pred_c, self.psi.to(pred_c.device))
        # cosine loss
        cos_loss = tools.compute_cosine_loss(pred_c,
                                             c_gt) * self.opt['lambda_cos']
        # colision loss
        pred_strand_len = tools.compute_strand_lengths(pred_strands)
        col_loss = tools.compute_collision_loss(
            pred_strands, pred_strand_len) * self.opt['lambda_coli']

        # reprojection loss
        lambda_proj = self.opt['lambda_proj']
        if lambda_proj > 0:
            dxdy, *_ = self.render_dxdy(mvps, pred_strands)
            reg_loss = self.dxdy_reg_loss(dxdy, dxdy_gt) * lambda_proj
        else:
            reg_loss = 0

        loss_ED = (cos_loss + col_loss + reg_loss).mean()
        loss_ED.backward()
        return loss_ED, dxdy.detach()

    def backward_ED_consis(self, cA, cB, mvps):
        loss_consistency = self.criterionCycle(cA, cB)
        strands_A = tools.c_to_strands(cA, self.psi.to(cA.device))
        strands_B = tools.c_to_strands(cB, self.psi.to(cA.device))
        dxdy_A, *_ = self.render_dxdy(mvps, strands_A)
        dxdy_B, *_ = self.render_dxdy(mvps, strands_B)

        loss_consistency.backward()
        return loss_consistency, dxdy_A, dxdy_B

    def optimize_parameters(self):
        for net in self.train_nets:
            net.train()

        self.forward()

        # backward ED_A and ED_B
        self.optimizer_ED.zero_grad()
        self.backward_ED()
        self.optimizer_ED.step()

        if not self.opt['pretrain']:
            self.set_requires_grad([self.netDA, self.netDB], False)
            # backward G_A and G_B
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            self.set_requires_grad([self.netDA, self.netDB], True)
            # backward D_A and D_B
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
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

    def render_dxdy(self, mvps, strands):
        body_mesh_vert_pos = self.full_body_mesh_vert_pos.expand(
            strands.size(0), -1, -1).to(strands.device)
        body_mesh_face_inds = self.full_body_mesh_face_inds.expand(
            strands.size(0), -1, -1).to(strands.device)
        return tools.render(mvps,
                            strands,
                            body_mesh_vert_pos,
                            body_mesh_face_inds,
                            self.opt['im_size'],
                            expansion=self.opt['expansion'],
                            align_face=self.opt['im_size'],
                            target_face_scale=self.opt['target_face_scale'])

    def inference(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)
            mask_ = mask.unsqueeze(1).type(image.dtype)
            pred_dxdy = self.netG(torch.cat([image, mask_], dim=1))
            masked_pred_dxdy = torch.where(mask_ > 0., pred_dxdy,
                                           -torch.ones_like(pred_dxdy))
            return {'image': image, 'mask': mask, 'pred_dxdy': pred_dxdy}