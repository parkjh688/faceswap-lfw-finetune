#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample
from pg_modules.projected_discriminator import ProjectedDiscriminator


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class fsModel(BaseModel):
    """SimSwap projected model wrapper (device-agnostic)."""

    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and str(opt.gpu_ids) != "-1") else "cpu"
        )

        # Generator
        self.netG = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep
        ).to(self.device)

        # ArcFace (checkpoint may contain a module or a dict)
        arc_ckpt_path = opt.Arc_path
        arc_obj = torch.load(arc_ckpt_path, map_location="cpu")
        if isinstance(arc_obj, nn.Module):
            self.netArc = arc_obj
        elif isinstance(arc_obj, dict) and "model" in arc_obj:
            self.netArc = arc_obj["model"]
        else:
            self.netArc = arc_obj  # fallback
        if isinstance(self.netArc, nn.Module):
            self.netArc = self.netArc.to(self.device)
            self.netArc.eval()
            self.netArc.requires_grad_(False)

        if not self.isTrain:
            pretrained_path = opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return

        # Discriminator (projected)
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{}).to(self.device)

        if self.isTrain:
            self.criterionFeat = nn.L1Loss()
            self.criterionRec  = nn.L1Loss()

            # Optimizers (can be overridden by caller)
            self.optimizer_G = torch.optim.Adam(
                list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8
            )
            self.optimizer_D = torch.optim.Adam(
                list(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8
            )

        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr