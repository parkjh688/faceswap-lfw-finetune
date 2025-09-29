#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from models.projected_model import fsModel
from data.data_loader_lfw import get_lfw_loader

import wandb

def str2bool(v: str) -> bool:
    return v.lower() in ('true', '1', 'yes')


class TrainOptions:
    """CLI options."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='experiment name')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        self.parser.add_argument('--batchSize', type=int, default=4)
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

        self.parser.add_argument('--dataset', type=str, default='./lfw_funneled')
        self.parser.add_argument('--continue_train', type=str2bool, default='False')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224')
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--niter', type=int, default=10000)
        self.parser.add_argument('--niter_decay', type=int, default=10000)
        self.parser.add_argument('--beta1', type=float, default=0.0)
        self.parser.add_argument('--lr', type=float, default=0.0004)
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        self.parser.add_argument('--lambda_feat', type=float, default=10.0)
        self.parser.add_argument('--lambda_id', type=float, default=30.0)
        self.parser.add_argument('--lambda_rec', type=float, default=10.0)

        self.parser.add_argument('--Arc_path', type=str, default='arcface_model/arcface_checkpoint.tar')
        self.parser.add_argument('--total_step', type=int, default=5)
        self.parser.add_argument('--log_frep', type=int, default=1)
        self.parser.add_argument('--sample_freq', type=int, default=1)
        self.parser.add_argument('--model_freq', type=int, default=10000)

        self.isTrain = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            if save and not self.opt.continue_train:
                with open(os.path.join(expr_dir, 'opt.txt'), 'wt') as f:
                    f.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        f.write(f'{k}: {v}\n')
                    f.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')
    os.makedirs(log_path, exist_ok=True)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except Exception:
            start_epoch, epoch_iter = 1, 0
        print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')
    else:
        start_epoch, epoch_iter = 1, 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print('Device:', device)

    cudnn.benchmark = torch.cuda.is_available()

    model = fsModel()
    model.initialize(opt)
    model.to(device)

    # Pretrained checkpoints (Generator required, Discriminator optional)
    if opt.load_pretrain and os.path.exists(opt.load_pretrain):
        try:
            model.load_network(model.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print(f'Loaded Generator weights from {opt.load_pretrain}')
            try:
                model.load_network(model.netD, 'D', opt.which_epoch, opt.load_pretrain)
                print(f'Loaded Discriminator weights from {opt.load_pretrain}')
            except Exception:
                print("No Discriminator checkpoint found, skipped.")
        except Exception as e:
            print("Checkpoint load failed:", e)

    if opt.Arc_path and os.path.exists(opt.Arc_path):
        arc_obj = torch.load(opt.Arc_path, map_location=device)
        if isinstance(arc_obj, dict) and "model" in arc_obj:
            arc_state = arc_obj["model"]
        elif isinstance(arc_obj, dict):
            arc_state = arc_obj
        elif isinstance(arc_obj, torch.nn.Module):
            arc_state = arc_obj.state_dict()
        else:
            raise ValueError(f"Unexpected ArcFace checkpoint format: {type(arc_obj)}")
        model.netArc.load_state_dict(arc_state, strict=False)
        print("Loaded ArcFace checkpoint")

    model.netArc.requires_grad_(False)
    model.netArc.eval()

    for p in model.netG.parameters():
        p.requires_grad = True

    for p in model.netD.parameters():
        p.requires_grad = False
    optimizer_D = None

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write(f'================ Training Loss ({now}) ================\n')

    optimizer_G = torch.optim.Adam(
        (p for p in model.netG.parameters() if p.requires_grad),
        lr=opt.lr, betas=(opt.beta1, 0.999)
    )

    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)

    train_loader = get_lfw_loader("./lfw_funneled", batch_size=4, self_prob=0.3)
    data_iter = iter(train_loader)

    # -------------------------------
    # Init WandB
    # -------------------------------
    wandb.init(
        project="faceswap-lfw",
        name=opt.name,
        config=vars(opt)
    )

    print("Start to run a quick test loop")
    for step in range(opt.total_step):
        src_image1, src_image2, is_same = next(data_iter)
        src_image1, src_image2, is_same = (
            src_image1.to(device),
            src_image2.to(device),
            is_same.to(device),
        )

        # latent vector (ArcFace)
        img_id_112 = F.interpolate(src_image2, size=(112, 112), mode="bicubic")
        with torch.no_grad():
            latent_id = model.netArc(img_id_112)
            latent_id = F.normalize(latent_id, p=2, dim=1)

        # forward
        model.netG.train()
        img_fake = model.netG(src_image1, latent_id)

        gen_logits, feat = model.netD(img_fake, None)
        loss_Gmain = (-gen_logits).mean()

        img_fake_down = F.interpolate(img_fake, size=(112, 112), mode="bicubic")
        latent_fake = model.netArc(img_fake_down)
        latent_fake = F.normalize(latent_fake, p=2, dim=1)
        loss_G_ID = (1 - model.cosin_metric(latent_fake, latent_id)).mean()

        with torch.no_grad():
            real_feat = model.netD.get_feature(src_image1)
        feat_match_loss = model.criterionFeat(feat["3"], real_feat["3"])

        loss_G = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat

        if is_same.any():
            loss_G_Rec = model.criterionRec(img_fake, src_image1) * opt.lambda_rec
            loss_G = loss_G + loss_G_Rec
        else:
            loss_G_Rec = torch.tensor(0.0, device=device)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # -------------------------------
        # Logging to WandB
        # -------------------------------
        if (step + 1) % opt.log_frep == 0:
            cos_sim = model.cosin_metric(latent_fake.detach(), latent_id.detach()).mean().item()
            wandb.log({
                "loss/total_G": loss_G.item(),
                "loss/G_main":  loss_Gmain.item(),
                "loss/G_ID":    loss_G_ID.item(),
                "loss/G_rec":   float(loss_G_Rec.item()),
                "loss/G_feat":  feat_match_loss.item(),
                "metric/id_cosine": cos_sim,
            }, step=step)

        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                save_tag = "self" if is_same.any() else "cross"
                denorm = lambda x: (x * imagenet_std + imagenet_mean).clamp(0, 1)

                # 4개 샘플만 로깅
                wandb.log({
                    f"samples/source": [wandb.Image(denorm(src_image1[i].cpu()), caption=f"src {i}") for i in range(4)],
                    f"samples/target": [wandb.Image(denorm(src_image2[i].cpu()), caption=f"tgt {i}") for i in range(4)],
                    f"samples/fake_{save_tag}": [wandb.Image(denorm(img_fake[i].cpu()), caption=f"fake {i}") for i in range(4)],
                }, step=step)