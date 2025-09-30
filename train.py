#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from models.projected_model import fsModel
from data.data_loader_lfw import get_lfw_loader_aligned

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
        self.parser.add_argument('--total_step', type=int, default=30000)
        self.parser.add_argument('--log_frep', type=int, default=100)
        self.parser.add_argument('--sample_freq', type=int, default=500)
        self.parser.add_argument('--model_freq', type=int, default=1000)

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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    cudnn.benchmark = torch.cuda.is_available()

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

    # ArcFace 로드 (다양한 형식 대응)
    if opt.Arc_path and os.path.exists(opt.Arc_path):
        arc_obj = torch.load(opt.Arc_path, map_location=device, weights_only=False)
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

    # G만 학습
    for p in model.netG.parameters():
        p.requires_grad = True

    # D는 고정 특징 추출기로만 사용 (프리트레인 로드 시)
    for p in model.netD.parameters():
        p.requires_grad = False
    model.netD.eval()
    optimizer_D = None

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write(f'================ Training Loss ({now}) ================\n')

    optimizer_G = torch.optim.Adam(
        (p for p in model.netG.parameters() if p.requires_grad),
        lr=opt.lr, betas=(opt.beta1, 0.999)
    )

    # NOTE: 로더 정규화가 ImageNet(mean/std)인 경우에만 denorm 필요
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    denorm = lambda x: (x * imagenet_std + imagenet_mean).clamp(0, 1)

    train_loader = get_lfw_loader_aligned(
                        root_dir="./lfw_aligned_224",
                        batch_size=opt.batchSize,
                        num_workers=4,
                        seed=1234,
                        self_prob=0.5,
                        image_size=224,     # 정렬 저장 크기와 맞추기
                        epoch_mul=10,
                    )
    data_iter = iter(train_loader)

    # -------------------------------
    # Init WandB
    # -------------------------------
    wandb.init(
        project="faceswap-lfw",
        name=opt.name,
        config=vars(opt)
    )

    print("Start training (G-only fine-tune)")
    for step in range(opt.total_step):
        # ---- 안전한 data_iter ----
        try:
            src_image1, src_image2, is_same = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            src_image1, src_image2, is_same = next(data_iter)

        src_image1, src_image2, is_same = (
            src_image1.to(device),
            src_image2.to(device),
            is_same.to(device),
        )

        # latent vector (ArcFace) : 입력 [-1,1] 정규화
        img_id_112 = F.interpolate(src_image2, size=(112, 112), mode="bicubic", align_corners=False)
        img_id_112 = (img_id_112 - 0.5) / 0.5          # ✅ ArcFace 입력 정규화
        with torch.no_grad():
            latent_id = model.netArc(img_id_112)
            latent_id = F.normalize(latent_id, p=2, dim=1)

        # forward
        model.netG.train()
        img_fake = model.netG(src_image1, latent_id)     # ✅ 먼저 생성

        # Discriminator 고정 특징 기반 (adv는 후반부만 소량)
        with torch.no_grad():
            real_feat = model.netD.get_feature(src_image1)
        gen_logits, feat = model.netD(img_fake, None)

        # ID loss
        img_fake_down = F.interpolate(img_fake, size=(112, 112), mode="bicubic", align_corners=False)
        img_fake_down = (img_fake_down - 0.5) / 0.5
        latent_fake = model.netArc(img_fake_down)
        latent_fake = F.normalize(latent_fake, p=2, dim=1)
        loss_G_ID = (1 - model.cosin_metric(latent_fake, latent_id)).mean()

        # feature matching
        feat_match_loss = model.criterionFeat(feat["3"], real_feat["3"])

        # reconstruction (self-pair에만 마스크 적용)
        is_same_f = is_same.float().view(-1, 1, 1, 1)  # [B,1,1,1]
        if is_same_f.sum() > 0:
            # L1 재구성 예시 (criterionRec가 L1이면 대체 가능)
            rec_per_px = (img_fake - src_image1).abs()
            loss_G_Rec = (rec_per_px * is_same_f).sum() / (is_same_f.sum() * rec_per_px.shape[1] * rec_per_px.shape[2] * rec_per_px.shape[3])
            loss_G_Rec = loss_G_Rec * opt.lambda_rec
        else:
            loss_G_Rec = torch.tensor(0.0, device=device)

        # adversarial: 초반 off, 후반 소량 on
        lambda_adv = 0.0 if step < max(10000, opt.total_step // 2) else 0.1
        loss_adv = (-gen_logits).mean() if lambda_adv > 0 else torch.tensor(0.0, device=device)

        # total G loss
        loss_G = (
            loss_adv * lambda_adv
            + loss_G_ID * opt.lambda_id
            + feat_match_loss * opt.lambda_feat
            + loss_G_Rec
        )

        optimizer_G.zero_grad(set_to_none=True)
        loss_G.backward()
        optimizer_G.step()

        # -------------------------------
        # Logging to WandB
        # -------------------------------
        if (step + 1) % opt.log_frep == 0:
            cos_sim = model.cosin_metric(latent_fake.detach(), latent_id.detach()).mean().item()
            wandb.log({
                "loss/total_G": loss_G.item(),
                "loss/G_adv":   float(loss_adv.item()) if lambda_adv > 0 else 0.0,
                "loss/G_ID":    loss_G_ID.item(),
                "loss/G_rec":   float(loss_G_Rec.item()),
                "loss/G_feat":  feat_match_loss.item(),
                "metric/id_cosine": cos_sim,
                "meta/lambda_adv": lambda_adv,
                "meta/step": step + 1,
            }, step=step+1)

        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                # 배치에서 self / cross 인덱스 분리
                idx_self  = (is_same == 1).nonzero(as_tuple=True)[0]
                idx_cross = (is_same == 0).nonzero(as_tuple=True)[0]

                logs = {}

                def log_panel(indices, tag, kmax=4):
                    if indices.numel() == 0:
                        return {}
                    k = min(kmax, indices.numel())
                    ii = indices[:k].tolist()
                    return {
                        f"samples/source_{tag}": [wandb.Image(denorm(src_image1[i].detach().cpu()), caption=f"src {i}") for i in ii],
                        f"samples/target_{tag}": [wandb.Image(denorm(src_image2[i].detach().cpu()), caption=f"tgt {i}") for i in ii],
                        f"samples/fake_{tag}"  : [wandb.Image(denorm(img_fake[i].detach().cpu()),   caption=f"fake {i}") for i in ii],
                    }

                # self / cross 각각 패널 생성
                logs.update(log_panel(idx_self,  "self"))
                logs.update(log_panel(idx_cross, "cross"))

                # 하나라도 있으면 로깅
                if logs:
                    wandb.log(logs, step=step+1)


        # (선택) 주기적 체크포인트
        if (step + 1) % opt.model_freq == 0:
            ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.netG.state_dict(), os.path.join(ckpt_dir, f'netG_step{step+1}.pth'))