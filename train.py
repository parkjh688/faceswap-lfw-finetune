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
        
        # Adversarial loss 설정
        self.parser.add_argument('--adv_warmup_start', type=int, default=3000)
        self.parser.add_argument('--adv_warmup_end', type=int, default=15000)
        self.parser.add_argument('--adv_max_weight', type=float, default=1.0)
        
        # Training stability
        self.parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping threshold (0=disabled)')
        self.parser.add_argument('--use_ema', type=str2bool, default='False', help='use EMA for G')
        self.parser.add_argument('--ema_decay', type=float, default=0.999)

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

    exp_dir = os.path.join(opt.checkpoints_dir, opt.name)
    iter_path = os.path.join(exp_dir, 'iter.txt')
    sample_path = os.path.join(exp_dir, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    log_path = os.path.join(exp_dir, 'summary')
    os.makedirs(log_path, exist_ok=True)

    # -------------------------
    # (A) 재개용 상태 기본값
    # -------------------------
    global_step_offset = 0           # 마지막까지 끝낸 step
    resume_run_id = None             # W&B run id
    latest_ckpt = os.path.join(exp_dir, "latest.pth")

    # -------------------------
    # (B) 모델 생성/초기화
    # -------------------------
    model = fsModel()
    model.initialize(opt)
    model.to(device)

    # Pretrained checkpoints (Generator required, Discriminator optional)
    if opt.load_pretrain and os.path.exists(opt.load_pretrain):
        try:
            # ✅ G만 확실히 로드 (D/optimizer는 여기서 건드리지 않음)
            model.load_network(model.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print(f'Loaded Generator weights from {opt.load_pretrain}')
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

    # D는 고정 특징 추출기로만 사용
    for p in model.netD.parameters():
        p.requires_grad = False
    model.netD.eval()
    optimizer_D = None

    log_name = os.path.join(exp_dir, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write(f'================ Training Loss ({now}) ================\n')

    optimizer_G = torch.optim.Adam(
        (p for p in model.netG.parameters() if p.requires_grad),
        lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    
    # EMA (선택적)
    netG_ema = None
    if opt.use_ema:
        from copy import deepcopy
        netG_ema = deepcopy(model.netG).eval().requires_grad_(False)
        print(f"Using EMA with decay={opt.ema_decay}")

    # -------------------------
    # (C) latest.pth 있으면 재개
    # -------------------------
    if opt.continue_train and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        # netG
        model.netG.load_state_dict(ckpt["netG"], strict=True)
        # EMA 복원
        if opt.use_ema and ckpt.get("netG_ema") is not None:
            from copy import deepcopy
            netG_ema = deepcopy(model.netG).eval().requires_grad_(False)
            netG_ema.load_state_dict(ckpt["netG_ema"], strict=True)
            print("[RESUME] EMA weights loaded")
        # optimizer (있으면)
        if ckpt.get("optimizer_G") is not None:
            optimizer_G.load_state_dict(ckpt["optimizer_G"])
            # 안전: 학습률 보정
            for pg in optimizer_G.param_groups:
                pg["lr"] = opt.lr
        # step & wandb id
        global_step_offset = int(ckpt.get("step", 0))
        resume_run_id = ckpt.get("wandb_run_id", None)
        print(f"[RESUME] latest.pth loaded @ step {global_step_offset}")
    elif opt.continue_train:
        print("[RESUME] latest.pth not found, starting fresh.")

    # NOTE: 로더 정규화가 ImageNet(mean/std)인 경우에만 denorm 필요
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    denorm = lambda x: (x * imagenet_std + imagenet_mean).clamp(0, 1)

    train_loader = get_lfw_loader_aligned(
        root_dir=opt.dataset,
        batch_size=opt.batchSize,
        num_workers=4,
        seed=1234 + global_step_offset,  # resume 시 다른 시작점
        self_prob=0.5,
        image_size=224,
        epoch_mul=10,
    )
    data_iter = iter(train_loader)
    
    # Resume 시 데이터 로더를 올바른 위치로 이동
    if global_step_offset > 0:
        print(f"Skipping {global_step_offset} batches to resume at correct data position...")
        for _ in range(global_step_offset % len(train_loader)):
            try:
                _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                _ = next(data_iter)

    # -------------------------------
    # Init WandB (resume-aware)
    # -------------------------------
    wandb.init(
        project="faceswap-lfw",
        name=opt.name,
        config=vars(opt),
        id=resume_run_id,     # 이전 run id가 있으면 이어붙임
        resume="allow"        # 없으면 새로 시작
    )
    WAND_RUN_ID = wandb.run.id

    print("Start training (G-only fine-tune)")
    remaining_steps = opt.total_step - global_step_offset
    print(f"Training from step {global_step_offset} to {opt.total_step} ({remaining_steps} steps remaining)")
    
    for step in range(remaining_steps):
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
        img_id_112 = (img_id_112 - 0.5) / 0.5
        with torch.no_grad():
            latent_id = model.netArc(img_id_112)
            latent_id = F.normalize(latent_id, p=2, dim=1)

        # forward
        model.netG.train()
        img_fake = model.netG(src_image1, latent_id)

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
        is_same_f = is_same.float().view(-1, 1, 1, 1)
        if is_same_f.sum() > 0:
            rec_per_px = (img_fake - src_image1).abs()
            loss_G_Rec = (rec_per_px * is_same_f).sum() / (is_same_f.sum() * rec_per_px.shape[1] * rec_per_px.shape[2] * rec_per_px.shape[3])
            loss_G_Rec = loss_G_Rec * opt.lambda_rec
        else:
            loss_G_Rec = torch.tensor(0.0, device=device)

        # adversarial: 절대 step 기준 워밍업 (resume-aware)
        abs_step = global_step_offset + step
        if abs_step < opt.adv_warmup_start:
            lambda_adv = 0.0
        elif abs_step < opt.adv_warmup_end:
            progress = (abs_step - opt.adv_warmup_start) / (opt.adv_warmup_end - opt.adv_warmup_start)
            lambda_adv = opt.adv_max_weight * progress
        else:
            lambda_adv = opt.adv_max_weight
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
        
        # Gradient clipping (폭주 방지)
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.netG.parameters(), opt.grad_clip)
        
        optimizer_G.step()
        
        # EMA 업데이트
        if netG_ema is not None:
            with torch.no_grad():
                for p_ema, p in zip(netG_ema.parameters(), model.netG.parameters()):
                    p_ema.data.mul_(opt.ema_decay).add_(p.data, alpha=1 - opt.ema_decay)

        # -------------------------------
        # Logging to WandB (resume-aware)
        # -------------------------------
        current_step = global_step_offset + step + 1

        if (step + 1) % opt.log_frep == 0:
            cos_sim = model.cosin_metric(latent_fake.detach(), latent_id.detach()).mean().item()
            
            # Gradient norm 체크 (폭주 감지)
            total_norm = 0.0
            for p in model.netG.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            wandb.log({
                "loss/total_G": loss_G.item(),
                "loss/G_adv":   float(loss_adv.item()) if lambda_adv > 0 else 0.0,
                "loss/G_ID":    loss_G_ID.item(),
                "loss/G_rec":   float(loss_G_Rec.item()),
                "loss/G_feat":  feat_match_loss.item(),
                "metric/id_cosine": cos_sim,
                "metric/grad_norm": total_norm,
                "meta/lambda_adv": lambda_adv,
                "meta/step": current_step,
            }, step=current_step)

            # 재개 보조 파일
            with open(iter_path, "w") as f:
                f.write(f"{current_step},0\n")

        if (step + 1) % opt.sample_freq == 0:
            # EMA 모델로 샘플 생성 (더 안정적)
            eval_G = netG_ema if netG_ema is not None else model.netG
            eval_G.eval()
            
            with torch.no_grad():
                # EMA 사용 시 재생성
                if netG_ema is not None:
                    img_fake = eval_G(src_image1, latent_id)
                
                idx_self  = (is_same == 1).nonzero(as_tuple=True)[0]
                idx_cross = (is_same == 0).nonzero(as_tuple=True)[0]

                logs = {}

                def log_panel(indices, tag, kmax=4):
                    """각 샘플을 가로로 묶어서 표시: [source | target | fake]"""
                    if indices.numel() == 0:
                        return {}
                    k = min(kmax, indices.numel())
                    combined_images = []
                    for i in indices[:k].tolist():
                        src_img = denorm(src_image1[i].detach().cpu())
                        tgt_img = denorm(src_image2[i].detach().cpu())
                        fake_img = denorm(img_fake[i].detach().cpu())
                        combined_images.append(
                            wandb.Image(
                                [src_img, tgt_img, fake_img],
                                caption=f"Sample {i}: [Pose | Identity | Result]"
                            )
                        )
                    return {f"samples/{tag}_combined": combined_images}

                logs.update(log_panel(idx_self,  "self"))
                logs.update(log_panel(idx_cross, "cross"))

                if logs:
                    wandb.log(logs, step=current_step)

        # -------------------------------
        # 체크포인트 저장 (resume-ready)
        # -------------------------------
        if (step + 1) % opt.model_freq == 0:
            os.makedirs(exp_dir, exist_ok=True)
            # 1) netG 단독 저장 (호환성)
            torch.save(model.netG.state_dict(), os.path.join(exp_dir, f'netG_step{current_step}.pth'))
            # 1-1) EMA 저장
            if netG_ema is not None:
                torch.save(netG_ema.state_dict(), os.path.join(exp_dir, f'netG_ema_step{current_step}.pth'))
            
            # 2) 완전 체크포인트 (resume 전용)
            ckpt = {
                "step": current_step,
                "netG": model.netG.state_dict(),
                "netG_ema": netG_ema.state_dict() if netG_ema is not None else None,
                "optimizer_G": optimizer_G.state_dict(),
                "wandb_run_id": WAND_RUN_ID,
                "opt": vars(opt),
            }
            torch.save(ckpt, os.path.join(exp_dir, f'ckpt_step{current_step}.pth'))
            torch.save(ckpt, latest_ckpt)  # 항상 덮어쓰기
            print(f"[CKPT] Saved @ step {current_step}")

    print("Training finished.")