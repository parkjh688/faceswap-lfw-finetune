# app/video.py
import cv2, numpy as np
import torch
import torch.nn.functional as F

from app.inference import _denorm, _to_tensor, _align224  # ✅ align 추가

@torch.inference_mode()
def swap_video_stream(pipeline, src_img_rgb: np.ndarray, in_path: str, out_path: str, smooth: float = 0.9):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # 1) 소스 얼굴 정렬 후 latent 추출 (한 번만)
    src_rgb_a  = _align224(src_img_rgb, ctx_id=-1)   # GPU면 0
    latent_src = pipeline._arc_embed(src_rgb_a)
    ema_latent = latent_src.clone()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        tgt_rgb_a = _align224(frame_rgb, ctx_id=-1)

        ema_latent = smooth * ema_latent + (1.0 - smooth) * latent_src
        ema_latent = F.normalize(ema_latent, p=2, dim=1)

        t = _to_tensor(tgt_rgb_a).to(pipeline.device)
        out = pipeline.model.netG(t, ema_latent)
        out_rgb_224 = _denorm(out)

        out_rgb = cv2.resize(out_rgb_224, (w, h), interpolation=cv2.INTER_CUBIC)
        writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()