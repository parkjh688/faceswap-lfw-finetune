# app/video.py
import cv2, numpy as np
import torch
import torch.nn.functional as F

from app.inference import _denorm, _to_tensor
from data.face_align_utils import _setup_insightface, _ARCFACE_DST_112

@torch.inference_mode()
def swap_video_stream(pipeline, src_img_rgb: np.ndarray, in_path: str, out_path: str,
                      smooth: float = 0.9, det_size=(320,320),
                      lm_beta: float = 0.7, flow_blend: float = 0.15):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    ctx_id = 0 if (pipeline.device.type == "cuda") else -1
    app = _setup_insightface(ctx_id=ctx_id, det_size=det_size)

    src_kps = _detect_kps(app, src_img_rgb)
    src_rgb_a, _ = _align224_with_kps(src_img_rgb, src_kps)
    latent_src = pipeline._arc_embed(src_rgb_a)
    ema_latent = latent_src.clone()

    prev_kps = None
    prev_out = None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        kps = _detect_kps(app, frame_rgb)
        if kps is None:
            tgt_a = cv2.resize(frame_rgb, (224,224), interpolation=cv2.INTER_CUBIC)
        else:
            if prev_kps is None:
                kps_s = kps
            else:
                kps_s = lm_beta * prev_kps + (1.0 - lm_beta) * kps
            prev_kps = kps_s
            tgt_a, _ = _align224_with_kps(frame_rgb, kps_s)

        ema_latent = smooth * ema_latent + (1.0 - smooth) * latent_src
        ema_latent = F.normalize(ema_latent, p=2, dim=1)

        t = _to_tensor(tgt_a).to(pipeline.device)
        out = pipeline.model.netG(t, ema_latent)
        out_224 = _denorm(out)
        curr = cv2.resize(out_224, (w, h), interpolation=cv2.INTER_CUBIC)

        # output stabilization
        if prev_out is not None and flow_blend > 0:
            g_prev = cv2.cvtColor(prev_out, cv2.COLOR_RGB2GRAY)
            g_curr = cv2.cvtColor(curr,    cv2.COLOR_RGB2GRAY)
            flow = dis.calc(g_prev, g_curr, None)
            h_grid, w_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = (w_grid + flow[...,0]).astype(np.float32)
            map_y = (h_grid + flow[...,1]).astype(np.float32)
            prev_warp = cv2.remap(prev_out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            curr = cv2.addWeighted(curr, 1.0 - flow_blend, prev_warp, flow_blend, 0.0)

        writer.write(cv2.cvtColor(curr, cv2.COLOR_RGB2BGR))
        prev_out = curr

    cap.release()
    writer.release()

def _detect_kps(app, img_rgb: np.ndarray):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    kps = getattr(face, "kps", None)
    if kps is None:
        return None
    return np.asarray(kps, dtype=np.float32)

def _align224_with_kps(img_rgb: np.ndarray, kps: np.ndarray):
    dst = (_ARCFACE_DST_112 * (224.0/112.0)).astype(np.float32)
    src = kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), M, (224,224), borderValue=0)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned, M