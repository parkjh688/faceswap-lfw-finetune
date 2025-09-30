#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
from PIL import Image

# lazy singleton
_face_app = None
_curr_cfg = {"ctx_id": None, "det_size": None}

def _setup_insightface(ctx_id=0, det_size=(640, 640)):
    """
    Initialize insightface FaceAnalysis once (lazy).
    Recreate only when ctx_id or det_size changes.
    """
    global _face_app, _curr_cfg
    try:
        need_new = (
            _face_app is None
            or _curr_cfg["ctx_id"] != ctx_id
            or _curr_cfg["det_size"] != tuple(det_size)
        )
        if need_new:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            _face_app = app
            _curr_cfg = {"ctx_id": ctx_id, "det_size": tuple(det_size)}
        return _face_app
    except Exception as e:
        raise RuntimeError(f"InsightFace init failed: {e}")

# ArcFace 표준 5점 좌표(112 기준)
_ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def _align_with_landmarks(img_bgr: np.ndarray, landmark_5pts: np.ndarray, output_size: int) -> Image.Image:
    dst = _ARCFACE_DST_112 * (output_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmark_5pts.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
    aligned_bgr = cv2.warpAffine(img_bgr, M, (output_size, output_size), borderValue=0)
    return Image.fromarray(aligned_bgr[:, :, ::-1])  # BGR->RGB

def align_face(
    pil_img: Image.Image,
    output_size: int = 224,
    ctx_id: int = 0,
    det_size=(640, 640),
    retry_det_size=(1024, 1024),
    retry: bool = True
) -> Image.Image:
    """
    얼굴을 ArcFace 표준으로 정렬하여 PIL.Image(RGB)를 반환.
    - ctx_id: GPU=0, CPU=-1
    - det_size: 1차 탐지 해상도
    - retry_det_size: 실패 시 재시도 해상도
    """
    # PIL RGB -> np BGR
    img_rgb = np.array(pil_img)
    img_bgr = img_rgb[:, :, ::-1]

    app = _setup_insightface(ctx_id=ctx_id, det_size=det_size)
    faces = app.get(img_bgr)
    if len(faces) == 0 and retry:
        # det_size 키워서 한 번 더
        app = _setup_insightface(ctx_id=ctx_id, det_size=retry_det_size)
        faces = app.get(img_bgr)

    if len(faces) == 0:
        print("[align_face] No face detected -> returning original")
        return pil_img

    # 가장 큰 얼굴 선택
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    kps = getattr(face, "kps", None)
    if kps is None:
        print("[align_face] No 5-point landmarks -> returning original")
        return pil_img

    landmark = np.asarray(kps, dtype=np.float32)  # (5,2)
    return _align_with_landmarks(img_bgr, landmark, output_size)

# -------------------------------
# Quick test (standalone main)
# -------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python data/face_align_utils.py <input.jpg> <output.jpg> [size] [ctx_id]")
        print("Ex)    python data/face_align_utils.py in.jpg out.jpg 224 0")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    ctx  = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    img = Image.open(in_path).convert("RGB")
    aligned = align_face(img, output_size=size, ctx_id=ctx)
    aligned.save(out_path, quality=95)
    print(f"[face_align_utils] saved -> {out_path}")

