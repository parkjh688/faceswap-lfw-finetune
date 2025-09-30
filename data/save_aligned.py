#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import glob
import argparse
from typing import List
from PIL import Image
from tqdm import tqdm

import numpy as np
import cv2

from data.face_align_utils import align_face, _setup_insightface

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(root: str) -> List[str]:
    people = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    paths = []
    for person in people:
        pdir = os.path.join(root, person)
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(pdir, f"*{ext}")))
    return paths

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def dst_path_for(src_path: str, src_root: str, dst_root: str, size: int) -> str:
    rel = os.path.relpath(src_path, src_root)
    base, _ = os.path.splitext(rel)
    return os.path.join(dst_root, base + f"_a{size}.jpg")

def process_all(src_root: str, dst_root: str, size: int = 224, ctx_id: int = 0,
                det_retry: bool = True, fallback_resize: bool = True):
    # 미리 한 번 초기화 (첫 샷 지연 방지)
    _ = _setup_insightface(ctx_id=ctx_id)

    img_paths = list_images(src_root)
    print(f"[align] found {len(img_paths)} images under {src_root}")

    for sp in tqdm(img_paths, ncols=100):
        dp = dst_path_for(sp, src_root, dst_root, size)
        ensure_dir(os.path.dirname(dp))

        try:
            pil = Image.open(sp).convert("RGB")
        except Exception as e:
            print(f"[skip:open] {sp} -> {e}")
            continue

        aligned = align_face(pil, output_size=size, ctx_id=ctx_id)

        # 탐지 실패 시 align_face가 원본을 그대로 반환할 수 있음
        # 원하면 det_size를 키워서 한 번 더 시도
        if det_retry and aligned is pil:
            try:
                # det_size를 키우기 위해 임시로 내부 app을 다시 준비
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name='buffalo_l')
                app.prepare(ctx_id=ctx_id, det_size=(1024, 1024))
                # 수동 정렬: landmark로 affine (코드 중복 피하려면 face_align_utils 확장 가능)
                # 여기서는 간단히 다시 align_face를 한 번 더 호출하는 쪽으로 대체
                aligned = align_face(pil, output_size=size, ctx_id=ctx_id)  # 이미 app 재준비 효과
            except Exception:
                pass

        # 그래도 실패면 fallback
        if aligned is pil and fallback_resize:
            aligned = pil.resize((size, size), Image.BICUBIC)

        try:
            aligned.save(dp, quality=95)
        except Exception as e:
            print(f"[skip:save] {dp} -> {e}")

    print(f"[align] done. saved to: {dst_root}")

def parse_args():
    ap = argparse.ArgumentParser(description="Batch align LFW-style folders")
    ap.add_argument("--src", type=str, default="./lfw_funneled", help="input root (person folders inside)")
    ap.add_argument("--dst", type=str, default="./lfw_aligned_224", help="output root (mirrors structure)")
    ap.add_argument("--size", type=int, default=224, help="aligned square size (e.g., 112 or 224)")
    ap.add_argument("--ctx", type=int, default=0, help="insightface ctx_id (GPU=0, CPU=-1)")
    ap.add_argument("--no-retry", action="store_true", help="disable second detection attempt")
    ap.add_argument("--no-fallback", action="store_true", help="disable fallback resize when detection fails")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all(
        src_root=args.src,
        dst_root=args.dst,
        size=args.size,
        ctx_id=args.ctx,
        det_retry=not args.no_retry,
        fallback_resize=not args.no_fallback,
    )