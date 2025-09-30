# data/data_loader_lfw_aligned.py
import os
import glob
import random
from typing import List, Dict
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class LFWDatasetAligned(data.Dataset):
    """
    LFW pair sampling for face swap fine-tuning (pre-aligned images).
    Assumes the directory structure:
        root/
          personA/*.jpg
          personB/*.jpg
          ...
    Returns:
        img1 (Tensor): source image (transformed)
        img2 (Tensor): target image (transformed)
        is_same (int): 1 if self-swap, 0 if cross-swap
    """
    def __init__(
        self,
        root_dir: str,
        transform: T.Compose,
        seed: int = 1234,
        self_prob: float = 0.5,
        use_aug_for_single: bool = True,
        epoch_mul: int = 10,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.self_prob = self_prob
        self.use_aug_for_single = use_aug_for_single
        self.epoch_mul = max(1, int(epoch_mul))

        self.id2imgs: Dict[str, List[str]] = {}
        people = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        for pid in people:
            pdir = os.path.join(root_dir, pid)
            imgs = []
            for ext in IMG_EXTS:
                imgs.extend(glob.glob(os.path.join(pdir, f"*{ext}")))
            if len(imgs) > 0:
                self.id2imgs[pid] = imgs

        self.ids = list(self.id2imgs.keys())
        random.seed(seed)
        print(f"[LFWDatasetAligned] Loaded {len(self.ids)} identities from {root_dir} (pre-aligned)")

        # 간단 품질 필터(선택): 이미지가 2장 미만인 ID는 self-swap에서 복제 사용
        # 필요하면 여기서 너무 작은 해상도/비율 이상치 걸러도 됨.

    def __len__(self):
        # 에폭 길이(샘플 수): 아이디 수 * epoch_mul
        return max(1, len(self.ids) * self.epoch_mul)

    def _open(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        if random.random() < self.self_prob and len(self.ids) > 0:
            # ---------- Self-swap ----------
            src_id = random.choice(self.ids)
            paths = self.id2imgs[src_id]
            if len(paths) >= 2:
                p1, p2 = random.sample(paths, 2)
                img1 = self._open(p1)
                img2 = self._open(p2)
            else:
                p1 = paths[0]
                img1 = self._open(p1)
                img2 = img1.copy()  # 단일 이미지일 때 복제
            is_same = 1
        else:
            # ---------- Cross-swap ----------
            if len(self.ids) >= 2:
                id1, id2 = random.sample(self.ids, 2)
            else:
                # ID가 1개뿐이면 self로 대체
                id1 = id2 = self.ids[0]
            p1 = random.choice(self.id2imgs[id1])
            p2 = random.choice(self.id2imgs[id2])
            img1 = self._open(p1)
            img2 = self._open(p2)
            is_same = int(id1 == id2)  # 보통 0이 됨

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, is_same


def _seed_worker(worker_id):
    """각 DataLoader worker에 고유 seed 부여 (torch>=1.8 권장 패턴)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def get_lfw_loader_aligned(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 1234,
    self_prob: float = 0.5,
    image_size: int = 224,
    epoch_mul: int = 10,
):
    """
    Pre-aligned LFW loader (no alignment inside).
    Note:
      - Generator/Discriminator 입력용으로 ImageNet 정규화.
      - ArcFace는 forward에서 별도로 112/[-1,1] 정규화로 처리하세요.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    dataset = LFWDatasetAligned(
        root_dir=root_dir,
        transform=transform,
        seed=seed,
        self_prob=self_prob,
        epoch_mul=epoch_mul,
    )

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    return loader