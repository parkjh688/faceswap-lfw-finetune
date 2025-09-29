import os
import glob
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class LFWDataset(data.Dataset):
    """
    LFW pair sampling dataset for face swap fine-tuning.

    Returns:
        img1 (Tensor): source image
        img2 (Tensor): target image
        is_same (int): 1 if self-swap, 0 if cross-swap
    """

    def __init__(self, root_dir, transform=None, seed=1234,
                 use_aug_for_single=True, self_prob=0.5):
        self.root_dir = root_dir
        self.transform = transform
        self.use_aug_for_single = use_aug_for_single
        self.self_prob = self_prob  # self-swap 확률

        self.id2imgs = {}
        for person in sorted(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, person)
            if not os.path.isdir(person_dir):
                continue
            imgs = glob.glob(os.path.join(person_dir, "*.jpg"))
            if len(imgs) > 0:
                self.id2imgs[person] = imgs

        self.ids = list(self.id2imgs.keys())
        random.seed(seed)
        print(f"Loaded {len(self.ids)} identities from {root_dir}")

    def __len__(self):
        return len(self.ids) * 10

    def __getitem__(self, index):
        if random.random() < self.self_prob:
            # ---------- Self-swap ----------
            src_id = random.choice(self.ids)
            src_imgs = self.id2imgs[src_id]
            if len(src_imgs) >= 2:
                img1_path, img2_path = random.sample(src_imgs, 2)
                img1 = Image.open(img1_path).convert("RGB")
                img2 = Image.open(img2_path).convert("RGB")
            else:
                img1_path = src_imgs[0]
                img1 = Image.open(img1_path).convert("RGB")
                if self.use_aug_for_single:
                    img2 = img1.copy()
                else:
                    img2 = img1.copy()
            is_same = 1
        else:
            # ---------- Cross-swap ----------
            src_id, tgt_id = random.sample(self.ids, 2)
            img1_path = random.choice(self.id2imgs[src_id])
            img2_path = random.choice(self.id2imgs[tgt_id])
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            is_same = 0

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, is_same


def get_lfw_loader(root_dir, batch_size=8, num_workers=4, seed=1234, self_prob=0.5):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset = LFWDataset(root_dir, transform, seed, self_prob=self_prob)
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=True)
    return loader