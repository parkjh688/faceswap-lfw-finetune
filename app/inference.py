# app/inference.py
import io
import torch, cv2, numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import Optional
from models.projected_model import fsModel
from data.face_align_utils import align_face

async def load_image_from_any(file: Optional[Image.Image], url: Optional[str]):
    if file is not None:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    if url:
        raise RuntimeError("URL download disabled; upload file instead.")
    raise RuntimeError("No input given")

def _align224(img_rgb: np.ndarray, ctx_id: int = -1) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    aligned = align_face(pil, output_size=224, ctx_id=ctx_id)
    return np.array(aligned)

class InferenceOpt:
    def __init__(self, device="cpu"):
        self.isTrain = True
        self.gpu_ids = [] if device == "cpu" else [0]
        self.Gdeep = False

        self.Arc_path = "arcface_model/arcface_checkpoint.tar"
        self.checkpoints_dir = "./checkpoints/simswap"
        self.which_epoch = "latest"
        self.lr = 0.0002
        self.beta1 = 0.5
        self.continue_train = False
        self.load_pretrain = None
        self.name = "inference"

def _resize_224(img_rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(img_rgb, (224,224), interpolation=cv2.INTER_CUBIC)

def _to_tensor(img_rgb):
    img_rgb = _resize_224(img_rgb)
    x = torch.from_numpy(img_rgb.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (x-mean)/std

def _denorm(x):
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x = (x*std+mean).clamp(0,1)
    return (x[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)

class FaceSwapPipeline:
    def __init__(self, gen_ckpt:str, arc_ckpt:str, device="cpu"):
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = fsModel()
        opt = InferenceOpt(self.device.type)
        self.model.initialize(opt)
        self.model.to(self.device)

        # netG
        sd = torch.load(gen_ckpt, map_location="cpu")
        self.model.netG.load_state_dict(sd, strict=False)

        # ArcFace
        arc_obj = torch.load(arc_ckpt, map_location="cpu")
        arc_state = arc_obj["model"] if isinstance(arc_obj, dict) and "model" in arc_obj else (
            arc_obj.state_dict() if hasattr(arc_obj, "state_dict") else arc_obj
        )
        self.model.netArc.load_state_dict(arc_state, strict=False)
        self.model.netArc.eval()

        self.model.netG.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _arc_embed(self, img_rgb: np.ndarray):
        img = cv2.resize(img_rgb, (112,112), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(img.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
        t = (t - 0.5) / 0.5
        t = t.to(self.device)
        with torch.no_grad():
            feat = self.model.netArc(t)
            feat = F.normalize(feat, p=2, dim=1)
        return feat

    def swap_image(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        src_rgb_a = _align224(source_rgb, ctx_id=-1)
        tgt_rgb_a = _align224(target_rgb,  ctx_id=-1)

        src_latent = self._arc_embed(src_rgb_a)
        tgt_t = _to_tensor(tgt_rgb_a).to(self.device)

        with torch.no_grad():
            out = self.model.netG(tgt_t, src_latent)
        return _denorm(out)