import torch
import lpips
from tqdm import tqdm
from app.inference import FaceSwapPipeline
from data.data_loader_lfw import get_lfw_loader_aligned
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(results['id_self'], alpha=0.5, label='Self-swap', bins=30)
    axes[0].hist(results['id_cross'], alpha=0.5, label='Cross-swap', bins=30)
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_title('Identity Preservation')
    axes[0].legend()
    
    axes[1].hist(results['lpips'], bins=30, color='orange')
    axes[1].set_xlabel('LPIPS Score')
    axes[1].set_title('Perceptual Quality')
    
    axes[2].boxplot([results['id_self'], results['id_cross']], 
                    labels=['Self', 'Cross'])
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('ID Preservation Comparison')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    plt.show()

def evaluate_model(checkpoint_path, test_data_dir, device='cpu', max_batches=50):
    pipeline = FaceSwapPipeline(
        gen_ckpt=checkpoint_path,
        arc_ckpt="arcface_model/arcface_checkpoint.tar",
        device=device
    )
    
    test_loader = get_lfw_loader_aligned(
        root_dir=test_data_dir,
        batch_size=4,
        num_workers=2,
        self_prob=0.5,
        epoch_mul=1
    )
    
    lpips_fn = lpips.LPIPS(net='alex')
    if device == 'cuda':
        lpips_fn = lpips_fn.cuda()
    
    id_sims_self = []
    id_sims_cross = []
    lpips_scores = []
    
    for batch_idx, (src_img, tgt_img, is_same) in enumerate(tqdm(test_loader, total=max_batches)):
        if batch_idx >= max_batches:
            break
            
        if device == 'cuda':
            src_img = src_img.cuda()
            tgt_img = tgt_img.cuda()
        
        with torch.no_grad():
            src_112 = F.interpolate(src_img, size=(112,112), mode='bicubic')
            src_latent = pipeline.model.netArc((src_112 - 0.5) / 0.5)
            src_latent = F.normalize(src_latent, p=2, dim=1)
            
            fake = pipeline.model.netG(tgt_img, src_latent)
            
            fake_112 = F.interpolate(fake, size=(112,112), mode='bicubic')
            fake_latent = pipeline.model.netArc((fake_112 - 0.5) / 0.5)
            fake_latent = F.normalize(fake_latent, p=2, dim=1)
            
            id_sim = F.cosine_similarity(src_latent, fake_latent, dim=1)
            
            for i in range(len(is_same)):
                if is_same[i] == 1:
                    id_sims_self.append(id_sim[i].item())
                else:
                    id_sims_cross.append(id_sim[i].item())
            
            lpips_score = lpips_fn(tgt_img, fake).squeeze()
            if lpips_score.dim() == 0:
                lpips_scores.append(lpips_score.item())
            else:
                lpips_scores.extend(lpips_score.cpu().tolist())
    
    print(f"\n=== Evaluation Results ({len(id_sims_self) + len(id_sims_cross)} samples) ===")
    print(f"ID Similarity (Self-swap):  {np.mean(id_sims_self):.4f} ± {np.std(id_sims_self):.4f}")
    print(f"ID Similarity (Cross-swap): {np.mean(id_sims_cross):.4f} ± {np.std(id_sims_cross):.4f}")
    print(f"LPIPS (lower is better):    {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    
    return {
        'id_self': id_sims_self,
        'id_cross': id_sims_cross,
        'lpips': lpips_scores
    }

if __name__ == "__main__":
    results = evaluate_model(
        checkpoint_path="checkpoints/simswap/netG_step13000.pth",
        test_data_dir="./lfw_aligned_224",
        device="cpu",
        max_batches=50
    )
    plot_metrics(results)