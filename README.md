# FaceSwap LFW Fine-tune

Face swap implementation using SimSwap architecture, fine-tuned on the LFW (Labeled Faces in the Wild) dataset.

## Features

- **Face alignment** using InsightFace for robust preprocessing
- **Identity preservation** with ArcFace embeddings
- **FastAPI server** for image and video face swapping
- **LFW dataset fine-tuning** with self-swap and cross-swap sampling
- **WandB integration** for training visualization

## Project Structure

```
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── inference.py        # Face swap pipeline
│   └── video.py            # Video processing
├── data/
│   ├── data_loader_lfw.py  # LFW dataset loader
│   └── face_align_utils.py # Face alignment utilities
├── models/
│   └── projected_model.py  # SimSwap model definition
├── train.py                # Training script
└── save_aligned.py         # Batch face alignment script
```

## Installation

```bash
pip install torch torchvision
pip install fastapi uvicorn
pip install insightface opencv-python pillow
pip install wandb tqdm
```

## Data Preparation

1. Download LFW dataset and place in `./lfw_funneled`
2. Run face alignment preprocessing:

```bash
python save_aligned.py --src ./lfw_funneled --dst ./lfw_aligned_224 --size 224 --ctx 0
```

Arguments:
- `--src`: Input directory with person folders
- `--dst`: Output directory for aligned faces
- `--size`: Output image size (default: 224)
- `--ctx`: GPU device ID (0 for GPU, -1 for CPU)

## Training

```bash
python train.py \
  --name simswap \
  --batchSize 4 \
  --total_step 30000 \
  --lr 0.0004 \
  --lambda_id 30.0 \
  --lambda_feat 10.0 \
  --lambda_rec 10.0 \
  --Arc_path arcface_model/arcface_checkpoint.tar \
  --load_pretrain ./people \
  --gpu_ids 0
```

Key hyperparameters:
- `--lambda_id`: Identity loss weight (30.0)
- `--lambda_feat`: Feature matching loss weight (10.0)
- `--lambda_rec`: Reconstruction loss for self-swap (10.0)
- `--self_prob`: Probability of self-swap vs cross-swap (0.5)

Training features:
- Generator-only fine-tuning (discriminator frozen)
- Self-swap and cross-swap pair sampling
- ID cosine similarity tracking
- Adversarial loss warm-up after 10k steps

## Inference API

Start the FastAPI server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Image Face Swap

```bash
curl -X POST http://localhost:8000/swap/image \
  -F "source=@source.jpg" \
  -F "target=@target.jpg" \
  -o output.png
```

### Video Face Swap

```bash
curl -X POST http://localhost:8000/swap/video \
  -F "source=@source.jpg" \
  -F "target=@video.mp4" \
  -o output.mp4
```

## Results

### Image Examples

<table>
<tr>
<td><img src="Image 1" width="200"/><br/><i>Source</i></td>
<td><img src="Image 2" width="200"/><br/><i>Target</i></td>
<td><img src="Image 3" width="200"/><br/><i>Result</i></td>
</tr>
</table>

## Model Architecture

- **Generator**: SimSwap architecture with identity injection
- **Discriminator**: Multi-scale patch discriminator (frozen during fine-tuning)
- **Identity Encoder**: ArcFace ResNet-50

## Loss Functions

```
L_total = λ_adv * L_adv + λ_id * L_id + λ_feat * L_feat + λ_rec * L_rec
```

- **L_adv**: Adversarial loss (warm-up after 10k steps)
- **L_id**: Identity cosine similarity loss
- **L_feat**: Feature matching loss (layer 3)
- **L_rec**: Reconstruction loss (self-swap only)

## Requirements

- Python 3.9+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)
- InsightFace models (buffalo_l)
- ArcFace checkpoint

## Citation

If you use this code, please cite the original SimSwap paper:

```bibtex
@article{chen2020simswap,
  title={SimSwap: An Efficient Framework For High Fidelity Face Swapping},
  author={Chen, Renwang and Chen, Xuanhong and Ni, Bingbing and Ge, Yanhao},
  journal={arXiv preprint arXiv:2106.06340},
  year={2020}
}
```

## License
This project is for research purposes only.