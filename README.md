# DeFT: Descriptor-Forked Test-Time Adaptation for Multi-Domain Medical Image Denoising

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange.svg)](https://huggingface.co/Lockbro/deft-checkpoints)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xiao-Zhanpeng/DeFT/blob/main/demo.ipynb)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

<!--
### 🗞️ News
- [202X-XX-XX] Code released
- [202X-XX-XX] Paper accepted at XXX 202X
-->

## 🏥 Overview

> DeFT (Descriptor-Forked Test-Time Adaptation) is a source-free, single-image test-time adaptation method for medical image denoising under domain shift. A shared degradation descriptor, extracted once per test image, forks into three complementary projections: **DCI** exposes the read-only conditioning interface, **PRM** performs polarized dual-route spatial routing between aggressive denoising and structure preservation, and **DCS** maps the same descriptor to an image-specific update schedule. The backbone remains frozen throughout; only lightweight episodic state (<1M parameters) is optimized per image.

---

## 📂 Project Structure

```
deft-open-source/
├── deft/                  # DeFT core package
│   ├── model.py           # DeFT main class + adapt() loop
│   ├── descriptor.py      # DescriptorState + noise estimation
│   ├── dci.py             # FiLM / PromptFiLM conditioning layers
│   ├── prm.py             # PolarizedRouteMixture dual-route spatial mixing
│   ├── dcs.py             # DescriptorConditionedScheduler
│   ├── backbone.py        # DeFTBackbone (31M U-Net)
│   └── loss.py            # Neighbor2Neighbor + Charbonnier
├── examples/              # Sample images for Quick Start
├── check_env.py           # Environment checker
├── environment.yaml        # Conda environment (mamba preferred)
├── requirements.txt        # pip fallback
├── scripts/               # One-click run scripts
└── checkpoints/           # Download checkpoints here
```

---

## 🛠️ Requirements

Hardware (reference)

| Item | Spec |
|------|------|
| CPU | Ryzen 7 5800X (8C16T) |
| RAM | 32 GB |
| GPU | RTX 4090 24 GB |
| Driver / CUDA | NVIDIA 535 / CUDA 12.2 |
| OS | Ubuntu 22.04 LTS |

One-click environment setup:

```bash
# Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
bash miniforge.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash

# Create and activate environment
mamba env create -f environment.yaml
mamba activate deft

# Verify
python check_env.py
```

---

## 📂 Data & Checkpoints

### Datasets (source $\mathcal{P}$ and targets $\mathcal{Q}_1, \mathcal{Q}_2, \mathcal{Q}_3$)

<table>
<tr><th>Role</th><th>Dataset</th><th>Modality</th><th>Download</th></tr>
<tr><td rowspan="3">$\mathcal{P}$</td><td>LIDC-IDRI</td><td align="center">CT</td><td><a href="https://www.cancerimagingarchive.net/collection/lidc-idri/">TCIA</a></td></tr>
<tr><td>IXI</td><td align="center">MRI</td><td><a href="https://brain-development.org/ixi-dataset/">brain-development</a></td></tr>
<tr><td>OASIS-1</td><td align="center">MRI</td><td><a href="https://sites.wustl.edu/oasisbrains/">oasis-brains</a></td></tr>
<tr><td>$\mathcal{Q}_1$</td><td>Mayo Low-Dose CT</td><td align="center">CT</td><td><a href="https://www.aapm.org/GrandChallenge/LowDoseCT/">AAPM</a></td></tr>
<tr><td>$\mathcal{Q}_2$</td><td>fastMRI Knee</td><td align="center">MRI</td><td><a href="https://fastmri.med.nyu.edu/">fastMRI</a></td></tr>
<tr><td>$\mathcal{Q}_3$</td><td>ChestX-ray14</td><td align="center">X-ray</td><td><a href="https://nihcc.app.box.com/v/ChestXray-NIHCC">NIHCC</a></td></tr>
</table>

Pre-built Google Drive archive (7z / tar compressed):
```bash
# Download all 6 dataset archives (Google Drive)
pip install gdown
gdown --folder 1J3rZ3AjbTTo3laSZmi1cqe886QQbk7Tw

# Extract each archive into raw_medical_datasets/
mkdir -p raw_medical_datasets
7z x IXI-Dataset.7z -o raw_medical_datasets          # Brain MRI (IXI)
7z x NIH-ChestX-ray14.7z -o raw_medical_datasets      # Chest X-ray
7z x LIDC-IDRI.7z -o raw_medical_datasets             # Lung CT
7z x OASIS-1.7z -o raw_medical_datasets               # Brain MRI (OASIS)
7z x 2016_mayo_CT.7z -o raw_medical_datasets          # Abdomen CT (Mayo)
tar -xf fastMRI_knee_singlecoil.tar -C raw_medical_datasets  # Knee MRI

# Build train/val/test splits
python tools/auto_build_datasets.py --raw-root raw_medical_datasets --out-root preprocessed_datasets
```

### Checkpoints (Google Drive / Hugging Face 🤗)

| File | Method | Size | Download |
|------|--------|------|----------|
| `unet_source_checkpoint.pt` | DeFT backbone (source pretrained) | ~356 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `swinir_checkpoint.pt` | SwinIR | ~422 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `restormer_checkpoint.pt` | Restormer | ~300 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `apbsn_checkpoint.pt` | AP-BSN | 44 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `blind2unblind_checkpoint.pt` | Blind2Unblind | 13 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `cyclegan_checkpoint.pt` | CycleGAN | ~324 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `reggan_checkpoint.pt` | RegGAN | ~332 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `unet_ctonly_checkpoint.pt` | CT-only source composition | ~356 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |
| `unet_mrionly_checkpoint.pt` | MRI-only source composition | ~356 MB | [HF](https://huggingface.co/Lockbro/deft-checkpoints) |

```bash
# Google Drive (bulk, all 9 checkpoints)
gdown --folder 1zmPcJQUqAmQtWhO2Ti-uqGtl8EpJFU7d

# Hugging Face (single file)
pip install huggingface_hub
huggingface-cli download Lockbro/deft-checkpoints unet_source_checkpoint.pt --local-dir checkpoints/

# Hugging Face (bulk)
huggingface-cli download Lockbro/deft-checkpoints --local-dir checkpoints/
```

---

## 🏋️ Training & Evaluation

### Training (source-domain pretraining)

```bash
# Pre-train a DeFT backbone on the source domain (LIDC-CT + IXI/OASIS-MRI)
python scripts/train_deft.py --data-root data/P_train --output-dir checkpoints

# The resulting checkpoint is used for downstream test-time adaptation.
# See "Evaluation" below for running DeFT on target domains.
```

| Parameter | Value |
|-----------|-------|
| Backbone | U-Net (~31M params) |
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Learning rate | $2 \times 10^{-4}$ |
| Batch size | 16 |
| Epochs | 100 |
| Loss | Charbonnier ($\varepsilon=10^{-3}$) + Neighbor2Neighbor |
| Data augmentation | Random H/V flip |
| Pretraining data | LIDC-IDRI (CT) + IXI/OASIS-1 (MRI) |

### Evaluation (test-time adaptation)

```bash
# Q1 Mayo abdomen CT (sigma=0.10)
python scripts/eval_deft.py --dataset Q0_mayo_eval_S10 --checkpoint checkpoints/unet_source_checkpoint.pt --output results/q1/

# Q2 fastMRI knee MRI (Rician sigma=0.07)
python scripts/eval_deft.py --dataset Q3_fastmri_eval_R07 --checkpoint checkpoints/unet_source_checkpoint.pt --output results/q2/

# Q3 Chest X-ray (sigma=0.10)
python scripts/eval_deft.py --dataset Q2_xray_eval_S10 --checkpoint checkpoints/unet_source_checkpoint.pt --output results/q3/
```

Canonical DeFT evaluation configuration:
| Parameter | Value |
|-----------|-------|
| Adapter type | `prm_prompt` |
| Routing mode | hybrid, spatial gate ON, hard route ON |
| Steps | 5 (adaptive) |
| Learning rate | $2 \times 10^{-4}$ |
| Loss | N2N + Charbonnier ($\varepsilon=10^{-3}$) |
| MAD reliability mask | $k=1.0$ |
| Prompt bank | $8 \times 32$ |
| FiLM levels | 3 (bottleneck + decoder1 + decoder2) |

---

## 🚀 Quick Start

Try DeFT in your browser — no local setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xiao-Zhanpeng/DeFT/blob/main/demo.ipynb)

Or run locally after completing the steps above:

```bash
# Download the backbone checkpoint (see Data & Checkpoints section)
huggingface-cli download Lockbro/deft-checkpoints unet_source_checkpoint.pt --local-dir checkpoints/

# Run DeFT on example images
python run_demo.py
```

---

## 📊 Results

### Main Comparison ($\mathcal{Q}_1$ / $\mathcal{Q}_2$ / $\mathcal{Q}_3$)

$\mathcal{Q}_1$ Mayo abdomen CT ($\sigma{=}0.10$), $\mathcal{Q}_2$ fastMRI knee MRI ($\sigma{=}0.07$, Rician), $\mathcal{Q}_3$ Chest X-ray ($\sigma{=}0.10$). **Bold** / <u>underline</u> mark best / second-best. Full table with RMSE and s.d. available in the manuscript.

| Method | $\mathcal{Q}_1$ PSNR / SSIM | $\mathcal{Q}_2$ PSNR / SSIM | $\mathcal{Q}_3$ PSNR / SSIM |
|--------|:---:|:---:|:---:|
| Restormer | 30.16 / 0.5957 | 29.03 / 0.7402 | <u>30.25</u> / 0.6570 |
| SwinIR | 29.05 / 0.5580 | <u>29.32</u> / 0.7525 | 28.73 / 0.5756 |
| B2U | 19.86 / 0.3531 | 27.65 / 0.7229 | 29.27 / 0.6482 |
| AP-BSN | <u>30.64</u> / 0.6822 | 27.01 / 0.6651 | 28.99 / <u>0.8508</u> |
| ZS-N2N | 28.19 / 0.5470 | 24.41 / 0.6018 | 29.96 / 0.6975 |
| DIP | 25.30 / 0.4592 | 26.12 / 0.6117 | 25.60 / 0.4660 |
| CycleGAN | 29.99 / <u>0.7732</u> | 23.70 / 0.5295 | 14.40 / 0.5358 |
| RegGAN | 30.09 / 0.7697 | 26.02 / 0.5690 | 18.26 / 0.6431 |
| CoTTA | 28.44 / 0.6405 | 29.03 / 0.7609 | 24.14 / 0.8229 |
| LAN | 28.07 / 0.6398 | 28.49 / <u>0.7658</u> | 22.39 / 0.8311 |
| **DeFT (Ours)** | **32.56** / **0.7841** | **29.38** / **0.7712** | **31.82** / **0.8535** |

> Full results including PSNR / RMSE / SSIM with standard deviations across 9 target configurations ($\mathcal{Q}_1{\times}3\sigma$, $\mathcal{Q}_2{\times}3\sigma$, $\mathcal{Q}_3{\times}3\sigma$) are available in the manuscript.

---

## 🙏 Acknowledgement

Motivated by recent advances in test-time adaptation and image restoration:

- [PromptIR](https://doi.org/10.52202/075280-3121) (Potlapalli et al., NeurIPS 2023) — prompt-conditioned image restoration
- [TTT-MIM](https://doi.org/10.1007/978-3-031-73254-6_20) (Mansour et al., ECCV 2024) — self-supervised test-time training for denoising
- [Task-adaptive routing](https://doi.org/10.1007/978-3-031-72104-5_7) (Yang et al., MICCAI 2024) — expert routing for medical image restoration
- [Denoising as Adaptation](https://openreview.net/forum?id=jsBhmOCKYs) (Liao et al., ICLR 2025) — noise-space domain adaptation
- [AdaIR](https://openreview.net/forum?id=M5t0WvjfCg) (Cui et al., ICLR 2025) — adaptive frequency modulation

Implementation and data:

- [MONAI](https://github.com/Project-MONAI/MONAI) — medical imaging pipelines
- [LAN](https://doi.org/10.1109/CVPR52733.2024.02380) (Kim et al., CVPR 2024) — single-image source-free test-time adaptation
- [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/), [IXI](https://brain-development.org/ixi-dataset/), [OASIS-1](https://sites.wustl.edu/oasisbrains/), [Mayo CT](https://www.aapm.org/GrandChallenge/LowDoseCT/), [fastMRI](https://fastmri.med.nyu.edu/), [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

<!--
## Citation

If you find this work useful, please consider citing:

```
@misc{deft2025,
      title={DeFT: Descriptor-Forked Test-Time Adaptation for Generalizable Medical Image Denoising},
      author={Zhanpeng Xiao and others},
      year={2025},
      howpublished={arXiv preprint, 2025},
}
```
-->

## License

This project is released under the [MIT License](LICENSE).
