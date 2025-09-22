# FSRCNN - Fast Super-Resolution Convolutional Neural Network

Implementation of the FSRCNN model for single image super-resolution based on the paper ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/pdf/1608.00367) by Chao Dong, Chen Change Loy, and Xiaoou Tang.

## Introduction

Single image super-resolution (SISR) is a fundamental computer vision task that aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs. This project implements FSRCNN, an accelerated version of the pioneering SRCNN model that addresses the computational efficiency limitations of its predecessor. The original SRCNN model was introduced in the paper ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/pdf/1501.00092) by Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang.

FSRCNN introduces several key architectural improvements:

- **Faster training and inference**: Achieves 40+ times speedup over SRCNN
- **Smaller model size**: More compact architecture while maintaining quality
- **End-to-end upsampling**: Learns upsampling filters directly instead of using pre-interpolation
- **Flexible architecture**: Supports multiple upscaling factors (2x, 3x, 4x) with a single model

This implementation provides a complete deep learning pipeline including training, evaluation, and inference capabilities with modern PyTorch features like mixed precision training and comprehensive evaluation metrics.

## Model Architecture

The FSRCNN model consists of five main components:

1. **Feature Extraction**: 5×5 convolution for initial feature extraction
2. **Shrinking**: 1×1 convolution to reduce feature map dimensions
3. **Mapping**: Multiple 3×3 convolutions for non-linear mapping
4. **Expanding**: 1×1 convolution to expand feature maps
5. **Deconvolution**: Transpose convolution for upsampling

Key advantages:

- Faster training and inference compared to SRCNN
- Smaller model size
- End-to-end learning of upsampling filters

## Implementation Differences

This implementation differs from the original paper in several key aspects to leverage modern deep learning practices and improve training efficiency:

### Modern PyTorch Features

- **Mixed Precision Training**: Uses `torch.amp.autocast` and `GradScaler` for faster training and reduced memory usage
- **DataLoader Optimizations**: Implements `pin_memory=True`, `persistent_workers=True`, and `num_workers=8` for efficient data loading
- **Modern Optimizers**: Uses Adam optimizer instead of SGD for better convergence

### Training Enhancements

- **Advanced Learning Rate Scheduling**: Implements MultiStepLR scheduler with milestones at 40%, 60%, 80%, and 90% of total epochs
- **Flexible Validation**: Configurable validation frequency (`--val_freq`) to balance training speed and monitoring
- **Checkpoint Management**: Automatic saving of both best and last checkpoints with complete optimizer and scheduler states
- **Resume Training**: Full support for resuming training from any checkpoint with proper epoch numbering

### Data Processing Improvements

- **On-the-fly Data Generation**: Dynamic HR→LR pair generation instead of pre-processed datasets
- **Flexible Patch Sampling**: Configurable patch sizes for training with automatic alignment
- **Image Caching**: Optional in-memory caching for faster data loading on systems with sufficient RAM
- **Dataset Repetition**: Configurable dataset repeat factor for extended training epochs

### Evaluation and Metrics

- **Comprehensive Metrics**: Implements both PSNR and SSIM evaluation with proper Y-channel conversion
- **Boundary Handling**: Standard evaluation cropping (shave=scale) for fair comparison with other methods
- **Inference Timing**: Detailed timing measurements for performance analysis
- **Batch Processing**: Efficient batch evaluation for faster testing

### Architecture Flexibility

- **Configurable Architecture**: All model parameters (d, s, m) are configurable at training time
- **Multi-scale Support**: Single codebase supports 2x, 3x, and 4x upscaling factors
- **Transfer Learning**: Support for loading pretrained weights and fine-tuning

These enhancements make the implementation more suitable for modern research and practical applications while maintaining the core FSRCNN architecture and performance characteristics.

## Datasets

- **T91**: Training dataset with 91 images
- **Set5**: 5 standard test images for super-resolution
- **Set14**: 14 standard test images for super-resolution
- **DIV2K**: Large-scale dataset with 2K resolution images (for advanced training)

## Infrastructure

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5080 16GB
- **CPU**: AMD Ryzen 7 9800X3D
- **Framework**: PyTorch with CUDA support

## Project Report

For detailed methodology, experiments, and analysis, see: [Project Report](docs/project-report.pdf)

## Results

### Set14 Dataset (Upscaling Factor = 4x)

| Image | FSRCNN PSNR (dB) | FSRCNN SSIM | Bicubic PSNR (dB) | Bicubic SSIM | Inference Time (s) |
|-------|-------------------|--------------|--------------------|--------------|--------------------|
| baboon.png | 21.42 | 0.5201 | 21.15 | 0.4668 | 0.2658 |
| barbara.png | 24.45 | 0.7302 | 23.88 | 0.6858 | 0.0036 |
| bridge.png | 23.78 | 0.6375 | 23.23 | 0.5822 | 0.0040 |
| coastguard.png | 24.76 | 0.5415 | 24.19 | 0.4997 | 0.0433 |
| comic.png | 21.29 | 0.6884 | 20.46 | 0.6125 | 0.0037 |
| face.png | 31.03 | 0.7705 | 30.35 | 0.7445 | 0.0034 |
| flowers.png | 25.74 | 0.7830 | 24.33 | 0.7279 | 0.0032 |
| foreman.png | 30.49 | 0.8970 | 28.17 | 0.8566 | 0.0010 |
| lenna.png | 30.12 | 0.8371 | 28.67 | 0.8054 | 0.0008 |
| man.png | 25.46 | 0.7272 | 24.48 | 0.6754 | 0.0012 |
| monarch.png | 28.55 | 0.9154 | 26.34 | 0.8810 | 0.0034 |
| pepper.png | 31.54 | 0.8527 | 29.44 | 0.8224 | 0.0010 |
| ppt3.png | 22.74 | 0.8744 | 20.75 | 0.8130 | 0.0029 |
| zebra.png | 24.80 | 0.7584 | 23.02 | 0.6927 | 0.0029 |
| **Average** | **26.15** | **0.7524** | **24.89** | **0.7047** | **0.0243** |

**Set14 FSRCNN improvements over Bicubic:**

- PSNR: +1.26 dB (5.1% improvement)
- SSIM: +0.0477 (6.8% improvement)

### Set5 Dataset (Upscaling Factor = 4x)

| Image | FSRCNN PSNR (dB) | FSRCNN SSIM | Bicubic PSNR (dB) | Bicubic SSIM | Inference Time (s) |
|-------|-------------------|--------------|--------------------|--------------|--------------------|
| baby.png | 31.99 | 0.8849 | 30.68 | 0.8571 | 0.1233 |
| bird.png | 31.28 | 0.9158 | 29.10 | 0.8737 | 0.0312 |
| butterfly.png | 23.50 | 0.8449 | 21.00 | 0.7388 | 0.0022 |
| head.png | 31.09 | 0.7724 | 30.38 | 0.7458 | 0.0022 |
| woman.png | 27.36 | 0.8835 | 25.35 | 0.8324 | 0.0020 |
| **Average** | **29.05** | **0.8603** | **27.30** | **0.8095** | **0.0322** |

**Set5 FSRCNN improvements over Bicubic:**

- PSNR: +1.78 dB (6.4% improvement)
- SSIM: +0.0507 (6.3% improvement)

## Installation

### Prerequisites

- Python 3.13. Make sure Python and pip are installed. You can download Python from [python.org](https://www.python.org/downloads/).
- PDM (Python Dependency Manager).
- NVIDIA GPU with CUDA support (recommended).

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd fsrcnn
```

2. Install dependencies using PDM:

If you don't have PDM installed, you can install it via pip:

```bash
pip install pdm
```

Then, install the project dependencies:

```bash
pdm install
```

> [!IMPORTANT]
> If you don't have a compatible GPU, you can install the CPU-only version of PyTorch. Modify the `pyproject.toml` file, replace the `url` in the `[tool.pdm.source]` section where the `name = "pytorch"` with the following URL: `https://download.pytorch.org/whl/cpu` and then run `pdm install` again.

```toml
[[tool.pdm.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
verify_ssl = true
```

## Project Structure

```text
fsrcnn/
├── src/fsrcnn/          # Main package
│   ├── models/          # FSRCNN model implementation
│   ├── data/            # Dataset classes
│   └── utils/           # Utility functions (metrics, image ops)
├── scripts/             # Training, evaluation, and inference scripts
├── data/               # Datasets (T91, Set5, Set14, DIV2K)
├── runs/               # Training checkpoints and logs
├── docs/               # Documentation and reports
├── pretrained/         # Pretrained models
└── pyproject.toml      # Project configuration
```

## Usage

### Training

The training script supports various parameters for customization:

```bash
python scripts/train.py --train_dir <path> --val_dir <path> [options]
```

**Parameters:**

- `--train_dir`: Path to HR training images (required)
- `--val_dir`: Path to HR validation images (required)
- `--test_dir`: Path to HR test images (optional)
- `--save_dir`: Path to save training checkpoints (default: "runs/fsrcnn")
- `--scale`: Upscale factor (2, 3, 4) (default: 4)
- `--d`: Number of feature maps for feature extraction/expanding layers (default: 56)
- `--s`: Number of feature maps for shrinking/mapping layers (default: 12)
- `--m`: Number of mapping layers (3x3 convs) operating on s channels (default: 4)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--patch_size`: LR patch size for training (HR is scale*patch) (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--resume`: Path to checkpoint to resume training from (default: "")
- `--val_freq`: Validation frequency (every N epochs) (default: 10)
- `--pretrained`: Path to pretrained weights for transfer learning (default: "")
- `--cache_images`: Cache images in memory for faster loading (uses more RAM)
- `--reduce_repeat`: Reduce dataset repeat factor for large datasets (default: 1)

**Example Training Commands:**

```bash
# Train on T91 dataset
pdm run train

# Train with custom parameters on T91 dataset
pdm run train_d64_s16_m6

# Train on DIV2K dataset (large dataset)
pdm run train_div2k

# Resume training from checkpoint
python scripts/train.py --train_dir data/div2k/train --val_dir data/div2k/val --resume runs/fsrcnn_div2k/best.ckpt --epochs 5000
```

### Evaluation

Evaluate a trained model on a test dataset:

```bash
python scripts/eval.py --data_dir <path> --weights <checkpoint_path> [options]
```

**Parameters:**

- `--data_dir`: Path to test images (required)
- `--weights`: Path to model checkpoint (required)
- `--scale`: Upscale factor (2, 3, 4) (default: 4)
- `--include_bicubic`: Include bicubic interpolation comparison
- `--save_images`: Save super-resolved images
- `--output_dir`: Directory to save output images (default: "output")

**Example:**

```bash
pdm run eval
```

### Inference

Run inference on images:

```bash
python scripts/infer.py --input_dir <path> --weights <checkpoint_path> [options]
```

**Parameters:**

- `--image`: Path to a single input image (optional, mutually exclusive with `--input_dir`)
- `--input_dir`: Path to input images directory (required if `--image` not provided)
- `--weights`: Path to model checkpoint (required)
- `--scale`: Upscale factor (2, 3, 4) (default: 4)
- `--output_dir`: Directory to save output images (default: "output")
- `--save_bicubic`: Also save bicubic interpolation results for comparison

**Example:**

```bash
pdm run infer
```

## Author

- **Sukhrobbek Ilyosbekov**
- CS 7180 Advanced Perception

## License

MIT License

## References

### Original Paper

```bibtex
@article{dong2016accelerating,
  title={Accelerating the super-resolution convolutional neural network},
  author={Dong, Chao and Loy, Chen Change and Tang, Xiaoou},
  journal={European conference on computer vision},
  pages={391--407},
  year={2016},
  organization={Springer}
}
```

Original Paper: [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/pdf/1608.00367)

### Implementation

```bibtex
@misc{ilyosbekov2025fsrcnn,
  author = {Ilyosbekov, Sukhrobbek},
  title = {FSRCNN PyTorch Implementation},
  year = {2025},
  note={CS 7180 Advanced Perception Course Project},
  howpublished = {\url{https://github.com/suxrobGM/fsrcnn}}
}
```
