# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Evaluate FSRCNN on a folder of HR images (PSNR/SSIM on Y channel)
#
# Parameters for the evaluation script:
# --data_dir: Path to HR images for evaluation.
# --weights: Path to model weights (.pth file).
# --scale: Upscale factor (2, 3, 4). Default is 4.
# --include_bicubic: Include bicubic interpolation metrics for comparison.

from __future__ import annotations
import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsrcnn.models import FSRCNN
from fsrcnn.data import SRFolderDataset
from fsrcnn.utils import psnr as psnr_np, ssim_y as ssim_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to HR images for evaluation"
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument(
        "--include_bicubic",
        action="store_true",
        help="Include bicubic interpolation metrics for comparison",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location=device)

    # Create model and load weights from checkpoint file
    # The checkpoint contains model args (d, s, m) and model state_dict
    model = FSRCNN(
        scale=args.scale, **{k: checkpoint["args"][k] for k in ["d", "s", "m"]}
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = SRFolderDataset(args.data_dir, scale=args.scale, patch_size=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnrs, ssims, inference_times = [], [], []
    bicubic_psnrs, bicubic_ssims = [], []
    shave = args.scale  # shave border for evaluation

    print(f"[bold blue]Evaluating {len(dataset)} images...[/bold blue]\n")

    if args.include_bicubic:
        print(f"{'Image':<20} {'FSRCNN PSNR':<12} {'FSRCNN SSIM':<12} {'Bicubic PSNR':<12} {'Bicubic SSIM':<12} {'Time (s)':<10}")
        print("-" * 85)
    else:
        print(f"{'Image':<20} {'PSNR (dB)':<12} {'SSIM':<8} {'Time (s)':<10}")
        print("-" * 55)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval")):
            lr: torch.Tensor = batch["lr"].to(device)
            hr: torch.Tensor = batch["hr"].to(device)

            # Measure inference time
            start_time = time.time()
            sr = torch.clamp(model(lr), 0.0, 1.0)
            inference_time = time.time() - start_time

            sr_np = (sr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")
            hr_np = (hr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")
            lr_np = (lr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")

            for j, (s, h, l) in enumerate(zip(sr_np, hr_np, lr_np)):
                # FSRCNN metrics
                psnr_val = psnr_np(h, s, shave=shave)
                ssim_val = ssim_np(h, s, shave=shave)

                psnrs.append(psnr_val)
                ssims.append(ssim_val)
                inference_times.append(inference_time)

                # Get image filename from batch path
                img_name = os.path.basename(batch["path"][j])

                if args.include_bicubic:
                    # Generate bicubic upscaled version
                    H_lr, W_lr = l.shape
                    H_hr, W_hr = H_lr * args.scale, W_lr * args.scale
                    bicubic_y = cv2.resize(l, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)

                    # Bicubic metrics
                    bicubic_psnr = psnr_np(h, bicubic_y, shave=shave)
                    bicubic_ssim = ssim_np(h, bicubic_y, shave=shave)

                    bicubic_psnrs.append(bicubic_psnr)
                    bicubic_ssims.append(bicubic_ssim)

                    print(
                        f"{img_name:<20} {psnr_val:<12.2f} {ssim_val:<12.4f} {bicubic_psnr:<12.2f} {bicubic_ssim:<12.4f} {inference_time:<10.4f}"
                    )
                else:
                    print(
                        f"{img_name:<20} {psnr_val:<12.2f} {ssim_val:<8.4f} {inference_time:<10.4f}"
                    )

    if args.include_bicubic:
        print("\n" + "=" * 85)
    else:
        print("\n" + "=" * 55)

    # FSRCNN Results
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    avg_time = sum(inference_times) / len(inference_times)

    print(f"[bold green]FSRCNN Results:[/bold green]")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Average Inference Time: {avg_time:.4f} seconds")

    if args.include_bicubic:
        # Bicubic Results
        avg_bicubic_psnr = sum(bicubic_psnrs) / len(bicubic_psnrs)
        avg_bicubic_ssim = sum(bicubic_ssims) / len(bicubic_ssims)

        print(f"\n[bold yellow]Bicubic Results:[/bold yellow]")
        print(f"  Average PSNR: {avg_bicubic_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_bicubic_ssim:.4f}")

        # Improvements
        psnr_improvement = avg_psnr - avg_bicubic_psnr
        ssim_improvement = avg_ssim - avg_bicubic_ssim

        print(f"\n[bold cyan]FSRCNN Improvements over Bicubic:[/bold cyan]")
        print(f"  PSNR Improvement: +{psnr_improvement:.2f} dB ({psnr_improvement/avg_bicubic_psnr*100:.1f}%)")
        print(f"  SSIM Improvement: +{ssim_improvement:.4f} ({ssim_improvement/avg_bicubic_ssim*100:.1f}%)")

    print(f"\n[bold green]Total Images:[/bold green] {len(psnrs)}")


if __name__ == "__main__":
    main()
