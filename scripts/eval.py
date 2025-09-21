# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Evaluate FSRCNN on a folder of HR images (PSNR/SSIM on Y channel)
#
# Parameters for the evaluation script:
# --data_dir: Path to HR images for evaluation.
# --weights: Path to model weights (.pth file).
# --scale: Upscale factor (2, 3, 4). Default is 4.

from __future__ import annotations
import os
import sys
import argparse
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

    loader = DataLoader(
        SRFolderDataset(args.data_dir, scale=args.scale, patch_size=None),
        batch_size=1,
        shuffle=False,
    )

    psnrs, ssims = [], []
    shave = args.scale  # shave border for evaluation

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = torch.clamp(model(lr), 0.0, 1.0)
            sr_np = (sr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")
            hr_np = (hr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")

            for s, h in zip(sr_np, hr_np):
                psnrs.append(psnr_np(h, s, shave=shave))
                ssims.append(ssim_np(h, s, shave=shave))

    print(f"[bold green]PSNR:[/bold green] {sum(psnrs)/len(psnrs):.2f} dB")
    print(f"[bold green]SSIM:[/bold green] {sum(ssims)/len(ssims):.4f}")


if __name__ == "__main__":
    main()
