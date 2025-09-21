# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Run FSRCNN inference on a single RGB image (processes Y channel)
#
# Parameters for the inference script:
# --image: Path to input RGB image.
# --weights: Path to model weights (.pth file).
# --scale: Upscale factor (2, 3, 4). Default is 4.
# --out: Path to save the output SR image. Default is "sr.png"

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsrcnn.models import FSRCNN
from fsrcnn.utils import (
    read_image,
    save_image,
    ycbcr_to_rgb,
    rgb_to_ycbcr,
    mod_crop,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--out", type=str, default="sr.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.weights, map_location=device)
    model = FSRCNN(scale=args.scale, **{k: ckpt["args"][k] for k in ["d", "s", "m"]})
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Read and prepare image
    rgb = read_image(args.image)
    rgb = mod_crop(rgb, args.scale)
    ycbcr = rgb_to_ycbcr(rgb)
    y = ycbcr[..., 0]
    y_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) / 255.0
    y_t = y_t.to(device)

    # Inference on Y channel
    with torch.no_grad():
        sr_y = model(y_t)
        sr_y = torch.clamp(sr_y, 0.0, 1.0)

    sr_y = (sr_y.squeeze(0).squeeze(0).cpu().numpy() * 255.0).round().astype("uint8")

    # Upscale CbCr with bicubic to match size
    H, W = sr_y.shape
    cb = cv2.resize(ycbcr[..., 1], (W, H), interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(ycbcr[..., 2], (W, H), interpolation=cv2.INTER_CUBIC)
    sr_ycbcr = np.stack([sr_y, cb, cr], axis=-1)
    sr_rgb = ycbcr_to_rgb(sr_ycbcr)

    save_image(sr_rgb, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
