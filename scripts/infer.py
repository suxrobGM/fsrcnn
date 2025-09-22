# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Run FSRCNN inference on RGB images (processes Y channel)
#
# Parameters for the inference script:
# --image: Path to input RGB image.
# --input_dir: Path to input directory containing images (alternative to --image).
# --weights: Path to model weights (.pth file).
# --scale: Upscale factor (2, 3, 4). Default is 4.
# --output_dir: Path to save the output SR image. Default is "sr.png"
# --save_bicubic: Whether to save bicubic upscaled image for comparison.

from __future__ import annotations
import os
import sys
import argparse
import time
import glob
import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsrcnn.models import FSRCNN
from fsrcnn.utils import (
    read_image,
    save_image,
    ycbcr_to_rgb,
    rgb_to_ycbcr,
    mod_crop,
)


def process_single_image(
    model: FSRCNN,
    device: torch.device,
    image_path: str,
    output_path: str,
    scale: int,
    save_bicubic: bool = False,
) -> float:
    """Process a single image through FSRCNN.
    Args:
        model: FSRCNN model.
        device: Device to run the model on.
        image_path: Path to the input image.
        output_path: Path to save the output image.
        scale: Upscaling factor.
        save_bicubic: Whether to save bicubic comparison image.
    Returns:
        Inference time in seconds.
    """
    # Read and prepare image
    rgb = read_image(image_path)
    rgb = mod_crop(rgb, scale)
    ycbcr = rgb_to_ycbcr(rgb)
    y = ycbcr[..., 0]
    y_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) / 255.0
    y_t = y_t.to(device)

    # Inference on Y channel
    start_time = time.time()
    with torch.no_grad():
        sr_y = model(y_t)
        sr_y = torch.clamp(sr_y, 0.0, 1.0)
    inference_time = time.time() - start_time

    sr_y = (sr_y.squeeze(0).squeeze(0).cpu().numpy() * 255.0).round().astype("uint8")

    # Upscale CbCr with bicubic to match size
    H, W = sr_y.shape
    cb = cv2.resize(ycbcr[..., 1], (W, H), interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(ycbcr[..., 2], (W, H), interpolation=cv2.INTER_CUBIC)
    sr_ycbcr = np.stack([sr_y, cb, cr], axis=-1)
    sr_rgb = ycbcr_to_rgb(sr_ycbcr)

    save_image(sr_rgb, output_path)

    # Create and save bicubic upscaled version
    if save_bicubic:
        H_lr, W_lr = y.shape
        H_hr, W_hr = H_lr * scale, W_lr * scale

        # Upscale the entire RGB image using bicubic interpolation
        bicubic_rgb = cv2.resize(rgb, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)

        # Generate bicubic output filename
        name, ext = os.path.splitext(output_path)
        bicubic_path = f"{name}_bicubic{ext}"
        save_image(bicubic_rgb, bicubic_path)

    return inference_time


def get_image_files(directory: str) -> list[str]:
    """Get list of image files from directory.
    Args:
        directory: Directory path.
    Returns:
        List of image file paths.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    return sorted(image_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to input RGB image")
    parser.add_argument(
        "--input_dir", type=str, help="Path to input directory containing images"
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save images",
    )
    parser.add_argument(
        "--save_bicubic",
        action="store_true",
        help="Save bicubic upscaled image for comparison",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.input_dir:
        parser.error("Either --image or --input_dir must be specified")
    if args.image and args.input_dir:
        parser.error("Cannot specify both --image and --input_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.weights, map_location=device)
    model = FSRCNN(scale=args.scale, **{k: ckpt["args"][k] for k in ["d", "s", "m"]})
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Model loaded from {args.weights}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # Single image processing
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} does not exist")
            return

        filename = os.path.basename(args.image)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(args.output_dir, f"{name}_sr_x{args.scale}{ext}")

        inference_time = process_single_image(
            model, device, args.image, output_path, args.scale, args.save_bicubic
        )
        print(f"Saved to {output_path}")

        if args.save_bicubic:
            name, ext = os.path.splitext(output_path)
            bicubic_path = f"{name}_bicubic{ext}"
            print(f"Bicubic comparison saved to {bicubic_path}")

        print(f"Inference time: {inference_time:.4f} seconds")
    else:
        # Directory processing
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            return

        # Get all image files
        image_files = get_image_files(args.input_dir)
        if not image_files:
            print(f"No image files found in {args.input_dir}")
            return

        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {args.output_dir}")

        if args.save_bicubic:
            print("Bicubic comparison images will also be generated")

        total_time = 0
        processed = 0

        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_sr_x{args.scale}{ext}"
                output_path = os.path.join(args.output_dir, output_filename)

                inference_time = process_single_image(
                    model,
                    device,
                    image_path,
                    output_path,
                    args.scale,
                    args.save_bicubic,
                )
                total_time += inference_time
                processed += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        print(f"Processed: {processed}/{len(image_files)} images")
        print(f"Average inference time: {total_time/processed:.4f} seconds per image")
        print(f"Total processing time: {total_time:.4f} seconds")
        print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
