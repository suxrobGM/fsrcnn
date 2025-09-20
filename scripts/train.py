# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Training script for FSRCNN
#
# Parameters for the training script:
# --train_dir: Path to HR training images.
# --val_dir: Path to HR validation images.
# --save_dir: Path to save training checkpoints. Default is "runs/fsrcnn"
# --scale: Upscale factor (2, 3, 4). Default is 4
# --d: Number of feature maps for the feature extraction/expanding layers. Default is 56
# --s: Number of feature maps for shrinking/mapping layers. Default is 12
# --m: Number of mapping layers (3x3 convs) operating on s channels. Default is 3
# --epochs: Number of training epochs. Default is 100
# --batch_size: Batch size. Default is 16
# --patch_size: LR patch size for training (HR patch is scale*patch). Default is 33.
# --lr: Learning rate. Default is 1e-3
# --seed: Random seed for reproducibility. Default is 42
# --resume: Path to checkpoint to resume training from. Default is "" (no resume)


from __future__ import annotations
import os
import sys
import argparse
import torch
import torch.nn as nn
import random, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fsrcnn.models import FSRCNN
from fsrcnn.data import SRFolderDataset
from fsrcnn.utils import psnr as psnr_np, ssim_y as ssim_np


def seed_everything(seed: int = 42) -> None:
    """
    Make results more reproducible.
    Args:
        seed (int): Random seed. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    path: str, scale: int, patch: int | None, batch: int, shuffle: bool, repeat: int = 1
) -> DataLoader:
    """
    Create a PyTorch DataLoader for SRFolderDataset.
    Args:
        path (str): Path to image folder.
        scale (int): Downscale factor (2, 3, 4).
        patch (int | None): LR patch size for training. If None, use full image for val/test.
        batch (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        repeat (int): Number of times to repeat the dataset (for more iterations per epoch).
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    ds = SRFolderDataset(path, scale=scale, patch_size=patch, repeat=repeat)
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def train_one_epoch(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """
    Train for one epoch and return average loss.
    Args:
        model (nn.Module): The FSRCNN model.
        optim (torch.optim.Optimizer): The optimizer.
        loader (DataLoader): DataLoader for training data.
        device (torch.device): Device to run the training on.
        criterion: Loss function.
    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        lr = batch["lr"].to(device)  # [B,1,h,w]
        hr = batch["hr"].to(device)  # [B,1,H,W]

        sr = model(lr)
        loss = criterion(sr, hr)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running += loss.item() * lr.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)  # type: ignore


def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, scale: int
) -> tuple[float, float]:
    """
    Validate PSNR/SSIM on full images (Y channel).
    Args:
        model (nn.Module): The FSRCNN model.
        loader (DataLoader): DataLoader for validation/test data.
        device (torch.device): Device to run the validation on.
        scale (int): Upscale factor (2, 3, 4).
    Returns:
        tuple[float, float]: Average PSNR and SSIM over the dataset.
    """
    model.eval()
    psnrs, ssims = [], []
    shave = scale  # standard evaluation crop

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)

            # Clamp to [0,1]
            sr = torch.clamp(sr, 0.0, 1.0)

            # Move to CPU numpy uint8
            sr_np = (sr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")
            hr_np = (hr.squeeze(1).cpu().numpy() * 255.0).round().astype("uint8")

            for s, h in zip(sr_np, hr_np):
                psnrs.append(psnr_np(h, s, shave=shave))
                ssims.append(ssim_np(h, s, shave=shave))
    return float(sum(psnrs) / len(psnrs)), float(sum(ssims) / len(ssims))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FSRCNN")
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Path to HR training images"
    )
    parser.add_argument(
        "--val_dir", type=str, required=True, help="Path to HR validation images"
    )
    parser.add_argument("--save_dir", type=str, default="runs/fsrcnn")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--d", type=int, default=56)
    parser.add_argument("--s", type=int, default=12)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--patch_size", type=int, default=32, help="LR patch size (HR is scale*patch)"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold green]Device:[/bold green] {device}")

    model = FSRCNN(scale=args.scale, d=args.d, s=args.s, m=args.m)
    model.to(device)

    criterion = nn.MSELoss()  # MSE loss function to be used for training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam optimizer

    # Learning rate scheduler: reduce LR by half at 50% and 80% of total epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.8)],
        gamma=0.5,
    )

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        print(f"Resumed from {args.resume}")

    train_loader = make_loader(
        args.train_dir,
        args.scale,
        args.patch_size,
        args.batch_size,
        shuffle=True,
        repeat=8,
    )
    val_loader = make_loader(args.val_dir, args.scale, None, batch=1, shuffle=False)

    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, criterion)
        psnr_val, ssim_val = validate(model, val_loader, device, args.scale)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] loss={loss:.4f} val_psnr={psnr_val:.2f} val_ssim={ssim_val:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(args.save_dir, "last.ckpt"))

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save(checkpoint, os.path.join(args.save_dir, "best.ckpt"))
            print(f"[bold cyan]New best PSNR[/bold cyan]: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
