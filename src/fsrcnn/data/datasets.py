# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: PyTorch datasets for HR->LR pair generation and random patching

from __future__ import annotations
import os
import random
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from fsrcnn.utils.image_ops import mod_crop, bicubic_downsample

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SRFolderDataset(Dataset):
    """
    (HR -> LR) dataset that generates LR/HR pairs on the fly.
    - Applies mod-crop so HR dims are divisible by scale.
    - Training: extracts random LR patches and the aligned HR patches.
    - Validation/Test: returns full-size LR/HR arrays (no random crops) when patch_size=None.
    """

    root: str
    """Path to the root directory containing images."""

    scale: int
    """Downscale factor (2, 3, 4)."""

    patch_size: int | None
    """LR patch size for training (HR is scale*patch)."""

    repeat: int
    """Number of times to repeat the dataset (for more iterations per epoch)."""

    cache_images: bool
    """Whether to cache images in memory for faster loading."""

    paths: list[str]
    """List of image file paths."""

    _cache: dict[str, tuple[Image.Image, Image.Image]] | None
    """Cache for loaded (LR, HR) image pairs if caching is enabled."""

    def __init__(
        self,
        root: str,
        scale: int = 4,
        patch_size: int | None = 32,
        repeat: int = 1,
        cache_images: bool = False,
    ) -> None:
        """
        Initialize the dataset.
        Args:
            root (str): Root directory containing images.
            scale (int): Downscale factor (2, 3, 4).
            patch_size (int | None): LR patch size for training. If None, use full image for val/test.
            repeat (int): Number of times to repeat the dataset (for more iterations per epoch).
            cache_images (bool): Whether to cache images in memory for faster loading.
        """
        super().__init__()
        self.root = root
        self.scale = scale
        self.patch_size = patch_size
        self.repeat = repeat
        self.cache_images = cache_images
        self.paths = self._scan(root)
        self._cache = {} if cache_images else None

    def _scan(self, root: str) -> list[str]:
        """
        Scan the specified folder for image files.
        Args:
            root (str): Root directory containing images.
        """
        out = []
        for dirpath, _, fnames in os.walk(root):
            for f in fnames:
                if os.path.splitext(f.lower())[1] in IMG_EXTS:
                    out.append(os.path.join(dirpath, f))

        out.sort()
        return out

    def __len__(self) -> int:
        if self.patch_size is None:
            return len(self.paths)
        return len(self.paths) * self.repeat

    def _load_pair(self, path: str) -> tuple[Image.Image, Image.Image]:
        # Use cache if enabled
        if self._cache is not None and path in self._cache:
            return self._cache[path]

        hr = Image.open(path).convert("RGB")
        hr_np = np.array(hr)
        hr_np = mod_crop(hr_np, self.scale)
        hr = Image.fromarray(hr_np)
        lr = bicubic_downsample(hr, self.scale)

        # Cache the result if caching is enabled
        if self._cache is not None:
            self._cache[path] = (lr, hr)

        return lr, hr

    def _random_crop_pair(
        self, lr: Image.Image, hr: Image.Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract random aligned patches from LR and HR images with augmentation.
        Crops a random patch of size (patch_size x patch_size) from LR and randomly flips/rotates it.
        Args:
            lr (PIL.Image): Low-resolution image.
            hr (PIL.Image): High-resolution image.
        Returns:
            tuple[np.ndarray, np.ndarray]: LR and HR patches as numpy arrays.
        """
        ps = self.patch_size
        assert ps is not None
        w, h = lr.size

        if w < ps or h < ps:
            # If the image is too small, resize HR slightly up to meet patch size requirements
            factor = max(ps / w, ps / h)
            new_lr = lr.resize(
                (int(w * factor) + 1, int(h * factor) + 1), Image.Resampling.BICUBIC
            )

            # Recreate HR by upscaling back (for alignment) – only used in rare small-image cases
            new_hr = hr.resize(
                (new_lr.size[0] * self.scale, new_lr.size[1] * self.scale),
                Image.Resampling.BICUBIC,
            )
            lr, hr = new_lr, new_hr
            w, h = lr.size

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        lr_patch = lr.crop((x, y, x + ps, y + ps))
        hr_patch = hr.crop(
            (
                x * self.scale,
                y * self.scale,
                (x + ps) * self.scale,
                (y + ps) * self.scale,
            )
        )

        # Data augmentation: random horizontal flip and rotation
        if random.random() < 0.5:
            lr_patch = lr_patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hr_patch = hr_patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random 90-degree rotations
        rot = random.choice([0, 90, 180, 270])
        if rot != 0:
            if rot == 90:
                lr_patch = lr_patch.transpose(Image.Transpose.ROTATE_90)
                hr_patch = hr_patch.transpose(Image.Transpose.ROTATE_90)
            elif rot == 180:
                lr_patch = lr_patch.transpose(Image.Transpose.ROTATE_180)
                hr_patch = hr_patch.transpose(Image.Transpose.ROTATE_180)
            elif rot == 270:
                lr_patch = lr_patch.transpose(Image.Transpose.ROTATE_270)
                hr_patch = hr_patch.transpose(Image.Transpose.ROTATE_270)

        return np.array(lr_patch), np.array(hr_patch)

    def __getitem__(self, idx: int):
        path = self.paths[idx % len(self.paths)]
        lr, hr = self._load_pair(path)

        if self.patch_size is not None:
            lr_np, hr_np = self._random_crop_pair(lr, hr)
        else:
            lr_np, hr_np = np.array(lr), np.array(hr)

        # Convert to Y channel only
        lr_y = _rgb_to_y(lr_np)
        hr_y = _rgb_to_y(hr_np)

        # To tensor [1,H,W], normalize to [0,1]
        lr_t = torch.from_numpy(lr_y).float().unsqueeze(0) / 255.0
        hr_t = torch.from_numpy(hr_y).float().unsqueeze(0) / 255.0
        return {"lr": lr_t, "hr": hr_t, "path": path}  # [1,H,W]  # [1,sH,sW]


def _rgb_to_y(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Utility function to convert RGB uint8 image to Y channel.
    Args:
        img_rgb_u8 (np.ndarray): Input RGB image (H, W, 3) in uint8.
    Returns:
        np.ndarray: Y channel (H, W) in uint8.
    """
    ycrcb = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0]
    return y
