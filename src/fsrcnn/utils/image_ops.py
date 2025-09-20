# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: Utilities for image I/O, color space, and resizing

from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0,255] to YCbCr (uint8).
    Args:
        img (np.ndarray): Input RGB image (H, W, 3) in uint8.
    Returns:
        np.ndarray: YCbCr image (H, W, 3) in uint8
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr [0,255] to RGB (uint8).
    Args:
        img (np.ndarray): Input YCbCr image (H, W, 3) in uint8.
    Returns:
        np.ndarray: RGB image (H, W, 3) in uint8.
    """
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def to_y(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Extract Y channel from an RGB uint8 image.
    Args:
        img_rgb_u8 (np.ndarray): Input RGB image (H, W, 3) in uint8.
    Returns:
        np.ndarray: Y channel (H, W) in uint8.
    """
    ycbcr = rgb_to_ycbcr(img_rgb_u8)
    y = ycbcr[..., 0]
    return y


def bicubic_downsample(img: Image.Image, scale: int) -> Image.Image:
    """
    Downsample PIL image by scale using bicubic interpolation.
    Args:
        img (PIL.Image): Input image.
        scale (int): Downscale factor.
    Returns:
        PIL.Image: Downsampled image.
    """
    w, h = img.size
    lr = img.resize((w // scale, h // scale), Image.Resampling.BICUBIC)
    return lr


def mod_crop(arr: np.ndarray, scale: int) -> np.ndarray:
    """
    Crop array so that its H and W are multiples of scale.
    Args:
        arr (np.ndarray): Input image array.
        scale (int): Scale factor.
    Returns:
        np.ndarray: Cropped image array.
    """
    h, w = arr.shape[:2]
    h_r = h - (h % scale)
    w_r = w - (w % scale)
    return arr[:h_r, :w_r]


def read_image(path: str) -> np.ndarray:
    """
    Read image as RGB uint8 ndarray.
    Args:
        path (str): Path to the image file.
    Returns:
        np.ndarray: Image array in RGB format (H, W, 3) as uint
    """
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)


def save_image(arr_rgb_u8: np.ndarray, path: str) -> None:
    """
    Save RGB uint8 ndarray to disk.
    Args:
        arr_rgb_u8 (np.ndarray): Input image array.
        path (str): Path to the output image file.
    """
    Image.fromarray(arr_rgb_u8).save(path)
