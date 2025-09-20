# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: PSNR / SSIM metrics (Y channel)

from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity as ssim


def psnr(
    y_true: np.ndarray, y_pred: np.ndarray, shave: int = 0, data_range: float = 255.0
) -> float:
    """
    Compute PSNR on uint8 Y images with optional border shave.
    Args:
        y_true (np.ndarray): Ground truth Y channel image (H, W) in uint8.
        y_pred (np.ndarray): Predicted Y channel image (H, W) in uint8.
        shave (int): Border pixels to shave off before computing PSNR. Default is 0.
        data_range (float): The data range of the input images. Default is 255.0 for uint8 images.
    Returns:
        float: PSNR value in dB.
    """
    if shave > 0:
        y_true = y_true[shave:-shave, shave:-shave]

    y_pred = y_pred[shave:-shave, shave:-shave]
    diff = y_true.astype(np.float32) - y_pred.astype(np.float32)
    mse = np.mean(diff**2)

    if mse <= 1e-10:
        return 99.0
    return 10.0 * np.log10((data_range**2) / mse)


def ssim_y(
    y_true: np.ndarray, y_pred: np.ndarray, shave: int = 0, data_range: float = 255.0
) -> float:
    """
    Compute SSIM on uint8 Y images with optional border shave.
    Args:
        y_true (np.ndarray): Ground truth Y channel image (H, W) in uint8.
        y_pred (np.ndarray): Predicted Y channel image (H, W) in uint8.
        shave (int): Border pixels to shave off before computing SSIM. Default is 0.
        data_range (float): The data range of the input images. Default is 255.0 for uint8 images.
    Returns:
        float: SSIM value in [0, 1].
    """
    if shave > 0:
        y_true = y_true[shave:-shave, shave:-shave]
    y_pred = y_pred[shave:-shave, shave:-shave]
    return float(ssim(y_true, y_pred, data_range=data_range))  # type: ignore
