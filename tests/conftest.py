"""Pytest fixtures for B&W scorer tests."""

import numpy as np
import cv2
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture
def high_contrast_gray() -> np.ndarray:
    """Create a high contrast grayscale image (black and white stripes)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, ::2] = 255  # Alternating white columns
    return img


@pytest.fixture
def low_contrast_gray() -> np.ndarray:
    """Create a low contrast grayscale image (uniform mid-gray)."""
    return np.full((100, 100), 128, dtype=np.uint8)


@pytest.fixture
def gradient_gray() -> np.ndarray:
    """Create a smooth gradient grayscale image."""
    return np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))


@pytest.fixture
def textured_gray() -> np.ndarray:
    """Create a textured grayscale image with edges."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Add checkerboard pattern
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i : i + 10, j : j + 10] = 200
            else:
                img[i : i + 10, j : j + 10] = 50
    return img


@pytest.fixture
def smooth_gray() -> np.ndarray:
    """Create a smooth grayscale image with no texture."""
    return np.full((100, 100), 100, dtype=np.uint8)


@pytest.fixture
def saturated_bgr() -> np.ndarray:
    """Create a highly saturated color image (pure red)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # Red channel
    return img


@pytest.fixture
def desaturated_bgr() -> np.ndarray:
    """Create a desaturated/grayscale-like BGR image."""
    gray_val = 128
    img = np.full((100, 100, 3), gray_val, dtype=np.uint8)
    return img


@pytest.fixture
def center_bright_gray() -> np.ndarray:
    """Create an image with bright center (portrait-like composition)."""
    img = np.full((100, 100), 50, dtype=np.uint8)
    img[25:75, 25:75] = 180  # Bright center
    return img


