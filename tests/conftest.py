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


@pytest.fixture
def landscape_gray() -> np.ndarray:
    """Create a landscape-like grayscale image: sky/ground split + texture."""
    img = np.zeros((200, 300), dtype=np.uint8)
    # Sky: bright gradient
    for row in range(100):
        img[row, :] = 180 + int(row * 0.5)
    # Ground: textured dark
    for row in range(100, 200):
        base = 60 + (row - 100)
        img[row, :] = base
    # Add some texture in the ground
    for i in range(100, 200, 8):
        img[i, :] = img[i, :] // 2
    return img


@pytest.fixture
def architecture_gray() -> np.ndarray:
    """Create an architecture-like grayscale image: strong geometric edges."""
    img = np.full((200, 200), 120, dtype=np.uint8)
    # Strong vertical and horizontal lines (building-like)
    for x in range(0, 200, 20):
        img[:, x : x + 3] = 30
    for y in range(0, 200, 30):
        img[y : y + 3, :] = 30
    # A bright rectangle "window"
    img[40:80, 60:140] = 220
    return img


@pytest.fixture
def thirds_gray() -> np.ndarray:
    """Create an image with strong edges along rule-of-thirds lines."""
    img = np.full((300, 300), 128, dtype=np.uint8)
    # Horizontal thirds: sharp transitions
    img[95:105, :] = 30
    img[195:205, :] = 30
    # Vertical thirds
    img[:, 95:105] = 30
    img[:, 195:205] = 30
    return img
