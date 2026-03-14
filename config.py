"""Scoring configuration with sensible defaults and optional JSON overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScoringConfig:
    """All tunable scoring parameters in one place."""

    # --- Final score weights (must sum to 1.0) ---
    weight_contrast: float = 0.35
    weight_texture: float = 0.25
    weight_saturation: float = 0.10
    weight_composition: float = 0.05
    weight_channel_separation: float = 0.25

    # --- Tonal contrast ---
    contrast_std_weight: float = 0.35
    contrast_extremes_weight: float = 0.25
    contrast_range_weight: float = 0.25
    contrast_balance_weight: float = 0.15

    # --- Texture ---
    sobel_threshold: float = 50.0
    canny_low: int = 100
    canny_high: int = 200
    texture_edge_weight: float = 0.25
    texture_canny_weight: float = 0.20
    texture_variance_weight: float = 0.30
    texture_sharpness_weight: float = 0.25

    # --- Composition ---
    composition_separation_weight: float = 0.50
    composition_highlight_weight: float = 0.50

    # --- Channel separation ---
    channel_sep_ceiling: float = 50.0

    @classmethod
    def from_file(cls, path: Path) -> ScoringConfig:
        """Load config from a JSON file, overriding only specified fields."""
        with open(path) as f:
            overrides: dict[str, Any] = json.load(f)

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        return cls(**{k: v for k, v in overrides.items() if k in valid_fields})
