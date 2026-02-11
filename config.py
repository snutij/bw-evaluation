"""Scoring configuration with sensible defaults and optional JSON overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoringConfig:
    """All tunable scoring parameters in one place."""

    # --- Final score weights ---
    weight_contrast: float = 0.28
    weight_texture: float = 0.22
    weight_saturation: float = 0.28
    weight_composition: float = 0.12
    weight_metadata: float = 0.10

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

    # --- Colorimetry ---
    high_sat_threshold: int = 150
    high_sat_avg_trigger: float = 100.0
    high_sat_ratio_trigger: float = 0.3
    high_sat_strong_penalty_factor: float = 40.0
    high_sat_weak_penalty_factor: float = 15.0
    low_sat_threshold: int = 30
    gamut_penalty_max: float = 15.0
    gamut_spread_divisor: float = 150.0

    # --- Composition ---
    composition_separation_weight: float = 0.25
    composition_gradient_weight: float = 0.20
    composition_center_weight: float = 0.20
    composition_thirds_weight: float = 0.15
    composition_highlight_weight: float = 0.20

    # --- Metadata ---
    metadata_iso_weight: float = 0.30
    metadata_focal_weight: float = 0.25
    metadata_light_weight: float = 0.25
    metadata_dof_weight: float = 0.20

    # --- Scene bonuses ---
    scene_portrait_bonus: int = 8
    scene_landscape_bonus: int = 5
    scene_architecture_bonus: int = 6
    scene_street_bonus: int = 4

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
