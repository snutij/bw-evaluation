"""Tests for config.py"""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ScoringConfig


class TestScoringConfig:
    """Tests for ScoringConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have expected weights."""
        cfg = ScoringConfig()
        assert cfg.weight_contrast == 0.28
        assert cfg.weight_texture == 0.22
        assert cfg.weight_saturation == 0.28
        assert cfg.weight_composition == 0.12
        assert cfg.weight_metadata == 0.10

    def test_weights_sum_to_one(self) -> None:
        """Final score weights must sum to 1.0."""
        cfg = ScoringConfig()
        total = (
            cfg.weight_contrast + cfg.weight_texture + cfg.weight_saturation
            + cfg.weight_composition + cfg.weight_metadata
        )
        assert total == pytest.approx(1.0)

    def test_from_file_overrides(self, tmp_path: Path) -> None:
        """Loading from JSON should override specified fields only."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"weight_contrast": 0.50, "sobel_threshold": 80.0}))

        cfg = ScoringConfig.from_file(config_path)
        assert cfg.weight_contrast == 0.50
        assert cfg.sobel_threshold == 80.0
        # Other defaults unchanged
        assert cfg.weight_texture == 0.22

    def test_from_file_unknown_key_raises(self, tmp_path: Path) -> None:
        """Unknown config keys should raise ValueError."""
        config_path = tmp_path / "bad_config.json"
        config_path.write_text(json.dumps({"nonexistent_key": 42}))

        with pytest.raises(ValueError, match="Unknown config keys"):
            ScoringConfig.from_file(config_path)

    def test_from_file_empty_json(self, tmp_path: Path) -> None:
        """Empty JSON object should return default config."""
        config_path = tmp_path / "empty.json"
        config_path.write_text("{}")

        cfg = ScoringConfig.from_file(config_path)
        default = ScoringConfig()
        assert cfg == default

    def test_metadata_weights_sum_to_one(self) -> None:
        """Metadata sub-weights must sum to 1.0."""
        cfg = ScoringConfig()
        total = (
            cfg.metadata_iso_weight + cfg.metadata_focal_weight
            + cfg.metadata_light_weight + cfg.metadata_dof_weight
        )
        assert total == pytest.approx(1.0)

    def test_composition_weights_sum_to_one(self) -> None:
        """Composition sub-weights must sum to 1.0."""
        cfg = ScoringConfig()
        total = (
            cfg.composition_separation_weight + cfg.composition_gradient_weight
            + cfg.composition_center_weight + cfg.composition_thirds_weight
            + cfg.composition_highlight_weight
        )
        assert total == pytest.approx(1.0)
