"""Tests for config.py."""

import json
from pathlib import Path

import pytest

from bw_evaluation.config import ScoringConfig


class TestScoringConfig:
    """Tests for ScoringConfig dataclass."""

    def test_default_weights(self) -> None:
        """Default config should have new expected weights."""
        cfg = ScoringConfig()
        assert cfg.weight_contrast == 0.35
        assert cfg.weight_texture == 0.25
        assert cfg.weight_saturation == 0.10
        assert cfg.weight_composition == 0.05
        assert cfg.weight_channel_separation == 0.25

    def test_weights_sum_to_one(self) -> None:
        """Final score weights must sum to 1.0."""
        cfg = ScoringConfig()
        total = (
            cfg.weight_contrast
            + cfg.weight_texture
            + cfg.weight_saturation
            + cfg.weight_composition
            + cfg.weight_channel_separation
        )
        assert total == pytest.approx(1.0)

    def test_no_metadata_parameters(self) -> None:
        """Config should not have metadata weight parameters."""
        cfg = ScoringConfig()
        assert not hasattr(cfg, "weight_metadata")
        assert not hasattr(cfg, "metadata_iso_weight")

    def test_no_scene_bonus_parameters(self) -> None:
        """Config should not have scene bonus parameters."""
        cfg = ScoringConfig()
        assert not hasattr(cfg, "scene_portrait_bonus")
        assert not hasattr(cfg, "scene_landscape_bonus")

    def test_no_saturation_penalty_parameters(self) -> None:
        """Config should not have saturation sub-penalty parameters."""
        cfg = ScoringConfig()
        assert not hasattr(cfg, "high_sat_threshold")
        assert not hasattr(cfg, "gamut_penalty_max")

    def test_channel_separation_parameters(self) -> None:
        """Config should have channel separation parameters."""
        cfg = ScoringConfig()
        assert cfg.channel_sep_ceiling == 50.0

    def test_composition_weights_sum_to_one(self) -> None:
        """Composition sub-weights must sum to 1.0."""
        cfg = ScoringConfig()
        total = cfg.composition_separation_weight + cfg.composition_highlight_weight
        assert total == pytest.approx(1.0)

    def test_from_file_overrides(self, tmp_path: Path) -> None:
        """Loading from JSON should override specified fields only."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"weight_contrast": 0.50, "sobel_threshold": 80.0})
        )

        cfg = ScoringConfig.from_file(config_path)
        assert cfg.weight_contrast == 0.50
        assert cfg.sobel_threshold == 80.0
        # Other defaults unchanged
        assert cfg.weight_texture == 0.25

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

    def test_channel_sep_ceiling_configurable(self, tmp_path: Path) -> None:
        """Channel separation ceiling should be overridable via JSON."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"channel_sep_ceiling": 30.0}))

        cfg = ScoringConfig.from_file(config_path)
        assert cfg.channel_sep_ceiling == 30.0
