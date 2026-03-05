"""Unit tests for oww_trainer.models — model discovery and resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from oww_trainer.models import get_available_models, get_custom_names, resolve_wakeword_path

##### GET CUSTOM NAMES #####


def test_get_custom_names_finds_onnx(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "base").mkdir(parents=True)

    eager_dir = models_dir / "eager"
    eager_dir.mkdir()
    (eager_dir / "eager.onnx").write_bytes(b"fake")

    robot_dir = models_dir / "robot"
    robot_dir.mkdir()
    (robot_dir / "robot.onnx").write_bytes(b"fake")

    with patch("oww_trainer.models.MODELS_DIR", models_dir):
        names = get_custom_names()
    assert names == ["eager", "robot"]


def test_get_custom_names_ignores_base(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "base").mkdir(parents=True)

    with patch("oww_trainer.models.MODELS_DIR", models_dir):
        names = get_custom_names()
    assert names == []


def test_get_custom_names_ignores_missing_onnx(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "base").mkdir(parents=True)
    (models_dir / "empty_model").mkdir()

    with patch("oww_trainer.models.MODELS_DIR", models_dir):
        names = get_custom_names()
    assert names == []


##### RESOLVE WAKEWORD PATH #####


def test_resolve_wakeword_path_custom(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    eager_dir = models_dir / "eager"
    eager_dir.mkdir(parents=True)
    onnx_path = eager_dir / "eager.onnx"
    onnx_path.write_bytes(b"fake")

    with patch("oww_trainer.models.MODELS_DIR", models_dir):
        result = resolve_wakeword_path("eager")
    assert result == str(onnx_path)


def test_resolve_wakeword_path_not_found(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "base").mkdir(parents=True)

    with (
        patch("oww_trainer.models.MODELS_DIR", models_dir),
        patch("openwakeword.get_pretrained_model_paths", return_value=[]),
        pytest.raises(ValueError, match="not found"),
    ):
        resolve_wakeword_path("nonexistent")


##### GET AVAILABLE MODELS #####


def test_get_available_models_structure(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "base").mkdir(parents=True)
    eager_dir = models_dir / "eager"
    eager_dir.mkdir()
    (eager_dir / "eager.onnx").write_bytes(b"fake")

    with (
        patch("oww_trainer.models.MODELS_DIR", models_dir),
        patch(
            "openwakeword.get_pretrained_model_paths",
            return_value=["/fake/alexa_v0.1.onnx", "/fake/hey_mycroft_v0.1.onnx"],
        ),
    ):
        result = get_available_models()

    assert "pretrained" in result
    assert "custom" in result
    assert "alexa" in result["pretrained"]
    assert "hey_mycroft" in result["pretrained"]
    assert "eager" in result["custom"]
