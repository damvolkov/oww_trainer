"""Unit tests for oww_trainer.trainer — mocks all subprocess/IO calls."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from oww_trainer.trainer import (
    StepTimer,
    _compute_samples_per_lang,
    _format_duration,
    build_config,
    run_pipeline,
)

##### FORMAT DURATION #####


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.5, "0.5s"),
        (30.0, "30.0s"),
        (59.9, "59.9s"),
        (60.0, "1m 0.0s"),
        (90.5, "1m 30.5s"),
        (125.3, "2m 5.3s"),
    ],
    ids=["half-sec", "30s", "just-under-min", "exact-min", "90s", "2min"],
)
def test_format_duration_outputs_correctly(seconds: float, expected: str) -> None:
    assert _format_duration(seconds) == expected


##### BUILD CONFIG #####


def test_build_config_produces_valid_dict(tmp_path: Path) -> None:
    config = build_config("hey robot", output_dir=tmp_path, n_samples=100, n_samples_val=10, steps=500)

    assert config["target_phrase"] == ["hey robot"]
    assert config["model_name"] == "hey_robot"
    assert config["n_samples"] == 100
    assert config["n_samples_val"] == 10
    assert config["steps"] == 500
    assert config["output_dir"] == str(tmp_path)
    assert config["model_type"] == "dnn"


def test_build_config_strips_whitespace(tmp_path: Path) -> None:
    config = build_config("  eager  ", output_dir=tmp_path)
    assert config["target_phrase"] == ["eager"]
    assert config["model_name"] == "eager"


def test_build_config_normalizes_spaces_to_underscores(tmp_path: Path) -> None:
    config = build_config("ok google", output_dir=tmp_path)
    assert config["model_name"] == "ok_google"


def test_build_config_defaults(tmp_path: Path) -> None:
    config = build_config("test", output_dir=tmp_path)
    assert config["n_samples"] == 5000
    assert config["n_samples_val"] == 1000
    assert config["steps"] == 10000
    assert config["layer_size"] == 32


##### STEP TIMER #####


def test_step_timer_no_exception() -> None:
    with StepTimer("test step", 1, 3) as timer:
        assert timer.step_name == "test step"
    # No exception — should not raise


def test_step_timer_with_exception() -> None:
    with pytest.raises(ValueError, match="boom"), StepTimer("failing step", 1, 1):
        raise ValueError("boom")


##### COMPUTE SAMPLES PER LANG #####


@pytest.mark.parametrize(
    ("total", "langs", "expected"),
    [
        (100, ["en"], {"en": 100}),
        (100, ["en", "de"], {"en": 50, "de": 50}),
        (100, ["en", "de", "fr"], {"en": 34, "de": 33, "fr": 33}),
        (10, ["en", "de", "fr", "nl"], {"en": 3, "de": 3, "fr": 2, "nl": 2}),
        (1, ["en", "de"], {"en": 1, "de": 0}),
    ],
    ids=["single", "two-even", "three-remainder", "four-langs", "minimum"],
)
def test_compute_samples_per_lang(total: int, langs: list[str], expected: dict[str, int]) -> None:
    result = _compute_samples_per_lang(total, langs)
    assert result == expected
    assert sum(result.values()) == total


##### RUN PIPELINE (fully mocked) #####


@patch("oww_trainer.trainer._finalize_model")
@patch("oww_trainer.trainer._run_train_model")
@patch("oww_trainer.trainer._run_augment_clips")
@patch("oww_trainer.trainer._run_generate_clips")
@patch("oww_trainer.trainer.download_all")
def test_run_pipeline_creates_dirs_and_config(
    mock_download: MagicMock,
    mock_generate: MagicMock,
    mock_augment: MagicMock,
    mock_train: MagicMock,
    mock_finalize: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oww_trainer.trainer.PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()

    result = run_pipeline("test word", n_samples=50, n_samples_val=10, steps=100)

    # Directories created
    assert (tmp_path / "models" / "test_word").is_dir()
    assert (tmp_path / "datasets" / "test_word").is_dir()

    # Config written
    config_path = tmp_path / "configs" / "test_word.yaml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text())
    assert config["target_phrase"] == ["test word"]
    assert config["n_samples"] == 50
    assert config["langs"] == ["en"]

    # All pipeline steps called
    mock_download.assert_called_once()
    mock_generate.assert_called_once_with(config_path)
    mock_augment.assert_called_once_with(config_path)
    mock_train.assert_called_once_with(config_path)
    mock_finalize.assert_called_once()

    assert result == tmp_path / "models" / "test_word"


@patch("oww_trainer.trainer._finalize_model")
@patch("oww_trainer.trainer._run_train_model")
@patch("oww_trainer.trainer._run_augment_clips")
@patch("oww_trainer.trainer._generate_multilang_clips")
@patch("oww_trainer.trainer.download_piper_lang_models")
@patch("oww_trainer.trainer.download_all")
def test_run_pipeline_multilang_uses_multilang_generator(
    mock_download: MagicMock,
    mock_dl_langs: MagicMock,
    mock_multilang: MagicMock,
    mock_augment: MagicMock,
    mock_train: MagicMock,
    mock_finalize: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oww_trainer.trainer.PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()

    result = run_pipeline("hola", n_samples=50, n_samples_val=10, steps=100, langs=["en", "de"])

    # Config should have langs
    config_path = tmp_path / "configs" / "hola.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["langs"] == ["en", "de"]

    # Multi-lang path: download_piper_lang_models called for extra langs
    mock_dl_langs.assert_called_once_with(["de"])

    # _generate_multilang_clips used instead of _run_generate_clips
    mock_multilang.assert_called_once()
    call_args = mock_multilang.call_args
    assert call_args[0][1] == ["en", "de"]

    # Rest of pipeline still runs
    mock_augment.assert_called_once_with(config_path)
    mock_train.assert_called_once_with(config_path)
    mock_finalize.assert_called_once()

    assert result == tmp_path / "models" / "hola"


@patch("oww_trainer.trainer._finalize_model")
@patch("oww_trainer.trainer._run_train_model")
@patch("oww_trainer.trainer._run_augment_clips")
@patch("oww_trainer.trainer._run_generate_clips")
@patch("oww_trainer.trainer.download_all")
def test_run_pipeline_propagates_step_failure(
    mock_download: MagicMock,
    mock_generate: MagicMock,
    mock_augment: MagicMock,
    mock_train: MagicMock,
    mock_finalize: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oww_trainer.trainer.PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()

    mock_generate.side_effect = RuntimeError("TTS failed")

    with pytest.raises(RuntimeError, match="TTS failed"):
        run_pipeline("fail_word", n_samples=10, n_samples_val=5, steps=50)

    # Steps after failure should NOT be called
    mock_augment.assert_not_called()
    mock_train.assert_not_called()
    mock_finalize.assert_not_called()


def test_run_pipeline_rejects_unsupported_lang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oww_trainer.trainer.PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()

    with pytest.raises(ValueError, match="Unsupported language 'xx'"):
        run_pipeline("test", langs=["en", "xx"])
