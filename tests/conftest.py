"""Shared fixtures for oww_trainer tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal project directory structure."""
    for d in ("configs", "datasets/base", "models/base", "models/base/piper-sample-generator"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def mock_subprocess(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock subprocess.run to avoid executing real commands."""
    mock = MagicMock()
    mock.return_value.returncode = 0
    monkeypatch.setattr("subprocess.run", mock)
    return mock
