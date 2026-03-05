"""Unit tests for oww_trainer.download — mocks all network/subprocess calls."""

from pathlib import Path

from oww_trainer.download import _dir_ready, _file_ready

##### FILE READY #####


def test_file_ready_nonexistent(tmp_path: Path) -> None:
    assert _file_ready(tmp_path / "missing.bin") is False


def test_file_ready_too_small(tmp_path: Path) -> None:
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"x" * 10)
    assert _file_ready(p, min_size=100) is False


def test_file_ready_large_enough(tmp_path: Path) -> None:
    p = tmp_path / "good.bin"
    p.write_bytes(b"x" * 200)
    assert _file_ready(p, min_size=100) is True


def test_file_ready_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 200)
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    assert _file_ready(link, min_size=100) is False


##### DIR READY #####


def test_dir_ready_nonexistent(tmp_path: Path) -> None:
    assert _dir_ready(tmp_path / "nope") is False


def test_dir_ready_empty(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    assert _dir_ready(d, min_files=1) is False


def test_dir_ready_enough_files(tmp_path: Path) -> None:
    d = tmp_path / "full"
    d.mkdir()
    for i in range(5):
        (d / f"f{i}.wav").write_bytes(b"x")
    assert _dir_ready(d, min_files=3) is True


def test_dir_ready_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real_dir"
    real.mkdir()
    (real / "file.txt").write_bytes(b"x")
    link = tmp_path / "link_dir"
    link.symlink_to(real)
    assert _dir_ready(link) is False
