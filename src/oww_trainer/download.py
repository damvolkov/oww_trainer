"""Download and cache base assets required for training."""

import logging
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from rich.console import Console
from tqdm import tqdm

##### CONSTANTS #####

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_BASE = PROJECT_ROOT / "datasets" / "base"
MODELS_BASE = PROJECT_ROOT / "models" / "base"
PIPER_REPO = "https://github.com/dscripka/piper-sample-generator"
PIPER_MODEL_URL = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"

_PIPER_RELEASE_BASE = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0"
_PIPER_RAW_BASE = "https://raw.githubusercontent.com/rhasspy/piper-sample-generator/master/models"
_PIPER_VOICES_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# lang_code -> (model_filename, espeak_voice, format)
# format: "pt" = PyTorch checkpoint (piper-sample-generator v2), "onnx" = Piper voice (piper-tts)
PIPER_LANG_MODELS: dict[str, tuple[str, str, str]] = {
    "en": ("en_US-libritts_r-medium.pt", "en-us", "pt"),
    "es": ("es/es_ES/davefx/medium/es_ES-davefx-medium.onnx", "es", "onnx"),
    "de": ("de_DE-mls-medium.pt", "de", "pt"),
    "fr": ("fr_FR-mls-medium.pt", "fr-fr", "pt"),
    "nl": ("nl_NL-mls-medium.pt", "nl", "pt"),
}

DEFAULT_LANG = "en"

OWW_MODEL_URLS = {
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    "embedding_model.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "melspectrogram.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
}

FEATURES_URLS = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation_set_features.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
}

RIR_DATASET_ID = "davidscripka/MIT_environmental_impulse_responses"

AUDIOSET_TAR_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"

console = Console()
log = logging.getLogger("oww_trainer.download")

##### HELPERS #####


def _wget(url: str, output: Path) -> None:
    """Download a file via wget with resume support."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "--show-progress", "-c", "-O", str(output), url],
        check=True,
    )


def _file_ready(path: Path, min_size: int = 100) -> bool:
    """Check if a file exists and has a reasonable size."""
    return path.exists() and not path.is_symlink() and path.stat().st_size > min_size


def _dir_ready(path: Path, min_files: int = 1) -> bool:
    """Check if a directory exists with at least min_files inside."""
    if not path.exists() or not path.is_dir():
        return False
    if path.is_symlink():
        return False
    return sum(1 for _ in path.iterdir()) >= min_files


##### DOWNLOAD STEPS #####


def download_piper(force: bool = False) -> None:
    """Clone dscripka's piper-sample-generator fork (includes TTS model)."""
    piper_dir = MODELS_BASE / "piper-sample-generator"

    # Remove stale symlink if present
    if piper_dir.is_symlink():
        piper_dir.unlink()

    if not force and piper_dir.exists() and (piper_dir / "generate_samples.py").exists():
        log.info("piper-sample-generator already present, skipping")
        return

    log.info("Cloning piper-sample-generator (dscripka fork)...")
    MODELS_BASE.mkdir(parents=True, exist_ok=True)

    # Remove dir if exists but incomplete
    if piper_dir.exists():
        import shutil
        shutil.rmtree(piper_dir)

    subprocess.run(
        ["git", "clone", "--depth=1", PIPER_REPO, str(piper_dir)],
        check=True,
    )

    # The dscripka fork bundles en-us-libritts-high.pt via git-lfs, which --depth=1 skips
    model_path = piper_dir / "models" / "en-us-libritts-high.pt"
    if not _file_ready(model_path, min_size=1_000_000):
        log.info("Downloading Piper TTS model (git-lfs not available with shallow clone)...")
        _wget(PIPER_MODEL_URL, model_path)

    log.info("piper-sample-generator ready")


def download_piper_lang_models(langs: list[str], force: bool = False) -> None:
    """Download Piper TTS models for the requested languages."""
    piper_models_dir = MODELS_BASE / "piper-sample-generator" / "models"
    piper_models_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        if lang not in PIPER_LANG_MODELS:
            available = ", ".join(sorted(PIPER_LANG_MODELS))
            raise ValueError(f"Unsupported language '{lang}'. Available: {available}")

        model_filename, _espeak_voice, fmt = PIPER_LANG_MODELS[lang]

        match fmt:
            case "pt":
                _download_piper_pt_model(lang, model_filename, piper_models_dir, force)
            case "onnx":
                _download_piper_onnx_model(lang, model_filename, piper_models_dir, force)

    log.info(f"Piper language models ready: {', '.join(langs)}")


def _download_piper_pt_model(lang: str, model_filename: str, models_dir: Path, force: bool) -> None:
    """Download a .pt checkpoint from piper-sample-generator releases."""
    model_path = models_dir / model_filename
    config_path = Path(f"{model_path}.json")

    if not force and _file_ready(model_path, min_size=1_000_000):
        log.info(f"Piper {lang} model already present, skipping")
    else:
        model_url = f"{_PIPER_RELEASE_BASE}/{model_filename}"
        log.info(f"Downloading Piper {lang} model: {model_filename}...")
        _wget(model_url, model_path)

    if not force and _file_ready(config_path, min_size=100):
        log.info(f"Piper {lang} config already present, skipping")
    else:
        config_url = f"{_PIPER_RAW_BASE}/{model_filename}.json"
        log.info(f"Downloading Piper {lang} config: {model_filename}.json...")
        _wget(config_url, config_path)


def _download_piper_onnx_model(lang: str, hf_path: str, models_dir: Path, force: bool) -> None:
    """Download a .onnx voice from rhasspy/piper-voices on HuggingFace."""
    onnx_filename = Path(hf_path).name
    model_path = models_dir / onnx_filename
    config_path = Path(f"{model_path}.json")

    if not force and _file_ready(model_path, min_size=100_000):
        log.info(f"Piper {lang} ONNX model already present, skipping")
    else:
        model_url = f"{_PIPER_VOICES_HF}/{hf_path}"
        log.info(f"Downloading Piper {lang} ONNX model: {onnx_filename}...")
        _wget(model_url, model_path)

    if not force and _file_ready(config_path, min_size=100):
        log.info(f"Piper {lang} ONNX config already present, skipping")
    else:
        config_url = f"{_PIPER_VOICES_HF}/{hf_path}.json"
        log.info(f"Downloading Piper {lang} ONNX config: {onnx_filename}.json...")
        _wget(config_url, config_path)


def get_piper_model_path(lang: str) -> Path:
    """Return the path to the Piper model for a given language."""
    if lang not in PIPER_LANG_MODELS:
        available = ", ".join(sorted(PIPER_LANG_MODELS))
        raise ValueError(f"Unsupported language '{lang}'. Available: {available}")

    model_filename, _espeak_voice, fmt = PIPER_LANG_MODELS[lang]

    match fmt:
        case "pt":
            if lang == "en":
                legacy = MODELS_BASE / "piper-sample-generator" / "models" / "en-us-libritts-high.pt"
                if legacy.exists():
                    return legacy
            return MODELS_BASE / "piper-sample-generator" / "models" / model_filename
        case "onnx":
            onnx_filename = Path(model_filename).name
            return MODELS_BASE / "piper-sample-generator" / "models" / onnx_filename
        case _:
            raise ValueError(f"Unknown model format: {fmt}")


def get_piper_model_format(lang: str) -> str:
    """Return the model format ('pt' or 'onnx') for a given language."""
    if lang not in PIPER_LANG_MODELS:
        available = ", ".join(sorted(PIPER_LANG_MODELS))
        raise ValueError(f"Unsupported language '{lang}'. Available: {available}")
    return PIPER_LANG_MODELS[lang][2]


def download_oww_models(force: bool = False) -> None:
    """Download openWakeWord ONNX/tflite base models."""
    MODELS_BASE.mkdir(parents=True, exist_ok=True)

    for filename, url in OWW_MODEL_URLS.items():
        dest = MODELS_BASE / filename
        if not force and _file_ready(dest, min_size=100_000):
            log.info(f"{filename} already present, skipping")
            continue
        log.info(f"Downloading {filename}...")
        _wget(url, dest)

    # Ensure models are also available inside the installed openwakeword package
    _sync_models_to_package()

    log.info("OWW base models ready")


def _sync_models_to_package() -> None:
    """Copy base ONNX/tflite models into openwakeword's package resources dir."""
    import shutil

    import openwakeword

    pkg_models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    pkg_models_dir.mkdir(parents=True, exist_ok=True)

    for filename in OWW_MODEL_URLS:
        src = MODELS_BASE / filename
        dst = pkg_models_dir / filename
        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
            log.info(f"Synced {filename} to openwakeword package")


def download_features(force: bool = False) -> None:
    """Download pre-computed openWakeWord feature files."""
    DATASETS_BASE.mkdir(parents=True, exist_ok=True)

    for filename, url in FEATURES_URLS.items():
        dest = DATASETS_BASE / filename
        # Remove stale symlink if present
        if dest.is_symlink():
            dest.unlink()
        if not force and _file_ready(dest, min_size=1_000_000):
            log.info(f"{filename} already present, skipping")
            continue
        log.info(f"Downloading {filename}...")
        _wget(url, dest)

    log.info("Feature files ready")


def download_rirs(force: bool = False) -> None:
    """Download MIT room impulse responses via HuggingFace datasets."""
    import datasets as hf_datasets

    rir_dir = DATASETS_BASE / "mit_rirs"

    # Remove stale symlink if present
    if rir_dir.is_symlink():
        rir_dir.unlink()

    if not force and _dir_ready(rir_dir, min_files=200):
        log.info(f"MIT RIRs already present ({sum(1 for _ in rir_dir.iterdir())} files), skipping")
        return

    log.info("Downloading MIT room impulse responses...")
    rir_dir.mkdir(parents=True, exist_ok=True)

    rir_dataset = hf_datasets.load_dataset(RIR_DATASET_ID, split="train", streaming=True)
    for row in tqdm(rir_dataset, desc="Saving RIRs"):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            str(rir_dir / name), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )

    log.info(f"MIT RIRs ready ({sum(1 for _ in rir_dir.iterdir())} files)")


def download_audioset(force: bool = False) -> None:
    """Download a subset of AudioSet and convert to 16kHz wav."""
    import shutil

    import datasets as hf_datasets

    audioset_dir = DATASETS_BASE / "audioset_16k"

    # Remove stale symlink if present
    if audioset_dir.is_symlink():
        audioset_dir.unlink()

    if not force and _dir_ready(audioset_dir, min_files=100):
        log.info(f"AudioSet already present ({sum(1 for _ in audioset_dir.iterdir())} files), skipping")
        return

    log.info("Downloading AudioSet subset...")
    audioset_dir.mkdir(parents=True, exist_ok=True)

    tmp_tar = DATASETS_BASE / "_audioset_tmp.tar"
    tmp_extract = DATASETS_BASE / "_audioset_extract"

    try:
        _wget(AUDIOSET_TAR_URL, tmp_tar)
    except subprocess.CalledProcessError:
        log.warning("Direct wget failed for AudioSet, trying HuggingFace datasets API...")
        tmp_tar.unlink(missing_ok=True)
        _download_audioset_via_hf(audioset_dir)
        return

    log.info("Extracting AudioSet tar...")
    tmp_extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_tar) as tar:
        tar.extractall(path=str(tmp_extract))

    log.info("Converting AudioSet to 16kHz wav...")
    flac_files = [str(p) for p in tmp_extract.rglob("*.flac")]
    audioset_dataset = hf_datasets.Dataset.from_dict({"audio": flac_files})
    audioset_dataset = audioset_dataset.cast_column("audio", hf_datasets.Audio(sampling_rate=16000))

    for row in tqdm(audioset_dataset, desc="Converting"):
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(
            str(audioset_dir / name), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )

    # Cleanup temp files
    tmp_tar.unlink(missing_ok=True)
    shutil.rmtree(tmp_extract, ignore_errors=True)

    log.info(f"AudioSet ready ({sum(1 for _ in audioset_dir.iterdir())} files)")


def _download_audioset_via_hf(audioset_dir: Path) -> None:
    """Fallback: stream AudioSet via HuggingFace datasets API."""
    import datasets as hf_datasets

    log.info("Streaming AudioSet via HuggingFace datasets (balanced train, ~500 clips)...")
    dataset = hf_datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True, trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", hf_datasets.Audio(sampling_rate=16000))

    count = 0
    max_clips = 500
    for row in tqdm(dataset, desc="Downloading AudioSet", total=max_clips):
        audio = row["audio"]
        name = f"audioset_{count:05d}.wav"
        scipy.io.wavfile.write(
            str(audioset_dir / name), 16000,
            (audio["array"] * 32767).astype(np.int16),
        )
        count += 1
        if count >= max_clips:
            break

    log.info(f"AudioSet ready ({count} files via streaming)")


##### ORCHESTRATOR #####


def download_all(force: bool = False, langs: list[str] | None = None) -> None:
    """Download all base assets. Skips already-cached files."""
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )

    console.rule("[bold cyan]Downloading base assets")
    start = time.monotonic()

    steps: list[tuple[str, ...]] = [
        ("OWW base models (ONNX/tflite)", "oww_models"),
        ("Piper TTS (sample generator)", "piper"),
        ("Pre-computed features (.npy)", "features"),
        ("MIT room impulse responses", "rirs"),
        ("AudioSet background noise", "audioset"),
    ]

    # Add extra language models if requested
    extra_langs = [l for l in (langs or []) if l != DEFAULT_LANG]
    if extra_langs:
        steps.append(("Piper TTS language models", "piper_langs"))

    dispatch = {
        "oww_models": lambda: download_oww_models(force=force),
        "piper": lambda: download_piper(force=force),
        "features": lambda: download_features(force=force),
        "rirs": lambda: download_rirs(force=force),
        "audioset": lambda: download_audioset(force=force),
        "piper_langs": lambda: download_piper_lang_models(extra_langs, force=force),
    }

    for i, (name, key) in enumerate(steps, 1):
        console.rule(f"[bold]{i}/{len(steps)}: {name}")
        step_start = time.monotonic()
        dispatch[key]()
        elapsed = time.monotonic() - step_start
        log.info(f"Completed in {elapsed:.1f}s")

    total = time.monotonic() - start
    console.rule(f"[bold green]All base assets ready ({total:.1f}s)")


##### ENTRYPOINT #####


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download base assets for OWW training")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    download_all(force=args.force)


if __name__ == "__main__":
    main()
