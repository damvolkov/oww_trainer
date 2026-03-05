"""openWakeWord trainer - automated pipeline for training wakeword models."""

import logging
import sys
import time
import uuid
from pathlib import Path

import scipy.io.wavfile
import scipy.signal
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from oww_trainer.download import (
    DEFAULT_LANG,
    PIPER_LANG_MODELS,
    download_all,
    download_piper_lang_models,
    get_piper_model_format,
    get_piper_model_path,
)

##### CONSTANTS #####

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_BASE = PROJECT_ROOT / "datasets" / "base"
MODELS_BASE = PROJECT_ROOT / "models" / "base"
PIPER_PATH = MODELS_BASE / "piper-sample-generator"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"

console = Console()
log = logging.getLogger("oww_trainer")

##### HELPERS #####


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("openwakeword").setLevel(logging.INFO)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


class StepTimer:
    """Context manager that logs step name, status, and elapsed time."""

    def __init__(self, step_name: str, step_number: int, total_steps: int):
        self.step_name = step_name
        self.step_number = step_number
        self.total_steps = total_steps
        self._start: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        console.rule(f"[bold cyan]Step {self.step_number}/{self.total_steps}: {self.step_name}")
        log.info(f"Starting: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.monotonic() - self._start
        match exc_type:
            case None:
                log.info(f"[green]DONE[/green] {self.step_name} ({_format_duration(elapsed)})", extra={"markup": True})
            case _:
                log.error(
                    f"[red]FAILED[/red] {self.step_name} after {_format_duration(elapsed)}",
                    extra={"markup": True},
                )


##### CONFIG #####


def build_config(wakeword: str, output_dir: Path, n_samples: int = 5000, n_samples_val: int = 1000,
                 steps: int = 10000) -> dict:
    """Build a training config dict for the given wakeword."""
    model_name = wakeword.strip().replace(" ", "_").lower()

    return {
        "target_phrase": [wakeword.strip().lower()],
        "model_name": model_name,
        "custom_negative_phrases": [],
        "n_samples": n_samples,
        "n_samples_val": n_samples_val,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(PIPER_PATH),
        "output_dir": str(output_dir),
        "rir_paths": [str(DATASETS_BASE / "mit_rirs")],
        "background_paths": [str(DATASETS_BASE / "audioset_16k")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(DATASETS_BASE / "validation_set_features.npy"),
        "augmentation_rounds": 1,
        "feature_data_files": {
            "ACAV100M_sample": str(DATASETS_BASE / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        },
        "batch_n_per_class": {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": steps,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "target_accuracy": 0.6,
        "target_recall": 0.25,
    }


##### MULTI-LANG CLIP GENERATION #####

_TARGET_SAMPLE_RATE = 16000


def _resample_dir_to_16k(directory: str) -> None:
    """Resample all wav files in a directory to 16kHz if needed."""
    for wav_path in Path(directory).glob("*.wav"):
        sr, data = scipy.io.wavfile.read(str(wav_path))
        if sr == _TARGET_SAMPLE_RATE:
            continue
        n_samples = int(len(data) * _TARGET_SAMPLE_RATE / sr)
        resampled = scipy.signal.resample(data.astype(float), n_samples).astype(data.dtype)
        scipy.io.wavfile.write(str(wav_path), _TARGET_SAMPLE_RATE, resampled)


def _compute_samples_per_lang(total: int, langs: list[str]) -> dict[str, int]:
    """Split total samples across languages. English gets the largest share."""
    n_langs = len(langs)
    base = total // n_langs
    remainder = total - base * n_langs
    result: dict[str, int] = {}
    for i, lang in enumerate(langs):
        result[lang] = base + (1 if i < remainder else 0)
    return result


def _generate_lang_clips_worker(
    config: dict, lang: str, n_train: int, n_val: int, dirs: dict[str, str],
) -> None:
    """Worker: generate TTS clips for a single language. Runs in subprocess for GPU isolation."""
    sys.path.insert(0, str(PIPER_PATH))
    from generate_samples import generate_samples
    from generate_samples import generate_samples_onnx
    from openwakeword.data import generate_adversarial_texts

    model_path = str(get_piper_model_path(lang))
    model_fmt = get_piper_model_format(lang)

    gen_fn = generate_samples_onnx if model_fmt == "onnx" else generate_samples
    pt_kwargs: dict = {"batch_size": config["tts_batch_size"], "auto_reduce_batch_size": True}
    pt_kwargs_neg: dict = {"batch_size": max(config["tts_batch_size"] // 7, 1), "auto_reduce_batch_size": True}

    # Positive train
    if len(list(Path(dirs["positive_train"]).glob("*.wav"))) < config["n_samples"] * 0.95:
        gen_fn(
            text=config["target_phrase"], max_samples=n_train, model=model_path,
            noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
            output_dir=dirs["positive_train"],
            file_names=[f"{lang}_{uuid.uuid4().hex}.wav" for _ in range(n_train)],
            **(pt_kwargs if model_fmt == "pt" else {}),
        )

    # Positive test
    if len(list(Path(dirs["positive_test"]).glob("*.wav"))) < config["n_samples_val"] * 0.95:
        gen_fn(
            text=config["target_phrase"], max_samples=n_val, model=model_path,
            noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
            output_dir=dirs["positive_test"],
            file_names=[f"{lang}_{uuid.uuid4().hex}.wav" for _ in range(n_val)],
            **(pt_kwargs if model_fmt == "pt" else {}),
        )

    # Adversarial negative texts
    adversarial_texts = list(config.get("custom_negative_phrases", []))
    for phrase in config["target_phrase"]:
        adversarial_texts.extend(generate_adversarial_texts(
            input_text=phrase,
            N=n_train // max(len(config["target_phrase"]), 1),
            include_partial_phrase=1.0, include_input_words=0.2,
        ))

    # Negative train
    if len(list(Path(dirs["negative_train"]).glob("*.wav"))) < config["n_samples"] * 0.95:
        gen_fn(
            text=adversarial_texts, max_samples=n_train, model=model_path,
            noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
            output_dir=dirs["negative_train"],
            file_names=[f"{lang}_{uuid.uuid4().hex}.wav" for _ in range(n_train)],
            **(pt_kwargs_neg if model_fmt == "pt" else {}),
        )

    # Adversarial negative texts for test (regenerate with n_val count)
    adversarial_texts_val = list(config.get("custom_negative_phrases", []))
    for phrase in config["target_phrase"]:
        adversarial_texts_val.extend(generate_adversarial_texts(
            input_text=phrase,
            N=n_val // max(len(config["target_phrase"]), 1),
            include_partial_phrase=1.0, include_input_words=0.2,
        ))

    # Negative test
    if len(list(Path(dirs["negative_test"]).glob("*.wav"))) < config["n_samples_val"] * 0.95:
        gen_fn(
            text=adversarial_texts_val, max_samples=n_val, model=model_path,
            noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
            output_dir=dirs["negative_test"],
            file_names=[f"{lang}_{uuid.uuid4().hex}.wav" for _ in range(n_val)],
            **(pt_kwargs_neg if model_fmt == "pt" else {}),
        )


def _run_lang_subprocess(config: dict, lang: str, n_train: int, n_val: int, dirs: dict[str, str]) -> None:
    """Run TTS generation for one language in an isolated subprocess (full GPU release on exit)."""
    import json
    import subprocess
    import tempfile

    payload = {"config": config, "lang": lang, "n_train": n_train, "n_val": n_val, "dirs": dirs}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name

    script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r}); "
        f"payload = json.load(open({payload_path!r})); "
        "from oww_trainer.trainer import _generate_lang_clips_worker; "
        "_generate_lang_clips_worker(**payload)"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(PROJECT_ROOT), capture_output=False, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"TTS generation for '{lang}' failed (exit code {result.returncode})")
    finally:
        Path(payload_path).unlink(missing_ok=True)


def _generate_multilang_clips(config: dict, langs: list[str]) -> None:
    """Generate positive and adversarial negative clips for all languages."""
    import torch

    use_subprocess = torch.cuda.is_available()

    model_name = config["model_name"]
    output_base = Path(config["output_dir"]) / model_name

    dirs = {
        "positive_train": str(output_base / "positive_train"),
        "positive_test": str(output_base / "positive_test"),
        "negative_train": str(output_base / "negative_train"),
        "negative_test": str(output_base / "negative_test"),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    train_per_lang = _compute_samples_per_lang(config["n_samples"], langs)
    val_per_lang = _compute_samples_per_lang(config["n_samples_val"], langs)

    for lang in langs:
        n_train = train_per_lang[lang]
        n_val = val_per_lang[lang]
        model_fmt = get_piper_model_format(lang)
        model_name_short = get_piper_model_path(lang).name

        log.info(f"[{lang.upper()}] Generating {n_train}+{n_val} clips "
                 f"(model: {model_name_short}, format: {model_fmt}, "
                 f"device: {'cuda/subprocess' if use_subprocess else 'cpu/in-process'})")

        if use_subprocess:
            _run_lang_subprocess(config, lang, n_train, n_val, dirs)
        else:
            _generate_lang_clips_worker(config, lang, n_train, n_val, dirs)

    # Resample all clips to 16kHz (Piper models may output at 22050Hz)
    log.info("Resampling clips to 16kHz...")
    for d in dirs.values():
        _resample_dir_to_16k(d)

    log.info(f"Multi-language clip generation complete: {', '.join(langs)}")


##### PIPELINE #####


def run_pipeline(
    wakeword: str,
    n_samples: int = 5000,
    n_samples_val: int = 1000,
    steps: int = 10000,
    langs: list[str] | None = None,
) -> Path:
    """Run the full training pipeline for a wakeword. Returns the output directory."""
    _configure_logging()
    pipeline_start = time.monotonic()

    langs = langs or [DEFAULT_LANG]
    is_multilang = len(langs) > 1 or langs != [DEFAULT_LANG]

    # Validate languages
    for lang in langs:
        if lang not in PIPER_LANG_MODELS:
            available = ", ".join(sorted(PIPER_LANG_MODELS))
            raise ValueError(f"Unsupported language '{lang}'. Available: {available}")

    model_name = wakeword.strip().replace(" ", "_").lower()
    output_dir = PROJECT_ROOT / "models" / model_name
    datasets_dir = PROJECT_ROOT / "datasets" / model_name
    config_path = PROJECT_ROOT / "configs" / f"{model_name}.yaml"

    console.print(Panel.fit(
        f"[bold]Wakeword:[/bold] {wakeword}\n"
        f"[bold]Model name:[/bold] {model_name}\n"
        f"[bold]Languages:[/bold] {', '.join(langs)}\n"
        f"[bold]Samples:[/bold] {n_samples} train / {n_samples_val} val\n"
        f"[bold]Steps:[/bold] {steps}\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="[bold magenta]OWW Trainer Pipeline[/bold magenta]",
        border_style="magenta",
    ))

    total_steps = 7 if is_multilang else 6

    # Step 1: Ensure base assets are downloaded
    with StepTimer("Verify base assets", 1, total_steps):
        download_all(langs=langs)

    # Step 1b: Download extra language models if needed
    step_offset = 0
    if is_multilang:
        step_offset = 1
        with StepTimer("Download language TTS models", 2, total_steps):
            extra_langs = [l for l in langs if l != DEFAULT_LANG]
            if extra_langs:
                download_piper_lang_models(extra_langs)

    # Step 2: Generate config
    with StepTimer("Generate training config", 2 + step_offset, total_steps):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)

        config = build_config(wakeword, output_dir=datasets_dir, n_samples=n_samples,
                              n_samples_val=n_samples_val, steps=steps)

        # Store langs in config for reference
        config["langs"] = langs

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        log.info(f"Config saved to {config_path}")

    # Step 3: Generate synthetic clips
    with StepTimer(f"Generate synthetic clips ({', '.join(langs)})", 3 + step_offset, total_steps):
        if is_multilang:
            _generate_multilang_clips(config, langs)
        else:
            _run_generate_clips(config_path)

    # Step 4: Augment clips and compute features
    with StepTimer("Augment clips & compute features", 4 + step_offset, total_steps):
        _run_augment_clips(config_path)

    # Step 5: Train model
    with StepTimer("Train model", 5 + step_offset, total_steps):
        _run_train_model(config_path)

    # Step 6: Copy final model to models/
    with StepTimer("Finalize model artifacts", 6 + step_offset, total_steps):
        _finalize_model(config, datasets_dir, output_dir)

    pipeline_elapsed = time.monotonic() - pipeline_start

    # Summary
    table = Table(title="Training Summary", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Wakeword", wakeword)
    table.add_row("Model name", model_name)
    table.add_row("Languages", ", ".join(langs))
    table.add_row("Total time", _format_duration(pipeline_elapsed))

    onnx_path = output_dir / f"{model_name}.onnx"
    table.add_row("ONNX model", str(onnx_path) if onnx_path.exists() else "[red]NOT FOUND[/red]")
    console.print(table)

    return output_dir


##### STEP IMPLEMENTATIONS #####


def _run_generate_clips(config_path: Path) -> None:
    """Run the clip generation step via openwakeword train.py."""
    import subprocess

    train_script = TRAIN_SCRIPT
    cmd = [
        sys.executable, str(train_script),
        "--training_config", str(config_path),
        "--generate_clips",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Clip generation failed with return code {result.returncode}")


def _run_augment_clips(config_path: Path) -> None:
    """Run the augmentation step via openwakeword train.py."""
    import subprocess

    train_script = TRAIN_SCRIPT
    cmd = [
        sys.executable, str(train_script),
        "--training_config", str(config_path),
        "--augment_clips",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Clip augmentation failed with return code {result.returncode}")


def _run_train_model(config_path: Path) -> None:
    """Run the training step via openwakeword train.py."""
    import subprocess

    train_script = TRAIN_SCRIPT
    cmd = [
        sys.executable, str(train_script),
        "--training_config", str(config_path),
        "--train_model",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, text=True)
    if result.returncode != 0:
        # Check if ONNX was saved despite tflite conversion failure (upstream bug: default="False" is truthy)
        config = yaml.safe_load(config_path.read_text())
        onnx_path = Path(config["output_dir"]) / f"{config['model_name']}.onnx"
        if onnx_path.exists():
            log.warning("Training subprocess exited non-zero but ONNX model was saved (tflite conversion skipped)")
        else:
            raise RuntimeError(f"Model training failed with return code {result.returncode}")


def _finalize_model(config: dict, datasets_dir: Path, output_dir: Path) -> None:
    """Copy the final model artifacts from the datasets dir to the models dir."""
    import shutil

    model_name = config["model_name"]
    onnx_src = datasets_dir / f"{model_name}.onnx"

    if onnx_src.exists():
        onnx_dst = output_dir / f"{model_name}.onnx"
        shutil.copy2(onnx_src, onnx_dst)
        log.info(f"ONNX model copied to {onnx_dst}")
    else:
        log.warning(f"ONNX model not found at {onnx_src}")

    # Copy config for reference
    config_dst = output_dir / f"{model_name}_config.yaml"
    with open(config_dst, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Config saved to {config_dst}")


##### ENTRYPOINT #####


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OWW Trainer - train openWakeWord models")
    parser.add_argument("wakeword", type=str, help="The wakeword or phrase to train")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of training samples (default: 5000)")
    parser.add_argument("--n-samples-val", type=int, default=1000, help="Number of validation samples (default: 1000)")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps (default: 10000)")
    parser.add_argument(
        "--langs", type=str, default="en",
        help=f"Comma-separated language codes (default: en). Available: {', '.join(sorted(PIPER_LANG_MODELS))}",
    )

    args = parser.parse_args()
    langs = [l.strip() for l in args.langs.split(",")]

    run_pipeline(
        wakeword=args.wakeword,
        n_samples=args.n_samples,
        n_samples_val=args.n_samples_val,
        steps=args.steps,
        langs=langs,
    )


if __name__ == "__main__":
    main()
