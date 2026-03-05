"""Model discovery and resolution for openWakeWord models."""

import os
from pathlib import Path

import openwakeword

##### CONSTANTS #####

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_BASE = MODELS_DIR / "base"

_INFRA_MODELS = frozenset({"embedding_model", "melspectrogram", "silero_vad"})

##### DISCOVERY #####


def get_pretrained_names() -> list[str]:
    """Return names of pretrained OWW models (without version suffix)."""
    names: list[str] = []
    for path in openwakeword.get_pretrained_model_paths(inference_framework="onnx"):
        basename = os.path.basename(path)
        name = basename.split("_v0")[0].replace(".onnx", "")
        if name not in _INFRA_MODELS:
            names.append(name)
    return sorted(set(names))


def get_custom_names() -> list[str]:
    """Return names of custom-trained models in models/<name>/<name>.onnx."""
    names: list[str] = []
    if not MODELS_DIR.exists():
        return names
    for d in sorted(MODELS_DIR.iterdir()):
        if d.name == "base" or not d.is_dir():
            continue
        onnx = d / f"{d.name}.onnx"
        if onnx.exists():
            names.append(d.name)
    return names


def get_available_models() -> dict[str, list[str]]:
    """Return all available models grouped by type."""
    return {
        "pretrained": get_pretrained_names(),
        "custom": get_custom_names(),
    }


def resolve_wakeword_path(name: str) -> str:
    """Resolve a wakeword name to its ONNX model path."""
    custom_path = MODELS_DIR / name / f"{name}.onnx"
    if custom_path.exists():
        return str(custom_path)

    for path in openwakeword.get_pretrained_model_paths(inference_framework="onnx"):
        if name in os.path.basename(path):
            return path

    available = get_available_models()
    all_names = available["pretrained"] + available["custom"]
    raise ValueError(f"Model '{name}' not found. Available: {all_names}")
