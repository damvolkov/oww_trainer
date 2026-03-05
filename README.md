# OWW Trainer

Automated training pipeline for [openWakeWord](https://github.com/dscripka/openWakeWord) custom wakeword models.

Takes a wakeword phrase, generates synthetic TTS clips via Piper, augments them with room impulse responses and background noise, then trains a lightweight ONNX model ready for real-time inference.

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)
- `wget`, `git`, `espeak-ng` (system packages)
- ~16 GB disk for base assets (features, AudioSet, RIRs)

## Quick Start

```bash
# Install deps + download base assets (~16 GB, cached)
make setup

# Train a wakeword model
make train eager

# Train with custom params
make train "hey robot" N_SAMPLES=10000 STEPS=20000

# List all models (base + trained)
make list

# Live acceptance test with microphone
make acceptance alexa
make acceptance eager
```

## Make Targets

| Target                | Description                                      |
|-----------------------|--------------------------------------------------|
| `make setup`          | Install deps, pre-commit hooks, download assets  |
| `make download`       | Download base assets (skips cached)              |
| `make train <w>`      | Train model for wakeword `<w>`                   |
| `make list`           | List base and custom trained models              |
| `make acceptance <m>` | Live mic test for model `<m>` (VAD + OWW)        |
| `make clean`          | Remove generated datasets/models (keeps base)    |
| `make lint`           | Run ruff linter + formatter                      |
| `make type`           | Run ty type checker                              |
| `make test`           | Run unit tests                                   |
| `make help`           | Show all targets                                 |

## Project Structure

```
oww_trainer/
├── src/oww_trainer/
│   ├── trainer.py      # Training pipeline (config, TTS, augment, train, finalize)
│   ├── download.py     # Base asset downloader (models, features, RIRs, AudioSet)
│   ├── models.py       # Model discovery and resolution
│   └── train.py        # Upstream openWakeWord training script
├── tests/
│   ├── unit/           # Mocked unit tests (28 tests)
│   └── acceptance/     # Live microphone acceptance test (VAD + OWW)
├── configs/            # Generated YAML training configs
├── datasets/
│   └── base/           # Downloaded base assets (gitignored)
├── models/
│   └── base/           # OWW base models + Piper TTS (gitignored)
├── Makefile
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Pipeline Steps

1. **Verify base assets** - downloads/caches OWW models, Piper TTS, features, RIRs, AudioSet
2. **Generate config** - builds YAML config for the target wakeword
3. **Generate clips** - synthesizes positive examples via Piper TTS
4. **Augment clips** - applies room impulse responses + background noise, computes features
5. **Train model** - trains a DNN classifier on the augmented features
6. **Finalize** - copies the ONNX model to `models/<wakeword>/`

## Output

After training, the model is at:

```
models/<wakeword>/<wakeword>.onnx
```

Use it with openWakeWord:

```python
from openwakeword.model import Model

model = Model(wakeword_models=["models/eager/eager.onnx"])
prediction = model.predict(audio_frame)
```

## Acceptance Testing

The acceptance test runs a 3-layer real-time audio pipeline:

1. **Energy gate** - skips dead silence (RMS < 0.01)
2. **Silero VAD** - detects voice activity (ONNX, releases GIL)
3. **OpenWakeWord** - detects wakeword (ONNX, parallel with VAD)

```bash
# Test with pretrained models
make acceptance alexa
make acceptance hey_mycroft

# Test with custom trained model
make acceptance eager
```

Color-coded output shows: SILENCE / NOISE / VOICE / WAKEWORD with energy and confidence scores.
