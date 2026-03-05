"""Live wakeword acceptance test — real-time audio pipeline with VAD + OWW."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum, auto
from queue import SimpleQueue

import numpy as np
import sounddevice as sd
import torch
from loguru import logger
from openwakeword.model import Model
from silero_vad import load_silero_vad

from oww_trainer.models import MODELS_BASE, get_available_models, resolve_wakeword_path

##### CONSTANTS #####

_CHUNK_SIZE = 512  # 32ms @ 16kHz — required by Silero VAD
_SAMPLE_RATE = 16000
_CHUNKS_PER_LOG = _SAMPLE_RATE // _CHUNK_SIZE  # ~31 chunks ~ 1s

_SILENCE_CEIL = 0.01  # RMS below this = dead silence (skip models entirely)
_VAD_THRESHOLD = 0.5  # Silero speech probability threshold
_WW_THRESHOLD = 0.5  # OpenWakeWord activation threshold

_SENTINEL = None  # Poison pill for worker shutdown

##### LABELS #####


class AudioLabel(StrEnum):
    SILENCE = auto()
    NOISE = auto()
    VOICE = auto()
    WAKEWORD = auto()


_PRIORITY: dict[AudioLabel, int] = {
    AudioLabel.SILENCE: 0,
    AudioLabel.NOISE: 1,
    AudioLabel.VOICE: 2,
    AudioLabel.WAKEWORD: 3,
}

_COLORS: dict[AudioLabel, str] = {
    AudioLabel.SILENCE: "25;25;112",
    AudioLabel.NOISE: "100;149;237",
    AudioLabel.VOICE: "0;206;209",
    AudioLabel.WAKEWORD: "34;139;34",
}

##### SILERO VAD CACHE #####


def _ensure_silero_vad_cached() -> None:
    """Ensure silero VAD ONNX model is cached under models/base/."""
    MODELS_BASE.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(MODELS_BASE / "torch_hub")


##### PROCESSOR #####


class ActiveListeningProcessor:
    """3-layer parallel pipeline: Energy gate -> [Silero VAD || OpenWakeWord]."""

    __slots__ = (
        "_chunk_count",
        "_executor",
        "_peak_energy",
        "_peak_label",
        "_peak_vad_score",
        "_peak_ww_score",
        "_queue",
        "_vad_model",
        "_worker",
        "_ww_model",
    )

    def __init__(self, wake_word_model: str = "alexa") -> None:
        _ensure_silero_vad_cached()

        # Layer 2: Silero VAD (ONNX — releases GIL during inference)
        self._vad_model = load_silero_vad(onnx=True)

        # Layer 3: OpenWakeWord (ONNX — releases GIL during inference)
        model_path = resolve_wakeword_path(wake_word_model)
        self._ww_model = Model(wakeword_models=[model_path], inference_framework="onnx")

        # Lock-free audio pipeline: callback -> queue -> worker thread
        self._queue: SimpleQueue[np.ndarray | None] = SimpleQueue()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._worker = threading.Thread(target=self._process_loop, daemon=True)

        # Logging accumulators (only accessed from worker thread — no lock needed)
        self._chunk_count = 0
        self._peak_label = AudioLabel.SILENCE
        self._peak_energy = 0.0
        self._peak_vad_score = 0.0
        self._peak_ww_score = 0.0

        logger.success(
            f"Wake Word: '{wake_word_model}' | "
            f"silence<{_SILENCE_CEIL} | vad>={_VAD_THRESHOLD} | ww>={_WW_THRESHOLD}"
        )

    def _promote(self, label: AudioLabel) -> None:
        if _PRIORITY[label] > _PRIORITY[self._peak_label]:
            self._peak_label = label

    def _infer_vad(self, audio: np.ndarray) -> float:
        """Silero VAD inference. ONNX releases GIL -> true parallel with OWW."""
        tensor = torch.from_numpy(audio).float()
        return float(self._vad_model(tensor, _SAMPLE_RATE).item())

    def _infer_ww(self, audio: np.ndarray) -> float:
        """OpenWakeWord inference. OWW requires int16 PCM internally."""
        audio_int16 = (audio * 32767).astype(np.int16)
        try:
            prediction = self._ww_model.predict(audio_int16)
        except Exception:
            return 0.0  # warmup: not enough frames accumulated yet
        return max(prediction.values(), default=0.0)

    def _process_chunk(self, audio: np.ndarray) -> None:
        energy = float(np.sqrt(np.mean(audio**2)))
        self._peak_energy = max(self._peak_energy, energy)

        ww_future = self._executor.submit(self._infer_ww, audio)

        if energy < _SILENCE_CEIL:
            ww_future.result()
            return

        vad_future = self._executor.submit(self._infer_vad, audio)

        vad_prob = vad_future.result()
        ww_score = ww_future.result()

        self._peak_vad_score = max(self._peak_vad_score, vad_prob)

        if vad_prob < _VAD_THRESHOLD:
            self._promote(AudioLabel.NOISE)
            return

        self._promote(AudioLabel.VOICE)

        if ww_score > _WW_THRESHOLD:
            self._promote(AudioLabel.WAKEWORD)
            self._peak_ww_score = max(self._peak_ww_score, ww_score)

    def _process_loop(self) -> None:
        """Worker thread: drains queue, runs parallel inference per chunk."""
        while (audio := self._queue.get()) is not _SENTINEL:
            self._process_chunk(audio)
            self._chunk_count += 1

            if self._chunk_count >= _CHUNKS_PER_LOG:
                self._flush_log()

    def _flush_log(self) -> None:
        label = self._peak_label
        color = _COLORS[label]
        tag = label.value.upper()
        ts = time.strftime("%H:%M:%S")

        extras: list[str] = []
        if self._peak_vad_score > 0:
            extras.append(f"vad={self._peak_vad_score:.2f}")
        if label == AudioLabel.WAKEWORD:
            extras.append(f"ww={self._peak_ww_score:.2f}")
        extra_str = f" {' '.join(extras)}" if extras else ""

        print(
            f"\033[38;2;{color}m{ts} [{tag}]{extra_str} e={self._peak_energy:.4f}\033[0m",
            file=sys.stderr,
        )

        self._peak_label = AudioLabel.SILENCE
        self._peak_energy = 0.0
        self._peak_vad_score = 0.0
        self._peak_ww_score = 0.0
        self._chunk_count = 0

    def process_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """PortAudio callback — O(1) enqueue only, zero inference here."""
        if status:
            logger.error(f"Audio status: {status}")
        self._queue.put_nowait(indata.flatten().astype(np.float32))

    def start(self) -> None:
        self._worker.start()
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                blocksize=_CHUNK_SIZE,
                callback=self.process_callback,
                dtype="float32",
            ):
                logger.info("Listening... Ctrl+C to stop")
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            logger.warning("\nStopped by user.")
        finally:
            self._queue.put(_SENTINEL)
            self._worker.join(timeout=2.0)
            self._executor.shutdown(wait=False)


##### MAIN #####


def main() -> None:
    available = get_available_models()
    all_models = available["pretrained"] + available["custom"]

    parser = argparse.ArgumentParser(
        description="Live wakeword acceptance test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available models:\n"
            f"  Pretrained: {', '.join(available['pretrained']) or '(none)'}\n"
            f"  Custom:     {', '.join(available['custom']) or '(none)'}"
        ),
    )
    parser.add_argument(
        "model",
        type=str,
        help="Wakeword model name to test",
    )
    args = parser.parse_args()

    if args.model not in all_models:
        parser.error(
            f"Model '{args.model}' not found.\n"
            f"  Pretrained: {', '.join(available['pretrained']) or '(none)'}\n"
            f"  Custom:     {', '.join(available['custom']) or '(none)'}"
        )

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="DEBUG",
    )

    processor = ActiveListeningProcessor(wake_word_model=args.model)
    processor.start()


if __name__ == "__main__":
    main()
