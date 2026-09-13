"""Sprint 3 - Voice modality (README S6.2).

Transcribe-then-reuse-the-text-model, deliberately. Open acoustic-emotion
datasets are trained on scripted, non-Nigerian speech and don't generalise to
Nigerian-accented, code-switched call audio - so this module does NOT train
a separate emotion model. It transcribes with Whisper (faster-whisper) and
then scores the transcript with the exact same TextModel from
src/text_model.py. Acoustic emotion recognition is future work, not part of
this build (# TODO(scope) if ever revisited).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_model import TextModel
from utils import load_config


class VoiceModel:
    def __init__(self, text_model: TextModel, config: dict | None = None):
        self.text_model = text_model
        self.cfg = (config or load_config())["voice_model"]
        self._whisper = None

    def _lazy_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(
                self.cfg["whisper_model_size"],
                device=self.cfg["device"],
                compute_type=self.cfg["compute_type"],
            )
        return self._whisper

    def transcribe(self, audio_path: str | Path) -> str:
        model = self._lazy_whisper()
        segments, _info = model.transcribe(str(audio_path), language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()

    def process(self, audio_path: str | Path | None = None, transcripts: list[str] | None = None) -> dict:
        """Null-safe. Accepts either pre-made transcript(s) or a path to a
        call-audio file to transcribe with Whisper (README S6.2 input)."""
        texts: list[str] = list(transcripts) if transcripts else []
        transcribed_here = None
        if audio_path:
            transcribed_here = self.transcribe(audio_path)
            texts = [transcribed_here] + texts

        if not texts:
            return {"voice_score": None, "voice_label": None, "confidence": None, "transcript": None}

        result = self.text_model.process(texts)
        return {
            "voice_score": result["text_score"],
            "voice_label": result["text_label"],
            "confidence": result["confidence"],
            "transcript": result["driving_text"],
        }
