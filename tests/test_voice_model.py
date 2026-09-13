import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voice_model import VoiceModel  # noqa: E402
from utils import load_config  # noqa: E402


class FakeTextModel:
    """Stands in for a trained TextModel so this test doesn't need to
    download/fine-tune a real BERT model - it only checks that VoiceModel
    correctly reuses whatever text model it is given (README S6.2)."""

    def process(self, texts):
        if not texts:
            return {"text_score": None, "text_label": None, "confidence": None, "driving_text": None}
        return {"text_score": 0.9, "text_label": "failed_transaction", "confidence": 0.9, "driving_text": texts[0]}


def test_process_is_null_safe_with_no_input():
    vm = VoiceModel(FakeTextModel(), load_config())
    result = vm.process()
    assert result == {"voice_score": None, "voice_label": None, "confidence": None, "transcript": None}


def test_process_reuses_the_given_text_model_on_transcripts():
    vm = VoiceModel(FakeTextModel(), load_config())
    result = vm.process(transcripts=["my transfer failed since yesterday"])
    assert result["voice_score"] == 0.9
    assert result["voice_label"] == "failed_transaction"
    assert result["transcript"] == "my transfer failed since yesterday"
