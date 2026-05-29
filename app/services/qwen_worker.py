"""Standalone Qwen3-TTS synthesis worker.

Runs inside the isolated ``.venv-qwen`` (created by ``setup-qwen.bat``), which
carries qwen-tts and its heavy/conflicting dependencies (gradio pulls a newer
FastAPI/Starlette, etc.). It is invoked as a subprocess by
``app.services.voice.qwen_tts`` so the main application's pinned environment
stays untouched.

Contract: read the text from a UTF-8 file, synthesize a wav, print a one-line
JSON result to stdout. Deliberately depends only on torch / qwen_tts / soundfile
so it can run in the slim isolated env without importing the rest of the app.

    python qwen_worker.py --text-file in.txt --language Russian \
        --speaker ryan --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --out out.wav
"""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True, help="UTF-8 file with the text to speak")
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--speaker", default="ryan")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--out", required=True, help="output wav path")
    args = parser.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print(json.dumps({"ok": False, "error": "empty text"}), file=sys.stderr)
        return 1

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Pascal GPUs (e.g. GTX 1070 Ti) have no native bf16 and flash-attn does not
    # build there; float32 + eager attention is correct on any GPU/CPU.
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    kwargs = dict(text=text, language=args.language, speaker=args.speaker.lower())
    if args.instruct:
        kwargs["instruct"] = args.instruct
    wavs, sr = model.generate_custom_voice(**kwargs)
    sf.write(args.out, wavs[0], sr)
    print(json.dumps({"ok": True, "sample_rate": int(sr), "device": device}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - report to caller as JSON
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        sys.exit(1)
