"""Record a sample of every voice for the voices page (/voices).

    python samples.py                 # only voices without a sample yet
    python samples.py --force         # re-record all
    python samples.py -v arthas bandit --force

Writes static/samples/<voice>.mp3 and static/samples/voices.json (what the page shows). Page metadata comes from
the optional voice.json keys: "title", "category", "gender" ("m"/"f"), "tags" (list), "sample_text".
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tts_pipeline import VOICES_DIR, Pipeline, list_voices

OUT = Path(__file__).parent / "static" / "samples"
SAMPLE_SR = 44100  # RVC models output 32/40/48 kHz; MP3 has no 40 kHz
DEFAULT_TEXT = "Привет, чат! Стример опять проиграл, но мы всё равно его любим. Давай, ещё одну катку!"


def meta(name: str) -> dict:
    p = VOICES_DIR / name / "voice.json"
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {
        "id": name,
        "title": cfg.get("title", name.replace("_", " ").title()),
        "category": cfg.get("category", "Другое"),
        "gender": cfg.get("gender"),
        "tags": cfg.get("tags", []),
        "text": cfg.get("sample_text", DEFAULT_TEXT),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--voices", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="re-record samples that already exist")
    ap.add_argument("--rvc-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--en-engine", choices=["kokoro", "silero"], default=None)
    args = ap.parse_args()

    import librosa

    OUT.mkdir(parents=True, exist_ok=True)
    voices = list_voices()
    todo = [v for v in (args.voices or voices) if args.force or not (OUT / f"{v}.mp3").exists()]
    pipe = Pipeline(rvc_device=args.rvc_device, en_engine=args.en_engine) if todo else None
    for v in todo:
        m = meta(v)
        torch.manual_seed(0)
        audio, sr, t = pipe.say(m["text"], v)
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_SR)
        audio *= 0.9 / max(float(np.abs(audio).max()), 1e-6)  # same peak level for every voice
        sf.write(OUT / f"{v}.mp3", audio, SAMPLE_SR, format="MP3")
        print(f"{v:14} {t}", file=sys.stderr)

    # the page lists every voice with a sample, also ones recorded earlier
    listed = [dict(meta(v), sample=f"{v}.mp3") for v in voices if (OUT / f"{v}.mp3").exists()]
    (OUT / "voices.json").write_text(json.dumps(listed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(listed)} voices in {OUT / 'voices.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
