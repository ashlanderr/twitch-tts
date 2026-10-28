"""Measure latency, VRAM and RAM of the pipeline on GPU vs CPU.

    python bench.py --rvc-device cuda
    python bench.py --rvc-device cpu --threads 4
"""

import argparse
import os
import time
from pathlib import Path

import psutil
import soundfile as sf
import torch

from tts_pipeline import Pipeline, list_voices

MESSAGES = [
    "Привет, сталкер!",
    "Эй, ты, иди сюда. Есть для тебя работёнка, заплачу 500 рублей.",
    "I used to be an adventurer like you, then I took an arrow in the knee.",
    "Ребята, это было просто gg, лучший клатч за весь стрим, 1 против 4!",
    "Стример, а когда будет Dead by Daylight? Мы ждём уже 2 часа, давай уже, пожалуйста, хватит сидеть в лобби.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--en-engine", choices=["kokoro", "silero"], default=None)
    ap.add_argument("--voices", nargs="*", default=None)
    args = ap.parse_args()

    proc = psutil.Process(os.getpid())
    out_dir = Path("out") / ("gpu" if args.rvc_device == "cuda" else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = Pipeline(rvc_device=args.rvc_device, en_engine=args.en_engine, threads=args.threads)
    print(f"init {time.perf_counter() - t0:.1f}s, RVC on {pipe.vc.device}, EN engine {pipe.tts.en_engine}, "
          f"torch threads {torch.get_num_threads()}")

    voices = args.voices or list_voices()
    t0 = time.perf_counter()
    pipe.warmup(voices[0])
    print(f"warmup {time.perf_counter() - t0:.1f}s")

    rows = []
    for voice in voices:
        for i, msg in enumerate(MESSAGES):
            audio, sr, t = pipe.say(msg, voice)
            sf.write(out_dir / f"{voice}_{i}.wav", audio, sr)
            total = t.tts + t.rvc
            rows.append((voice, i, t.tts, t.rvc, total, t.audio_sec))
            print(f"{voice:14} #{i}  {t}  (RTF {total / max(t.audio_sec, 1e-6):.2f})")

    print()
    tts = sum(r[2] for r in rows) / len(rows)
    rvc = sum(r[3] for r in rows) / len(rows)
    worst = max(r[4] for r in rows)
    print(f"avg tts {tts:.2f}s, avg rvc {rvc:.2f}s, worst message {worst:.2f}s")
    print(f"RAM (RSS) {proc.memory_info().rss / 2**30:.2f} GiB")
    if torch.cuda.is_available() and args.rvc_device == "cuda":
        print(f"VRAM peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB "
              f"(+ ~0.3-0.5 GiB CUDA context)")


if __name__ == "__main__":
    main()
