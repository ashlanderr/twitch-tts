"""Memory footprint of the pipeline with all voices loaded, plus per-message speed.

    python memtest.py --label before
    python memtest.py --label after
    python memtest.py --compare before after     # audio difference between two runs

Seeds are fixed, so two runs with identical settings produce identical audio.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

from bench import MESSAGES

OUT = Path("out") / "memtest"


def windows_vram_mib(pid: int) -> float | None:
    """Dedicated VRAM the OS attributes to the process: what competes with the game."""
    try:
        from metrics import GpuCounters
        _, mem = GpuCounters().sample()
        return mem.get(pid, 0) / 2**20
    except Exception:
        return None


def snapshot(proc) -> dict:
    gc.collect()
    torch.cuda.synchronize()
    return {
        "ram_mib": round(proc.memory_info().rss / 2**20),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20),
        "windows_vram_mib": round(windows_vram_mib(proc.pid) or 0),
    }


def run(label: str):
    from tts_pipeline import Pipeline, list_voices

    proc = psutil.Process()
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    voices = list_voices()

    t0 = time.perf_counter()
    pipe = Pipeline(rvc_device="cuda")
    pipe.warmup(*voices)
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    idle = snapshot(proc)

    times, devices = [], []
    for v in voices:
        for i, msg in enumerate(MESSAGES):
            torch.manual_seed(i)
            audio, sr, t = pipe.say(msg, v)
            times.append(t.tts + t.rvc)
            devices.append(t.extra.get("device"))
            sf.write(out / f"{v}_{i}.wav", audio, sr)
    # every message in a different voice than the previous one: worst case for swapping voices in and out
    switching = []
    for i, msg in enumerate(MESSAGES * 2):
        t0 = time.perf_counter()
        pipe.say(msg, voices[i % len(voices)])
        switching.append(time.perf_counter() - t0)
    # the CPU fallback used when VRAM is short; alternate voices so CPU copies get built and evicted too
    fallback = []
    if pipe.vc.gpu:
        for i, msg in enumerate(MESSAGES * 2):
            t0 = time.perf_counter()
            pipe.say(msg, voices[i % len(voices)], force_cpu=True)
            fallback.append(time.perf_counter() - t0)
    after = snapshot(proc)
    res = {
        "label": label,
        "voices": len(voices),
        "load_s": round(load_s, 1),
        "idle": idle,
        "after_messages": after,
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "msg_avg_s": round(statistics.mean(times), 3),
        "msg_max_s": round(max(times), 3),
        "msg_switching_voice_avg_s": round(statistics.mean(switching), 3),
        "msg_cpu_fallback_avg_s": round(statistics.mean(fallback), 3) if fallback else None,
        "msg_cpu_fallback_max_s": round(max(fallback), 3) if fallback else None,
        "devices": {d: devices.count(d) for d in set(devices)},
        "en_engine": pipe.tts.en_engine,
        "vram_cap_mib": getattr(pipe.vc, "vram_cap_mb", None),
    }
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def compare(a: str, b: str):
    import librosa

    diffs = []
    for fa in sorted((OUT / a).glob("*.wav")):
        fb = OUT / b / fa.name
        xa, sr = sf.read(fa)
        xb, _ = sf.read(fb)
        n = min(len(xa), len(xb))
        ma = librosa.power_to_db(librosa.feature.melspectrogram(y=xa[:n].astype(np.float32), sr=sr, n_mels=80))
        mb = librosa.power_to_db(librosa.feature.melspectrogram(y=xb[:n].astype(np.float32), sr=sr, n_mels=80))
        d = float(np.mean(np.abs(ma - mb)))
        diffs.append(d)
        print(f"{fa.name:22} log-mel diff {d:5.2f} dB   length {len(xa)} vs {len(xb)}")
    print(f"mean log-mel diff {statistics.mean(diffs):.2f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.label)


if __name__ == "__main__":
    main()
