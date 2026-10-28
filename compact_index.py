"""Make voices/<name>/model.compact.index: the same IVF index with vectors stored in fp16 (half the size).

Keeps the original coarse clustering (same nlist/nprobe and centroids), so lookups return almost the same
neighbours. At runtime vectors are read straight from the index, without RVC's extra float32 copy.

    python compact_index.py            # all voices
    python compact_index.py bandit
"""

import sys
import time
from pathlib import Path

import faiss  # no torch here: torch and faiss ship different OpenMP runtimes that clash in faiss training

VOICES_DIR = Path(__file__).resolve().parent / "voices"


def list_voices() -> list[str]:
    return sorted(p.name for p in VOICES_DIR.iterdir() if (p / "model.pth").exists())


def compact(name: str):
    d = VOICES_DIR / name
    src, dst = d / "model.index", d / "model.compact.index"
    if not src.exists():
        print(f"{name}: no index")
        return
    t0 = time.perf_counter()
    orig = faiss.read_index(str(src))
    ivf = faiss.extract_index_ivf(orig)
    vectors = orig.reconstruct_n(0, orig.ntotal)

    quantizer = faiss.clone_index(ivf.quantizer)
    new = faiss.IndexIVFScalarQuantizer(quantizer, orig.d, ivf.nlist, faiss.ScalarQuantizer.QT_fp16, ivf.metric_type)
    new.train(vectors)  # centroids are already trained; this only sets up the fp16 codec
    new.add(vectors)
    new.nprobe = ivf.nprobe
    faiss.write_index(new, str(dst))
    print(f"{name}: {orig.ntotal} vectors, {src.stat().st_size / 2**20:.0f} MiB -> {dst.stat().st_size / 2**20:.0f} MiB "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    for n in sys.argv[1:] or list_voices():
        compact(n)
