"""Download voice models from the sources recorded in voices/<name>/voice.json.

    python fetch_voices.py              # voices whose model files are missing
    python fetch_voices.py arthas bandit
    python fetch_voices.py --check      # verify sha256 of the files already on disk

voice.json "source":

    "source": {
      "page": "https://huggingface.co/Warlock700/Thrall",
      "files": {
        "model.pth":   {"url": ".../resolve/<commit>/thrall.zip", "member": "thrall/thrall_598e.pth", "sha256": "..."},
        "model.index": {"url": ".../resolve/<commit>/thrall.zip", "member": "thrall/thrall.index", "sha256": "..."}
      }
    }

URLs point at a fixed commit, so a download gets the same bytes as when the voice was added; sha256 checks it.
A new voice can be added without sha256: the first download records it in voice.json.
With "member", only that file is read out of the zip (HTTP range requests), not the whole archive: some packs are
tens of GB. After a download the index is compacted (compact_index.py).
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from compact_index import compact

VOICES_DIR = Path(__file__).resolve().parent / "voices"
CHUNK = 8 << 20


class HttpFile(io.RawIOBase):
    """A remote file opened for reading; every read is one HTTP range request."""

    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0  # r.url: after the CDN redirect

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req, timeout=300).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(spec: dict, dst: Path):
    tmp = dst.with_suffix(dst.suffix + ".part")
    if spec.get("member"):
        with zipfile.ZipFile(io.BufferedReader(HttpFile(spec["url"]), buffer_size=CHUNK)) as z:
            with z.open(spec["member"]) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
    else:
        with urllib.request.urlopen(spec["url"], timeout=300) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, CHUNK)
    got = sha256(tmp)
    if spec.get("sha256") and got != spec["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"{dst}: sha256 {got}, expected {spec['sha256']}")
    tmp.replace(dst)
    return got


def load_cfg(name: str) -> dict:
    p = VOICES_DIR / name / "voice.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voices", nargs="*")
    ap.add_argument("--check", action="store_true", help="only verify files on disk against the recorded sha256")
    args = ap.parse_args()

    names = args.voices or sorted(p.name for p in VOICES_DIR.iterdir() if (p / "voice.json").exists())
    failed = 0
    for name in names:
        cfg = load_cfg(name)
        src = cfg.get("source")
        if not src:
            print(f"{name}: no source in voice.json")
            continue
        for fn, spec in src["files"].items():
            dst = VOICES_DIR / name / fn
            if args.check:
                ok = dst.exists() and sha256(dst) == spec.get("sha256")
                failed += not ok
                print(f"{name}/{fn}: {'ok' if ok else 'MISSING' if not dst.exists() else 'DIFFERENT'}")
            elif not dst.exists():
                print(f"{name}/{fn}: downloading", flush=True)
                try:
                    got = download(spec, dst)
                except Exception as e:
                    failed += 1
                    print(f"{name}/{fn}: {e}")
                    continue
                if not spec.get("sha256"):
                    # a newly added voice: pin what we got, so later downloads must match it
                    spec["sha256"] = got
                    (VOICES_DIR / name / "voice.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                                                 encoding="utf-8")
                if fn == "model.index":
                    compact(name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
