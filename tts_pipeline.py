"""Text -> base TTS (Silero RU / Kokoro EN) -> RVC voice conversion."""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VOICES_DIR = ROOT / "voices"

BASE_SR = 24000  # Silero and Kokoro both synthesize at 24 kHz
SPEECH_PAUSE_SEC = 0.12  # after each language run of a message

SILERO_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_6_ru.pt"
SILERO_EN_MODEL_URL = "https://models.silero.ai/models/tts/en/v3_en.pt"
RMVPE_URL = "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/rmvpe.pt"

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
URL = re.compile(r"https?://\S+|www\.\S+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
PLUSES = re.compile(r"\++")
REPEATED_SHORT = re.compile(r"\b([А-Яа-яЁё]{1,3})(?:\s+\1\b)+", re.IGNORECASE)
RU_VOWEL = re.compile(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]")


@dataclass
class Voice:
    name: str
    model: Path
    index: Path | None
    pitch: int = 0                # semitones, applied by RVC
    index_rate: float = 0.66      # how strongly to pull timbre from the index
    protect: float = 0.33         # protects consonants/breaths from artifacts
    ru_speaker: str = "eugene"    # Silero base voice
    en_speaker: str = "am_michael"  # Kokoro base voice
    en_silero_speaker: str = "en_0"  # Silero EN base voice (CPU-only mode)

    @classmethod
    def load(cls, name: str) -> "Voice":
        d = VOICES_DIR / name
        cfg = {}
        if (d / "voice.json").exists():
            cfg = json.loads((d / "voice.json").read_text(encoding="utf-8"))
        index = d / "model.compact.index"  # see compact_index.py
        if not index.exists():
            index = d / "model.index"
        return cls(
            name=name,
            model=d / "model.pth",
            index=index if index.exists() else None,
            **cfg,
        )


def list_voices() -> list[str]:
    return sorted(p.name for p in VOICES_DIR.iterdir() if (p / "model.pth").exists())


def split_by_language(text: str, default: str = "ru") -> list[tuple[str, str]]:
    """Split a chat message into (lang, text) runs by script: Cyrillic -> ru, Latin -> en.

    Tokens without letters (numbers, punctuation) stick to the current run.
    """
    segments: list[tuple[str, str]] = []
    cur_lang, buf = None, []
    for tok in re.findall(r"\S+|\s+", text):
        lang = "ru" if CYRILLIC.search(tok) else "en" if LATIN.search(tok) else None
        if lang and cur_lang and lang != cur_lang:
            segments.append((cur_lang, "".join(buf).strip()))
            buf = []
        if lang:
            cur_lang = lang
        buf.append(tok)
    if buf:
        segments.append((cur_lang or default, "".join(buf).strip()))
    return [(lang, t) for lang, t in segments if re.search(r"[\w+]", t)]  # "+" alone is read as "плюс"


def normalize(text: str, lang: str) -> str:
    from num2words import num2words

    def spell(m: re.Match) -> str:
        try:
            return num2words(m.group().replace(",", "."), lang=lang)
        except Exception:
            return m.group()

    def plus(m: re.Match) -> str:
        # a single "+" before a vowel, at a word start or inside a word, is Silero's stress mark ("т+орты")
        before = m.string[m.start() - 1] if m.start() else " "
        after = m.string[m.end()] if m.end() < len(m.string) else ""
        if lang == "ru" and len(m.group()) == 1 and RU_VOWEL.match(after) and (before.isspace() or CYRILLIC.match(before)):
            return "+"
        word = " плюс " if lang == "ru" else " plus "
        # "C++", "Notepad++": each one is read; a standalone "+++" (agreeing in chat) once
        return word * len(m.group()) if before.isalnum() else word

    def percent(m: re.Match) -> str:
        num = m.group(1)
        if lang != "ru":
            return f"{num} percent"
        if not num.isdigit():
            word = "процента"  # 2,5 процента
        elif num[-2:] in ("11", "12", "13", "14") or num[-1] in "056789":
            word = "процентов"
        else:
            word = "процент" if num[-1] == "1" else "процента"
        return f"{num} {word}"

    text = URL.sub(" ссылка " if lang == "ru" else " link ", text)
    if lang == "ru":
        # "го го го": Silero reads a run of short words as one word ending in "-го" ("во го го"); commas keep them apart
        text = REPEATED_SHORT.sub(lambda m: ", ".join(m.group().split()), text)
    text = PERCENT.sub(percent, text)
    text = PLUSES.sub(plus, text)
    return NUMBER.sub(spell, text)


def load_silero(url: str):
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / Path(url).name
    if not path.exists():
        torch.hub.download_url_to_file(url, str(path))
    model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    return model


class BaseTTS:
    """Silero for Russian (always CPU, it is fast enough).

    English: Kokoro (better intonation; slow on CPU, ~0.1s on GPU) or Silero EN (fast on CPU).
    """

    def __init__(self, en_engine: str = "kokoro", kokoro_device: str = "cuda"):
        self.silero = load_silero(SILERO_MODEL_URL)
        self.en_engine = en_engine
        if en_engine == "kokoro":
            from kokoro import KPipeline
            if kokoro_device == "cuda" and not torch.cuda.is_available():
                kokoro_device = "cpu"
            self.kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=kokoro_device)
        else:
            self.silero_en = load_silero(SILERO_EN_MODEL_URL)

    def synth(self, lang: str, text: str, voice: Voice) -> np.ndarray:
        text = normalize(text, lang)
        if lang == "ru":
            # Silero only knows Cyrillic and basic punctuation
            text = re.sub(r"[^А-Яа-яЁё0-9 .,!?:;\-—+]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return np.zeros(0, dtype=np.float32)
            audio = self.silero.apply_tts(text=text, speaker=voice.ru_speaker, sample_rate=BASE_SR)
            return audio.cpu().numpy().astype(np.float32)
        if self.en_engine == "silero":
            text = re.sub(r"[^A-Za-z0-9 .,!?:;'\-]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return np.zeros(0, dtype=np.float32)
            audio = self.silero_en.apply_tts(text=text, speaker=voice.en_silero_speaker, sample_rate=BASE_SR)
            return audio.cpu().numpy().astype(np.float32)
        chunks = [a.cpu().numpy() for _, _, a in self.kokoro(text, voice=voice.en_speaker) if a is not None]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    def synth_message(self, text: str, voice: Voice) -> np.ndarray:
        pause = np.zeros(int(BASE_SR * SPEECH_PAUSE_SEC), dtype=np.float32)
        parts = []
        for lang, seg in split_by_language(text):
            audio = self.synth(lang, seg, voice)
            if audio.size:
                parts += [audio, pause]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


RVC_EDGE_PAD_SEC = 0.3

# RVC's content encoder (ContentVec) needs neighbouring sounds: alone, a short message loses its consonants
# ("ахаха" -> "асаса", "го, го, го" -> "дадада"), the same words inside a phrase are fine. So a short message goes
# through RVC after a neutral phrase, which is cut off again.
CONTEXT_BELOW_SEC = 2.5
CONTEXT_CARRIER = {"ru": "Ну ты даёшь.", "en": "Oh, come on."}


def _patch_rvc_padding():
    """infer_rvc_python pads every clip with 1s (CPU) or 3s (GPU) of reflected audio on each side.

    That is meant for long songs; for 1-10s chat messages it multiplies the work, so trim it.
    """
    from infer_rvc_python import root_pipe

    if getattr(root_pipe.VC, "_short_pad", False):
        return
    orig_init = root_pipe.VC.__init__

    def init(self, tgt_sr, config):
        orig_init(self, tgt_sr, config)
        self.t_pad = int(self.sr * RVC_EDGE_PAD_SEC)
        self.t_pad_tgt = int(tgt_sr * RVC_EDGE_PAD_SEC)
        self.t_pad2 = self.t_pad * 2

    root_pipe.VC.__init__ = init
    root_pipe.VC._short_pad = True


class VoiceConverter:
    """RVC on the GPU, falling back to the CPU for a message when VRAM is short.

    Running out of VRAM next to a game is the dangerous case: Windows starts paging GPU memory to RAM and both
    the game and the TTS crawl (a 22 s stall in testing). So before each message we check free VRAM and, if it is
    below `vram_reserve_mb`, convert that message on the CPU instead: slower (~2x real time), but the game keeps
    its memory. A PyTorch allocation cap (see `cap_vram`) turns any unexpected growth into an OOM error, which
    also falls back to the CPU instead of eating the game's VRAM.
    """

    CPU_VOICES_KEPT = 3  # CPU copies of voices are fp32 (~110 MB each); keep only the recently used ones

    def __init__(self, device: str = "cuda", vram_reserve_mb: int = 1024):
        from infer_rvc_python import BaseLoader

        _patch_rvc_padding()

        rmvpe_path = MODELS_DIR / "rmvpe.pt"
        if not rmvpe_path.exists():
            MODELS_DIR.mkdir(exist_ok=True)
            torch.hub.download_url_to_file(RMVPE_URL, str(rmvpe_path))
        self.loader = BaseLoader(only_cpu=(device == "cpu"), rmvpe_path=str(rmvpe_path))
        self.device = self.loader.config.device
        self.gpu = self.device != "cpu"
        # keep idle voice models in RAM and move one to the GPU only while it speaks:
        # VRAM then stays at one voice no matter how many are loaded
        self.offload = self.gpu
        self.resident: str | None = None
        # BaseLoader keeps only one voice loaded and re-reads model + index (hundreds of MB) on every switch.
        # Keep each voice's loaded state and swap it back in instead.
        self.loaded: dict[str, tuple[dict, dict]] = {}

        self.vram_reserve_mb = vram_reserve_mb
        self.cpu_loader = BaseLoader(only_cpu=True, rmvpe_path=str(rmvpe_path)) if self.gpu else None
        self.cpu_voices: dict[str, tuple[dict, dict]] = {}  # insertion order = LRU order
        self.nvml = None
        if self.gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml = None

    def free_vram_mb(self) -> float | None:
        """Free VRAM on the card across all processes (the game included)."""
        if not self.nvml:
            return None
        import pynvml
        return pynvml.nvmlDeviceGetMemoryInfo(self.nvml).free / 2**20

    def cap_vram(self, headroom_mb: int = 400):
        """Cap PyTorch's allocator at what warmup needed plus headroom; going over raises OOM instead of growing."""
        if not self.gpu:
            return
        total = torch.cuda.get_device_properties(0).total_memory
        cap = torch.cuda.max_memory_reserved() + headroom_mb * 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
        self.vram_cap_mb = round(cap / 2**20)

    def _load(self, voice: Voice):
        # the index is attached by hand below, so BaseLoader doesn't build its float32 copy of all vectors
        self.loader.apply_conf(
            tag=voice.name,
            file_model=str(voice.model),
            pitch_algo="rmvpe+",
            pitch_lvl=voice.pitch,
            file_index="",
            index_influence=voice.index_rate,
            consonant_breath_protection=voice.protect,
        )
        self.loader.model_vc = {}
        self.loader.generate_from_cache(audio_data=(np.zeros(BASE_SR // 2, dtype=np.float32), BASE_SR),
                                        tag=voice.name)
        vc = self.loader.model_vc
        vc["cpt"]["weight"] = None  # the weights already live in net_g; don't keep a second copy
        # weight norm keeps derived weights as plain attributes that .to() doesn't move (~40 MB of VRAM per voice
        # left behind after offloading); folding it in is the standard inference step anyway
        vc["net_g"].dec.remove_weight_norm()
        vc["net_g"].flow.remove_weight_norm()
        if voice.index:
            import faiss
            index = faiss.read_index(str(voice.index))
            vc["index"], vc["big_npy"], vc["index_rate"] = index, IndexVectors(index), voice.index_rate
        if self.offload:
            vc["net_g"] = PinnedModule(vc["net_g"])
            torch.cuda.empty_cache()
        self.loaded[voice.name] = (vc, self.loader.cache_model)

    def convert(self, audio: np.ndarray, sr: int, voice: Voice, force_cpu: bool = False) -> tuple[np.ndarray, int, str]:
        """Returns audio, sample rate and where it ran: "gpu", "cpu" or "cpu-oom" (GPU ran out of its cap)."""
        if voice.name not in self.loaded:
            self._load(voice)
        if not self.gpu:
            self.loader.model_vc, self.loader.cache_model = self.loaded[voice.name]
            return *self.loader.generate_from_cache(audio_data=(audio, sr), tag=voice.name), "cpu"
        free = self.free_vram_mb()
        if force_cpu or (free is not None and free < self.vram_reserve_mb):
            return *self._convert_cpu(audio, sr, voice), "cpu"
        try:
            return *self._convert_gpu(audio, sr, voice), "gpu"
        except torch.OutOfMemoryError:
            self._evict_resident()
            return *self._convert_cpu(audio, sr, voice), "cpu-oom"

    def _convert_gpu(self, audio, sr, voice):
        self.loader.model_vc, self.loader.cache_model = self.loaded[voice.name]
        if self.resident != voice.name:
            # keep only the last used voice on the GPU: people often send several messages in a row
            self._evict_resident()
            self.loader.model_vc["net_g"].to_device(self.device)
            self.resident = voice.name
        try:
            return self.loader.generate_from_cache(audio_data=(audio, sr), tag=voice.name)
        finally:
            torch.cuda.empty_cache()  # hand cached activations back so the game can use them

    def _evict_resident(self):
        if self.resident:
            self.loaded[self.resident][0]["net_g"].to_cpu()
            self.resident = None
        torch.cuda.empty_cache()

    def _convert_cpu(self, audio, sr, voice):
        state, cache = self._cpu_voice(voice)
        self.cpu_loader._load_rmvpe()
        state["pipe"].model_rmvpe = self.cpu_loader.model_pitch_estimator
        self.cpu_loader.model_vc, self.cpu_loader.cache_model = state, cache
        return self.cpu_loader.generate_from_cache(audio_data=(audio, sr), tag=voice.name)

    def _cpu_voice(self, voice: Voice) -> tuple[dict, dict]:
        """CPU (fp32) twin of a loaded voice, built from its pinned weights; shares the index."""
        import copy

        from infer_rvc_python.root_pipe import VC

        name = voice.name
        if name in self.cpu_voices:
            self.cpu_voices[name] = self.cpu_voices.pop(name)  # mark as recently used
            return self.cpu_voices[name]
        vc, cache = self.loaded[name]
        if self.resident == name:
            self._evict_resident()
        net_g = copy.deepcopy(vc["net_g"].module).float().eval()  # weights point at pinned RAM, so this copies RAM
        for m in net_g.modules():
            if hasattr(m, "is_half"):
                m.is_half = False
        state = dict(vc, net_g=net_g, pipe=VC(vc["tgt_sr"], self.cpu_loader.config))
        self.cpu_loader.model_config[name] = dict(self.loader.model_config[name])
        self.cpu_voices[name] = (state, copy.deepcopy(cache))
        while len(self.cpu_voices) > self.CPU_VOICES_KEPT:
            self.cpu_voices.pop(next(iter(self.cpu_voices)))
        return self.cpu_voices[name]


class PinnedModule:
    """A model whose weights live in pinned RAM and are copied to the GPU only while in use.

    Moving back to RAM is free: the parameters are just pointed at the pinned copy again.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module.to("cpu")
        self.tensors = [t for _, t in module.named_parameters()] + [t for _, t in module.named_buffers()]
        self.pinned = [t.data.pin_memory() for t in self.tensors]
        self.to_cpu()

    def to_device(self, device):
        for t, p in zip(self.tensors, self.pinned):
            t.data = p.to(device, non_blocking=True)
        torch.cuda.synchronize()

    def to_cpu(self):
        for t, p in zip(self.tensors, self.pinned):
            t.data = p

    def __getattr__(self, name):
        return getattr(self.module, name)


class IndexVectors:
    """Stands in for RVC's `big_npy` (a float32 copy of every index vector): reads vectors from the index itself."""

    def __init__(self, index):
        self.index = index
        index.make_direct_map()

    def __getitem__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        flat = ids.reshape(-1).copy()
        flat[flat < 0] = self.index.ntotal - 1  # faiss pads missing neighbours with -1; numpy big_npy[-1] = last row
        return self.index.reconstruct_batch(flat).reshape(*ids.shape, -1)


@dataclass
class Timings:
    tts: float = 0.0
    rvc: float = 0.0
    audio_sec: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"tts {self.tts:.2f}s + rvc {self.rvc:.2f}s = {self.tts + self.rvc:.2f}s "
                f"for {self.audio_sec:.1f}s of audio")


class Pipeline:
    def __init__(self, rvc_device: str = "cuda", en_engine: str | None = None, threads: int = 4,
                 vram_reserve_mb: int = 1024):
        # English defaults to Silero on the CPU: Kokoro sounds better but costs ~0.5 GB of VRAM next to the game
        # (and is too slow on the CPU). RVC replaces the timbre anyway.
        self.tts = BaseTTS(en_engine=en_engine or "silero", kokoro_device=rvc_device)
        self.vc = VoiceConverter(rvc_device, vram_reserve_mb=vram_reserve_mb)
        # set after loading: one of the model packages resets torch to a single thread
        torch.set_num_threads(threads)
        self.voices: dict[str, Voice] = {}
        self.carriers: dict[tuple, np.ndarray] = {}

    def _carrier(self, text: str, voice: Voice) -> np.ndarray:
        """Base audio of a neutral phrase in the message's first language and the voice's base speaker, cached."""
        segments = split_by_language(text)
        lang = segments[0][0] if segments else "ru"
        speaker = voice.ru_speaker if lang == "ru" else (voice.en_speaker, voice.en_silero_speaker)
        key = (lang, speaker)
        if key not in self.carriers:
            self.carriers[key] = self.tts.synth_message(CONTEXT_CARRIER[lang], voice)  # ends with a pause
        return self.carriers[key]

    def warmup(self, *voice_names: str):
        """First calls are slow (JIT compilation, CUDA kernels, HuBERT/RMVPE loading); also loads every voice.

        Also preloads the CPU fallback (HuBERT/RMVPE on the CPU) so the first fallback isn't a multi-second stall,
        then caps PyTorch's VRAM at what this needed plus headroom.
        """
        for _ in range(2):
            self.say("Прогрев. Warm up.", voice_names[0])
        for name in voice_names[1:]:
            self.say("Прогрев. Warm up.", name)
        if self.vc.gpu:
            self.say("Прогрев.", voice_names[0], force_cpu=True)
            self.vc.cap_vram()

    def voice(self, name: str) -> Voice:
        if name not in self.voices:
            self.voices[name] = Voice.load(name)
        return self.voices[name]

    def say(self, text: str, voice_name: str, raw: bool = False,
            force_cpu: bool = False) -> tuple[np.ndarray, int, Timings]:
        """Returns int16 audio, sample rate and timings (t.extra["device"]: where RVC ran).

        raw=True skips RVC (base TTS only).
        """
        voice = self.voice(voice_name)
        t = Timings()
        t0 = time.perf_counter()
        base = self.tts.synth_message(text, voice)
        t.tts = time.perf_counter() - t0
        if not base.size:
            return np.zeros(0, dtype=np.int16), BASE_SR, t
        if raw:
            t.audio_sec = base.size / BASE_SR
            return (np.clip(base, -1, 1) * 32767).astype(np.int16), BASE_SR, t
        carrier = self._carrier(text, voice) if base.size < CONTEXT_BELOW_SEC * BASE_SR else None
        t0 = time.perf_counter()
        if carrier is None:
            out, sr, t.extra["device"] = self.vc.convert(base, BASE_SR, voice, force_cpu=force_cpu)
        else:
            out, sr, t.extra["device"] = self.vc.convert(np.concatenate([carrier, base]), BASE_SR, voice,
                                                         force_cpu=force_cpu)
            # RVC keeps durations to within ~1%: cut in the middle of the carrier's trailing pause
            ratio = out.size / sr / ((carrier.size + base.size) / BASE_SR)
            out = out[round((carrier.size / BASE_SR - SPEECH_PAUSE_SEC / 2) * ratio * sr):]
            fade = min(out.size, int(0.01 * sr))
            out = out.copy()
            out[:fade] = (out[:fade] * np.linspace(0, 1, fade)).astype(out.dtype)
            t.extra["context"] = True
        t.rvc = time.perf_counter() - t0
        t.audio_sec = out.size / sr
        return out, sr, t
