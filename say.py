"""Speak text in an RVC voice.

    python say.py -v sidorovich "Привет, сталкер! Welcome to the Zone"
    python say.py -v skyrim_guard --rvc-device cpu --out hello.wav "I used to be an adventurer like you"
    python say.py -v sidorovich --play          # interactive: one line = one message, "voice: text" switches voice
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

from tts_pipeline import Pipeline, list_voices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="message; omit for interactive mode (reads stdin)")
    ap.add_argument("-v", "--voice", default=None)
    ap.add_argument("--rvc-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--en-engine", choices=["kokoro", "silero"], default=None,
                    help="English base TTS (default silero; kokoro sounds better, ~0.5 GB of VRAM)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (leave some for the game)")
    ap.add_argument("--out", help="output .wav (single message) or folder (interactive)")
    ap.add_argument("--play", action="store_true", help="play to an audio device")
    ap.add_argument("--audio-device", default=None, help="output device name/index for --play (e.g. CABLE Input)")
    ap.add_argument("--raw", action="store_true", help="skip RVC, base TTS only")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--list-audio-devices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        print("\n".join(list_voices()))
        return
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    voice = args.voice or list_voices()[0]
    pipe = Pipeline(rvc_device=args.rvc_device, en_engine=args.en_engine, threads=args.threads)
    print(f"RVC on {pipe.vc.device}, voices: {', '.join(list_voices())}", file=sys.stderr)

    def speak(text: str, voice: str, out_path: str | None):
        audio, sr, t = pipe.say(text, voice, raw=args.raw)
        print(f"[{voice}] {t}", file=sys.stderr)
        if out_path:
            sf.write(out_path, audio, sr)
            print(out_path, file=sys.stderr)
        if args.play and audio.size:
            import sounddevice as sd
            sd.play(audio, sr, device=args.audio_device, blocking=True)

    if args.text:
        speak(args.text, voice, args.out or (None if args.play else "out.wav"))
        return

    pipe.warmup(voice)
    print("ready", file=sys.stderr)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        head, sep, rest = line.partition(":")
        if sep and head.strip() in list_voices():
            voice, line = head.strip(), rest.strip()
        n += 1
        speak(line, voice, str(out_dir / f"{n:04d}_{voice}.wav") if out_dir else None)


if __name__ == "__main__":
    main()
