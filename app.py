"""Chat simulator + TTS queue + live metrics, with a web UI.

    python app.py                       # http://127.0.0.1:8765
    python app.py --rvc-device cpu --audio-device "CABLE Input"
    python app.py --host 0.0.0.0 --output web   # TTS box: OBS on another PC plays http://<this box>:8765/overlay
"""

import argparse
import csv
import json
import os
import random
import socket
import statistics
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import psutil

import chat_sim
from audio_player import AudioPlayer, WebAudio
from metrics import MetricsRecorder

ROOT = Path(__file__).resolve().parent
MAX_TTS_CHARS = 300


class App:
    def __init__(self, args):
        # imported here, not at module level: the audio player process re-imports this module on spawn,
        # and it should not pull in torch
        from tts_pipeline import list_voices

        self.args = args
        self.voices = list_voices()
        self.settings = {
            "sim_running": False,
            "speed": 1.0,           # chat pace multiplier; 1.0 is ~10 messages/min
            "tts_enabled": True,
            "voice": "per_user",    # each chatter has a fixed voice; or "random", or a voice name
            "max_queue": 8,
        }
        self.status = "loading models..."
        self.pipe = None
        self.messages: list[dict] = []
        self.next_id = 1
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.synth_q: deque[dict] = deque()
        self.synth_cv = threading.Condition(self.lock)
        self.play_q: deque[tuple[dict, object, int]] = deque()
        self.play_cv = threading.Condition(self.lock)
        self.busy = False
        self.dropped = 0

        stamp = time.strftime("%Y%m%d_%H%M%S")
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        self.msg_log_path = logs / f"messages_{stamp}.csv"
        self.msg_log = open(self.msg_log_path, "w", newline="", encoding="utf-8")
        self.msg_writer = csv.writer(self.msg_log)
        self.msg_writer.writerow(["time", "user", "voice", "text", "status", "queue_wait_s", "tts_s", "rvc_s",
                                  "audio_s", "latency_s", "device"])
        self.metrics = MetricsRecorder(logs / f"metrics_{stamp}.csv", self._state_for_metrics)
        self.audio = WebAudio() if args.output == "web" else AudioPlayer(args.audio_device)

        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._simulate, daemon=True).start()
        threading.Thread(target=self._synth_worker, daemon=True).start()
        threading.Thread(target=self._player, daemon=True).start()

    def _state_for_metrics(self) -> dict:
        return {"tts_enabled": int(self.settings["tts_enabled"]), "busy": int(self.busy),
                "queue": len(self.synth_q) + len(self.play_q)}

    def _load(self):
        from tts_pipeline import Pipeline

        t0 = time.perf_counter()
        pipe = Pipeline(rvc_device=self.args.rvc_device, en_engine=self.args.en_engine, threads=self.args.threads,
                        vram_reserve_mb=self.args.vram_reserve_mb)
        self.status = "warming up, loading all voices..."
        pipe.warmup(*self.voices)
        self.pipe = pipe
        cap = f", VRAM cap {pipe.vc.vram_cap_mb} MB, CPU fallback below {pipe.vc.vram_reserve_mb} MB free"             if pipe.vc.gpu else ""
        self.status = f"ready in {time.perf_counter() - t0:.0f}s, RVC on {pipe.vc.device}, EN {pipe.tts.en_engine}{cap}"

    # --- chat -------------------------------------------------------------

    def add_message(self, user: str, text: str, voice: str | None = None):
        if text.startswith("!"):
            cmd, _, rest = text[1:].partition(" ")
            if cmd in self.voices and rest.strip():
                voice, text = cmd, rest.strip()
        if not voice:
            voice = self.settings["voice"]
            if voice == "per_user":
                voice = chat_sim.VOICE_OF.get(user)
                if voice not in self.voices:
                    voice = self.voices[hash(user) % len(self.voices)]
            elif voice == "random":
                voice = random.choice(self.voices)
        with self.lock:
            msg = {"id": self.next_id, "user": user, "text": text, "voice": voice, "t_in": time.time(),
                   "status": "queued"}
            self.next_id += 1
            self.messages.append(msg)
            del self.messages[:-200]
            if not self.settings["tts_enabled"]:
                msg["status"] = "muted"
                self._log(msg)
                return
            self.synth_q.append(msg)
            while len(self.synth_q) + len(self.play_q) > self.settings["max_queue"]:
                old = self.synth_q.popleft()
                old["status"] = "dropped"
                self.dropped += 1
                self._log(old)
            self.synth_cv.notify()

    def add_event(self, text: str):
        """A game moment the chat reacts to; shown in chat, not voiced."""
        with self.lock:
            self.messages.append({"id": self.next_id, "user": "", "text": text, "voice": "", "t_in": time.time(),
                                  "status": "event"})
            self.next_id += 1

    def _simulate(self):
        sim = None
        while True:
            time.sleep(0.2)
            if not (self.settings["sim_running"] and self.pipe):
                sim = None  # paused: start fresh on resume instead of replaying a backlog
                continue
            now = time.time()
            if sim is None:
                sim = chat_sim.ChatSimulator(now, self.settings["speed"])
            sim.speed = self.settings["speed"]
            for kind, user, text in sim.poll(now):
                if kind == "event":
                    self.add_event(text)
                else:
                    self.add_message(user, text)

    # --- tts: synthesis overlaps with playback of the previous message -----

    def _synth_worker(self):
        while True:
            with self.synth_cv:
                while not self.synth_q or not self.pipe:
                    self.synth_cv.wait(timeout=0.5)
                msg = self.synth_q.popleft()
                msg["status"] = "synth"
                self.busy = True
            msg["t_synth"] = time.time()
            try:
                text = chat_sim.tts_text(msg["text"])[:MAX_TTS_CHARS]
                audio, sr, t = self.pipe.say(text, msg["voice"])
                msg.update(tts_s=round(t.tts, 2), rvc_s=round(t.rvc, 2), audio_s=round(t.audio_sec, 1),
                           device=t.extra.get("device", ""))
            except Exception as e:
                msg["status"] = f"error: {e}"
                self.busy = False
                self._log(msg)
                continue
            self.busy = False
            with self.play_cv:
                if audio.size:
                    msg["status"] = "ready"
                    self.play_q.append((msg, audio, sr))
                    self.play_cv.notify()
                else:
                    msg["status"] = "empty"
                    self._log(msg)

    def _player(self):
        while True:
            with self.play_cv:
                while not self.play_q:
                    self.play_cv.wait()
                msg, audio, sr = self.play_q.popleft()
            self.audio.play(msg["id"], audio, sr)
            while True:
                kind, msg_id, err = self.audio.next_event()
                if msg_id != msg["id"]:
                    continue
                if kind == "playing":
                    msg["t_play"] = time.time()
                    msg["latency_s"] = round(msg["t_play"] - msg["t_in"], 2)
                    msg["status"] = "playing"
                    continue
                msg["status"] = "done" if kind == "done" else f"audio error: {err}"
                break
            self._log(msg)

    def _log(self, m: dict):
        wait = round(m["t_synth"] - m["t_in"], 2) if "t_synth" in m else ""
        with self.log_lock:
            self.msg_writer.writerow([time.strftime("%H:%M:%S", time.localtime(m["t_in"])), m["user"], m["voice"],
                                      m["text"], m["status"], wait, m.get("tts_s", ""), m.get("rvc_s", ""),
                                      m.get("audio_s", ""), m.get("latency_s", ""), m.get("device", "")])
            self.msg_log.flush()

    # --- api --------------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            msgs = [{k: v for k, v in m.items() if not k.startswith("t_")} for m in self.messages[-80:]]
            lat = [m["latency_s"] for m in self.messages if "latency_s" in m]
            synth = [m["tts_s"] + m["rvc_s"] for m in self.messages if "rvc_s" in m]
            stats = {
                "voiced": len(lat),
                "dropped": self.dropped,
                "cpu_fallbacks": sum(1 for m in self.messages if str(m.get("device", "")).startswith("cpu")),
                "queue": len(self.synth_q) + len(self.play_q),
                "lat_avg": round(statistics.mean(lat), 2) if lat else None,
                "lat_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
                "lat_max": round(max(lat), 2) if lat else None,
                "synth_avg": round(statistics.mean(synth), 2) if synth else None,
                "synth_max": round(max(synth), 2) if synth else None,
            }
        return {"status": self.status, "settings": self.settings, "voices": self.voices, "messages": msgs,
                "users": chat_sim.VOICE_OF, "output": {**self.audio.info(), "lan_ip": lan_ip()},
                "stats": stats, "metrics": self.metrics.latest, "history": self.metrics.history[-120:],
                "logs": [str(self.msg_log_path.name), str(self.msg_log_path.name).replace("messages", "metrics")]}

    def update_settings(self, data: dict):
        for k in ("sim_running", "tts_enabled"):
            if k in data:
                self.settings[k] = bool(data[k])
        if "speed" in data:
            self.settings["speed"] = min(4.0, max(0.25, float(data["speed"])))
        if "max_queue" in data:
            self.settings["max_queue"] = min(50, max(1, int(data["max_queue"])))
        if data.get("voice") in ["per_user", "random", *self.voices]:
            self.settings["voice"] = data["voice"]


def make_handler(app: App):
    pages = {"/": ROOT / "static" / "index.html", "/overlay": ROOT / "static" / "overlay.html"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in pages:
                self._send(200, pages[url.path].read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(app.state())
            elif url.path == "/api/audio/next" and isinstance(app.audio, WebAudio):
                q = parse_qs(url.query)
                after = int((q.get("after") or ["0"])[0])
                clip = app.audio.next_clip(after, (q.get("client") or [self.client_address[0]])[0][:64])
                if clip is None:
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                msg_id, wav = clip
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Msg-Id", str(msg_id))
                self.end_headers()
                self.wfile.write(wav)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if self.path == "/api/settings":
                app.update_settings(data)
                self._json(app.settings)
            elif self.path == "/api/audio/event" and isinstance(app.audio, WebAudio):
                app.audio.report(int(data.get("id", 0)), str(data.get("kind", "")), data.get("error"))
                self._json({"ok": True})
            elif self.path == "/api/send":
                text = str(data.get("text", "")).strip()[:500]
                if text:
                    app.add_message(str(data.get("user") or "you")[:25], text)
                self._json({"ok": bool(text)})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--en-engine", choices=["kokoro", "silero"], default="silero",
                    help="English base TTS: kokoro sounds better but takes ~0.5 GB of VRAM; silero runs on the CPU")
    ap.add_argument("--output", choices=["local", "web"], default="local",
                    help="local: this machine's sound card; web: overlay pages at /overlay (OBS Browser Source)")
    ap.add_argument("--audio-device", default=None, help="output device name/index (see say.py --list-audio-devices)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the UI and /overlay from other machines")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--vram-reserve-mb", type=int, default=1024,
                    help="if less VRAM than this is free (game included), voice the message on the CPU")
    ap.add_argument("--priority", choices=["normal", "below_normal"], default="below_normal",
                    help="CPU priority of this process; below_normal lets the game win CPU contention")
    args = ap.parse_args()

    if args.priority == "below_normal":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)

    app = App(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    host = lan_ip() if args.host == "0.0.0.0" else args.host
    print(f"UI: http://{host}:{args.port}", file=sys.stderr, flush=True)
    if args.output == "web":
        print(f"OBS Browser Source: http://{host}:{args.port}/overlay", file=sys.stderr, flush=True)
    server.serve_forever()


def lan_ip() -> str:
    """This machine's address on the local network (no packets are sent: UDP connect only picks a route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
