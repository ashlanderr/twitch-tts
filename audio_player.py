"""Audio output: a local sound card (AudioPlayer) or the network (WebAudio, for OBS on another PC).

Both take clips with play() and report ("playing" | "done" | "error", msg_id, error) through next_event().
"""

import io
import multiprocessing as mp
import queue
import threading
import time


def _player_loop(jobs: mp.Queue, events: mp.Queue, device):
    import sounddevice as sd

    while True:
        job = jobs.get()
        if job is None:
            return
        msg_id, audio, sr = job
        events.put(("playing", msg_id, None))
        try:
            sd.play(audio, sr, device=device, blocking=True, latency="high")
            events.put(("done", msg_id, None))
        except Exception as e:
            events.put(("error", msg_id, str(e)))


class AudioPlayer:
    """Plays on this machine, in a separate process, so synthesis of the next message can't starve the audio callback."""

    def __init__(self, device=None):
        ctx = mp.get_context("spawn")
        self.jobs = ctx.Queue()
        self.events = ctx.Queue()
        self.proc = ctx.Process(target=_player_loop, args=(self.jobs, self.events, device), daemon=True)
        self.proc.start()

    def play(self, msg_id: int, audio, sr: int):
        self.jobs.put((msg_id, audio, sr))

    def next_event(self):
        """Blocks until the player reports ("playing" | "done" | "error", msg_id, error)."""
        return self.events.get()

    def info(self) -> dict:
        return {"mode": "local"}


class WebAudio:
    """Hands clips to overlay pages (/overlay, an OBS Browser Source on the streaming PC) over HTTP.

    The page long-polls for the next clip, plays it and reports back when it starts and ends. If no page picks a clip
    up, or a page never reports the end, the clip is given up on, so the queue never stalls on a closed OBS.
    """

    START_TIMEOUT = 5.0   # s for a page to start playing a clip
    END_GRACE = 5.0       # s past the clip's length before it counts as done anyway
    LISTENER_TTL = 30.0   # a page that polled this recently counts as connected (polls wait up to POLL_WAIT)
    POLL_WAIT = 20.0

    def __init__(self):
        self.cv = threading.Condition()
        self.clip: tuple[int, bytes] | None = None
        self.pending: dict[int, dict] = {}  # msg_id -> {"playing": Event, "done": Event}
        self.events: queue.Queue = queue.Queue()
        self.listeners: dict[str, float] = {}  # page id -> last poll

    def play(self, msg_id: int, audio, sr: int):
        import soundfile as sf

        if not self.connected():
            self.events.put(("error", msg_id, "no overlay connected"))
            return
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        p = {"playing": threading.Event(), "done": threading.Event()}
        with self.cv:
            self.pending[msg_id] = p
            self.clip = (msg_id, buf.getvalue())
            self.cv.notify_all()
        threading.Thread(target=self._watch, args=(msg_id, p, len(audio) / sr), daemon=True).start()

    def next_event(self):
        return self.events.get()

    def _watch(self, msg_id: int, p: dict, duration: float):
        if not p["playing"].wait(self.START_TIMEOUT):
            self.report(msg_id, "error", "overlay did not start playing")
        elif not p["done"].wait(duration + self.END_GRACE):
            self.report(msg_id, "done")

    def report(self, msg_id: int, kind: str, error: str | None = None):
        """Called by the overlay page (and the watchdog); the first report of each kind wins."""
        with self.cv:
            p = self.pending.get(msg_id)
            if not p:
                return
            if kind == "playing":
                if p["playing"].is_set():
                    return
                p["playing"].set()
            else:
                kind = "done" if kind == "done" else "error"
                p["playing"].set()
                p["done"].set()
                del self.pending[msg_id]
                if self.clip and self.clip[0] == msg_id:
                    self.clip = None
        self.events.put((kind, msg_id, error))

    def next_clip(self, after: int, client: str) -> tuple[int, bytes] | None:
        """Long poll: the clip waiting to be played if it is newer than `after`, else None after POLL_WAIT."""
        self.listeners[client] = time.time()
        with self.cv:
            self.cv.wait_for(lambda: self.clip and self.clip[0] > after, timeout=self.POLL_WAIT)
            clip = self.clip if self.clip and self.clip[0] > after else None
        self.listeners[client] = time.time()
        return clip

    def connected(self) -> int:
        now = time.time()
        return sum(1 for t in self.listeners.values() if now - t < self.LISTENER_TTL)

    def info(self) -> dict:
        return {"mode": "web", "listeners": self.connected()}
