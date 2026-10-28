"""Read-only Twitch chat over anonymous IRC (no account or token: the justinfan<digits> nick only reads).

Hands the app every message of a real user with the text to voice (no emotes, links or @mentions), plus moderation:
a deleted message (CLEARMSG) and a ban/timeout or a cleared chat (CLEARCHAT).

    python twitch_chat.py <channel>      # print what would be voiced
"""

import gzip
import json
import random
import re
import socket
import ssl
import sys
import threading
import time
import urllib.request

HOST, PORT = "irc.chat.twitch.tv", 6697

# chat bots seen on most channels; more with --twitch-ignore
BOTS = {"nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "wizebot", "botrixoficial", "sery_bot",
        "commanderroot", "soundalerts", "kofistreambot", "buttsbot", "pokemoncommunitygame", "jeetbot",
        "streamelementsbot", "own3d", "frostytoolsdotcom", "tangiabot", "blerp", "playwithviewersbot", "creatisbot"}

LINK = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|ru|net|org|tv|gg|io|me|be|ly|co|xyz|info|app|"
                  r"dev|su|ua|by|kz|рф)(?:/\S*)?(?=$|[\s,!?)])", re.IGNORECASE)


def _tags(raw: str) -> dict:
    def unescape(v: str) -> str:
        return re.sub(r"\\(.)", lambda m: {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}.get(m[1], m[1]), v)

    return {k: unescape(v) for k, _, v in (kv.partition("=") for kv in raw.split(";"))}


def parse(line: str) -> tuple[dict, str, str, list[str], str]:
    """IRC line -> (tags, prefix nick, command, params, trailing)."""
    tags, prefix = {}, ""
    if line.startswith("@"):
        raw, _, line = line[1:].partition(" ")
        tags = _tags(raw)
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
        prefix = prefix.split("!")[0]
    line, _, trailing = line.partition(" :")
    cmd, *params = line.split()
    return tags, prefix, cmd, params, trailing


def strip_emotes(text: str, emotes_tag: str) -> str:
    """Cut Twitch emotes out by the positions in the `emotes` tag ("25:0-4,12-16/1902:6-10", code point offsets)."""
    spans = sorted((int(a), int(b)) for part in emotes_tag.split("/") if ":" in part
                   for a, _, b in (r.partition("-") for r in part.split(":")[1].split(",")))
    for a, b in reversed(spans):
        text = text[:a] + " " + text[b + 1:]
    return text


class ThirdPartyEmotes:
    """7TV, BTTV and FFZ emote names (global + the channel's), looked up by the channel's id; they aren't in IRC tags."""

    def __init__(self):
        self.names: set[str] = set()
        self.loaded_for = None

    def load(self, room_id: str):
        if room_id == self.loaded_for:
            return
        self.loaded_for = room_id
        sources = [
            ("https://7tv.io/v3/emote-sets/global", lambda d: [e["name"] for e in d.get("emotes", [])]),
            (f"https://7tv.io/v3/users/twitch/{room_id}",
             lambda d: [e["name"] for e in (d.get("emote_set") or {}).get("emotes", [])]),
            ("https://api.betterttv.net/3/cached/emotes/global", lambda d: [e["code"] for e in d]),
            (f"https://api.betterttv.net/3/cached/users/twitch/{room_id}",
             lambda d: [e["code"] for e in d.get("channelEmotes", []) + d.get("sharedEmotes", [])]),
            ("https://api.betterttv.net/3/cached/frankerfacez/emotes/global", lambda d: [e["code"] for e in d]),
            (f"https://api.betterttv.net/3/cached/frankerfacez/users/twitch/{room_id}", lambda d: [e["code"] for e in d]),
        ]
        self.names = set()

        def fetch(url, pick):
            try:
                # gzip: a big channel's 7TV set is ~2.5 MB of JSON, ~0.3 MB compressed
                req = urllib.request.Request(url, headers={"User-Agent": "twitch-tts", "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                self.names = self.names | set(pick(json.loads(body)))  # each source applies as soon as it arrives
            except Exception:
                pass  # channel has no 7TV/BTTV/FFZ account (404) or the service is down: fewer emotes, still works

        threads = [threading.Thread(target=fetch, args=s, daemon=True) for s in sources]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        print(f"twitch: {len(self.names)} 7TV/BTTV/FFZ emotes", file=sys.stderr, flush=True)


class TwitchChat:
    """Background reader of one channel's chat. Reconnects on its own.

    on_message(msg): msg has id, user_id, user (login), name (display name), text (as written), tts (what to voice)
    on_delete(msg_id): a moderator deleted one message
    on_clear(user_id | None): a user was banned/timed out, or (None) the whole chat was cleared
    """

    def __init__(self, channel: str, on_message, on_delete, on_clear, ignore: set[str] = frozenset()):
        self.channel = channel.lower().lstrip("#")
        self.on_message, self.on_delete, self.on_clear = on_message, on_delete, on_clear
        self.ignore = BOTS | {u.lower() for u in ignore}
        self.emotes = ThirdPartyEmotes()
        self.status = "connecting"
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1
        while True:
            try:
                self._session()
                backoff = 1  # the server asked us to reconnect, or the connection dropped after working
            except Exception as e:
                self.status = f"disconnected: {e}"
                print(f"twitch: {self.status}, retry in {backoff}s", file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _session(self):
        sock = ssl.create_default_context().wrap_socket(socket.create_connection((HOST, PORT), timeout=30),
                                                         server_hostname=HOST)
        sock.settimeout(360)  # the server pings every ~5 minutes
        send = lambda s: sock.sendall((s + "\r\n").encode("utf-8"))
        send("CAP REQ :twitch.tv/tags twitch.tv/commands")
        send(f"NICK justinfan{random.randint(10000, 99999)}")
        send(f"JOIN #{self.channel}")
        buf = b""
        with sock:
            while True:
                data = sock.recv(65536)
                if not data:
                    raise ConnectionError("closed by server")
                buf += data
                *lines, buf = buf.split(b"\r\n")
                for raw in lines:
                    try:
                        if raw and self._handle(raw.decode("utf-8", "replace"), send) == "reconnect":
                            return
                    except OSError:
                        raise
                    except Exception as e:  # one odd message must not drop the connection
                        print(f"twitch: {e!r} on {raw[:200]!r}", file=sys.stderr, flush=True)

    def _handle(self, line: str, send):
        tags, nick, cmd, params, text = parse(line)
        if cmd == "PING":
            send(f"PONG :{text}")
        elif cmd == "RECONNECT":
            self.status = "reconnecting"
            return "reconnect"
        elif cmd == "ROOMSTATE" and "room-id" in tags:
            self.status = f"connected to #{self.channel}"
            print(f"twitch: {self.status}", file=sys.stderr, flush=True)
            threading.Thread(target=self.emotes.load, args=(tags["room-id"],), daemon=True).start()
        elif cmd == "PRIVMSG":
            self._message(tags, nick, text)
        elif cmd == "CLEARMSG" and tags.get("target-msg-id"):
            self.on_delete(tags["target-msg-id"])
        elif cmd == "CLEARCHAT":
            self.on_clear(tags.get("target-user-id"))
        elif cmd == "NOTICE" and "msg_channel_suspended" in tags.get("msg-id", ""):
            self.status = f"#{self.channel} does not exist or is suspended"

    def _message(self, tags: dict, nick: str, text: str):
        if nick in self.ignore or text.startswith("!") and not text.startswith(("!!!#", "!tts")):
            return  # bots, and commands to bots (!discord, !song…)
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text = text[8:-1]  # /me
        tts = strip_emotes(text, tags.get("emotes", ""))
        tts = LINK.sub(" ", tts)
        tts = " ".join(w for w in tts.split() if w not in self.emotes.names and not w.startswith("@"))
        self.on_message({"id": tags.get("id", ""), "user_id": tags.get("user-id", nick), "user": nick,
                         "name": tags.get("display-name") or nick, "text": text, "tts": tts})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    TwitchChat(sys.argv[1], lambda m: print(f"{m['name']}: {m['text']!r} -> {m['tts']!r}", flush=True),
               lambda i: print(f"deleted {i}", flush=True), lambda u: print(f"cleared {u or 'all'}", flush=True))
    threading.Event().wait()
