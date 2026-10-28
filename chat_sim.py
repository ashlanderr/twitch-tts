"""Fake Twitch chat for a small stream: ~10 regulars, each with a fixed voice and a habit of writing.

- people write in sessions: a few messages in a row, then go quiet for a while;
- some pairs talk to each other (@mentions, back-and-forth);
- when something happens in the game, several people react within seconds;
- lurkers show up rarely.

Overall pace at speed 1.0 is ~10 messages per minute.
"""

import heapq
import random
from dataclasses import dataclass, field

EMOTES = {"KEKW", "LUL", "PogChamp", "Kappa", "monkaS", "PepeHands", "Sadge", "OMEGALUL", "5Head", "EZ",
          "catJAM", "PepeLaugh", "ResidentSleeper", "WutFace", "BibleThump", "FeelsGoodMan", "Clap", "POGGERS"}


@dataclass
class Persona:
    name: str
    voice: str
    session_every: float          # mean seconds between "sessions" (bursts of activity)
    session_size: tuple[int, int]  # messages per session, min..max
    gap: tuple[float, float]       # seconds between messages inside a session
    lines: list[str]
    hype: list[str]                # reactions to a game event
    hype_chance: float             # probability to react to an event


PERSONAS = [
    Persona("dimon_gg", "bandit", 50, (1, 4), (2, 6), [
        "го го го", "ахахах ну ты даёшь", "давай давай давай", "слышь, а чё он там делает",
        "это база", "ну всё, сейчас начнётся", "братва, я пришёл", "крипово", "изи катка",
        "кто со мной в лобби после стрима?", "я бы там побежал направо, отвечаю", "жиза",
    ], ["ААААА", "КРАСАВА", "ахахаха", "ну ты зверь", "вот это да KEKW"], 0.9),
    Persona("SunnySveta", "jaina", 75, (1, 3), (4, 9), [
        "Всем привет! Как настроение?", "Стример, а ты сегодня долго будешь?", "Ой, как красиво тут",
        "А что за музыка играет? Очень приятная", "Я сегодня первый раз смотрю, мне нравится",
        "Стример, попей водички", "Спасибо за стрим, ты лучший", "Ой, мне страшно за тебя monkaS",
    ], ["Ура!!", "Молодец!", "Вау, как ты это сделал?", "Я в шоке"], 0.6),
    Persona("ded_inside", "sidorovich", 160, (1, 2), (6, 15), [
        "Опять ты в шкаф полез. Ну кто так играет, а?",
        "Я в эту игру с релиза играю, и скажу тебе: так только новички делают",
        "Генератор крути, а не бегай кругами",
        "Ну и зачем ты туда пошёл, там же было видно, что маньяк рядом",
        "Эх, молодёжь", "Скучно. Давай что-нибудь посложнее",
    ], ["Ну, неплохо. Для тебя.", "Повезло", "Хм. Сойдёт."], 0.4),
    Persona("valera_entity", "thrall", 90, (1, 3), (5, 10), [
        "Совет: в этой ситуации лучше сразу идти к выходу",
        "У него перк на обнаружение, будь аккуратнее",
        "Там под лестницей есть сундук, загляни",
        "Если что, на этой карте генераторы спавнятся у стен",
        "Попробуй поменять сборку, эта не очень",
    ], ["Вот это я понимаю игра", "Чисто", "Это было по учебнику"], 0.6),
    Persona("anime_enjoyer", "tyrande", 120, (1, 3), (3, 8), [
        "kawaii", "this map is so pretty", "стример, ты смотрел новую серию?", "I love this game catJAM",
        "ну это прям как в аниме было", "omg", "можно в следующий раз в милую игру поиграть?",
    ], ["SUGOI", "no way LUL", "это было красиво"], 0.7),
    Persona("MrPelmen", "grunt", 110, (1, 2), (2, 5), [
        "лол", "кек", "ору", "пельмени варю, смотрю стрим", "а", "чё", "+", "гыгы",
    ], ["КУДААА", "ОРУ", "+++", "ЛЕГЕНДА", "ПОГ"], 1.0),
    Persona("FrostyJoe", "arthas", 120, (1, 2), (5, 10), [
        "Hey from Canada!", "what's your rank?", "your aim is crazy today", "how many hours do you have in this game?",
        "can you speak English sometimes? lol", "this build looks strong", "GG man",
    ], ["LETS GO", "insane", "clip it!"], 0.6),
    Persona("nightowl_ru", "saharov", 300, (1, 1), (5, 10), [
        "Я тут с самого начала, просто молчу. Отличный стрим, спасибо",
        "Держи 300 рублей на развитие канала, ты молодец",
        "Подписка на месяц, продолжай в том же духе",
        "Только что проснулся, а ты всё ещё стримишь. Уважаю",
    ], ["Достойно"], 0.2),
    Persona("kotik_v_tapkah", "illidan", 95, (1, 3), (4, 9), [
        "Мой кот сейчас сидит у монитора и смотрит с вами", "Кто-нибудь знает, как отучить кота драть диван?",
        "У меня чай остыл, пока смотрел этот раунд", "Кот одобряет", "Мяу",
    ], ["Кот подпрыгнул от неожиданности", "МЯУ!"], 0.5),
    Persona("tired_survivor", "skyrim_guard", 140, (1, 2), (5, 12), [
        "Я тоже раньше был выжившим, пока не словил крюк в колено",
        "Опять работа завтра, а я тут сижу", "Ещё одну катку и спать. Наверное",
        "Стример, а ты когда спишь вообще?",
    ], ["Вот это сила", "Я бы так не смог"], 0.5),
]
VOICE_OF = {p.name: p.voice for p in PERSONAS}
BY_NAME = {p.name: p for p in PERSONAS}

# scripted back-and-forth between regulars
DIALOGUES = [
    [("SunnySveta", "@kotik_v_tapkah а как твоего кота зовут?"),
     ("kotik_v_tapkah", "@SunnySveta Барсик! Ему уже 7 лет"),
     ("SunnySveta", "@kotik_v_tapkah ой, какой милый, передай ему привет"),
     ("kotik_v_tapkah", "Барсик передаёт мяу")],
    [("ded_inside", "Опять маньяк его в угол зажал. Сам виноват"),
     ("valera_entity", "@ded_inside да ладно, там без шансов было, у маньяка обнаружение"),
     ("ded_inside", "@valera_entity шансы есть всегда, если голова на месте"),
     ("valera_entity", "@ded_inside ну покажи как надо, раз такой умный"),
     ("dimon_gg", "ахахах деды воюют")],
    [("FrostyJoe", "anyone here from Europe?"),
     ("anime_enjoyer", "@FrostyJoe i'm from Russia lol"),
     ("FrostyJoe", "@anime_enjoyer nice, what time is it there?"),
     ("anime_enjoyer", "@FrostyJoe 2 am, I should sleep")],
    [("dimon_gg", "@MrPelmen ты чё, опять пельмени?"),
     ("MrPelmen", "@dimon_gg а то"),
     ("dimon_gg", "@MrPelmen скинь рецепт брат"),
     ("MrPelmen", "@dimon_gg вода, пельмени, всё")],
    [("tired_survivor", "Кто завтра работает, ставьте плюс"),
     ("MrPelmen", "+"), ("kotik_v_tapkah", "+"), ("dimon_gg", "+ к сожалению")],
]

EVENTS = [
    "клатч 1 против 3",
    "стримера повесили на крюк",
    "маньяк упал с лестницы",
    "стример сбежал в последнюю секунду",
    "скример из-за угла",
    "хедшот через всю карту",
]


SESSION_SLOWDOWN = 1.4  # tunes the overall pace to ~10 messages/min at speed 1.0


@dataclass(order=True)
class _Item:
    at: float
    seq: int
    kind: str = field(compare=False)  # "session" | "msg" | "event" | "dialogue"
    data: object = field(compare=False, default=None)


class ChatSimulator:
    """Call poll(now) often; it returns ("msg", user, text) and ("event", None, text) items that are due."""

    def __init__(self, now: float, speed: float = 1.0):
        self.speed = speed
        self.q: list[_Item] = []
        self.seq = 0
        for p in PERSONAS:
            self._push(now + random.uniform(2, p.session_every * SESSION_SLOWDOWN), "session", p.name)
        self._push(now + random.uniform(40, 90), "event")
        self._push(now + random.uniform(30, 80), "dialogue")

    def _push(self, at: float, kind: str, data=None):
        self.seq += 1
        heapq.heappush(self.q, _Item(at, self.seq, kind, data))

    def _scaled(self, seconds: float) -> float:
        return seconds / max(self.speed, 0.05)

    def poll(self, now: float) -> list[tuple[str, str | None, str]]:
        out = []
        while self.q and self.q[0].at <= now:
            it = heapq.heappop(self.q)
            if it.kind == "msg":
                out.append(("msg", *it.data))
            elif it.kind == "session":
                p = BY_NAME[it.data]
                t = now
                for i in range(random.randint(*p.session_size)):
                    text = random.choice(p.lines)
                    if random.random() < 0.12:
                        text += " " + random.choice(sorted(EMOTES))
                    self._push(t, "msg", (p.name, text))
                    t += random.uniform(*p.gap)
                self._push(t + random.expovariate(1 / self._scaled(p.session_every * SESSION_SLOWDOWN)), "session", p.name)
            elif it.kind == "event":
                out.append(("event", None, random.choice(EVENTS)))
                # the whole chat reacts at once: first reactions within a second or two
                for p in PERSONAS:
                    if random.random() < p.hype_chance:
                        self._push(now + random.uniform(0.8, 6), "msg", (p.name, random.choice(p.hype)))
                self._push(now + random.uniform(*map(self._scaled, (90, 200))), "event")
            elif it.kind == "dialogue":
                t = now
                for user, text in random.choice(DIALOGUES):
                    self._push(t, "msg", (user, text))
                    t += random.uniform(5, 14)  # time to read and type a reply
                self._push(t + random.uniform(*map(self._scaled, (60, 150))), "dialogue")
        return out


def tts_text(text: str) -> str:
    """What the TTS reads: no emotes, no @mentions."""
    return " ".join(w for w in text.split() if w not in EMOTES and not w.startswith("@"))
