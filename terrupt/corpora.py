"""Source sentences for dataset generation.

A corpus comes either from a plain-text file (free text is split into
sentences) or from the built-in curated sentences.
"""

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")

BUILTIN = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "The scholar pressed her lips to the warm cup of tea.",
        "Every morning the baker wakes before the sun and kneads dough by hand.",
        "She told the committee that the proposal needed more work.",
        "If you want to succeed, you have to learn from your failures.",
        "The old lighthouse stood alone on the jagged cliff.",
        "Nobody expected the storm to arrive before noon, but it did.",
        "Please send the invoice to accounting by Friday afternoon.",
        "The children built a fort from blankets and pillows in the living room.",
        "A good book is like a friend that stays by your side.",
        "He opened the door and found a package on the doorstep.",
        "The orchestra played a beautiful piece that moved the entire audience.",
        "We should never underestimate the power of a simple apology.",
        "The scientists discovered a new species in the depths of the ocean.",
        "Can you believe they walked the whole distance without stopping?",
        "I have been trying to reach you for hours; where have you been?",
        "The museum is closed on Mondays, so plan your visit accordingly.",
        "Winter in the mountains is harsh, but the views are unforgettable.",
        "The programmer fixed the bug, then ran the tests twice to be sure.",
        "Her grandmother taught her how to knit warm woolen socks.",
        "The cat sat on the windowsill and watched the rain fall.",
        "Many people forget that success is built on small daily habits.",
        "The train to Berlin departs from platform four at exactly ten o'clock.",
        "What time does the last ferry leave the harbor tonight?",
        "The detective studied the note for several minutes before speaking.",
        "Coffee, tea, or hot chocolate, what would you like?",
        "The garden was full of roses, tulips, and fragrant lavender.",
        "Despite the noise, the baby finally fell asleep in the car.",
        "Reading expands the mind and enriches the imagination.",
        "The report, which took three months to write, was rejected.",
        "Throw the ball gently so the little dog can catch it.",
        "A stitch in time saves nine, as the old saying goes.",
        "The concert was sold out within hours of the announcement.",
        "Every student handed in their assignment before the deadline.",
        "The river winds slowly through the valley toward the sea.",
        "Why did you leave the meeting without saying goodbye?",
        "The chef sprinkled fresh basil over the steaming pasta.",
        "We camped under a sky full of stars near the lake.",
        "The company announced a new policy for remote workers.",
        "History repeats itself, but nobody seems to learn from it.",
    ],
    "ru": [
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Мы идём в театр завтра вечером, не опаздывай.",
        "Утром на улице было холодно, но солнце всё равно светило.",
        "Он прочитал письмо дважды, прежде чем ответить.",
        "Дети играли во дворе до самого заката.",
        "Эта книга изменила моё представление о мире.",
        "Куда ты положил ключи от машины?",
        "В парке цвели розы и пахло свежей листвой.",
    ],
    "de": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Ich habe den Bericht gestern Abend fertig geschrieben.",
        "Können Sie mir bitte den Weg zum Bahnhof erklären?",
        "Das Wetter ist heute wunderbar, also gehen wir spazieren.",
        "Sie hat das Buch in einer einzigen Nacht gelesen.",
        "Die alte Brücke führt über den Fluss ins Dorf.",
    ],
    "fr": [
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Je pense donc je suis, écrivait Descartes.",
        "Où se trouve la gare la plus proche, s'il vous plaît ?",
        "Il a plu toute la journée, mais la soirée fut magnifique.",
        "Elle prépare un gâteau au chocolat pour l'anniversaire.",
    ],
    "es": [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "¿Dónde está la estación de tren más cercana?",
        "Me encanta caminar por la playa al atardecer.",
        "La reunión empieza a las nueve en punto, no llegues tarde.",
        "Ese restaurante sirve la mejor paella de la ciudad.",
    ],
    "uk": [
        "Швидкий бурий лис стрибає через лінивого пса.",
        "Де ти був учора ввечері, я тобі дзвонив?",
        "Ця пісня нагадує мені про дитинство в селі.",
        "Завтра буде дощ, тому візьми з собою парасольку.",
    ],
}


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _valid(sentence):
    words = sentence.split()
    return 3 <= len(words) <= 64


def builtin(language="en"):
    if language in BUILTIN:
        return [s for s in BUILTIN[language] if _valid(s)]
    return []


def load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    sentences = []
    for part in re.split(r"\n+", raw):
        for sentence in _SENT_SPLIT.split(part):
            sentence = _clean(sentence)
            if sentence and _valid(sentence):
                sentences.append(sentence)
    return sentences


def get_sentences(language="en", path=None):
    if path:
        sentences = load_corpus(path)
    else:
        sentences = builtin(language)
    if not sentences:
        raise ValueError("corpus is empty; pass --corpus with a plain-text file")
    return sentences
