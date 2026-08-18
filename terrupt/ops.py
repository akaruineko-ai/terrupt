"""Atomic corruption operations.

Each operation transforms ``text`` into a corrupted variant. Operations are
pure with respect to randomness: they receive a ``random.Random`` instance and
return the corrupted string. When no change could be applied they return the
input unchanged, so the engine can skip them.
"""

_ASCII = "abcdefghijklmnopqrstuvwxyz"

QWERTY_NEIGHBORS = {
    "q": "was", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsx", "s": "aewdx", "d": "serfc", "f": "drtgc", "g": "ftyhb",
    "h": "gyujb", "j": "huikn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

LEET_MAP = {
    "a": "4", "b": "8", "e": "3", "g": "9", "i": "1", "l": "1",
    "o": "0", "s": "5", "t": "7", "z": "2",
}

HOMOGLYPHS = {
    "a": "\u0430", "c": "\u0441", "e": "\u0435", "h": "\u04bb",
    "i": "\u0456", "k": "\u043a", "m": "\u043c", "o": "\u043e",
    "p": "\u0440", "s": "\u0455", "t": "\u0442", "u": "\u03c5",
    "x": "\u0445", "y": "\u0443",
}

DIACRITICS = {
    "a": "\u00e4", "c": "\u00e7", "e": "\u00eb", "i": "\u00ef",
    "n": "\u00f1", "o": "\u00f6", "s": "\u015f", "u": "\u00fc",
}

COMBINING_ACUTE = "\u0301"
PUNCT_CHARS = set(".,!?;:'\"")


def _positions(chars, predicate):
    return [i for i, ch in enumerate(chars) if predicate(ch)]


def typo(text, rng, k=1):
    chars = list(text)
    idxs = _positions(chars, lambda c: c.lower() in QWERTY_NEIGHBORS)
    if not idxs:
        return text
    for i in rng.sample(idxs, min(k, len(idxs))):
        ch = chars[i]
        repl = rng.choice(QWERTY_NEIGHBORS[ch.lower()])
        chars[i] = repl.upper() if ch.isupper() else repl
    return "".join(chars)


def deletion(text, rng, k=1):
    idxs = _positions(text, lambda c: c.isalnum())
    if len(idxs) < 2:
        return text
    k = min(k, len(idxs) - 1)
    drop = set(rng.sample(idxs, k))
    return "".join(ch for i, ch in enumerate(text) if i not in drop)


def insertion(text, rng, k=1):
    chars = list(text)
    for _ in range(k):
        pos = rng.randrange(len(chars) + 1)
        chars.insert(pos, rng.choice(_ASCII))
    return "".join(chars)


def swap(text, rng, k=1):
    chars = list(text)
    for _ in range(k):
        idxs = _positions(chars, lambda c: c.isalpha())
        if len(idxs) < 2:
            break
        pairs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        i, j = rng.choice(pairs)
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def repeated_chars(text, rng, k=1):
    idxs = _positions(text, lambda c: c.isalpha())
    if not idxs:
        return text
    chars = list(text)
    for i in sorted(rng.sample(idxs, min(k, len(idxs))), reverse=True):
        chars.insert(i + 1, text[i] * rng.randint(1, 2))
    return "".join(chars)


def bad_spacing(text, rng):
    words = text.split()
    if len(words) < 2:
        return text
    mode = rng.random()
    if mode < 0.45:
        i = rng.randrange(len(words) - 1)
        words[i] = words[i] + words[i + 1]
        del words[i + 1]
        return " ".join(words)
    if mode < 0.75:
        i = rng.randrange(len(words))
        if len(words[i]) < 3:
            return text
        j = rng.randrange(1, len(words[i]))
        words[i] = words[i][:j] + " " + words[i][j:]
        return " ".join(words)
    i = rng.randrange(1, len(words))
    return " ".join(words[:i]) + "  " + " ".join(words[i:])


def case(text, rng):
    words = text.split()
    if not words:
        return text
    mode = rng.random()
    if mode < 0.45:
        idxs = [i for i, w in enumerate(words) if w and w[0].isalpha()]
        if not idxs:
            return text
        i = rng.choice(idxs)
        words[i] = words[i][0].swapcase() + words[i][1:]
    elif mode < 0.75:
        i = rng.randrange(len(words))
        words[i] = words[i].lower()
    else:
        i = rng.randrange(len(words))
        words[i] = words[i].upper()
    return " ".join(words)


def punctuation(text, rng):
    present = [i for i, ch in enumerate(text) if ch in PUNCT_CHARS]
    if not present:
        return text
    mode = rng.random()
    if mode < 0.5:
        drop = set(rng.sample(present, rng.randint(1, min(2, len(present)))))
        return "".join(ch for i, ch in enumerate(text) if i not in drop)
    if mode < 0.8:
        mapping = {"?": ".", "!": ".", ",": ";", ";": ",", ":": ";", ".": ","}
        chars = list(text)
        i = rng.choice(present)
        chars[i] = mapping.get(chars[i], rng.choice([".", ","]))
        return "".join(chars)
    chars = list(text)
    chars.insert(rng.randrange(len(chars) + 1), rng.choice([".", ",", "!", "?"]))
    return "".join(chars)


def unicode(text, rng):
    idxs = _positions(text, lambda c: c.isascii() and c.isalpha())
    if not idxs:
        return text
    i = rng.choice(idxs)
    ch = text[i].lower()
    chars = list(text)
    if ch in HOMOGLYPHS and rng.random() < 0.75:
        repl = HOMOGLYPHS[ch]
        chars[i] = repl.upper() if text[i].isupper() else repl
        return "".join(chars)
    if ch in DIACRITICS:
        repl = DIACRITICS[ch]
        chars[i] = repl.upper() if text[i].isupper() else repl
        return "".join(chars)
    chars[i] = text[i] + COMBINING_ACUTE
    return "".join(chars)


def leetspeak(text, rng, k=1):
    idxs = [i for i, ch in enumerate(text) if ch.lower() in LEET_MAP]
    if not idxs:
        return text
    chars = list(text)
    for i in rng.sample(idxs, min(k, len(idxs))):
        chars[i] = LEET_MAP[chars[i].lower()]
    return "".join(chars)


def word_shuffle(text, rng):
    words = text.split()
    if len(words) < 3:
        return text
    if len(words) <= 5 or rng.random() < 0.5:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)
    start = 0 if rng.random() < 0.3 else 1
    end = len(words) if rng.random() < 0.3 else len(words) - 1
    if end - start >= 2:
        middle = words[start:end]
        rng.shuffle(middle)
        words[start:end] = middle
    return " ".join(words)


OPS = {
    "typo": typo,
    "deletion": deletion,
    "insertion": insertion,
    "swap": swap,
    "repeated_chars": repeated_chars,
    "bad_spacing": bad_spacing,
    "case": case,
    "punctuation": punctuation,
    "unicode": unicode,
    "leetspeak": leetspeak,
    "word_shuffle": word_shuffle,
}
