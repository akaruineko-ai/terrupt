"""Dataset builders.

Each task turns a list of sentences (``str`` or :class:`~terrupt.sources.Sentence`)
into a list of records. Records are plain dicts ready to be serialized as JSON
lines or exported to a Hugging Face dataset folder.
"""

import math
import multiprocessing
import os
import random
import re
import sys
from multiprocessing import Pool

from terrupt.engine import corrupt, corruption_type
from terrupt.progress import Progress
from terrupt.sources import Sentence

_PUNCT_RE = re.compile(r"[.,!?]")
_PUNCT_STRIP = re.compile(r"[.,!?;:'\"()\[\]]")
_WRONG_MAP = {'.': ',', ',': '.', '?': '.', '!': '.', ';': ',', ':': ';'}

# Shared worker state. Under the fork start method (the Linux default) workers
# inherit this copy-on-write, so the (potentially huge) sentence pool is never
# pickled and shipped over pipes.
_STATE = {}


def _text(sentence):
    return sentence.text if isinstance(sentence, Sentence) else sentence


def _source(sentence):
    return sentence.source if isinstance(sentence, Sentence) else "builtin"


def _parallel_workers(workers):
    workers = workers or os.cpu_count() or 1
    return max(1, int(workers))


def _can_parallel(workers, count):
    """Parallel corruption needs >1 worker, enough records, and fork (Linux)."""
    if _parallel_workers(workers) <= 1 or count < 1000:
        return False
    try:
        method = multiprocessing.get_start_method(allow_none=True)
    except Exception:
        method = None
    if method is None:
        method = "fork" if sys.platform not in ("win32", "darwin") else "spawn"
    return method == "fork"


def _run_parallel(chunk_fn, pool, extra, language, task_count, total, seed,
                  workers, desc, unit, progress):
    """Run ``chunk_fn`` over disjoint index ranges in a process pool."""
    span = math.ceil(task_count / workers)
    tasks = [(start, min(start + span, task_count))
             for start in range(0, task_count, span)]
    _STATE.update(pool=pool, extra=extra, language=language, base_seed=seed)
    pbar = Progress(total=total, desc=desc, unit=unit, disable=not progress)
    records = []
    with pbar:
        executor = Pool(processes=workers)
        try:
            for chunk in executor.imap_unordered(chunk_fn, tasks):
                records.extend(chunk)
                pbar.update(len(chunk))
        finally:
            executor.terminate()
            executor.join()
    return records


def _textcorrupt_chunk(task):
    """Worker: produce textcorrupt records for index range ``[start, end)``."""
    start, end = task
    pool = _STATE["pool"]
    severities = _STATE["extra"]
    language = _STATE["language"]
    rng = random.Random(_STATE["base_seed"] + start)
    records = []
    for i in range(start, end):
        sentence = pool[i % len(pool)]
        severity = rng.choice(severities)
        c = corrupt(_text(sentence), severity, rng)
        records.append({
            "original": c.original,
            "corrupted": c.corrupted,
            "corruptions": c.corruptions,
            "corruption_type": corruption_type(c.corruptions),
            "severity": c.severity,
            "op_count": c.op_count,
            "language": language,
            "source": _source(sentence),
        })
    return records


def _textcorrupt_serial(pool, severities, language, count, rng, progress):
    records = []
    i = 0
    pbar = Progress(total=count, desc="corrupting", unit="records",
                    disable=not progress)
    with pbar:
        while len(records) < count:
            sentence = pool[i % len(pool)]
            i += 1
            severity = rng.choice(list(severities))
            c = corrupt(_text(sentence), severity, rng)
            records.append({
                "original": c.original,
                "corrupted": c.corrupted,
                "corruptions": c.corruptions,
                "corruption_type": corruption_type(c.corruptions),
                "severity": c.severity,
                "op_count": c.op_count,
                "language": language,
                "source": _source(sentence),
            })
            pbar.update(1)
    return records


def make_textcorrupt(sentences, severities, rng, language="en", count=None,
                     progress=True, workers=None):
    """Task 1: corrupted text -> original text.

    If ``count`` is ``None`` or ``0``, every available sentence is used.
    """
    count = count or len(sentences)
    if count <= 0 or not sentences:
        return []
    pool = list(sentences)
    rng.shuffle(pool)
    if _can_parallel(workers, count):
        seed = rng.randrange(1 << 32)
        return _run_parallel(_textcorrupt_chunk, pool, severities, language,
                             count, count, seed, _parallel_workers(workers),
                             "corrupting", "records", progress)
    return _textcorrupt_serial(pool, severities, language, count, rng, progress)


def make_classification(sentences, severities, rng, language="en", count=None,
                        include_clean=False, clean_fraction=0.1, progress=True,
                        workers=None):
    """Task 3: text -> corruption type(s)."""
    count = count or len(sentences)
    if count <= 0 or not sentences:
        return []
    records = []
    for rec in make_textcorrupt(sentences, severities, rng, language, count,
                                progress=progress, workers=workers):
        records.append({
            "text": rec["corrupted"],
            "original": rec["original"],
            "labels": rec["corruptions"],
            "corruption_type": rec["corruption_type"],
            "severity": rec["severity"],
            "language": rec["language"],
            "source": rec["source"],
        })
    if include_clean:
        clean_src = list(sentences)
        rng.shuffle(clean_src)
        n_clean = max(1, int(count * clean_fraction))
        for i in range(n_clean):
            sent = _text(clean_src[i % len(clean_src)])
            records.append({
                "text": sent,
                "original": sent,
                "labels": [],
                "corruption_type": "none",
                "severity": 0.0,
                "language": language,
                "source": _source(clean_src[i % len(clean_src)]),
            })
    return records


def _strip_punct(text, rng):
    return re.sub(r"\s+", " ", _PUNCT_STRIP.sub("", text)).strip().lower()


def _wrong_punct(text, rng):
    pos = [i for i, ch in enumerate(text) if ch in _WRONG_MAP]
    if not pos:
        return text
    chars = list(text)
    for i in rng.sample(pos, rng.randint(1, min(3, len(pos)))):
        chars[i] = _WRONG_MAP.get(chars[i], rng.choice([",", "."]))
    return "".join(chars)


def _random_punct(text, rng):
    words = text.split()
    if not words:
        return text
    for _ in range(rng.randint(1, 2)):
        words[rng.randrange(len(words))] += rng.choice([".", ",", ";", "!", "?"])
    return " ".join(words)


def _spacing_punct(text, rng):
    chars = list(text)
    pos = [i for i, ch in enumerate(chars) if ch in ",."]
    if not pos:
        return text
    i = rng.choice(pos)
    mode = rng.random()
    if mode < 0.4 and i + 1 < len(chars) and chars[i + 1] == " ":
        del chars[i + 1]
    elif mode < 0.7:
        chars.insert(i, " ")
    elif i > 0 and chars[i - 1] == " ":
        del chars[i - 1]
    else:
        del chars[i]
    return "".join(chars)


_PUNCT_MODES = {
    "strip": _strip_punct,
    "wrong": _wrong_punct,
    "random": _random_punct,
    "spacing": _spacing_punct,
}


def _punctuation_chunk(task):
    """Worker: produce punctuation records over a range of attempts."""
    start, end = task
    pool = _STATE["pool"]
    modes = _STATE["extra"]
    language = _STATE["language"]
    rng = random.Random(_STATE["base_seed"] + start)
    records = []
    for a in range(start, end):
        sentence = pool[a % len(pool)]
        original = _text(sentence)
        mode = rng.choice(modes)
        input_text = _PUNCT_MODES[mode](original, rng)
        if input_text == original:
            continue
        records.append({
            "input": input_text,
            "original": original,
            "mode": mode,
            "language": language,
            "source": _source(sentence),
        })
    return records


def _punctuation_serial(pool, modes, language, count, rng, progress):
    records = []
    i = 0
    attempts = 0
    max_attempts = max(count * 10, len(pool) * len(modes) * 2)
    pbar = Progress(total=count, desc="corrupting punctuation", unit="records",
                    disable=not progress)
    with pbar:
        while len(records) < count and attempts < max_attempts:
            attempts += 1
            sentence = pool[i % len(pool)]
            i += 1
            original = _text(sentence)
            mode = rng.choice(modes)
            input_text = _PUNCT_MODES[mode](original, rng)
            if input_text == original:
                continue
            records.append({
                "input": input_text,
                "original": original,
                "mode": mode,
                "language": language,
                "source": _source(sentence),
            })
            pbar.update(1)
    if len(records) < count:
        raise ValueError(
            f"could only create {len(records)} of {count} punctuation records "
            f"after {attempts} attempts; check the selected modes and corpus"
        )
    return records


def make_punctuation(sentences, rng, language="en", count=None,
                     modes=("strip", "wrong", "random", "spacing"), progress=True,
                     workers=None):
    """Task 2: unpunctuated / mispunctuated text -> punctuated text.

    A ``count`` of ``None`` or ``0`` uses ``pool * modes`` (every combination).
    """
    modes = tuple(modes)
    pool = [s for s in sentences if _PUNCT_RE.search(_text(s))]
    if not pool:
        raise ValueError("punctuation task needs sentences that contain punctuation")
    count = count or len(pool) * len(modes)
    if count <= 0:
        return []
    rng.shuffle(pool)
    if _can_parallel(workers, count):
        seed = rng.randrange(1 << 32)
        n = _parallel_workers(workers)
        attempts = max(count * 2, n)
        records = _run_parallel(_punctuation_chunk, pool, modes, language,
                                attempts, count, seed, n,
                                "corrupting punctuation", "records", progress)
        if len(records) < count:
            raise ValueError(
                f"could only create {len(records)} of {count} punctuation records "
                f"after {attempts} attempts; check the selected modes and corpus"
            )
        return records[:count]
    return _punctuation_serial(pool, modes, language, count, rng, progress)
