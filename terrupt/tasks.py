"""Dataset builders.

Each task turns a list of sentences (``str`` or :class:`~terrupt.sources.Sentence`)
into a list of records. Records are plain dicts ready to be serialized as JSON
lines or exported to a Hugging Face dataset folder.
"""

import math
import mmap
import multiprocessing
import os
import random
import re
import struct
import sys
import tempfile
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
_DISK_CACHE = {}


def _text(sentence):
    return sentence.text if isinstance(sentence, Sentence) else sentence


def _source(sentence):
    return sentence.source if isinstance(sentence, Sentence) else "builtin"


def _parallel_workers(workers):
    workers = workers or min(os.cpu_count() or 1, 8)
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
    return list(_iter_parallel(chunk_fn, pool, extra, language, task_count,
                               total, seed, workers, desc, unit, progress))


def _iter_parallel(chunk_fn, pool, extra, language, task_count, total, seed,
                    workers, desc, unit, progress):
    """Yield bounded worker batches instead of retaining the whole result."""
    span = max(1, min(1000, math.ceil(task_count / workers)))
    tasks = [(start, min(start + span, task_count))
             for start in range(0, task_count, span)]
    _STATE.update(pool=pool, extra=extra, language=language, base_seed=seed)
    pbar = Progress(total=total, desc=desc, unit=unit, disable=not progress)
    with pbar:
        with Pool(processes=workers) as executor:
            for chunk in executor.imap_unordered(chunk_fn, tasks):
                pbar.update(len(chunk))
                yield from chunk


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


def _disk_sentence(index):
    corpus_path, offsets_path = _STATE["disk"]
    corpus = _DISK_CACHE.get(corpus_path)
    if corpus is None:
        corpus_file = open(corpus_path, "rb")
        offsets_file = open(offsets_path, "rb")
        corpus = (corpus_file, mmap.mmap(corpus_file.fileno(), 0, access=mmap.ACCESS_READ),
                  mmap.mmap(offsets_file.fileno(), 0, access=mmap.ACCESS_READ))
        _DISK_CACHE[corpus_path] = corpus
    offset = struct.unpack_from("<Q", corpus[2], index * 8)[0]
    corpus[1].seek(offset)
    source, text = corpus[1].readline().rstrip(b"\n").split(b"\t", 1)
    return text.decode("utf-8"), source.decode("utf-8")


def _textcorrupt_disk_chunk(task):
    """Worker variant that reads sentences from shared memory-mapped files."""
    start, end = task
    severities = _STATE["extra"]
    language = _STATE["language"]
    rng = random.Random(_STATE["base_seed"] + start)
    records = []
    for i in range(start, end):
        text, source = _disk_sentence(i)
        severity = rng.choice(severities)
        c = corrupt(text, severity, rng)
        records.append({
            "original": c.original,
            "corrupted": c.corrupted,
            "corruptions": c.corruptions,
            "corruption_type": corruption_type(c.corruptions),
            "severity": c.severity,
            "op_count": c.op_count,
            "language": language,
            "source": source,
        })
    return records


def _disk_corpus(sentences):
    """Materialize text outside Python objects so fork workers share pages."""
    directory = tempfile.TemporaryDirectory(prefix="terrupt-corrupt-")
    corpus_path = os.path.join(directory.name, "sentences.bin")
    offsets_path = os.path.join(directory.name, "offsets.bin")
    with open(corpus_path, "wb") as corpus, open(offsets_path, "wb") as offsets:
        position = 0
        for sentence in sentences:
            text = _text(sentence).replace("\t", " ").replace("\n", " ")
            source = _source(sentence)
            line = f"{source}\t{text}\n".encode("utf-8")
            offsets.write(struct.pack("<Q", position))
            corpus.write(line)
            position += len(line)
    return directory, corpus_path, offsets_path


def _textcorrupt_serial(pool, severities, language, count, rng, progress):
    return list(_textcorrupt_serial_iter(pool, severities, language, count,
                                         rng, progress))


def _textcorrupt_serial_iter(pool, severities, language, count, rng, progress):
    i = 0
    pbar = Progress(total=count, desc="corrupting", unit="records",
                    disable=not progress)
    with pbar:
        while i < count:
            sentence = pool[i % len(pool)]
            i += 1
            severity = rng.choice(list(severities))
            c = corrupt(_text(sentence), severity, rng)
            yield {
                "original": c.original,
                "corrupted": c.corrupted,
                "corruptions": c.corruptions,
                "corruption_type": corruption_type(c.corruptions),
                "severity": c.severity,
                "op_count": c.op_count,
                "language": language,
                "source": _source(sentence),
            }
            pbar.update(1)


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
    return list(iter_textcorrupt(pool, severities, rng, language, count,
                                 progress, workers, already_shuffled=True))


def iter_textcorrupt(sentences, severities, rng, language="en", count=None,
                     progress=True, workers=None, already_shuffled=False):
    """Yield textcorrupt records while keeping only bounded batches in memory."""
    count = count or len(sentences)
    if count <= 0 or not sentences:
        return
    pool = sentences if already_shuffled else sentences
    if not already_shuffled and len(sentences) <= 1_000_000:
        pool = list(sentences)
        rng.shuffle(pool)
    use_parallel = _can_parallel(workers, count)
    if use_parallel:
        seed = rng.randrange(1 << 32)
        if len(pool) > 1_000_000:
            directory, corpus_path, offsets_path = _disk_corpus(pool)
            _STATE["disk"] = (corpus_path, offsets_path)
            try:
                yield from _iter_parallel(
                    _textcorrupt_disk_chunk, None, severities, language,
                    count, count, seed, _parallel_workers(workers),
                    "corrupting", "records", progress)
            finally:
                _STATE.pop("disk", None)
                directory.cleanup()
        else:
            yield from _iter_parallel(_textcorrupt_chunk, pool, severities,
                                      language, count, count, seed,
                                      _parallel_workers(workers), "corrupting",
                                      "records", progress)
        return
    yield from _textcorrupt_serial_iter(pool, severities, language, count,
                                       rng, progress)


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
