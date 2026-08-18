"""Corruption engine.

Scales the number and intensity of corruption operations to a severity level
and records which corruption types were actually applied. Severity controls
both how many operations run and how aggressive each one is:

* ``0.1``  -> one minor typo
* ``0.25`` -> a couple of small errors
* ``0.5``  -> several errors plus punctuation
* ``0.75`` -> heavy corruption
* ``1.0``  -> the text survived a small civil war
"""

import random
from dataclasses import dataclass, field

from terrupt.ops import OPS

SEVERITIES = (0.1, 0.25, 0.5, 0.75, 1.0)

K_OPS = {"deletion", "insertion", "swap", "repeated_chars", "typo", "leetspeak"}

BASE_WEIGHTS = {
    "typo": 14, "deletion": 9, "insertion": 7, "swap": 10,
    "repeated_chars": 6, "bad_spacing": 7, "case": 9,
    "punctuation": 7, "unicode": 4, "leetspeak": 4, "word_shuffle": 2,
}

SEVERITY_SCALE = (
    "0.1: one minor typo; "
    "0.25: a couple of small errors; "
    "0.5: several errors plus punctuation; "
    "0.75: heavy corruption; "
    "1.0: the text survived a small civil war."
)

HEAVY_OPS = {"word_shuffle", "unicode", "leetspeak", "repeated_chars"}

OP_DESCRIPTIONS = {
    "typo": "replace a letter with a neighboring keyboard key",
    "deletion": "delete characters",
    "insertion": "insert random characters",
    "swap": "transpose adjacent characters",
    "repeated_chars": "stutter (repeat) characters",
    "bad_spacing": "merge words, split words, or add double spaces",
    "case": "flip, lowercase, or uppercase words",
    "punctuation": "remove, replace, or insert punctuation",
    "unicode": "replace letters with homoglyphs or diacritics",
    "leetspeak": "convert letters to leet-style symbols",
    "word_shuffle": "reorder words",
}


@dataclass
class Corruption:
    original: str
    corrupted: str
    corruptions: list
    severity: float
    op_count: int = field(default=0)


def n_ops(severity, word_count):
    return max(1, round(severity * (2.0 + 0.6 * word_count)))


def corruption_type(corruptions):
    return corruptions[0] if len(corruptions) == 1 else "mixed"


def _weights(severity):
    return {name: base * (1.0 + severity) if name in HEAVY_OPS else base
            for name, base in BASE_WEIGHTS.items()}


def _op_k(severity, name):
    if name in K_OPS:
        return 1 + int(severity * 2)
    return 1


def _cumulative(weights):
    names = list(weights)
    cum = []
    total = 0.0
    for name in names:
        total += weights[name]
        cum.append(total)
    return names, cum, total


def _weighted_choice(rng, names, cum, total):
    r = rng.random() * total
    for name, upto in zip(names, cum):
        if r <= upto:
            return name
    return names[-1]


def _dedupe(seq):
    out = []
    for item in seq:
        if item not in out:
            out.append(item)
    return out


def corrupt(text, severity, rng):
    """Apply corruption to ``text`` at the given severity.

    ``severity`` is clamped to ``[0.1, 1.0]`` and preserved verbatim when it
    already falls inside the canonical :data:`SEVERITIES` set. Returns a
    :class:`Corruption` record whose ``corruptions`` field lists the distinct
    operation types that actually modified the text.
    """
    severity = min(max(float(severity), 0.1), 1.0)
    target = n_ops(severity, len(text.split()))
    weights = _weights(severity)
    names, cum, total = _cumulative(weights)
    result = text
    applied = []
    attempts = 0
    while len(applied) < target and attempts < target * 4:
        attempts += 1
        name = _weighted_choice(rng, names, cum, total)
        op = OPS[name]
        try:
            out = op(result, rng, k=_op_k(severity, name)) if name in K_OPS else op(result, rng)
        except Exception:
            continue
        if out and out != result:
            result = out
            applied.append(name)
    if result == text:
        for name in ("case", "typo", "insertion"):
            out = OPS[name](result, rng)
            if out and out != result:
                result = out
                applied.append(name)
                break
    return Corruption(
        original=text,
        corrupted=result,
        corruptions=_dedupe(applied),
        severity=severity,
        op_count=len(applied),
    )
