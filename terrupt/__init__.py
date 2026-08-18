"""Terrupt: dataset + benchmark for restoring intentionally corrupted text."""

from terrupt.corpora import builtin, get_sentences, load_corpus
from terrupt.engine import OPS, SEVERITIES, Corruption, corrupt, n_ops
from terrupt.sources import Sentence, load_sentences
from terrupt.tasks import (iter_textcorrupt, make_classification,
                           make_punctuation, make_textcorrupt)

__version__ = "0.1.0"

__all__ = [
    "OPS", "SEVERITIES", "Corruption", "corrupt", "n_ops",
    "builtin", "get_sentences", "load_corpus",
    "Sentence", "load_sentences",
    "iter_textcorrupt", "make_classification", "make_punctuation",
    "make_textcorrupt",
]
