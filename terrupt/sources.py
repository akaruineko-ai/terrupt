"""Sentence sourcing from Hugging Face datasets.

Terrupt draws its base sentences from public Hugging Face corpora. Each
source is streamed (no full download), cleaned, and split into sentences with
a per-source cap so the corpus can scale to any budget.
"""

import json
import os
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

from terrupt.corpora import _SENT_SPLIT, _clean, _valid
from terrupt.progress import Progress

DEFAULT_SOURCES = ("wikipedia", "reddit", "textbooks", "stories")

SOURCE_NAMES = {
    "wikipedia": "Wikipedia",
    "reddit": "Reddit",
    "textbooks": "Tiny Textbooks",
    "stories": "TinyStories",
}


@dataclass(frozen=True)
class Sentence:
    text: str
    source: str


def _remove_braces(text, opener, closer):
    out = []
    depth = 0
    for ch in text:
        if ch == opener:
            depth += 1
            if depth == 1:
                continue
            continue
        if depth:
            if ch == closer:
                depth -= 1
            continue
        out.append(ch)
    return "".join(out)


_FLAT_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")


def _clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = _FLAT_TEMPLATE.sub(" ", text)
    if "{{" in text:
        text = _remove_braces(text, "{", "}")
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"<ref[^>]*/>", " ", text, flags=re.I)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[(?:[^|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://[^\s\]]*\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://[^\]]*\]", "", text)
    text = re.sub(r"(?:https?|ftp)://\S+", "", text)
    text = re.sub(r"[\n\r]+", " ", text)
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r"''+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_reddit(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?:https?|ftp)://\S+", "", text)
    text = re.sub(r"[*_`>#~^]", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<")
    text = text.replace("&gt;", ">").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _clean_basic(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


SOURCES = {
    "wikipedia": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.en",
        "column": "text",
        "clean": _clean_wikitext,
        "description": "Encyclopedia articles with wikitext markup stripped.",
    },
    "reddit": {
        "hf_id": "trungnam299/reddit_dataset_44",
        "config": None,
        "column": "text",
        "clean": _clean_reddit,
        "description": "Reddit posts (44 datasets merged) with markdown and URLs stripped.",
    },
    "textbooks": {
        "hf_id": "nampdn-ai/tiny-textbooks",
        "config": None,
        "column": "text",
        "clean": _clean_basic,
        "description": "Deduplicated textbook-style passages.",
    },
    "stories": {
        "hf_id": "roneneldan/TinyStories",
        "config": None,
        "column": "text",
        "clean": _clean_basic,
        "description": "Short synthetic stories for young children.",
    },
}


def _iter_sentences(text):
    for sentence in _SENT_SPLIT.split(text):
        cleaned = _clean(sentence)
        if cleaned and _valid(cleaned):
            yield cleaned


def _raw_stream(ds, column):
    for row in ds:
        raw = row.get(column)
        if isinstance(raw, str) and raw.strip():
            yield raw


_PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"


def _resolve_parquet_shards(spec, wikipedia_config):
    """Metadata for a source's parquet shards, or ``None`` if unresolvable.

    Enumerates shards through the datasets-server ``/parquet`` endpoint. The
    actual shard files are downloaded lazily inside the worker processes.
    Returns ``None`` when the endpoint is unavailable (gated datasets, network
    errors) so callers can fall back to the streaming path.
    """
    config = wikipedia_config if spec["hf_id"] == "wikimedia/wikipedia" else spec["config"]
    config = config or "default"
    url = f"{_PARQUET_ENDPOINT}?dataset={spec['hf_id']}&config={config}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except Exception:
        return None
    shards = []
    for entry in payload.get("parquet_files") or []:
        if entry.get("split") != "train":
            continue
        raw_url = entry.get("url")
        if not raw_url or "/resolve/" not in raw_url:
            continue
        revision, path = raw_url.split("/resolve/", 1)[1].split("/", 1)
        shards.append({
            "repo_id": spec["hf_id"],
            "revision": urllib.parse.unquote(revision),
            "filename": path,
            "size": entry.get("size") or 0,
        })
    return shards or None


def _read_shards(task):
    """Worker: decode, clean, and split one parquet shard."""
    shard, column, clean, cap = task
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(shard["repo_id"], shard["filename"],
                           repo_type="dataset", revision=shard["revision"])
    out = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=[column]):
        for raw in batch[column].to_pylist():
            if not isinstance(raw, str) or not raw.strip():
                continue
            cleaned = clean(raw)
            if not cleaned:
                continue
            for sentence in _iter_sentences(cleaned):
                out.append(sentence)
                if cap and len(out) >= cap:
                    return out
    return out


def _collect(spec, per_source, wikipedia_config, desc="loading", progress=True,
             workers=None):
    """Clean and split sentences from a source.

    When ``workers > 1`` and the source's parquet shards can be resolved, each
    worker decodes and cleans its own shards in parallel (this scales both the
    decode and the pure-Python regex work). Otherwise a serial streaming path
    is used (gated sources, ``workers == 1``, or no network).
    """
    workers = workers or os.cpu_count() or 1
    workers = max(1, int(workers))
    clean = spec["clean"]
    column = spec["column"]
    out = []
    seen = set()
    pbar = Progress(total=per_source if per_source else None,
                    desc=desc, unit="sentences", disable=not progress)

    def accumulate(sentences):
        added = 0
        for sentence in sentences:
            if sentence in seen:
                continue
            seen.add(sentence)
            out.append(sentence)
            added += 1
        if added:
            pbar.update(added)

    def finish():
        return out[:per_source] if per_source else out

    shards = None
    if workers > 1:
        shards = _resolve_parquet_shards(spec, wikipedia_config)
    if shards:
        from multiprocessing import Pool

        pool = Pool(processes=workers)
        try:
            with pbar:
                for sentences in pool.imap_unordered(
                        _read_shards, [(s, column, clean, per_source) for s in shards]):
                    accumulate(sentences)
                    if per_source and len(out) >= per_source:
                        break
        finally:
            pool.terminate()
            pool.join()
        return finish()

    try:
        from datasets import load_dataset

        config = wikipedia_config if spec["hf_id"] == "wikimedia/wikipedia" else spec["config"]
        ds = load_dataset(spec["hf_id"], config, split="train", streaming=True)
        with pbar:
            batch = []
            for raw in _raw_stream(ds, column):
                cleaned = clean(raw)
                if not cleaned:
                    continue
                for sentence in _iter_sentences(cleaned):
                    batch.append(sentence)
                    if len(batch) >= 100:
                        accumulate(batch)
                        batch = []
                        if per_source and len(out) >= per_source:
                            return finish()
            if batch:
                accumulate(batch)
            return finish()
    finally:
        pass


def load_sentences(sources=DEFAULT_SOURCES, per_source=20000, language="en",
                   wikipedia_config=None, rng=None, progress=True, workers=None):
    """Load sentences from the given HF sources.

    Returns ``(sentences, breakdown)`` where ``sentences`` is a list of
    :class:`Sentence` and ``breakdown`` maps each source to loading status,
    sentence count, and language.
    """
    rng = rng or random.Random()
    all_sentences = []
    breakdown = {}
    for key in sources:
        if key not in SOURCES:
            breakdown[key] = {"language": language, "status": "unknown source"}
            continue
        spec = SOURCES[key]
        try:
            collected = _collect(spec, per_source, wikipedia_config,
                                 desc=f"source: {key}", progress=progress,
                                 workers=workers)
        except Exception as exc:
            breakdown[key] = {
                "language": language,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            collected = []
        breakdown[key] = {
            "language": language,
            "status": "ok" if collected else "empty",
            "sentences": len(collected),
        }
        all_sentences.extend(Sentence(s, key) for s in collected)
    rng.shuffle(all_sentences)
    if not all_sentences:
        raise ValueError(
            "no sentences could be loaded from the given sources; "
            "check --sources or use --corpus with a plain-text file"
        )
    return all_sentences, breakdown


def load_sentences_cli(sources, corpus, language, per_source, wikipedia_config, seed,
                       progress=True, workers=None):
    """CLI wrapper: prefer a local corpus file, otherwise load HF sources."""
    rng = random.Random(seed)
    if corpus:
        from terrupt.corpora import load_corpus

        sentences = [Sentence(s, os.path.basename(corpus)) for s in load_corpus(corpus)]
        breakdown = {
            os.path.basename(corpus): {
                "language": language,
                "status": "ok",
                "sentences": len(sentences),
            }
        }
        return sentences, breakdown
    return load_sentences(sources, per_source, language, wikipedia_config, rng,
                          progress, workers)
