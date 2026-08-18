"""Convert a directory of JSONL split files into a Hugging Face dataset folder.

Writes arrow files, ``dataset_info.json``, and an auto-generated ``README.md``
dataset card that documents the source corpora.
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrupt.engine import OP_DESCRIPTIONS, SEVERITY_SCALE
from terrupt.hfcard import build_meta
from terrupt.hfexport import export_hf, export_hf_stream
from terrupt.sources import SOURCES, SOURCE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Terrupt: export JSONL splits to HF dataset folder")
    parser.add_argument("--in", dest="input_dir", type=str, required=True,
                        help="directory with *_train.jsonl / *_val.jsonl / *_test.jsonl")
    parser.add_argument("--out", type=str, required=True,
                        help="output dataset folder")
    parser.add_argument("--name", type=str, default=None,
                        help="dataset name (default: input dir basename)")
    parser.add_argument("--description", type=str, default="",
                        help="dataset summary")
    parser.add_argument("--task", type=str, default=None,
                        choices=["textcorrupt", "punctuation", "classification"],
                        help="task id (auto-detected from schema by default)")
    parser.add_argument("--task-title", type=str, default=None)
    parser.add_argument("--task-category", type=str, default=None,
                        choices=["text2text-generation", "text-classification", "other"])
    parser.add_argument("--sources", type=str, default="wikipedia,reddit,textbooks,stories",
                        help="HF source keys used to build the corpus (for the card)")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--license", type=str, default="mit")
    parser.add_argument("--tags", type=str, default="text-corruption,text-restoration,corruption,benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-source", type=int, default=20000)
    parser.add_argument("--wikipedia-config", type=str, default="20231101.en")
    parser.add_argument("--modes", type=str, default="strip,wrong,random,spacing")
    parser.add_argument("--severities", type=str, default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--stream", action="store_true",
                        help="stream JSONL row-by-row (memory-safe, recommended for >1M rows)")
    parser.add_argument("--max-shard-size", type=str, default="2GB",
                        help="max arrow shard size (default: 2GB)")
    parser.add_argument("--version", type=str, default="0.1.0")
    args = parser.parse_args()

    if args.name is None:
        args.name = os.path.basename(args.input_dir.rstrip("/"))
    task = args.task or _detect_task(args.input_dir)
    if not args.description:
        args.description = _default_description(task)

    sources = tuple(s.strip() for s in args.sources.split(","))
    source_meta = []
    for key in sources:
        if key not in SOURCES:
            continue
        spec = SOURCES[key]
        source_meta.append({
            "name": SOURCE_NAMES.get(key, key),
            "hf_id": spec["hf_id"],
            "config": spec["config"] or "default",
            "description": spec["description"],
            "status": "ok",
            "sentences": None,
        })

    meta_args = argparse.Namespace(
        name=args.name, out=args.out, description=args.description,
        task=task,
        task_title=args.task_title or _task_title(task),
        task_category=args.task_category or _task_category(task),
        tags=args.tags, license=args.license,
        severities=[float(s) for s in args.severities.split(",")],
        severity_scale=SEVERITY_SCALE,
        seed=args.seed, per_source=args.per_source,
        wikipedia_config=args.wikipedia_config,
        modes=args.modes, include_clean=args.include_clean,
        version=args.version, date=datetime.date.today().isoformat(),
    )
    stats = _load_stats(args.input_dir)
    meta = build_meta(meta_args, stats, None, source_meta,
                      [{"name": n, "description": d} for n, d in OP_DESCRIPTIONS.items()],
                      _example(task), args.language)
    if args.stream:
        export_hf_stream(args.input_dir, args.out, meta,
                         max_shard_size=args.max_shard_size)
    else:
        export_hf(args.input_dir, args.out, meta)
    print(f"exported {args.name} ({task}) -> {args.out}")


def _detect_task(input_dir):
    from terrupt.hfexport import discover_splits, _read_records

    splits = discover_splits(input_dir)
    if not splits:
        raise SystemExit("no JSONL split files found; pass --task")
    first_paths = next(iter(splits.values()))
    sample = _read_records(first_paths, limit=1)
    if not sample:
        raise SystemExit("no records found to detect task; pass --task")
    keys = set(sample[0])
    if "labels" in keys:
        return "classification"
    if "corrupted" in keys:
        return "textcorrupt"
    if "mode" in keys:
        return "punctuation"
    raise SystemExit("cannot detect task from schema; pass --task")


def _task_title(task):
    return {
        "textcorrupt": "Corrupted text to original text",
        "punctuation": "Unpunctuated text to punctuated text",
        "classification": "Corruption type classification",
    }[task]


def _task_category(task):
    return "text-classification" if task == "classification" else "text2text-generation"


def _default_description(task):
    return {
        "textcorrupt": (
            "A dataset of sentences whose text has been intentionally corrupted by a "
            "controlled corruption engine. The task is to restore the original text "
            "from the corrupted input."
        ),
        "punctuation": (
            "A dataset of sentences whose punctuation has been stripped, replaced, "
            "randomized, or mis-spaced. The task is to restore correct punctuation."
        ),
        "classification": (
            "A dataset of corrupted sentences labeled with the corruption types that "
            "were applied. The task is to classify which corruption operations are "
            "present in the input text."
        ),
    }[task]


def _example(task):
    return {
        "textcorrupt": ("i l0ve pr0gramm1ng", "I love programming"),
        "punctuation": ("hello my friend how are you today", "Hello, my friend. How are you today?"),
        "classification": ("i l0ve pr0gramm1ng", "leetspeak, case"),
    }[task]


def _load_stats(input_dir):
    path = os.path.join(input_dir, "stats.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


if __name__ == "__main__":
    main()