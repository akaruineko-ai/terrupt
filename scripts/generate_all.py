"""Generate all three Terrupt datasets from a single source corpus.

By default writes JSONL splits per task. With ``--format hf`` it additionally
builds a Hugging Face dataset folder (arrow + dataset_info.json + README card)
for each task.
"""

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrupt.cli import add_common_args, add_source_args, finalize, finalize_stream
from terrupt.hfexport import export_hf
from terrupt.sources import load_sentences_cli
from terrupt.tasks import (iter_textcorrupt, make_classification,
                           make_punctuation)


def main():
    parser = argparse.ArgumentParser(description="Terrupt: generate all three datasets")
    add_common_args(parser)
    add_source_args(parser)
    parser.add_argument("--severities", type=str,
                        default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--modes", type=str,
                        default="strip,wrong,random,spacing")
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--format", choices=["jsonl", "hf"], default="hf",
                        help="hf also writes arrow dataset folders + README cards")
    parser.add_argument("--keep-jsonl", action="store_true",
                        help="keep intermediate JSONL when --format hf")
    args = parser.parse_args()

    severities = tuple(float(s) for s in args.severities.split(","))
    modes = tuple(m.strip() for m in args.modes.split(","))
    sources = tuple(s.strip() for s in args.sources.split(","))
    sentences, breakdown = load_sentences_cli(
        sources, args.corpus, args.language, args.per_source,
        args.wikipedia_config, args.seed, workers=args.workers)

    jsonl_root = args.out if args.format == "jsonl" else os.path.join(args.out, "_jsonl")
    os.makedirs(jsonl_root, exist_ok=True)

    def run_task(name, build, fields):
        task_out = os.path.join(jsonl_root, name)
        os.makedirs(task_out, exist_ok=True)
        task_args = argparse.Namespace(**vars(args))
        task_args.out = task_out
        records = build()
        if name == "textcorrupt":
            stats = finalize_stream(records, task_args, name, fields,
                                    args.count or len(sentences))
        else:
            stats = finalize(records, task_args, name, fields)
        print(f"{name}: {stats['count']} records -> {task_out}")
        return task_out, stats

    rng = random.Random(args.seed)
    tc_dir, _ = run_task(
        "textcorrupt",
         lambda: iter_textcorrupt(sentences, severities, rng, args.language,
                                  args.count, workers=args.workers),
        ["corruption_type", "severity", "op_count", "language", "source"])

    rng = random.Random(args.seed + 1)
    punc_dir, _ = run_task(
        "punctuation",
        lambda: make_punctuation(sentences, rng, args.language, args.count, modes,
                                 workers=args.workers),
        ["mode", "language", "source"])

    rng = random.Random(args.seed + 2)
    cls_dir, _ = run_task(
        "classification",
        lambda: make_classification(sentences, severities, rng, args.language,
                                    args.count, args.include_clean,
                                    workers=args.workers),
        ["corruption_type", "severity", "language", "source"])

    if args.format == "hf":
        from terrupt.hfcard import build_meta
        from terrupt.engine import OP_DESCRIPTIONS, SEVERITY_SCALE

        for name, jsonl_dir in (("textcorrupt", tc_dir), ("punctuation", punc_dir),
                                ("classification", cls_dir)):
            meta = build_meta(_export_args(args, name), _load_stats(jsonl_dir),
                              None, _source_meta(breakdown, sources),
                              _corruption_types(), _example(name),
                              args.language)
            meta["severity_scale"] = SEVERITY_SCALE
            export_hf(jsonl_dir, os.path.join(args.out, name), meta)
            print(f"exported {name} dataset folder -> {os.path.join(args.out, name)}")
        if not args.keep_jsonl:
            shutil.rmtree(jsonl_root, ignore_errors=True)

    print(f"all datasets written -> {args.out}")


def _export_args(args, name):
    severities = tuple(float(s) for s in args.severities.split(","))
    meta_args = argparse.Namespace(
        name=f"Terrupt-{name.title()}" if name != "classification" else "Terrupt-Classification",
        out=os.path.join(args.out, name),
        description="",
        task=name,
        task_title=_task_title(name),
        task_category="text-classification" if name == "classification" else "text2text-generation",
        tags="text-corruption,text-restoration,corruption,benchmark",
        license="mit",
        severities=list(severities),
        severity_scale="",
        seed=args.seed,
        per_source=args.per_source,
        wikipedia_config=args.wikipedia_config,
        modes=args.modes,
        include_clean=args.include_clean,
        version="0.1.0",
        date="",
    )
    if name == "textcorrupt":
        meta_args.description = (
            "A dataset of sentences whose text has been intentionally corrupted by a "
            "controlled corruption engine. The task is to restore the original text "
            "from the corrupted input."
        )
        meta_args.task_title = "Corrupted text to original text"
    elif name == "punctuation":
        meta_args.description = (
            "A dataset of sentences whose punctuation has been stripped, replaced, "
            "randomized, or mis-spaced. The task is to restore correct punctuation."
        )
        meta_args.task_title = "Unpunctuated text to punctuated text"
    else:
        meta_args.description = (
            "A dataset of corrupted sentences labeled with the corruption types that "
            "were applied. The task is to classify which corruption operations are "
            "present in the input text."
        )
        meta_args.task_title = "Corruption type classification"
    return meta_args


def _task_title(name):
    return {
        "textcorrupt": "Corrupted text to original text",
        "punctuation": "Unpunctuated text to punctuated text",
        "classification": "Corruption type classification",
    }[name]


def _load_stats(jsonl_dir):
    import json

    with open(os.path.join(jsonl_dir, "stats.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _source_meta(breakdown, sources):
    from terrupt.sources import SOURCES, SOURCE_NAMES

    out = []
    for key in sources:
        if key not in SOURCES:
            continue
        spec = SOURCES[key]
        info = breakdown.get(key, {})
        out.append({
            "name": SOURCE_NAMES.get(key, key),
            "hf_id": spec["hf_id"],
            "config": spec["config"] or "default",
            "description": spec["description"],
            "status": info.get("status", "-"),
            "sentences": info.get("sentences", "-"),
        })
    return out


def _corruption_types():
    from terrupt.engine import OP_DESCRIPTIONS

    return [{"name": name, "description": desc}
            for name, desc in OP_DESCRIPTIONS.items()]


def _example(name):
    if name == "textcorrupt":
        return ("i l0ve pr0gramm1ng", "I love programming")
    if name == "punctuation":
        return ("hello my friend how are you today", "Hello, my friend. How are you today?")
    return ("i l0ve pr0gramm1ng", "leetspeak, case")


if __name__ == "__main__":
    main()
