"""Shared CLI helpers for the dataset generator scripts."""

import json
import math
import os
import random

from terrupt.progress import Progress


def parse_splits(spec):
    splits = {}
    for part in spec.split(","):
        name, ratio = part.rsplit("=", 1)
        splits[name.strip()] = float(ratio)
    return splits


def assign_splits(records, splits, rng):
    total = len(records)
    names = list(splits)
    raw = {n: total * splits[n] for n in names}
    counts = {n: math.floor(raw[n]) for n in names}
    remainder = total - sum(counts.values())
    order = sorted(names, key=lambda n: raw[n] - counts[n], reverse=True)
    for n in order[:remainder]:
        counts[n] += 1
    counts[names[-1]] += total - sum(counts.values())
    rng.shuffle(records)
    assigned = []
    pos = 0
    for n in names:
        assigned.append((n, records[pos:pos + counts[n]]))
        pos += counts[n]
    return assigned


def write_records(path, records):
    pbar = Progress(total=len(records),
                    desc=f"writing {os.path.basename(path)}", unit="rows")
    with open(path, "w", encoding="utf-8") as fh:
        with pbar:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False))
                fh.write("\n")
                pbar.update(1)


def write_split(path_prefix, split_name, records, shards):
    if shards <= 1 or not records:
        write_records(f"{path_prefix}_{split_name}.jsonl", records)
        return
    size = math.ceil(len(records) / shards)
    for s in range(shards):
        chunk = records[s * size:(s + 1) * size]
        if chunk:
            write_records(f"{path_prefix}_{split_name}_{s:02d}.jsonl", chunk)


def write_stats(path, stats):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)


def summarize(records, fields):
    stats = {"count": len(records)}
    for field in fields:
        counts = {}
        for rec in records:
            value = rec.get(field)
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        stats[field] = counts
    return stats


def add_source_args(parser):
    parser.add_argument("--sources", type=str,
                        default="wikipedia,reddit,textbooks,stories",
                        help="comma-separated Hugging Face source keys "
                             "(wikipedia, reddit, textbooks, stories)")
    parser.add_argument("--per-source", type=int, default=20000,
                        help="max sentences to draw per source (0 = unlimited)")
    parser.add_argument("--wikipedia-config", type=str, default="20231101.en",
                        help="wikipedia dump config, e.g. 20231101.en")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers for sentence extraction "
                             "(default: number of CPU cores)")


def add_common_args(parser):
    parser.add_argument("--count", type=int, default=10000,
                        help="number of examples to generate (0 = all available)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=str, default=None,
                        help="plain-text corpus file (otherwise built-in sentences)")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    parser.add_argument("--prefix", type=str, default=None,
                        help="output file prefix (default: directory name)")
    parser.add_argument("--splits", type=str,
                        default="train=0.9,val=0.05,test=0.05")
    parser.add_argument("--shards", type=int, default=1,
                        help="number of shard files per split")


def finalize(records, args, prefix, summary_fields):
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed + 999)
    splits = parse_splits(args.splits)
    assigned = assign_splits(records, splits, rng)
    prefix = prefix or os.path.basename(args.out.rstrip("/"))
    path_prefix = os.path.join(args.out, prefix)
    split_counts = {}
    for name, chunk in assigned:
        write_split(path_prefix, name, chunk, args.shards)
        split_counts[name] = len(chunk)
    stats = summarize(records, summary_fields)
    stats["splits"] = split_counts
    write_stats(os.path.join(args.out, "stats.json"), stats)
    import gc
    gc.collect()
    return stats
