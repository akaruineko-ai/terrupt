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


def finalize_stream(records, args, prefix, summary_fields, total):
    """Write a known-size record iterator without retaining all records."""
    os.makedirs(args.out, exist_ok=True)
    splits = parse_splits(args.splits)
    raw = {name: total * splits[name] for name in splits}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(splits, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    prefix = prefix or os.path.basename(args.out.rstrip("/"))
    handles = {}
    stats = {"count": 0}
    field_counts = {field: {} for field in summary_fields}
    split_seen = {name: 0 for name in splits}
    boundaries = []
    end = 0
    for name in splits:
        end += counts[name]
        boundaries.append((end, name))

    def path_for(name, index):
        if args.shards <= 1:
            return os.path.join(args.out, f"{prefix}_{name}.jsonl")
        size = math.ceil(counts[name] / args.shards)
        return os.path.join(args.out, f"{prefix}_{name}_{index // size:02d}.jsonl")

    try:
        for index, record in enumerate(records):
            for boundary, name in boundaries:
                if index < boundary:
                    break
            local_index = split_seen[name]
            split_seen[name] += 1
            path = path_for(name, local_index)
            if path not in handles:
                handles[path] = open(path, "w", encoding="utf-8")
            handles[path].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["count"] += 1
            for field in summary_fields:
                key = str(record.get(field))
                field_counts[field][key] = field_counts[field].get(key, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    stats.update(field_counts)
    stats["splits"] = split_seen
    write_stats(os.path.join(args.out, "stats.json"), stats)
    return stats
