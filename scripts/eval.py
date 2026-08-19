"""Evaluate a fine-tuned model on terrupt-textcorrupt.

Loads a trained checkpoint and evaluates on a val subsample with
exact-match (normalized) + chrF, broken down per severity, corruption
type, and source.

Usage:
    # Local
    python scripts/eval.py --model models/terrupt-t5-small

    # HF Hub dataset
    python scripts/eval.py --model models/terrupt-t5-small \
        --data akaruineko/terrupt-textcorrupt

    # From HF bucket checkpoint
    python scripts/eval.py --model hf://buckets/akaruineko/restratext/checkpoint-1234 \
        --precision bf16

    # Full 1.62M val (expensive ~15-45 min on A100)
    python scripts/eval.py --model models/terrupt-t5-small --n-rows 1623670
"""

import argparse
import os
import sys
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from terrupt.metrics import exact_match, build_quality_report, print_quality_report


def resolve_precision(precision_str):
    if precision_str == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return "bf16" if major >= 8 else "fp16"
        return "fp32"
    return precision_str


def load_data(data_path):
    if os.path.isdir(data_path):
        return load_from_disk(data_path)["val"]
    if os.path.isfile(data_path):
        raise ValueError(f"Expected a directory, got file: {data_path}")
    print(f"Loading val split from HuggingFace Hub: {data_path}")
    return load_dataset(data_path, split="val")


def load_model_from_path(model_path, precision):
    """Load model from local path, HF Hub ID, or HF bucket URL."""
    if model_path.startswith("hf://"):
        tmp_dir = tempfile.mkdtemp(prefix="terrupt_ckpt_")
        print(f"Downloading checkpoint from {model_path} to {tmp_dir}...")
        subprocess.run(["hf", "sync", model_path, tmp_dir], check=True)
        model_path = tmp_dir

    dtype = None
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate terrupt fine-tuned model")
    parser.add_argument("--model", type=str, required=True,
                        help="Local path, HF Hub ID, or HF bucket URL")
    parser.add_argument("--data", type=str, default="data/terrupt-textcorrupt",
                        help="Local path or HF Hub repo ID")
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--precision", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    precision = resolve_precision(args.precision)
    print(f"Loading model: {args.model} (precision={precision})")
    tokenizer, model = load_model_from_path(args.model, precision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print(f"Loading val set from {args.data}...")
    val_full = load_data(args.data)
    n = min(args.n_rows, len(val_full))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(val_full), n, replace=False)
    val_ds = val_full.select(idx.tolist())

    print(f"Running inference on {n} rows (beams={args.num_beams})...")
    preds = []
    refs = []
    severities = []
    corruption_types = []
    sources = []

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = [val_ds[i] for i in range(start, end)]

        inputs = ["restore: " + b["corrupted"] for b in batch]
        targets = [b["original"] for b in batch]

        enc = tokenizer(
            inputs, return_tensors="pt", max_length=args.max_length,
            truncation=True, padding=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc, max_length=args.max_length,
                num_beams=args.num_beams, early_stopping=True
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend(targets)
        severities.extend([b["severity"] for b in batch])
        corruption_types.extend([b["corruption_type"] for b in batch])
        sources.extend([b["source"] for b in batch])

        if (start // args.batch_size) % 5 == 0:
            print(f"  {min(start + args.batch_size, n)}/{n} done")

    # Build and print report
    print("\n" + "=" * 60)
    report = build_quality_report(preds, refs, severities=severities,
                                  corruption_types=corruption_types)
    print_quality_report(report, title="eval results", severity_key="severity")

    # Per-source breakdown (not in shared report, so do manually)
    print("-" * 60)
    print("PER SOURCE")
    print("-" * 60)
    from collections import defaultdict
    src_groups = defaultdict(lambda: {"preds": [], "refs": []})
    for i in range(len(preds)):
        src_groups[sources[i]]["preds"].append(preds[i])
        src_groups[sources[i]]["refs"].append(refs[i])
    for src in sorted(src_groups):
        g = src_groups[src]
        em = sum(exact_match(p, r) for p, r in zip(g["preds"], g["refs"])) / len(g["preds"])
        print(f"  {src:<12}: EM={em:.4f}  n={len(g['preds'])}")

    # Show sample failures
    print()
    print("-" * 60)
    print("SAMPLE FAILURES (first 5)")
    print("-" * 60)
    failures = [
        (i, preds[i], refs[i], severities[i], corruption_types[i])
        for i in range(len(preds))
        if exact_match(preds[i], refs[i]) == 0
    ]
    for i, pred, ref, sev, ct in failures[:5]:
        print(f"  sev={sev} type={ct}")
        print(f"    corrupted: {val_ds[i]['corrupted'][:100]}")
        print(f"    reference: {ref[:100]}")
        print(f"    predicted: {pred[:100]}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
