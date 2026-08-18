"""Benchmark T5 training throughput on terrupt-textcorrupt.

Trains ~200 steps on a small subsample and reports samples/sec so you
can extrapolate exact wall-clock time before committing to the full run.

Usage:
    # Local 1660 Ti
    python scripts/benchmark.py

    # HF dataset on A100
    python scripts/benchmark.py --data akaruineko/terrupt-textcorrupt --precision bf16
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def resolve_precision(precision_str):
    if precision_str == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return "bf16" if major >= 8 else "fp16"
        return "fp32"
    return precision_str


def load_data(data_path):
    if os.path.isdir(data_path):
        return load_from_disk(data_path)["train"]
    print(f"Loading dataset from HuggingFace Hub: {data_path}")
    ds = load_dataset(data_path, split="train")
    return ds


def build_dataset(data_dir, tokenizer, n_rows=10_000):
    ds = load_data(data_dir)
    rng = random.Random(42)
    idx = rng.sample(range(len(ds)), n_rows)
    ds = ds.select(idx)

    def preprocess(examples):
        inputs = ["restore: " + c for c in examples["corrupted"]]
        targets = examples["original"]
        model_inputs = tokenizer(
            inputs, max_length=64, truncation=True, padding=False
        )
        labels = tokenizer(
            targets, max_length=64, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(preprocess, batched=True, batch_size=512, num_proc=1)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Benchmark T5 training throughput")
    parser.add_argument("--data", type=str, default="data/terrupt-textcorrupt",
                        help="Local path or HF Hub repo ID")
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.gradient_checkpointing_enable()

    print(f"Building 10k-row dataset from {args.data}...")
    train_ds = build_dataset(args.data, tokenizer, n_rows=10_000)

    precision = resolve_precision(args.precision)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    out_dir = "/tmp/terrupt_benchmark"
    training_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_steps=args.steps,
        dataloader_num_workers=2,
    )
    if precision == "fp16":
        training_kwargs["fp16"] = True
    elif precision == "bf16":
        training_kwargs["bf16"] = True

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"Running {args.steps} steps (batch {args.batch_size} x grad_accum {args.grad_accum}, precision {precision})...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    effective_batch = args.batch_size * args.grad_accum
    total_samples = args.steps * effective_batch
    samples_per_sec = total_samples / elapsed

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Steps:            {args.steps}")
    print(f"Effective batch:  {effective_batch}")
    print(f"Total samples:    {total_samples}")
    print(f"Wall time:        {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Samples/sec:      {samples_per_sec:.1f}")
    print(f"Model:            {args.model}")
    print(f"Precision:        {precision}")
    print()

    # Extrapolation table
    print("-" * 60)
    print("EXTRAPOLATION (hours)")
    print("-" * 60)
    print(f"{'Rows':>10}  {'1 ep':>8}  {'2 ep':>8}  {'3 ep':>8}")
    print("-" * 60)
    for n_rows in [1_000_000, 3_000_000, 5_000_000, 10_000_000, 29_000_000]:
        row = []
        for n_epochs in [1, 2, 3]:
            total = n_rows * n_epochs
            hours = total / samples_per_sec / 3600
            row.append(f"{hours:.1f}h")
        print(f"{n_rows:>10,}  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
