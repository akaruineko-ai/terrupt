"""Fine-tune T5-small on terrupt-textcorrupt (corrupted -> original).

Loads the paired dataset from local disk or HuggingFace Hub, builds a
hard-weighted subsample (or uses all 29M rows), tokenizes, and trains
with the HuggingFace Seq2SeqTrainer. Supports LoRA and HF bucket sync.

Usage:
    # Local 1660 Ti (5M hard-weighted, ~5-20h)
    python scripts/finetune.py --data data/terrupt-textcorrupt

    # Full 29M on A100 (1 epoch, ~1.5h)
    python scripts/finetune.py --data akaruineko/terrupt-textcorrupt \
        --target-rows -1 --epochs 1 --precision bf16

    # Colab with HF bucket checkpoint sync
    python scripts/finetune.py --data akaruineko/terrupt-textcorrupt \
        --target-rows -1 --precision bf16 \
        --hf-bucket hf://buckets/akaruineko/restratext

    # LoRA on T5-large
    python scripts/finetune.py --model t5-large --lora --lora-r 16 \
        --lora-alpha 32 --target-rows -1 --precision bf16
"""

import argparse
import os
import random
import subprocess
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
    TrainerCallback,
)


# ---------------------------------------------------------------------------
# HF bucket sync
# ---------------------------------------------------------------------------

def sync_to_hf_bucket(local_dir, bucket_url):
    """Sync a local directory to an HF bucket via ``hf sync``."""
    try:
        subprocess.run(
            ["hf", "sync", local_dir, bucket_url],
            check=True, capture_output=True, text=True,
        )
        print(f"  Synced {local_dir} -> {bucket_url}")
    except FileNotFoundError:
        print("  Warning: 'hf' CLI not found, skipping bucket sync")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: bucket sync failed: {e.stderr.strip()}")


class HFSyncCallback(TrainerCallback):
    """Sync checkpoint dir to HF bucket after every save."""

    def __init__(self, bucket_url, local_dir):
        self.bucket_url = bucket_url
        self.local_dir = local_dir

    def on_save(self, args, state, control, **kwargs):
        sync_to_hf_bucket(self.local_dir, self.bucket_url)


# ---------------------------------------------------------------------------
# Precision helper
# ---------------------------------------------------------------------------

def resolve_precision(precision_str):
    """Map --precision flag to TrainingArguments fp16/bf16 kwargs."""
    if precision_str == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return "bf16" if major >= 8 else "fp16"
        return "fp32"
    return precision_str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path):
    """Load dataset from local path or HuggingFace Hub repo ID."""
    if os.path.isdir(data_path):
        return load_from_disk(data_path)
    # Assume HF Hub repo ID (e.g. "akaruineko/terrupt-textcorrupt")
    print(f"Loading dataset from HuggingFace Hub: {data_path}")
    ds = load_dataset(data_path)
    # Normalize: HF datasets may return train-only if no split info
    if "train" not in ds:
        key = list(ds.keys())[0]
        ds = ds.rename_column(key, "train") if key != "train" else ds
    return ds


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def build_subsample(dataset, target_n=5_000_000, seed=42):
    """Build a hard-weighted subsample: keep severity 1.0 heavily, thin out 0.1/0.25.

    Returns list of indices, or None if target_n < 0 (= use all rows).
    """
    if target_n < 0:
        print(f"Using full dataset: {len(dataset)} rows")
        return None

    rng = random.Random(seed)
    total = len(dataset)

    # Extract severities in bulk (fast column access)
    severities = dataset["severity"]

    # Bucket indices by severity
    buckets = {0.1: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    for i, sev in enumerate(severities):
        buckets[sev].append(i)

    # Target 5M: 3M from severity 1.0, 1.5M from 0.75, 500k from 0.5
    plan = {
        1.0: min(len(buckets[1.0]), 3_000_000),
        0.75: min(len(buckets[0.75]), 1_500_000),
        0.5: min(len(buckets[0.5]), 500_000),
        0.25: min(len(buckets[0.25]), 0),
        0.1: min(len(buckets[0.1]), 0),
    }

    selected = []
    for sev, n_keep in plan.items():
        pool = buckets[sev]
        if n_keep >= len(pool):
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, n_keep))

    rng.shuffle(selected)
    print(f"Subsample: {len(selected)} rows from {total}")
    for sev in sorted(plan):
        actual = sum(1 for i in selected if severities[i] == sev)
        print(f"  severity {sev}: {actual}")
    return selected


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_length=64, n_proc=1,
                     cache_file_prefix=None, force_retokenize=False):
    """Tokenize corrupted->original with 'restore:' prefix."""
    cache_files = None
    load_cache = not force_retokenize
    if cache_file_prefix:
        cache_dir = os.path.dirname(cache_file_prefix)
        os.makedirs(cache_dir, exist_ok=True)
        cache_files = [f"{cache_file_prefix}_proc{i}" for i in range(n_proc)]

    def preprocess(examples):
        inputs = ["restore: " + c for c in examples["corrupted"]]
        targets = examples["original"]
        model_inputs = tokenizer(
            inputs, max_length=max_length, truncation=True, padding=False
        )
        labels = tokenizer(
            targets, max_length=max_length, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess,
        batched=True,
        batch_size=512,
        num_proc=n_proc,
        load_from_cache_file=load_cache if cache_file_prefix else True,
        cache_file_names={f"proc{i}": cf for i, cf in enumerate(cache_files)} if cache_files else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 on terrupt-textcorrupt (corrupted -> original)"
    )
    # Data
    parser.add_argument("--data", type=str, default="data/terrupt-textcorrupt",
                        help="Local path or HF Hub repo ID (e.g. akaruineko/terrupt-textcorrupt)")
    parser.add_argument("--out", type=str, default="models/terrupt-t5-small",
                        help="Output directory for checkpoints and final model")
    parser.add_argument("--model", type=str, default="t5-small",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--target-rows", type=int, default=5_000_000,
                        help="Number of train rows (-1 = use all 29M)")
    parser.add_argument("--val-rows", type=int, default=20_000,
                        help="Number of val rows for eval")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Max token length for inputs and labels")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="Number of workers for tokenization")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--precision", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])

    # LoRA
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA adapter (requires peft)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Tokenization cache
    parser.add_argument("--force-retokenize", action="store_true",
                        help="Ignore tokenization cache and recompute")
    parser.add_argument("--tokenize-cache-dir", type=str, default=None,
                        help="Directory for tokenization cache files")

    # HuggingFace bucket sync
    parser.add_argument("--hf-bucket", type=str, default=None,
                        help="HF bucket URL for checkpoint sync (e.g. hf://buckets/akaruineko/restratext)")

    # Resume
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path or HF bucket URL to resume training from")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -----------------------------------------------------------------------
    # Model + tokenizer
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # LoRA
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        print(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "v"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"Loading dataset: {args.data}")
    ds = load_data(args.data)
    train_full = ds["train"]
    val_full = ds["val"]

    # Subsample train
    print("Building training split...")
    train_idx = build_subsample(train_full, args.target_rows, args.seed)
    if train_idx is not None:
        train_ds = train_full.select(train_idx)
    else:
        train_ds = train_full

    # Val subset
    val_rng = random.Random(args.seed)
    val_idx = val_rng.sample(range(len(val_full)), min(args.val_rows, len(val_full)))
    val_ds = val_full.select(val_idx)

    # Tokenize (cached by default)
    n_proc = args.num_proc if args.num_proc > 0 else (os.cpu_count() or 1)
    cache_prefix = args.tokenize_cache_dir
    print(f"Tokenizing train ({len(train_ds)}) and val ({len(val_ds)}) [workers={n_proc}]...")
    train_ds = tokenize_dataset(
        train_ds, tokenizer, max_length=args.max_length, n_proc=n_proc,
        cache_file_prefix=cache_prefix,
        force_retokenize=args.force_retokenize,
    )
    val_ds = tokenize_dataset(
        val_ds, tokenizer, max_length=args.max_length, n_proc=n_proc,
        cache_file_prefix=f"{cache_prefix}_val" if cache_prefix else None,
        force_retokenize=args.force_retokenize,
    )

    # -----------------------------------------------------------------------
    # Training args
    # -----------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    effective_batch = args.batch_size * args.grad_accum
    precision = resolve_precision(args.precision)

    training_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=min(n_proc, 8),
        predict_with_generate=False,
        seed=args.seed,
    )
    if precision == "fp16":
        training_kwargs["fp16"] = True
    elif precision == "bf16":
        training_kwargs["bf16"] = True
    # fp32: no flag needed (default)

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # HF bucket sync callback
    if args.hf_bucket:
        print(f"Checkpoint sync enabled: {args.hf_bucket}")
        trainer.add_callback(HFSyncCallback(args.hf_bucket, args.out))

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("\nStarting training...")
    print(f"  Model:          {args.model}")
    print(f"  Train rows:     {len(train_ds)}")
    print(f"  Val rows:       {len(val_ds)}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Effective batch:{effective_batch}")
    print(f"  LR:             {args.lr}")
    print(f"  Precision:      {precision}")
    print(f"  LoRA:           {'yes' if args.lora else 'no'}")
    print(f"  Output:         {args.out}")
    if args.hf_bucket:
        print(f"  HF bucket:      {args.hf_bucket}")
    print()

    t0 = time.time()
    if args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # Save + sync
    # -----------------------------------------------------------------------
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Saving final model to {args.out}...")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    if args.hf_bucket:
        sync_to_hf_bucket(args.out, args.hf_bucket)

    print("Done.")


if __name__ == "__main__":
    main()
