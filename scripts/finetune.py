"""Fine-tune T5 on terrupt-textcorrupt (corrupted -> original).

Loads the paired dataset from local disk or HuggingFace Hub, builds a
hard-weighted subsample (or uses all 29M rows), tokenizes, and trains
with the HuggingFace Seq2SeqTrainer. Supports LoRA, HF bucket sync,
configurable checkpoint frequency with storage-limit pruning, and
per-severity generation quality evaluation.

Usage:
    # Local 1660 Ti (5M hard-weighted)
    python scripts/finetune.py --data data/terrupt-textcorrupt

    # Full 29M on A100 (1 epoch), save every 5k steps
    python scripts/finetune.py --data akaruineko/terrupt-textcorrupt \
        --target-rows -1 --epochs 1 --precision bf16 \
        --save-steps 5000

    # Colab with bucket sync + 50GB checkpoint cap + end-of-training gen eval
    python scripts/finetune.py --data akaruineko/terrupt-textcorrupt \
        --target-rows -1 --precision bf16 \
        --hf-bucket hf://buckets/akaruineko/restratext \
        --limit-checkpoints-folder 50gb \
        --full-eval-at-end

    # LoRA on T5-large with 20GB checkpoint cap
    python scripts/finetune.py --model t5-large --lora --lora-r 16 \
        --lora-alpha 32 --target-rows -1 --precision bf16 \
        --limit-checkpoints-folder 20gb
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from terrupt.metrics import build_quality_report, print_quality_report, quality_report_to_log


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^([\d.]+)\s*(b|kb|mb|gb|tb|k|m|g|t)?$", re.IGNORECASE)
_UNITS = {
    "b": 1, "k": 1 << 10, "kb": 1 << 10,
    "m": 1 << 20, "mb": 1 << 20,
    "g": 1 << 30, "gb": 1 << 30,
    "t": 1 << 40, "tb": 1 << 40,
}


def parse_size(s):
    """Parse human-readable size string ('50gb', '2.5tb', '100mb', '4096') to bytes."""
    m = _SIZE_RE.match(s.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid size '{s}': use format like '50gb', '2.5tb', '100mb', or bytes"
        )
    value = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    return int(value * _UNITS[unit])


def dir_size(path):
    """Recursive total file size of a directory in bytes."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _checkpoint_step(name):
    """Extract numeric step from 'checkpoint-1234' directory name."""
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else -1


def checkpoint_dirs(output_dir):
    """Return checkpoint directory names sorted by step (oldest first)."""
    dirs = []
    if not os.path.isdir(output_dir):
        return dirs
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-") and os.path.isdir(
            os.path.join(output_dir, name)
        ):
            dirs.append(name)
    dirs.sort(key=_checkpoint_step)
    return dirs


# ---------------------------------------------------------------------------
# HF bucket sync
# ---------------------------------------------------------------------------

def sync_to_hf_bucket(local_dir, bucket_url, delete_remote=False):
    """Sync a local directory to an HF bucket via ``hf sync``."""
    cmd = ["hf", "sync"]
    if delete_remote:
        cmd.append("--delete")
    cmd.extend([local_dir, bucket_url])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Synced {local_dir} -> {bucket_url}")
    except FileNotFoundError:
        print("  Warning: 'hf' CLI not found, skipping bucket sync")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: bucket sync failed: {e.stderr.strip()}")


class HFSyncCallback(TrainerCallback):
    """Sync checkpoint dir to HF bucket after every save."""

    def __init__(self, bucket_url, local_dir, delete_remote=False):
        self.bucket_url = bucket_url
        self.local_dir = local_dir
        self.delete_remote = delete_remote

    def on_save(self, args, state, control, **kwargs):
        sync_to_hf_bucket(self.local_dir, self.bucket_url, self.delete_remote)


# ---------------------------------------------------------------------------
# Storage-limit callback
# ---------------------------------------------------------------------------

class StorageLimitCallback(TrainerCallback):
    """Delete oldest checkpoint-* dirs when output_dir exceeds a byte-size cap."""

    def __init__(self, output_dir, limit_bytes):
        self.output_dir = output_dir
        self.limit_bytes = limit_bytes

    def on_save(self, args, state, control, **kwargs):
        limit = self.limit_bytes
        protected = set()
        if state.best_model_checkpoint:
            protected.add(os.path.basename(state.best_model_checkpoint))
        protected.add(f"checkpoint-{state.global_step}")

        total = dir_size(self.output_dir)
        if total <= limit:
            return

        print(f"\n  [storage-limit] checkpoint dir = {total / (1 << 30):.1f} GB "
              f"> {limit / (1 << 30):.1f} GB limit — pruning old checkpoints")

        while total > limit:
            oldest = next(
                (d for d in checkpoint_dirs(self.output_dir) if d not in protected),
                None,
            )
            if oldest is None:
                print(f"  [storage-limit] cannot prune further: only protected "
                      f"checkpoints remain ({total / (1 << 30):.1f} GB)")
                break
            target = os.path.join(self.output_dir, oldest)
            freed = dir_size(target)
            shutil.rmtree(target, ignore_errors=True)
            total -= freed
            print(f"  [storage-limit] deleted {oldest} (freed {freed / (1 << 30):.1f} GB, "
                  f"now {total / (1 << 30):.1f} GB)")
        print()


# ---------------------------------------------------------------------------
# Quality-eval callback (generation-based per-severity metrics)
# ---------------------------------------------------------------------------

class QualityEvalCallback(TrainerCallback):
    """Run generation-based quality eval on a fixed stratified subset each epoch.

    Produces per-severity exact-match + chrF tables printed to stdout and
    logged to the Trainer metric history.

    Optionally runs a final full-val generation pass after training.
    """

    def __init__(self, trainer, tokenizer, quality_subset, severity_list,
                 eval_num_beams=4, eval_batch_size=32, eval_max_length=64,
                 eval_every=1, full_eval_at_end=False, full_val_ds=None,
                 full_val_severities=None, full_val_refs=None):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.quality_subset = quality_subset
        self.severity_list = severity_list
        self.eval_num_beams = eval_num_beams
        self.eval_batch_size = eval_batch_size
        self.eval_max_length = eval_max_length
        self.eval_every = eval_every
        self.full_eval_at_end = full_eval_at_end
        self.full_val_ds = full_val_ds
        self.full_val_severities = full_val_severities
        self.full_val_refs = full_val_refs

    def _run_generation_eval(self, dataset, severity_list, refs, label):
        """Run generate() over a dataset, return quality report dict."""
        self.trainer.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.trainer.data_collator,
            shuffle=False,
            num_workers=2,
        )
        all_preds = []
        all_refs = []
        all_sevs = []
        idx = 0
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if k != "severity"}
            labels = batch.pop("labels", None)
            with torch.no_grad():
                generated = self.trainer.model.generate(
                    **batch,
                    max_length=self.eval_max_length,
                    num_beams=self.eval_num_beams,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            n = len(decoded)
            all_preds.extend(decoded)
            all_refs.extend(refs[idx:idx + n])
            all_sevs.extend(severity_list[idx:idx + n])
            idx += n

        report = build_quality_report(all_preds, all_refs, severities=all_sevs)
        print_quality_report(report, title=label)
        return report

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch % self.eval_every != 0:
            return
        print(f"\n  [quality] generation eval (epoch {epoch})...")
        report = self._run_generation_eval(
            self.quality_subset, self.severity_list,
            [self.quality_subset[i]["original"] for i in range(len(self.quality_subset))],
            label=f"quality eval (epoch {epoch})",
        )
        # Log metrics
        log_metrics = quality_report_to_log(report, prefix="quality")
        log_metrics["epoch"] = state.epoch
        self.trainer.log(log_metrics)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.full_eval_at_end or self.full_val_ds is None:
            return
        print("\n  [quality] full-val generation eval (end of training)...")
        report = self._run_generation_eval(
            self.full_val_ds, self.full_val_severities, self.full_val_refs,
            label="full-val quality (end of training)",
        )
        log_metrics = quality_report_to_log(report, prefix="full_quality")
        log_metrics["epoch"] = state.epoch
        self.trainer.log(log_metrics)


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
    print(f"Loading dataset from HuggingFace Hub: {data_path}")
    ds = load_dataset(data_path)
    if "train" not in ds:
        key = list(ds.keys())[0]
        ds = ds.rename_column(key, "train") if key != "train" else ds
    if "val" not in ds and "validation" in ds:
        ds["val"] = ds["validation"]
        del ds["validation"]
    return ds


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def build_subsample(dataset, target_n=5_000_000, seed=42):
    """Build a hard-weighted subsample: keep severity 1.0 heavily, thin out 0.1/0.25."""
    if target_n < 0:
        print(f"Using full dataset: {len(dataset)} rows")
        return None

    rng = random.Random(seed)
    total = len(dataset)
    severities = dataset["severity"]
    buckets = {0.1: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    for i, sev in enumerate(severities):
        buckets[sev].append(i)

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


def build_quality_subset(val_full, tokenizer, per_severity_n=1000, seed=42,
                         max_length=64, n_proc=1, cache_prefix=None):
    """Build a fixed stratified subset for generation-based quality eval.

    Returns (tokenized_dataset, severity_list) where severity_list is aligned
    by row index.
    """
    rng = random.Random(seed)
    buckets = {0.1: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    val_severities = val_full["severity"]
    for i, sev in enumerate(val_severities):
        buckets[sev].append(i)

    selected = []
    for sev, pool in sorted(buckets.items()):
        n = min(per_severity_n, len(pool))
        selected.extend(rng.sample(pool, n) if n < len(pool) else pool)
    rng.shuffle(selected)

    subset = val_full.select(selected)
    severity_list = [val_severities[i] for i in selected]

    # Tokenize
    def preprocess(examples):
        inputs = ["restore: " + c for c in examples["corrupted"]]
        targets = examples["original"]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=False)
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cache_files = None
    if cache_prefix:
        cache_files = [f"{cache_prefix}_quality{i}" for i in range(n_proc)]

    subset = subset.map(
        preprocess, batched=True, batch_size=512, num_proc=n_proc,
        cache_file_name=cache_files if cache_files else None,
    )
    print(f"Quality subset: {len(subset)} rows (per-sev: {per_severity_n})")
    return subset, severity_list


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
        cache_file_name=cache_files if cache_files else None,
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
    parser.add_argument("--val-rows", type=int, default=None,
                        help="Number of val rows for loss eval (default: full val)")
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

    # Checkpoint control
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: save per epoch)")
    parser.add_argument("--limit-checkpoints-folder", type=str, default=None,
                        help="Max local+remote checkpoint storage (e.g. '50gb', '200mb')")

    # Quality evaluation
    parser.add_argument("--eval-subset", type=int, default=1000,
                        help="Rows per severity for the generation quality eval subset (default: 1000)")
    parser.add_argument("--quality-eval-every", type=int, default=1,
                        help="Run generation quality eval every N epochs (default: 1)")
    parser.add_argument("--eval-num-beams", type=int, default=4,
                        help="Beams for generation quality eval (1=greedy, default: 4)")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Batch size for generation quality eval (default: 32)")
    parser.add_argument("--full-eval-at-end", action="store_true",
                        help="Run full-val generation eval after training completes")

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
    train_ds = train_full.select(train_idx) if train_idx is not None else train_full

    # Val subset for loss eval (full or partial)
    if args.val_rows is not None and args.val_rows < len(val_full):
        val_rng = random.Random(args.seed)
        val_idx = val_rng.sample(range(len(val_full)), args.val_rows)
        val_ds = val_full.select(val_idx)
    else:
        val_ds = val_full

    # Tokenize train + val for loss eval
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

    # Build quality subset (stratified by severity, for generation eval)
    quality_sub_sevs = None
    quality_ds, quality_sevs = build_quality_subset(
        val_full, tokenizer,
        per_severity_n=args.eval_subset, seed=args.seed,
        max_length=args.max_length, n_proc=n_proc,
        cache_prefix=cache_prefix,
    )

    # -----------------------------------------------------------------------
    # Training args
    # -----------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    effective_batch = args.batch_size * args.grad_accum
    precision = resolve_precision(args.precision)

    if args.save_steps is not None:
        save_strategy, save_steps = "steps", args.save_steps
        eval_strategy, eval_steps = "steps", args.save_steps
    else:
        save_strategy, save_steps = "epoch", None
        eval_strategy, eval_steps = "epoch", None

    has_storage_limit = args.limit_checkpoints_folder is not None
    storage_limit_bytes = parse_size(args.limit_checkpoints_folder) if has_storage_limit else None

    total_steps = int(np.ceil(len(train_ds) / effective_batch)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=1 if not has_storage_limit else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=min(n_proc, 8),
        seed=args.seed,
    )
    if save_steps is not None:
        training_kwargs["save_steps"] = save_steps
        training_kwargs["eval_steps"] = eval_steps
    if precision == "fp16":
        training_kwargs["fp16"] = True
    elif precision == "bf16":
        training_kwargs["bf16"] = True

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    if has_storage_limit:
        print(f"Storage limit: {args.limit_checkpoints_folder} ({storage_limit_bytes:,} bytes)")
        trainer.add_callback(StorageLimitCallback(args.out, storage_limit_bytes))

    if args.hf_bucket:
        print(f"Checkpoint sync: {args.hf_bucket}")
        trainer.add_callback(HFSyncCallback(args.hf_bucket, args.out, delete_remote=has_storage_limit))

    # Quality eval callback (generation-based per-severity metrics)
    quality_callback = QualityEvalCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        quality_subset=quality_ds,
        severity_list=quality_sevs,
        eval_num_beams=args.eval_num_beams,
        eval_batch_size=args.eval_batch_size,
        eval_max_length=args.max_length,
        eval_every=args.quality_eval_every,
        full_eval_at_end=args.full_eval_at_end,
    )
    trainer.add_callback(quality_callback)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("\nStarting training...")
    print(f"  Model:            {args.model}")
    print(f"  Train rows:       {len(train_ds)}")
    print(f"  Val rows (loss):  {len(val_ds)}")
    print(f"  Quality subset:   {len(quality_ds)} ({args.eval_subset}/severity)")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Effective batch:  {effective_batch}")
    print(f"  LR:               {args.lr}")
    print(f"  Precision:        {precision}")
    print(f"  Save strategy:    {save_strategy}"
          + (f" every {save_steps} steps" if save_steps else ""))
    print(f"  Storage limit:    {args.limit_checkpoints_folder or 'none'}")
    print(f"  Quality eval:     every {args.quality_eval_every} epoch(s), "
          f"beams={args.eval_num_beams}")
    print(f"  Full eval at end: {'yes' if args.full_eval_at_end else 'no'}")
    print(f"  LoRA:             {'yes' if args.lora else 'no'}")
    print(f"  Output:           {args.out}")
    if args.hf_bucket:
        print(f"  HF bucket:        {args.hf_bucket}")
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
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Saving final model to {args.out}...")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    if args.hf_bucket:
        sync_to_hf_bucket(args.out, args.hf_bucket, delete_remote=has_storage_limit)

    print("Done.")


if __name__ == "__main__":
    main()
