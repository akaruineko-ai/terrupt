"""Export generated JSONL records to a Hugging Face dataset folder.

Produces a folder compatible with ``datasets.load_from_disk``: arrow files,
``dataset_info.json``, and an auto-generated ``README.md`` dataset card.
"""

import glob
import hashlib
import json
import os
import re

from terrupt.hfcard import render_card
from terrupt.progress import Progress

_SPLIT_RE = re.compile(r"_(train|val|test|validation|dev)(?:_\d+)?\.jsonl$")
_SHARD_RE = re.compile(r"data-\d{5}-of-(\d{5})\.arrow$")


def _infer_features(records):
    from datasets import Features, Sequence, Value

    keys = set()
    for rec in records:
        keys.update(rec.keys())
    features = {}
    for key in sorted(keys):
        values = [rec[key] for rec in records if key in rec]
        if all(isinstance(v, str) for v in values):
            features[key] = Value("string")
        elif all(isinstance(v, bool) for v in values):
            features[key] = Value("bool")
        elif all(isinstance(v, int) for v in values):
            features[key] = Value("int64")
        elif all(isinstance(v, float) for v in values):
            features[key] = Value("float64")
        elif (all(isinstance(v, list) for v in values)
              and all(isinstance(x, str) for v in values for x in v)):
            features[key] = Sequence(Value("string"))
        else:
            features[key] = Value("string")
    return Features(features)


def _dtype_str(feature):
    from datasets import Sequence, Value

    if isinstance(feature, Value):
        return feature.dtype
    if isinstance(feature, Sequence):
        return f"Sequence[{_dtype_str(feature.feature)}]"
    return type(feature).__name__


def _iter_records(paths, progress=False):
    """Yield records line-by-line without holding all in memory."""
    pbar = Progress(desc=f"streaming {len(paths)} file(s)", unit="rows",
                    disable=not progress)
    with pbar:
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
                    pbar.update(1)


def _infer_features_from_paths(paths, sample_limit=200):
    """Infer HF features by sampling a few hundred records."""
    sample = _read_records(paths, limit=sample_limit)
    return _infer_features(sample) if sample else None


def _parse_size(size_str):
    """Parse human-readable size like '2GB' into bytes."""
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    size_str = size_str.strip()
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.upper().endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


def export_hf_stream(input_dir, out_dir, meta, max_shard_size="2GB", sample_limit=200):
    """Build an HF dataset folder using memory-safe streaming with direct Arrow IPC writing.

    Reads JSONL line-by-line, writes Arrow IPC stream files directly into the
    output directory structure expected by ``datasets.load_from_disk``.  No
    intermediate cache or in-memory buffering beyond one batch of records.
    """
    import pyarrow.ipc as ipc

    BATCH_SIZE = 10_000
    max_shard_bytes = _parse_size(max_shard_size)

    splits = discover_splits(input_dir)
    if not splits:
        raise ValueError(
            f"no *_train.jsonl / *_val.jsonl / *_test.jsonl files in {input_dir}"
        )

    schema = None
    for paths in splits.values():
        schema = _infer_features_from_paths(paths, sample_limit=sample_limit)
        break
    if schema is None:
        raise ValueError("could not infer features from any split")

    arrow_schema = schema.arrow_schema
    os.makedirs(out_dir, exist_ok=True)

    split_counts = {}

    for split_name, paths in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        shard_files = []
        shard_idx = 0
        batch = []
        current_shard_bytes = 0
        writer = None
        total_count = 0

        for record in _iter_records(paths, progress=True):
            batch.append(record)
            total_count += 1

            if len(batch) >= BATCH_SIZE:
                shard_files, shard_idx, writer, current_shard_bytes = _flush_shard(
                    batch, arrow_schema, split_dir, shard_idx, shard_files, writer,
                    max_shard_bytes, current_shard_bytes,
                )
                batch = []

        if batch:
            shard_files, shard_idx, writer, current_shard_bytes = _flush_shard(
                batch, arrow_schema, split_dir, shard_idx, shard_files, writer,
                max_shard_bytes, current_shard_bytes,
            )
        if writer:
            writer.close()
            writer = None

        total_shards = len(shard_files)
        for i, old_path in enumerate(shard_files):
            new_name = f"data-{i:05d}-of-{total_shards:05d}.arrow"
            new_path = os.path.join(split_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
            shard_files[i] = new_path

        _write_state(split_dir, [os.path.basename(p) for p in shard_files])
        _write_split_dataset_info(split_dir, schema)

        split_counts[split_name] = total_count
        print(f"  {split_name}: {total_count:,} rows, {len(shard_files)} shard(s)")

    _write_dataset_dict(out_dir, sorted(splits.keys()))
    _write_top_dataset_info(out_dir, schema, split_counts)

    if meta:
        meta["splits"] = split_counts
        meta["schema"] = [
            {"name": key, "dtype": _dtype_str(feature)}
            for key, feature in schema.items()
        ]
        readme = render_card(meta)
        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme)

    return split_counts


def _flush_shard(batch, arrow_schema, split_dir, shard_idx, shard_files, writer,
                 max_shard_bytes, current_shard_bytes):
    """Write the current batch to the active shard, opening a new shard if needed."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.Table.from_pylist(batch, schema=arrow_schema)
    batch_bytes = table.nbytes

    if writer is None or current_shard_bytes + batch_bytes >= max_shard_bytes:
        if writer is not None:
            writer.close()
        shard_name = f"data-{shard_idx:05d}-of-XXXXX.arrow"
        shard_path = os.path.join(split_dir, shard_name)
        writer = ipc.new_stream(shard_path, arrow_schema)
        shard_files.append(shard_path)
        shard_idx += 1
        current_shard_bytes = 0

    writer.write_table(table)
    current_shard_bytes += batch_bytes
    return shard_files, shard_idx, writer, current_shard_bytes


def _write_state(split_dir, shard_names):
    state = {
        "_data_files": [{"filename": name} for name in shard_names],
        "_fingerprint": hashlib.md5(split_dir.encode()).hexdigest()[:16],
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    with open(os.path.join(split_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _write_split_dataset_info(split_dir, schema):
    from datasets import Sequence, Value

    features = {}
    for key, feature in schema.items():
        if isinstance(feature, Sequence):
            features[key] = {
                "feature": {"dtype": "string", "_type": "Value"},
                "_type": "Sequence",
            }
        else:
            features[key] = {"dtype": feature.dtype, "_type": "Value"}
    info = {
        "citation": "",
        "description": "",
        "features": features,
        "homepage": "",
        "license": "",
    }
    with open(os.path.join(split_dir, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)


def _write_dataset_dict(out_dir, split_names):
    path = os.path.join(out_dir, "dataset_dict.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"splits": split_names}, fh)


def _write_top_dataset_info(out_dir, schema, split_counts):
    from datasets import Sequence, Value

    features = {}
    for key, feature in schema.items():
        if isinstance(feature, Sequence):
            features[key] = {
                "feature": {"dtype": "string", "_type": "Value"},
                "_type": "Sequence",
            }
        else:
            features[key] = {"dtype": feature.dtype, "_type": "Value"}
    info = {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": features,
        "splits": {
            name: {"name": name, "num_bytes": -1, "num_examples": count}
            for name, count in split_counts.items()
        },
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)


def _read_records(paths, limit=None, progress=False):
    records = []
    pbar = Progress(desc=f"reading {len(paths)} file(s)", unit="rows",
                    disable=not progress)
    with pbar:
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
                    pbar.update(1)
                    if limit and len(records) >= limit:
                        return records
    return records


def discover_splits(input_dir):
    """Group JSONL files in ``input_dir`` by split name."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.jsonl"))):
        match = _SPLIT_RE.search(os.path.basename(path))
        if match:
            groups.setdefault(match.group(1), []).append(path)
    return groups


def _write_dataset_info(out_dir, schema, splits):
    from datasets import Sequence

    features = {}
    for key, feature in schema.items():
        if isinstance(feature, Sequence):
            features[key] = {"feature": {"dtype": "string", "_type": "Value"},
                             "_type": "List"}
        else:
            features[key] = {"dtype": feature.dtype, "_type": "Value"}
    info = {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": features,
        "splits": {name: {"name": name, "num_bytes": -1, "num_examples": count}
                   for name, count in splits.items()},
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)


def export_hf(input_dir, out_dir, meta):
    """Build an HF dataset folder from JSONL split files in ``input_dir``."""
    from datasets import Dataset, DatasetDict

    splits = discover_splits(input_dir)
    if not splits:
        raise ValueError(f"no *_train.jsonl / *_val.jsonl / *_test.jsonl files in {input_dir}")

    dataset_dict = {}
    schema = None
    for split_name, paths in splits.items():
        records = _read_records(paths, progress=True)
        if not records:
            continue
        if schema is None:
            schema = _infer_features(records)
        dataset_dict[split_name] = Dataset.from_list(records, features=schema)

    ddict = DatasetDict(dataset_dict)
    ddict.save_to_disk(out_dir)

    if meta:
        meta["splits"] = {name: len(ds) for name, ds in ddict.items()}
        if schema is not None:
            meta["schema"] = [
                {"name": key, "dtype": _dtype_str(feature)}
                for key, feature in schema.items()
            ]
            _write_dataset_info(out_dir, schema, meta["splits"])
        readme = render_card(meta)
        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme)
    return ddict
