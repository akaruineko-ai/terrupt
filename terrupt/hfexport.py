"""Export generated JSONL records to a Hugging Face dataset folder.

Produces a folder compatible with ``datasets.load_from_disk``: arrow files,
``dataset_info.json``, and an auto-generated ``README.md`` dataset card.
"""

import glob
import json
import os
import re

from terrupt.hfcard import render_card
from terrupt.progress import Progress

_SPLIT_RE = re.compile(r"_(train|val|test|validation|dev)(?:_\d+)?\.jsonl$")


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