"""Shared quality-metric helpers for terrupt evaluation.

Used by both ``scripts/finetune.py`` (in-training generation eval)
and ``scripts/eval.py`` (standalone evaluation).
"""

import re
from collections import defaultdict

from sacrebleu.metrics import CHRF


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred, ref):
    """1 if normalized strings match, else 0."""
    return int(normalize(pred) == normalize(ref))


def chrf_score(preds, refs):
    """Corpus-level chrF++ (word_order=2) via sacrebleu."""
    chrf = CHRF(word_order=2)
    return chrf.corpus_score(preds, [refs])


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def build_quality_report(preds, refs, severities=None, corruption_types=None):
    """Compute per-severity (+ optional per-type) quality metrics.

    Returns::

        {
            "overall": {"exact_match": float, "chrF": float, "n": int},
            "per_severity": {float: {"exact_match": ..., "chrF": ..., "n": ...}, ...},
            "per_corruption_type": {str: {...}, ...} | None,
        }
    """
    overall_em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)
    overall_chrf = chrf_score(preds, refs).score

    report = {
        "overall": {
            "exact_match": overall_em,
            "chrF": overall_chrf,
            "n": len(preds),
        },
        "per_severity": {},
        "per_corruption_type": None,
    }

    if severities is not None:
        groups = defaultdict(lambda: {"preds": [], "refs": []})
        for i in range(len(preds)):
            groups[severities[i]]["preds"].append(preds[i])
            groups[severities[i]]["refs"].append(refs[i])
        for sev in sorted(groups):
            g = groups[sev]
            em = sum(exact_match(p, r) for p, r in zip(g["preds"], g["refs"])) / len(g["preds"])
            ch = chrf_score(g["preds"], g["refs"]).score
            report["per_severity"][sev] = {"exact_match": em, "chrF": ch, "n": len(g["preds"])}

    if corruption_types is not None:
        groups = defaultdict(lambda: {"preds": [], "refs": []})
        for i in range(len(preds)):
            groups[corruption_types[i]]["preds"].append(preds[i])
            groups[corruption_types[i]]["refs"].append(refs[i])
        report["per_corruption_type"] = {}
        for ct in sorted(groups):
            g = groups[ct]
            em = sum(exact_match(p, r) for p, r in zip(g["preds"], g["refs"])) / len(g["preds"])
            ch = chrf_score(g["preds"], g["refs"]).score
            report["per_corruption_type"][ct] = {"exact_match": em, "chrF": ch, "n": len(g["preds"])}

    return report


def print_quality_report(report, title="quality", severity_key="severity"):
    """Pretty-print a quality report table to stdout."""
    overall = report["overall"]
    print(f"\n  [{title}]  exact_match={overall['exact_match']:.4f}  "
          f"chrF={overall['chrF']:.2f}  n={overall['n']}")

    if report["per_severity"]:
        print(f"  {'severity':<12} {'exact_match':>12} {'chrF':>8} {'n':>8}")
        print("  " + "-" * 42)
        for sev, m in sorted(report["per_severity"].items()):
            print(f"  {sev:<12} {m['exact_match']:>12.4f} {m['chrF']:>8.2f} {m['n']:>8}")

    if report.get("per_corruption_type"):
        print(f"\n  {'corruption_type':<16} {'exact_match':>12} {'chrF':>8} {'n':>8}")
        print("  " + "-" * 46)
        for ct, m in sorted(report["per_corruption_type"].items()):
            print(f"  {ct:<16} {m['exact_match']:>12.4f} {m['chrF']:>8.2f} {m['n']:>8}")

    print()


def quality_report_to_log(report, prefix=""):
    """Flatten a quality report into a dict suitable for ``trainer.log()``."""
    metrics = {}
    overall = report["overall"]
    tag = f"{prefix}/" if prefix else ""
    metrics[f"{tag}exact_match"] = overall["exact_match"]
    metrics[f"{tag}chrF"] = overall["chrF"]
    if report["per_severity"]:
        for sev, m in report["per_severity"].items():
            s = str(sev).replace(".", "_")
            metrics[f"{tag}sev{s}/exact_match"] = m["exact_match"]
            metrics[f"{tag}sev{s}/chrF"] = m["chrF"]
    return metrics
