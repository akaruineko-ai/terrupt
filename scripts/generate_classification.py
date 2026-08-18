"""Generate the corruption classification dataset: text -> corruption type(s)."""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrupt.cli import add_common_args, add_source_args, finalize
from terrupt.sources import load_sentences_cli
from terrupt.tasks import make_classification


def main():
    parser = argparse.ArgumentParser(description="Terrupt corruption classification dataset")
    add_common_args(parser)
    add_source_args(parser)
    parser.add_argument("--severities", type=str,
                        default="0.1,0.25,0.5,0.75,1.0",
                        help="comma-separated severity levels")
    parser.add_argument("--include-clean", action="store_true",
                        help="add uncorrupted examples labeled as no-op")
    args = parser.parse_args()

    severities = tuple(float(s) for s in args.severities.split(","))
    sources = tuple(s.strip() for s in args.sources.split(","))
    sentences, breakdown = load_sentences_cli(
        sources, args.corpus, args.language, args.per_source,
        args.wikipedia_config, args.seed, workers=args.workers)
    rng = random.Random(args.seed)
    records = make_classification(sentences, severities, rng, args.language,
                                  args.count, args.include_clean,
                                  workers=args.workers)
    stats = finalize(records, args, "classification",
                     ["corruption_type", "severity", "language", "source"])
    print(f"generated {stats['count']} classification records -> {args.out}")
    print(f"  corpus sentences: {len(sentences)}")
    for key, info in breakdown.items():
        print(f"  source {key}: {info}")
    print(f"  splits: {stats['splits']}")
    print(f"  corruption_type: {stats['corruption_type']}")
    print(f"  source distribution: {stats['source']}")


if __name__ == "__main__":
    main()