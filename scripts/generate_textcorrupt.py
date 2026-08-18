"""Generate the TextCorrupt dataset: corrupted text -> original text."""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrupt.cli import add_common_args, add_source_args, finalize
from terrupt.sources import SOURCES, load_sentences_cli
from terrupt.tasks import make_textcorrupt


def main():
    parser = argparse.ArgumentParser(description="Terrupt TextCorrupt dataset")
    add_common_args(parser)
    add_source_args(parser)
    parser.add_argument("--severities", type=str,
                        default="0.1,0.25,0.5,0.75,1.0",
                        help="comma-separated severity levels")
    args = parser.parse_args()

    severities = tuple(float(s) for s in args.severities.split(","))
    sources = tuple(s.strip() for s in args.sources.split(","))
    sentences, breakdown = load_sentences_cli(
        sources, args.corpus, args.language, args.per_source,
        args.wikipedia_config, args.seed, workers=args.workers)
    rng = random.Random(args.seed)
    records = make_textcorrupt(sentences, severities, rng, args.language, args.count,
                               workers=args.workers)
    stats = finalize(records, args, "textcorrupt",
                     ["corruption_type", "severity", "op_count", "language", "source"])
    print(f"generated {stats['count']} textcorrupt records -> {args.out}")
    print(f"  corpus sentences: {len(sentences)}")
    for key, info in breakdown.items():
        print(f"  source {key}: {info}")
    print(f"  splits: {stats['splits']}")
    print(f"  source distribution: {stats['source']}")


if __name__ == "__main__":
    main()