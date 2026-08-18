"""Generate the punctuation correction dataset: unpunctuated/mispunctuated
text -> punctuated text."""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrupt.cli import add_common_args, add_source_args, finalize
from terrupt.sources import load_sentences_cli
from terrupt.tasks import make_punctuation


def main():
    parser = argparse.ArgumentParser(description="Terrupt punctuation dataset")
    add_common_args(parser)
    add_source_args(parser)
    parser.add_argument("--modes", type=str,
                        default="strip,wrong,random,spacing",
                        help="comma-separated corruption modes")
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(","))
    sources = tuple(s.strip() for s in args.sources.split(","))
    sentences, breakdown = load_sentences_cli(
        sources, args.corpus, args.language, args.per_source,
        args.wikipedia_config, args.seed, workers=args.workers)
    rng = random.Random(args.seed)
    records = make_punctuation(sentences, rng, args.language, args.count, modes,
                               workers=args.workers)
    stats = finalize(records, args, "punctuation", ["mode", "language", "source"])
    print(f"generated {stats['count']} punctuation records -> {args.out}")
    print(f"  corpus sentences: {len(sentences)}")
    for key, info in breakdown.items():
        print(f"  source {key}: {info}")
    print(f"  splits: {stats['splits']}")
    print(f"  mode: {stats['mode']}")
    print(f"  source distribution: {stats['source']}")


if __name__ == "__main__":
    main()