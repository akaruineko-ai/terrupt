import argparse
import json
import tempfile
import unittest
from random import Random
from unittest.mock import patch

from terrupt.cli import finalize_stream
from terrupt.tasks import iter_textcorrupt, make_punctuation


class PunctuationGenerationTests(unittest.TestCase):
    def test_generates_requested_records(self):
        sentences = ["One sentence has punctuation."]

        records = make_punctuation(
            sentences, Random(1), count=3, modes=("strip",), progress=False
        )

        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["input"] != record["original"] for record in records))

    def test_stops_when_mode_cannot_change_corpus(self):
        sentences = ["One sentence has punctuation."]
        unchanged = lambda text, rng: text

        with patch.dict("terrupt.tasks._PUNCT_MODES", {"broken": unchanged}):
            with self.assertRaisesRegex(ValueError, "could only create 0 of 2"):
                make_punctuation(
                    sentences, Random(1), count=2, modes=("broken",), progress=False
                )


class StreamingTextCorruptTests(unittest.TestCase):
    def test_iter_textcorrupt_yields_requested_records(self):
        records = list(iter_textcorrupt(
            ["The quick brown fox jumps over the lazy dog."],
            (0.1,), Random(1), count=5, progress=False, workers=1
        ))

        self.assertEqual(len(records), 5)
        self.assertTrue(all("original" in record for record in records))

    def test_finalize_stream_writes_without_materializing_input(self):
        with tempfile.TemporaryDirectory() as output:
            args = argparse.Namespace(
                out=output, splits="train=0.8,val=0.2", shards=1
            )
            records = ({"text": str(i)} for i in range(5))
            stats = finalize_stream(records, args, "sample", ["text"], 5)

            self.assertEqual(stats["splits"], {"train": 4, "val": 1})
            with open(f"{output}/sample_train.jsonl", encoding="utf-8") as fh:
                self.assertEqual(len(fh.readlines()), 4)
            with open(f"{output}/stats.json", encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["count"], 5)


if __name__ == "__main__":
    unittest.main()
