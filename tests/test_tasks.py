import unittest
from random import Random
from unittest.mock import patch

from terrupt.tasks import make_punctuation


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


if __name__ == "__main__":
    unittest.main()
