import unittest

from core.io_utils import build_output_filename
from core.segmenter import segment_text


class SmokeTests(unittest.TestCase):
    def test_build_output_filename(self) -> None:
        self.assertEqual(build_output_filename("dokument1.txt", "deu"), "dokument1.deu.txt")

    def test_segment_text_auto_mode(self) -> None:
        text = "Erster Absatz.\n\nZweiter Absatz mit mehr Inhalt."
        segments = segment_text(text, mode="Auto", max_chars=120)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Erster Absatz.")
        self.assertEqual(segments[0].separator, "\n\n")
        self.assertEqual(segments[1].text, "Zweiter Absatz mit mehr Inhalt.")


if __name__ == "__main__":
    unittest.main()
