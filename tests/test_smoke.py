import unittest

from core.io_utils import build_output_filename
from core.reassembler import reassemble_segments
from core.segmenter import segment_text
from core.translator import split_structured_prefix


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

    def test_segment_text_preserves_structured_lines(self) -> None:
        text = (
            "19193;не хватает только плз 28259\n"
            "19194;сергей приезжал\n"
            "19195;деньги привёз\n"
        )

        segments = segment_text(text, mode="Auto", max_chars=120)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].text, "19193;не хватает только плз 28259")
        self.assertEqual(segments[0].separator, "\n")
        self.assertEqual(segments[1].text, "19194;сергей приезжал")
        self.assertEqual(segments[1].separator, "\n")
        self.assertEqual(segments[2].text, "19195;деньги привёз")
        self.assertEqual(segments[2].separator, "\n")

    def test_split_structured_prefix(self) -> None:
        prefix, body = split_structured_prefix("19193; не хватает только плз 28259")

        self.assertEqual(prefix, "19193; ")
        self.assertEqual(body, "не хватает только плз 28259")

    def test_reassemble_segments_preserves_newlines(self) -> None:
        document = reassemble_segments(
            [
                "19193; fehlt nur noch bitte 28259\n",
                "19194; Sergei kam an\n",
                "19195; brachte das Geld\n",
            ]
        )

        self.assertEqual(
            document,
            "19193; fehlt nur noch bitte 28259\n"
            "19194; Sergei kam an\n"
            "19195; brachte das Geld\n",
        )


if __name__ == "__main__":
    unittest.main()
