"""Reassembles translated segments into a final document."""

from __future__ import annotations


def reassemble_segments(translated_segments: list[str], separator: str = "") -> str:
    cleaned = [segment for segment in translated_segments]
    document = separator.join(cleaned)
    return document.replace("\r\n", "\n").strip() + "\n"
