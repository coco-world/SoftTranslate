"""Robust text segmentation for TXT translation workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。！？])(?:[\"'»“”)\]]+)?\s+")
STRUCTURED_LINE_RE = re.compile(r"^\s*\d+\s*;\s*.*$")


@dataclass
class Segment:
    text: str
    separator: str
    index: int


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = SENTENCE_SPLIT_RE.split(stripped)
    return [part.strip() for part in parts if part.strip()]


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    words = text.split()
    current: list[str] = []
    current_len = 0

    for word in words:
        projected = current_len + len(word) + (1 if current else 0)
        if current and projected > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = projected

    if current:
        chunks.append(" ".join(current))
    return chunks


def segment_text(text: str, mode: str, max_chars: int) -> list[Segment]:
    normalized = text.replace("\r\n", "\n")
    if _looks_like_structured_lines(normalized):
        base_segments = _line_mode(normalized, max_chars)
    elif mode == "Satz":
        base_segments = _sentence_mode(normalized, max_chars)
    elif mode == "Auto":
        base_segments = _paragraph_mode(normalized, max_chars, allow_sentence_fallback=True)
    else:
        base_segments = _paragraph_mode(normalized, max_chars, allow_sentence_fallback=True)

    return [Segment(text=segment_text, separator=separator, index=index) for index, (segment_text, separator) in enumerate(base_segments)]


def _looks_like_structured_lines(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    structured_count = sum(1 for line in lines if STRUCTURED_LINE_RE.match(line))
    return structured_count / len(lines) >= 0.8


def _line_mode(text: str, max_chars: int) -> list[tuple[str, str]]:
    parts = re.split(r"(\n)", text)
    segments: list[tuple[str, str]] = []
    current_separator = ""

    for index in range(0, len(parts), 2):
        line = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""

        if not line and not separator:
            continue

        if not line and separator:
            if segments:
                last_text, last_separator = segments[-1]
                segments[-1] = (last_text, last_separator + separator)
            else:
                current_separator += separator
            continue

        if current_separator:
            line = current_separator + line
            current_separator = ""

        if len(line) <= max_chars:
            segments.append((line, separator))
            continue

        chunks = split_long_text(line, max_chars)
        for chunk_index, chunk in enumerate(chunks):
            chunk_separator = separator if chunk_index == len(chunks) - 1 else "\n"
            segments.append((chunk, chunk_separator))

    return segments


def _paragraph_mode(text: str, max_chars: int, allow_sentence_fallback: bool) -> list[tuple[str, str]]:
    paragraphs = re.split(r"(\n\s*\n)", text)
    segments: list[tuple[str, str]] = []

    for idx in range(0, len(paragraphs), 2):
        paragraph = paragraphs[idx]
        separator = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else ""
        if not paragraph.strip():
            if separator:
                segments.append(("", separator))
            continue

        if len(paragraph) <= max_chars:
            segments.append((paragraph.strip(), separator))
            continue

        if allow_sentence_fallback:
            sentence_chunks = _sentence_mode(paragraph, max_chars)
            if sentence_chunks:
                if sentence_chunks:
                    *body, last = sentence_chunks
                    segments.extend((chunk, "") for chunk, _ in body)
                    segments.append((last[0], separator))
                continue

        long_chunks = split_long_text(paragraph.strip(), max_chars)
        for chunk_index, chunk in enumerate(long_chunks):
            chunk_separator = separator if chunk_index == len(long_chunks) - 1 else ""
            segments.append((chunk, chunk_separator))

    return segments


def _sentence_mode(text: str, max_chars: int) -> list[tuple[str, str]]:
    sentences = split_sentences(text)
    segments: list[tuple[str, str]] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            segments.append((sentence, "\n"))
        else:
            for chunk in split_long_text(sentence, max_chars):
                segments.append((chunk, "\n"))
    if segments:
        last_text, _ = segments[-1]
        segments[-1] = (last_text, "")
    return segments
