"""Translation pipeline orchestration."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import torch

from core.glossary import GlossaryManager
from core.model_manager import get_quality_preset, load_model
from core.reassembler import reassemble_segments
from core.segmenter import Segment, segment_text


LOGGER = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    text: str
    source_language_code: str
    target_language_code: str
    model_name: str
    quality_mode: str
    segmentation_mode: str
    max_segment_length: int
    use_context_overlap: bool


@dataclass
class TranslationResult:
    translated_text: str
    segments: list[Segment]
    translated_segments: list[str]
    processing_seconds: float
    warnings: list[str]


class Translator:
    """Stateful translator that reuses loaded model weights."""

    def __init__(self, glossary_manager: GlossaryManager | None = None) -> None:
        self.glossary_manager = glossary_manager or GlossaryManager()

    def translate_document(self, request: TranslationRequest, progress_callback=None) -> TranslationResult:
        started_at = time.perf_counter()
        warnings: list[str] = []
        segments = segment_text(
            request.text,
            mode=request.segmentation_mode,
            max_chars=request.max_segment_length,
        )
        if not segments:
            raise ValueError("Dokument enthaelt keine uebersetzbaren Segmente")

        loaded = load_model(request.model_name)
        quality = get_quality_preset(request.quality_mode)
        tokenizer = loaded.tokenizer
        model = loaded.model
        tokenizer.src_lang = request.source_language_code
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(request.target_language_code)

        translated_segments: list[str] = []

        for index, segment in enumerate(segments, start=1):
            prompt_text = segment.text

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=min(quality["max_length"], 1024),
            )
            inputs = {key: value.to(loaded.device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=quality["max_length"],
                    num_beams=quality["num_beams"],
                )

            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            translated = clean_translation_output(translated)
            translated = self.glossary_manager.apply(translated)
            translated_segments.append(translated + segment.separator)

            if progress_callback:
                progress_callback(index, len(segments), segment.text, translated)

        if request.use_context_overlap:
            warnings.append(
                "Kontextmodus laeuft derzeit im stabilen Fallback ohne Prompt-Vermischung, um Dopplungen und Artefakte zu vermeiden."
            )

        translated_text = reassemble_segments(translated_segments)
        translated_text = clean_translation_output(translated_text)
        duration = time.perf_counter() - started_at
        LOGGER.info(
            "Translated %s segments from %s to %s in %.2fs",
            len(segments),
            request.source_language_code,
            request.target_language_code,
            duration,
        )
        return TranslationResult(
            translated_text=translated_text,
            segments=segments,
            translated_segments=translated_segments,
            processing_seconds=duration,
            warnings=warnings,
        )


def clean_translation_output(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"(?:\s*~\s*){4,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"([.!?])([A-ZА-ЯÄÖÜ])", r"\1 \2", cleaned)

    lines: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            if not lines or lines[-1] != "":
                lines.append("")
            continue
        if lines and lines[-1] == stripped:
            continue
        lines.append(stripped)

    return "\n".join(lines).strip()
