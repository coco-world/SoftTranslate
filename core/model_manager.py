"""Model loading and caching for local NLLB translation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, str] = {
    "NLLB 1.3B Distilled": "facebook/nllb-200-distilled-1.3B",
    "NLLB 3.3B": "facebook/nllb-200-3.3B",
}


QUALITY_PRESETS = {
    "Schnell": {"max_length": 256, "num_beams": 2},
    "Ausgewogen": {"max_length": 384, "num_beams": 4},
    "Maximal": {"max_length": 512, "num_beams": 5},
}


@dataclass
class LoadedModel:
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    device: str


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from cache or downloaded."""


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_offline_mode_enabled() -> bool:
    return os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def _load_tokenizer(model_name: str, *, local_files_only: bool) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=local_files_only)


def _load_seq2seq_model(model_name: str, *, local_files_only: bool) -> AutoModelForSeq2SeqLM:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_files_only)


@lru_cache(maxsize=2)
def load_model(model_name: str) -> LoadedModel:
    device = detect_device()
    LOGGER.info("Loading translation model %s on %s", model_name, device)
    try:
        tokenizer = _load_tokenizer(model_name, local_files_only=True)
        model = _load_seq2seq_model(model_name, local_files_only=True)
        LOGGER.info("Loaded translation model %s from local Hugging Face cache", model_name)
    except Exception as local_error:
        if _is_offline_mode_enabled():
            raise ModelLoadError(
                f"Modell {model_name} wurde nicht im lokalen Hugging-Face-Cache gefunden. "
                "Offline-Modus ist aktiv; bitte das Modell einmal mit Internetverbindung herunterladen."
            ) from local_error
        LOGGER.info("Local cache miss for %s, falling back to Hugging Face download", model_name)
        try:
            tokenizer = _load_tokenizer(model_name, local_files_only=False)
            model = _load_seq2seq_model(model_name, local_files_only=False)
        except Exception as remote_error:
            raise ModelLoadError(
                f"Modell {model_name} konnte weder lokal geladen noch von Hugging Face heruntergeladen werden."
            ) from remote_error
    model.to(device)
    model.eval()
    return LoadedModel(model_name=model_name, tokenizer=tokenizer, model=model, device=device)


def resolve_model_name(model_label: str | None) -> str:
    if not model_label:
        return MODEL_REGISTRY["NLLB 1.3B Distilled"]
    return MODEL_REGISTRY.get(model_label, model_label)


def get_quality_preset(label: str) -> dict[str, int]:
    return QUALITY_PRESETS[label]
