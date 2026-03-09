"""Model loading and caching for local NLLB translation."""

from __future__ import annotations

import logging
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


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def load_model(model_name: str) -> LoadedModel:
    device = detect_device()
    LOGGER.info("Loading translation model %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return LoadedModel(model_name=model_name, tokenizer=tokenizer, model=model, device=device)


def resolve_model_name(model_label: str | None) -> str:
    if not model_label:
        return MODEL_REGISTRY["NLLB 1.3B Distilled"]
    return MODEL_REGISTRY.get(model_label, model_label)


def get_quality_preset(label: str) -> dict[str, int]:
    return QUALITY_PRESETS[label]
