"""Central language definitions for the translation UI and model mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageOption:
    label: str
    nllb_code: str


LANGUAGE_OPTIONS: dict[str, LanguageOption] = {
    "Deutsch": LanguageOption(label="Deutsch", nllb_code="deu_Latn"),
    "Englisch": LanguageOption(label="Englisch", nllb_code="eng_Latn"),
    "Franzoesisch": LanguageOption(label="Franzoesisch", nllb_code="fra_Latn"),
    "Polnisch": LanguageOption(label="Polnisch", nllb_code="pol_Latn"),
    "Russisch": LanguageOption(label="Russisch", nllb_code="rus_Cyrl"),
    "Spanisch": LanguageOption(label="Spanisch", nllb_code="spa_Latn"),
    "Ukrainisch": LanguageOption(label="Ukrainisch", nllb_code="ukr_Cyrl"),
}


def get_language_labels() -> list[str]:
    return list(LANGUAGE_OPTIONS.keys())


def get_nllb_code(language_label: str) -> str:
    return LANGUAGE_OPTIONS[language_label].nllb_code
