"""Glossary hooks for future terminology control."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class GlossaryError(Exception):
    """Raised when the glossary file cannot be parsed."""


class GlossaryManager:
    """Loads simple source -> target replacement pairs."""

    def __init__(self, glossary_path: str | None = None) -> None:
        self.glossary_path = glossary_path
        self.replacements: dict[str, str] = {}

    def load(self) -> dict[str, str]:
        if not self.glossary_path:
            self.replacements = {}
            return self.replacements

        path = Path(self.glossary_path)
        if not path.exists():
            raise GlossaryError(f"Glossary file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            self.replacements = self._load_json(path)
        elif suffix == ".csv":
            self.replacements = self._load_csv(path)
        else:
            raise GlossaryError("Glossary format must be .json or .csv")
        return self.replacements

    def apply(self, text: str) -> str:
        updated = text
        for source_term, target_term in self.replacements.items():
            updated = updated.replace(source_term, target_term)
        return updated

    @staticmethod
    def _load_json(path: Path) -> dict[str, str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise GlossaryError(f"Failed to read glossary JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise GlossaryError("Glossary JSON must contain a flat object mapping")
        return {str(key): str(value) for key, value in data.items()}

    @staticmethod
    def _load_csv(path: Path) -> dict[str, str]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                replacements: dict[str, str] = {}
                for row in reader:
                    if len(row) < 2:
                        continue
                    replacements[str(row[0]).strip()] = str(row[1]).strip()
                return replacements
        except Exception as exc:  # noqa: BLE001
            raise GlossaryError(f"Failed to read glossary CSV: {exc}") from exc
