"""Input/output helpers for TXT batch processing."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PREFERRED_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "latin-1")


class FileReadError(Exception):
    """Raised when uploaded content cannot be decoded."""


@dataclass
class FileResultMetadata:
    source_filename: str
    output_filename: str
    source_language: str
    target_language: str
    model_name: str
    segment_count: int
    processing_seconds: float
    char_count_input: int
    char_count_output: int
    warnings: list[str]
    session_id: str


def ensure_runtime_directories(base_dir: Path) -> None:
    for folder in ("assets", "config", "core", "logs", "output", "temp", "tests"):
        (base_dir / folder).mkdir(parents=True, exist_ok=True)


def decode_uploaded_text(file_name: str, raw_bytes: bytes) -> tuple[str, str]:
    if not raw_bytes:
        raise FileReadError(f"Datei ist leer: {file_name}")

    for encoding in PREFERRED_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise FileReadError(f"Datei konnte nicht dekodiert werden: {file_name}")


def build_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_session_output_dir(base_dir: Path, session_id: str) -> Path:
    session_dir = base_dir / "output" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def build_output_filename(source_name: str, target_code: str) -> str:
    source_path = Path(source_name)
    stem = source_path.stem or "translated"
    suffix = source_path.suffix or ".txt"
    return f"{stem}.{target_code}{suffix}"


def write_text_output(output_dir: Path, filename: str, content: str) -> Path:
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def write_metadata(output_dir: Path, metadata: FileResultMetadata) -> Path:
    path = output_dir / f"{Path(metadata.output_filename).stem}.meta.json"
    path.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def create_zip_archive(file_paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            archive.write(file_path, arcname=file_path.name)
    buffer.seek(0)
    return buffer.read()
