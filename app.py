from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

import streamlit as st

from config.languages import get_language_labels, get_nllb_code
from core.glossary import GlossaryError, GlossaryManager
from core.io_utils import (
    FileReadError,
    build_output_filename,
    build_session_id,
    create_session_output_dir,
    create_zip_archive,
    decode_uploaded_text,
    ensure_runtime_directories,
    write_text_output,
)
from core.model_manager import MODEL_REGISTRY, detect_device, resolve_model_name
from core.translator import TranslationCancelledError, TranslationRequest, Translator


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PASTED_TEXT_NAME = "pasted_text.txt"
POLL_INTERVAL_SECONDS = 0.75
INPUT_MODES = ("upload", "text")
SEGMENTATION_MODES = ("Auto", "Absatz", "Satz")

ensure_runtime_directories(BASE_DIR)
LOG_PATH = BASE_DIR / "logs" / "app.log"


TRANSLATIONS = {
    "de": {
        "lang_toggle": "EN",
        "powered_locally_by": "Powered locally by",
        "banner_idle": "Bitte TXT-Dateien hochladen oder Freitext einfuegen, um die Uebersetzung zu starten.",
        "input": "Input",
        "output": "Output",
        "input_mode": "Eingabemodus",
        "mode_upload": "Datei-Upload",
        "mode_text": "Freitext",
        "upload_label": "TXT-Dateien hochladen",
        "upload_help": "Mehrere strukturierte TXT-Dateien koennen gemeinsam verarbeitet werden.",
        "text_label": "Text zum Uebersetzen",
        "text_help": "Fuer schnelle Einzeltexte kann der Inhalt direkt hier eingefuegt werden, statt eine TXT-Datei hochzuladen.",
        "save_txt": "TXT",
        "save_txt_help": "Wenn aktiviert, wird der eingefuegte Originaltext zusaetzlich als TXT im Session-Outputordner gespeichert.",
        "save_as_label": "Speichern als TXT in Output im Projektordner?",
        "save_as_help": "Dateiname fuer den eingefuegten Text, falls er zusammen mit den Uebersetzungen im Output-Ordner abgelegt werden soll.",
        "source_language": "Quellsprache",
        "source_help": "Sprache des Originaltexts. Sie wird auf den passenden NLLB-Sprachcode gemappt.",
        "target_language": "Zielsprache",
        "target_help": "Sprache, in die der Text lokal uebersetzt wird.",
        "advanced_options": "Erweiterte Uebersetzungsoptionen",
        "model": "Modell",
        "model_help": "Standard ist das lokal empfohlene 1.3B-NLLB-Modell. Die 3.3B-Option ist fuer spaetere Erweiterung vorbereitet.",
        "segmentation_mode": "Segmentierungsmodus",
        "segmentation_help": "Auto arbeitet primaer absatzweise und faellt bei langen Bloecken auf Satzsegmente zurueck. Satz ist feiner, Absatz erhaelt mehr Struktur.",
        "segmentation_auto": "Auto",
        "segmentation_paragraph": "Absatz",
        "segmentation_sentence": "Satz",
        "max_segment_length": "Maximale Segmentlaenge",
        "max_segment_help": "420 Zeichen ist ein sinnvoller Startwert fuer NLLB 1.3B. Wenn Artefakte auftreten, eher kleiner waehlen; wenn zu hart getrennt wird, etwas groesser.",
        "context_toggle": "Vorheriges Segment als Kontext verwenden",
        "context_help": "Gedacht fuer bessere begriffliche Konsistenz zwischen Segmenten. In der stabilen Umsetzung bleibt die Prompt-Vermischung bewusst sehr defensiv.",
        "glossary": "Optionales Glossar (CSV oder JSON)",
        "glossary_help": "Damit laesst sich feste Terminologie vorgeben, etwa Firmennamen, Rechtsformen oder Fachbegriffe.",
        "start": "Uebersetzung starten",
        "start_help": "Startet die Batch-Uebersetzung fuer den aktiven Eingabemodus.",
        "stop": "Stop",
        "stop_help": "Bricht eine laufende Uebersetzung nach dem aktuell bearbeiteten Segment ab.",
        "device_caption": "Beschleunigung: `{device}`. Modell bleibt fuer weitere Dateien geladen.",
        "current_translation": "Aktuelle Uebersetzung",
        "overall_job": "Gesamtjob",
        "chars": "Zeichen",
        "segments": "Segmente",
        "runtime": "Laufzeit",
        "original": "Original",
        "translation": "Uebersetzung",
        "partial_result": "Teilresultat",
        "partial_file_caption": "Diese Datei wurde nicht vollstaendig verarbeitet.",
        "download_txt": "TXT herunterladen",
        "download_zip": "Alle Ergebnisse als ZIP herunterladen",
        "warn_missing_input": "Bitte eine TXT-Datei hochladen oder Freitext eingeben.",
        "warn_same_language": "Quell- und Zielsprache muessen verschieden sein.",
        "status_init": "Initialisiere Uebersetzung...",
        "status_cancel_requested": "Abbruch angefordert. Das aktuelle Segment wird noch sauber abgeschlossen.",
        "status_processing_file": "Verarbeite Datei {current}/{total}: {name}",
        "status_cancelled": "Uebersetzung vom Nutzer abgebrochen.",
        "status_completed": "Session {session_id} abgeschlossen. Ausgabeordner: {output_dir}",
        "status_glossary_error": "Glossar konnte nicht geladen werden: {error}",
        "status_unexpected_error": "Unerwarteter Fehler: {error}",
        "warning_encoding_fallback": "Datei wurde mit Fallback-Encoding gelesen: {encoding}",
        "warning_partial_result": "Vom Nutzer abgebrochen. Teilresultat bis zum letzten abgeschlossenen Segment.",
        "warning_context_fallback": "Kontextmodus laeuft derzeit im stabilen Fallback ohne Prompt-Vermischung, um Dopplungen und Artefakte zu vermeiden.",
        "error_prefix": "{name}: {error}",
        "zip_name": "translations_{session_id}.zip",
        "zip_name_partial": "translations_partial_{session_id}.zip",
        "language_de": "Deutsch",
        "language_en": "Englisch",
        "language_fr": "Franzoesisch",
        "language_pl": "Polnisch",
        "language_ru": "Russisch",
        "language_es": "Spanisch",
        "language_uk": "Ukrainisch",
    },
    "en": {
        "lang_toggle": "DE",
        "powered_locally_by": "Powered locally by",
        "banner_idle": "Upload TXT files or paste free text to start the translation.",
        "input": "Input",
        "output": "Output",
        "input_mode": "Input mode",
        "mode_upload": "File upload",
        "mode_text": "Free text",
        "upload_label": "Upload TXT files",
        "upload_help": "Process multiple structured TXT files together.",
        "text_label": "Text to translate",
        "text_help": "For quick single texts, paste content here instead of uploading a TXT file.",
        "save_txt": "TXT",
        "save_txt_help": "If enabled, the pasted source text is also saved as a TXT file in the session output folder.",
        "save_as_label": "Save as TXT in the project output folder?",
        "save_as_help": "File name for the pasted text if it should be stored together with the translations.",
        "source_language": "Source language",
        "source_help": "Language of the source text. It is mapped to the matching NLLB language code.",
        "target_language": "Target language",
        "target_help": "Language the text should be translated into locally.",
        "advanced_options": "Advanced translation options",
        "model": "Model",
        "model_help": "The recommended local default is the 1.3B NLLB model. The 3.3B option is prepared for later extension.",
        "segmentation_mode": "Segmentation mode",
        "segmentation_help": "Auto mainly works paragraph-wise and falls back to sentence segmentation for long blocks. Sentence is finer-grained, paragraph keeps more structure.",
        "segmentation_auto": "Auto",
        "segmentation_paragraph": "Paragraph",
        "segmentation_sentence": "Sentence",
        "max_segment_length": "Maximum segment length",
        "max_segment_help": "420 characters is a solid starting point for NLLB 1.3B. If artifacts appear, choose smaller values; if splits are too aggressive, choose a larger one.",
        "context_toggle": "Use previous segment as context",
        "context_help": "Intended for better terminology consistency between segments. In the stable implementation, prompt mixing remains intentionally conservative.",
        "glossary": "Optional glossary (CSV or JSON)",
        "glossary_help": "Use this to enforce fixed terminology such as company names, legal forms, or domain-specific terms.",
        "start": "Start translation",
        "start_help": "Starts batch translation for the active input mode.",
        "stop": "Stop",
        "stop_help": "Stops a running translation after the currently processed segment.",
        "device_caption": "Acceleration: `{device}`. The model stays loaded for additional files.",
        "current_translation": "Current translation",
        "overall_job": "Overall job",
        "chars": "Characters",
        "segments": "Segments",
        "runtime": "Runtime",
        "original": "Original",
        "translation": "Translation",
        "partial_result": "Partial result",
        "partial_file_caption": "This file was not processed completely.",
        "download_txt": "Download TXT",
        "download_zip": "Download all results as ZIP",
        "warn_missing_input": "Please upload a TXT file or enter free text.",
        "warn_same_language": "Source and target language must be different.",
        "status_init": "Initializing translation...",
        "status_cancel_requested": "Cancellation requested. The current segment will finish cleanly first.",
        "status_processing_file": "Processing file {current}/{total}: {name}",
        "status_cancelled": "Translation cancelled by user.",
        "status_completed": "Session {session_id} completed. Output folder: {output_dir}",
        "status_glossary_error": "Glossary could not be loaded: {error}",
        "status_unexpected_error": "Unexpected error: {error}",
        "warning_encoding_fallback": "File was read with fallback encoding: {encoding}",
        "warning_partial_result": "Cancelled by user. Partial result up to the last completed segment.",
        "warning_context_fallback": "Context mode currently runs in a stable fallback without prompt mixing to avoid duplicates and artifacts.",
        "error_prefix": "{name}: {error}",
        "zip_name": "translations_{session_id}.zip",
        "zip_name_partial": "translations_partial_{session_id}.zip",
        "language_de": "German",
        "language_en": "English",
        "language_fr": "French",
        "language_pl": "Polish",
        "language_ru": "Russian",
        "language_es": "Spanish",
        "language_uk": "Ukrainian",
    },
}

LANGUAGE_NAME_KEYS = {
    "Deutsch": "language_de",
    "Englisch": "language_en",
    "Franzoesisch": "language_fr",
    "Polnisch": "language_pl",
    "Russisch": "language_ru",
    "Spanisch": "language_es",
    "Ukrainisch": "language_uk",
}

RUNTIME_MESSAGE_PATTERNS = [
    (re.compile(r"^Datei wurde mit Fallback-Encoding gelesen: (.+)$"), "warning_encoding_fallback", "encoding"),
    (
        re.compile(r"^Kontextmodus laeuft derzeit im stabilen Fallback ohne Prompt-Vermischung, um Dopplungen und Artefakte zu vermeiden\.$"),
        "warning_context_fallback",
        None,
    ),
    (
        re.compile(r"^Vom Nutzer abgebrochen\. Teilresultat bis zum letzten abgeschlossenen Segment\.$"),
        "warning_partial_result",
        None,
    ),
]


@dataclass
class TranslationJob:
    job_id: str
    stop_event: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    thread: Thread | None = None
    status: str = "running"
    status_key: str = "status_init"
    status_params: dict = field(default_factory=dict)
    session_id: str = ""
    output_dir: str = ""
    outputs: list[dict] = field(default_factory=list)
    zip_bytes: bytes | None = None
    file_progress: float = 0.0
    segment_progress: float = 0.0
    current_file: str = ""
    current_file_index: int = 0
    total_files: int = 0
    cancel_requested: bool = False
    error_key: str | None = None
    error_params: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "status_key": self.status_key,
                "status_params": dict(self.status_params),
                "session_id": self.session_id,
                "output_dir": self.output_dir,
                "outputs": [dict(item) for item in self.outputs],
                "zip_bytes": self.zip_bytes,
                "file_progress": self.file_progress,
                "segment_progress": self.segment_progress,
                "current_file": self.current_file,
                "current_file_index": self.current_file_index,
                "total_files": self.total_files,
                "cancel_requested": self.cancel_requested,
                "error_key": self.error_key,
                "error_params": dict(self.error_params),
            }


def setup_logging() -> None:
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


setup_logging()


def get_page_icon() -> str:
    logo_path = BASE_DIR / "assets" / "logo.png"
    return str(logo_path) if logo_path.exists() else "LT"


st.set_page_config(
    page_title="SoftTranslate",
    page_icon=get_page_icon(),
    layout="wide",
)


def initialize_session_state() -> None:
    defaults = {
        "ui_language": "de",
        "input_mode": "upload",
        "pasted_text": "",
        "pasted_text_name": DEFAULT_PASTED_TEXT_NAME,
        "save_pasted_input": False,
        "upload_widget_nonce": 0,
        "current_job_id": None,
        "translation_jobs": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


def current_ui_language() -> str:
    return st.session_state.ui_language


def tr(key: str, **kwargs) -> str:
    return TRANSLATIONS[current_ui_language()][key].format(**kwargs)


def localize_language_label(label: str) -> str:
    return tr(LANGUAGE_NAME_KEYS.get(label, label))


def localize_runtime_message(message: str) -> str:
    for pattern, key, group_name in RUNTIME_MESSAGE_PATTERNS:
        match = pattern.match(message)
        if match:
            if group_name is None:
                return tr(key)
            return tr(key, **{group_name: match.group(1)})
    return message


def get_current_job() -> TranslationJob | None:
    job_id = st.session_state.get("current_job_id")
    if not job_id:
        return None
    return st.session_state.translation_jobs.get(job_id)


def get_current_job_snapshot() -> dict | None:
    job = get_current_job()
    if job is None:
        return None
    return job.snapshot()


def clear_job_view() -> None:
    st.session_state.current_job_id = None


def toggle_ui_language() -> None:
    st.session_state.ui_language = "en" if current_ui_language() == "de" else "de"


def handle_input_mode_change() -> None:
    if st.session_state.input_mode == "upload":
        st.session_state.pasted_text = ""
        st.session_state.pasted_text_name = DEFAULT_PASTED_TEXT_NAME
        st.session_state.save_pasted_input = False
    else:
        st.session_state.upload_widget_nonce += 1
    clear_job_view()


def start_translation_job(payload: dict) -> None:
    job_id = uuid4().hex
    job = TranslationJob(job_id=job_id)
    thread = Thread(target=run_translation_job, args=(job, payload), daemon=True)
    job.thread = thread
    st.session_state.translation_jobs[job_id] = job
    st.session_state.current_job_id = job_id
    thread.start()


def request_stop_current_job() -> None:
    job = get_current_job()
    if job is None:
        return
    with job.lock:
        if job.status != "running":
            return
        job.cancel_requested = True
        job.status_key = "status_cancel_requested"
        job.status_params = {}
    job.stop_event.set()


def update_job(job: TranslationJob, **changes) -> None:
    with job.lock:
        for key, value in changes.items():
            setattr(job, key, value)


def append_job_output(job: TranslationJob, item: dict) -> None:
    with job.lock:
        job.outputs.append(item)


def run_translation_job(job: TranslationJob, payload: dict) -> None:
    output_files: list[Path] = []
    session_id = build_session_id()
    output_dir = create_session_output_dir(BASE_DIR, session_id)
    update_job(
        job,
        session_id=session_id,
        output_dir=str(output_dir),
        total_files=len(payload["inputs"]),
        status_key="status_init",
        status_params={},
    )

    try:
        glossary_manager = GlossaryManager()
        glossary_file = payload.get("glossary_file")
        if glossary_file is not None:
            glossary_path = BASE_DIR / "temp" / f"{job.job_id}_{glossary_file['name']}"
            glossary_path.write_bytes(glossary_file["bytes"])
            glossary_manager = GlossaryManager(str(glossary_path))
            glossary_manager.load()

        translator = Translator(glossary_manager=glossary_manager)

        for file_index, input_item in enumerate(payload["inputs"], start=1):
            if job.stop_event.is_set():
                break

            input_name = input_item["name"]
            update_job(
                job,
                current_file=input_name,
                current_file_index=file_index,
                file_progress=(file_index - 1) / len(payload["inputs"]),
                segment_progress=0.0,
                status_key="status_processing_file",
                status_params={"current": file_index, "total": len(payload["inputs"]), "name": input_name},
            )

            try:
                source_text, detected_encoding = decode_uploaded_text(input_name, input_item["bytes"])
                warnings: list[str] = []
                if detected_encoding != "utf-8":
                    warnings.append(f"Datei wurde mit Fallback-Encoding gelesen: {detected_encoding}")

                def progress_callback(current: int, total: int, _source: str, _translated: str) -> None:
                    update_job(job, segment_progress=current / total)

                request = TranslationRequest(
                    text=source_text,
                    source_language_code=payload["source_language_code"],
                    target_language_code=payload["target_language_code"],
                    model_name=payload["model_name"],
                    quality_mode="Ausgewogen",
                    segmentation_mode=payload["segmentation_mode"],
                    max_segment_length=payload["max_segment_length"],
                    use_context_overlap=payload["use_context_overlap"],
                )
                result = translator.translate_document(
                    request,
                    progress_callback=progress_callback,
                    should_cancel=job.stop_event.is_set,
                )
                warnings.extend(result.warnings)

                output_item, written_files = build_output_item(
                    output_dir=output_dir,
                    input_name=input_name,
                    source_text=source_text,
                    translated_text=result.translated_text,
                    target_language_code=payload["target_language_code"],
                    processing_seconds=result.processing_seconds,
                    segment_count=len(result.segments),
                    warnings=warnings,
                    is_uploaded_file=input_item["kind"] == "upload",
                    should_write_pasted_input=payload["save_pasted_input"] and input_item["kind"] == "text",
                    pasted_text_name=payload["pasted_text_name"],
                    partial=False,
                )
                output_files.extend(written_files)
                append_job_output(job, output_item)
            except TranslationCancelledError as exc:
                if exc.partial_text:
                    partial_warnings = list(exc.warnings)
                    partial_warnings.append("Vom Nutzer abgebrochen. Teilresultat bis zum letzten abgeschlossenen Segment.")
                    output_item, written_files = build_output_item(
                        output_dir=output_dir,
                        input_name=input_name,
                        source_text=source_text,
                        translated_text=exc.partial_text,
                        target_language_code=payload["target_language_code"],
                        processing_seconds=exc.processing_seconds,
                        segment_count=exc.translated_segment_count,
                        warnings=partial_warnings,
                        is_uploaded_file=input_item["kind"] == "upload",
                        should_write_pasted_input=payload["save_pasted_input"] and input_item["kind"] == "text",
                        pasted_text_name=payload["pasted_text_name"],
                        partial=True,
                    )
                    output_files.extend(written_files)
                    append_job_output(job, output_item)
                break
            except (FileReadError, ValueError, RuntimeError) as exc:
                logging.exception("Fehler bei Datei %s", input_name)
                append_job_output(job, {"name": input_name, "error": str(exc)})
            finally:
                update_job(job, segment_progress=0.0)

        zip_bytes = create_zip_archive(output_files) if output_files else None
        if job.stop_event.is_set():
            update_job(job, status="cancelled", status_key="status_cancelled", status_params={}, zip_bytes=zip_bytes)
        else:
            update_job(
                job,
                status="completed",
                status_key="status_completed",
                status_params={"session_id": session_id, "output_dir": str(output_dir)},
                file_progress=1.0,
                zip_bytes=zip_bytes,
            )
    except GlossaryError as exc:
        update_job(
            job,
            status="error",
            status_key="status_glossary_error",
            status_params={"error": str(exc)},
            error_key="status_glossary_error",
            error_params={"error": str(exc)},
        )
    except Exception as exc:  # pragma: no cover
        logging.exception("Unerwarteter Fehler in der Uebersetzungssession")
        update_job(
            job,
            status="error",
            status_key="status_unexpected_error",
            status_params={"error": str(exc)},
            error_key="status_unexpected_error",
            error_params={"error": str(exc)},
        )


def build_output_item(
    output_dir: Path,
    input_name: str,
    source_text: str,
    translated_text: str,
    target_language_code: str,
    processing_seconds: float,
    segment_count: int,
    warnings: list[str],
    is_uploaded_file: bool,
    should_write_pasted_input: bool,
    pasted_text_name: str,
    partial: bool,
) -> tuple[dict, list[Path]]:
    written_files: list[Path] = []
    if should_write_pasted_input:
        original_output_name = pasted_text_name.strip() or DEFAULT_PASTED_TEXT_NAME
        if not original_output_name.lower().endswith(".txt"):
            original_output_name = f"{original_output_name}.txt"
        original_output_path = write_text_output(output_dir, original_output_name, source_text)
        written_files.append(original_output_path)

    output_filename = build_output_filename(input_name, target_language_code.split("_")[0])
    output_path = None
    if is_uploaded_file or should_write_pasted_input:
        output_path = write_text_output(output_dir, output_filename, translated_text)
        written_files.append(output_path)

    return (
        {
            "name": input_name,
            "translation_name": output_filename,
            "original": source_text,
            "translated": translated_text,
            "segments": segment_count,
            "processing_seconds": processing_seconds,
            "char_count_input": len(source_text),
            "char_count_output": len(translated_text),
            "warnings": warnings,
            "output_path": output_path,
            "partial": partial,
        },
        written_files,
    )


def render_header() -> None:
    toggle_cols = st.columns([1, 8])
    with toggle_cols[0]:
        st.button(tr("lang_toggle"), on_click=toggle_ui_language, use_container_width=True)

    logo_path = BASE_DIR / "assets" / "logo.png"
    if logo_path.exists():
        encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <div style="width:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 0 8px 0;">
                <img src="data:image/png;base64,{encoded_logo}" alt="SoftTranslate logo" style="width:260px;max-width:42vw;height:auto;display:block;margin:0 auto;" />
                <div style="font-size:11px;color:#6a7280;margin-top:2px;text-align:center;">
                    {tr("powered_locally_by")} <strong>SoftTrade Innovations</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="width:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 0 8px 0;">
                <div style="display:inline-block;border:1px solid #d9dce1;border-radius:16px;padding:18px 28px;text-align:center;background:#f7f8fa;">
                    <div style="font-size:56px;">TXT</div>
                    <div style="font-size:12px;color:#5c6470;">Logo</div>
                </div>
                <div style="font-size:11px;color:#6a7280;margin-top:2px;text-align:center;">
                    {tr("powered_locally_by")} <strong>SoftTrade Innovations</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div style="margin-top:28px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:12px;color:#6a7280;text-align:right;">
            Copyright © SoftTrade Innovations
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_column_heading(label: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center;font-size:1.55rem;font-weight:700;margin-bottom:0.85rem;color:var(--text-color);">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_layout_styles() -> None:
    st.markdown(
        """
        <style>
        .st-key-input_panel,
        .st-key-output_panel {
            border: 1px solid var(--text-color);
            border-radius: 14px;
            padding: 18px 18px 14px 18px;
            box-sizing: border-box;
            height: 100%;
            width: 100%;
            background: transparent;
            color: var(--text-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(snapshot: dict | None, transient_message: tuple[str, dict] | None = None) -> None:
    level = "info"
    message = tr("banner_idle")

    if snapshot is not None:
        status = snapshot["status"]
        message = tr(snapshot["status_key"], **snapshot["status_params"])
        if status == "completed":
            level = "success"
        elif status in {"cancelled", "running"} and snapshot["cancel_requested"]:
            level = "warning"
        elif status == "cancelled":
            level = "warning"
        elif status == "error":
            level = "error"
    if transient_message is not None:
        level, payload = transient_message
        message = tr(payload["key"], **payload.get("params", {}))

    styles = {
        "info": ("#e7f0ff", "#1d4ed8", "#dbeafe"),
        "success": ("#e8f7ea", "#166534", "#bbf7d0"),
        "warning": ("#fff7df", "#92400e", "#fde68a"),
        "error": ("#feeceb", "#b91c1c", "#fecaca"),
    }
    background, text_color, border_color = styles[level]
    st.markdown(
        f"""
        <div style="width:100%;margin:0 0 18px 0;padding:12px 18px;border:1px solid {border_color};border-radius:14px;background:{background};color:{text_color};text-align:center;font-size:0.98rem;font-weight:600;box-sizing:border-box;">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(snapshot: dict | None, file_progress, segment_progress, result_area) -> None:
    if snapshot is None:
        segment_progress.progress(0.0)
        file_progress.progress(0.0)
        return

    segment_progress.progress(float(snapshot["segment_progress"]))
    file_progress.progress(float(snapshot["file_progress"]))

    with result_area:
        for item in snapshot["outputs"]:
            if item.get("error"):
                st.error(tr("error_prefix", name=item["name"], error=localize_runtime_message(item["error"])))
                continue

            title = f"{item['name']} -> {item['translation_name']}"
            if item.get("partial"):
                title += f" ({tr('partial_result')})"
            with st.expander(title, expanded=True):
                stat_cols = st.columns(3)
                stat_cols[0].metric(tr("chars"), item["char_count_input"])
                stat_cols[1].metric(tr("segments"), item["segments"])
                stat_cols[2].metric(tr("runtime"), f"{item['processing_seconds']:.2f}s")

                if item.get("partial"):
                    st.caption(tr("partial_file_caption"))

                preview_cols = st.columns(2)
                with preview_cols[0]:
                    st.markdown(f"**{tr('original')}**")
                    st.text_area(
                        f"original-{item['name']}",
                        item["original"],
                        height=260,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                with preview_cols[1]:
                    st.markdown(f"**{tr('translation')}**")
                    st.text_area(
                        f"translation-{item['name']}",
                        item["translated"],
                        height=260,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                for warning in item["warnings"]:
                    st.warning(localize_runtime_message(warning))

                download_cols = st.columns([4, 1.2])
                with download_cols[1]:
                    st.download_button(
                        tr("download_txt"),
                        data=item["translated"].encode("utf-8"),
                        file_name=item["translation_name"],
                        mime="text/plain",
                        key=f"download-{snapshot['job_id']}-{item['translation_name']}",
                        use_container_width=True,
                    )

        if snapshot["zip_bytes"]:
            name_key = "zip_name_partial" if snapshot["status"] == "cancelled" else "zip_name"
            st.download_button(
                tr("download_zip"),
                data=snapshot["zip_bytes"],
                file_name=tr(name_key, session_id=snapshot["session_id"] or snapshot["job_id"]),
                mime="application/zip",
                use_container_width=True,
            )


def build_runtime_inputs(input_mode: str, uploaded_files, pasted_text: str, pasted_text_name: str) -> list[dict]:
    if input_mode == "upload":
        return [{"name": file.name, "bytes": file.getvalue(), "kind": "upload"} for file in (uploaded_files or [])]

    if not pasted_text.strip():
        return []
    return [{"name": pasted_text_name.strip() or DEFAULT_PASTED_TEXT_NAME, "bytes": pasted_text.encode("utf-8"), "kind": "text"}]


def main() -> None:
    render_header()
    inject_layout_styles()
    banner_placeholder = st.empty()
    job_snapshot = get_current_job_snapshot()
    job_running = bool(job_snapshot and job_snapshot["status"] == "running")
    transient_message: tuple[str, dict] | None = None
    left_col, right_col = st.columns([1, 1.2], gap="large")

    available_languages = get_language_labels()

    with left_col:
        with st.container(key="input_panel"):
            render_column_heading(tr("input"))
            st.radio(
                tr("input_mode"),
                INPUT_MODES,
                horizontal=True,
                key="input_mode",
                format_func=lambda mode: tr("mode_upload") if mode == "upload" else tr("mode_text"),
                on_change=handle_input_mode_change,
                disabled=job_running,
            )

            uploaded_files = []
            glossary_file = None
            input_mode = st.session_state.input_mode

            if input_mode == "upload":
                uploaded_files = st.file_uploader(
                    tr("upload_label"),
                    type=["txt"],
                    accept_multiple_files=True,
                    help=tr("upload_help"),
                    key=f"uploaded_files_{st.session_state.upload_widget_nonce}",
                    disabled=job_running,
                )
            else:
                st.text_area(
                    tr("text_label"),
                    key="pasted_text",
                    height=220,
                    help=tr("text_help"),
                    disabled=job_running,
                )
                paste_save_cols = st.columns([1, 4])
                with paste_save_cols[0]:
                    st.checkbox(
                        tr("save_txt"),
                        key="save_pasted_input",
                        help=tr("save_txt_help"),
                        disabled=job_running,
                    )
                with paste_save_cols[1]:
                    st.text_input(
                        tr("save_as_label"),
                        key="pasted_text_name",
                        help=tr("save_as_help"),
                        disabled=job_running,
                    )

            language_cols = st.columns(2)
            with language_cols[0]:
                source_label = st.selectbox(
                    tr("source_language"),
                    available_languages,
                    index=available_languages.index("Russisch"),
                    format_func=localize_language_label,
                    help=tr("source_help"),
                    disabled=job_running,
                )
            with language_cols[1]:
                target_label = st.selectbox(
                    tr("target_language"),
                    available_languages,
                    index=available_languages.index("Deutsch"),
                    format_func=localize_language_label,
                    help=tr("target_help"),
                    disabled=job_running,
                )
            with st.expander(tr("advanced_options")):
                model_label = st.selectbox(
                    tr("model"),
                    list(MODEL_REGISTRY.keys()),
                    index=0,
                    help=tr("model_help"),
                    disabled=job_running,
                )
                segmentation_mode = st.selectbox(
                    tr("segmentation_mode"),
                    SEGMENTATION_MODES,
                    index=0,
                    format_func=lambda mode: {
                        "Auto": tr("segmentation_auto"),
                        "Absatz": tr("segmentation_paragraph"),
                        "Satz": tr("segmentation_sentence"),
                    }[mode],
                    help=tr("segmentation_help"),
                    disabled=job_running,
                )
                max_segment_length = st.slider(
                    tr("max_segment_length"),
                    min_value=180,
                    max_value=900,
                    value=420,
                    step=20,
                    help=tr("max_segment_help"),
                    disabled=job_running,
                )
                use_context_overlap = st.toggle(
                    tr("context_toggle"),
                    value=False,
                    help=tr("context_help"),
                    disabled=job_running,
                )
                glossary_file = st.file_uploader(
                    tr("glossary"),
                    type=["csv", "json"],
                    accept_multiple_files=False,
                    help=tr("glossary_help"),
                    disabled=job_running,
                )

            button_cols = st.columns(2)
            with button_cols[0]:
                start_clicked = st.button(
                    tr("start"),
                    type="primary",
                    use_container_width=True,
                    disabled=job_running,
                    help=tr("start_help"),
                )
            with button_cols[1]:
                stop_clicked = st.button(
                    tr("stop"),
                    type="secondary",
                    use_container_width=True,
                    disabled=not job_running,
                    help=tr("stop_help"),
                )

            st.caption(tr("device_caption", device=detect_device()))

    if stop_clicked:
        request_stop_current_job()
        job_snapshot = get_current_job_snapshot()
        job_running = bool(job_snapshot and job_snapshot["status"] == "running")

    if start_clicked:
        runtime_inputs = build_runtime_inputs(
            input_mode=st.session_state.input_mode,
            uploaded_files=uploaded_files,
            pasted_text=st.session_state.pasted_text,
            pasted_text_name=st.session_state.pasted_text_name,
        )
        if not runtime_inputs:
            transient_message = ("warning", {"key": "warn_missing_input"})
        elif source_label == target_label:
            transient_message = ("warning", {"key": "warn_same_language"})
        else:
            payload = {
                "inputs": runtime_inputs,
                "source_language_code": get_nllb_code(source_label),
                "target_language_code": get_nllb_code(target_label),
                "model_name": resolve_model_name(model_label),
                "segmentation_mode": segmentation_mode,
                "max_segment_length": max_segment_length,
                "use_context_overlap": use_context_overlap,
                "glossary_file": None if glossary_file is None else {"name": glossary_file.name, "bytes": glossary_file.getvalue()},
                "save_pasted_input": st.session_state.save_pasted_input,
                "pasted_text_name": st.session_state.pasted_text_name,
            }
            start_translation_job(payload)
            job_snapshot = get_current_job_snapshot()
            job_running = bool(job_snapshot and job_snapshot["status"] == "running")

    with banner_placeholder.container():
        render_status_banner(job_snapshot, transient_message=transient_message)

    with right_col:
        with st.container(key="output_panel"):
            render_column_heading(tr("output"))
            st.caption(tr("current_translation"))
            segment_progress = st.progress(0)
            st.caption(tr("overall_job"))
            file_progress = st.progress(0)
            result_area = st.container()
            render_results(job_snapshot, file_progress, segment_progress, result_area)

    render_footer()

    if job_running:
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
