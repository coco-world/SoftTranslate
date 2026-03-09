from __future__ import annotations

import base64
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
from core.translator import TranslationRequest, Translator


BASE_DIR = Path(__file__).resolve().parent
ensure_runtime_directories(BASE_DIR)
LOG_PATH = BASE_DIR / "logs" / "app.log"


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


if "is_translating" not in st.session_state:
    st.session_state.is_translating = False
if "run_translation" not in st.session_state:
    st.session_state.run_translation = False


def request_translation_run() -> None:
    st.session_state.is_translating = True
    st.session_state.run_translation = True


def render_header() -> None:
    logo_path = BASE_DIR / "assets" / "logo.png"
    if logo_path.exists():
        encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <div style="width:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 0 8px 0;">
                <img src="data:image/png;base64,{encoded_logo}" alt="SoftTranslate logo" style="width:260px;max-width:42vw;height:auto;display:block;margin:0 auto;" />
                <div style="font-size:11px;color:#6a7280;margin-top:2px;text-align:center;">
                    Powered locally by <strong>SoftTrade Innovations</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="width:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 0 8px 0;">
                <div style="display:inline-block;border:1px solid #d9dce1;border-radius:16px;padding:18px 28px;text-align:center;background:#f7f8fa;">
                    <div style="font-size:56px;">TXT</div>
                    <div style="font-size:12px;color:#5c6470;">Logo</div>
                </div>
                <div style="font-size:11px;color:#6a7280;margin-top:2px;text-align:center;">
                    Powered locally by <strong>SoftTrade Innovations</strong>
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
        <div style="text-align:center;font-size:1.55rem;font-weight:700;margin-bottom:0.85rem;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        render_column_heading("Input")
        uploaded_files = st.file_uploader(
            "TXT-Dateien hochladen",
            type=["txt"],
            accept_multiple_files=True,
            help="Mehrere strukturierte TXT-Dateien koennen gemeinsam verarbeitet werden.",
        )
        pasted_text = ""
        pasted_text_name = "pasted_text.txt"
        save_pasted_input = False
        with st.expander("Text direkt einfuegen"):
            pasted_text = st.text_area(
                "Text zum Uebersetzen",
                height=220,
                help="Fuer schnelle Einzeltexte kannst du den Inhalt direkt hier einfuegen, statt eine TXT-Datei hochzuladen.",
            )
            paste_save_cols = st.columns([1, 4])
            with paste_save_cols[0]:
                save_pasted_input = st.checkbox(
                    "TXT",
                    value=False,
                    help="Wenn aktiviert, wird der eingefuegte Originaltext zusaetzlich als TXT im Session-Outputordner gespeichert.",
                )
            with paste_save_cols[1]:
                pasted_text_name = st.text_input(
                    "Speichern als TXT in Output im Projektordner?",
                    value="pasted_text.txt",
                    help="Dateiname fuer den eingefuegten Text, falls du ihn zusammen mit den Uebersetzungen im Output-Ordner ablegen willst.",
                )

        source_label = st.selectbox(
            "Quellsprache",
            get_language_labels(),
            index=get_language_labels().index("Russisch"),
            help="Sprache des hochgeladenen Originaltexts. Sie wird auf den passenden NLLB-Sprachcode gemappt.",
        )
        target_label = st.selectbox(
            "Zielsprache",
            get_language_labels(),
            index=get_language_labels().index("Deutsch"),
            help="Sprache, in die der Text lokal uebersetzt wird.",
        )
        with st.expander("Erweiterte Uebersetzungsoptionen"):
            model_label = st.selectbox(
                "Modell",
                list(MODEL_REGISTRY.keys()),
                index=0,
                help="Standard ist das lokal empfohlene 1.3B-NLLB-Modell. Die 3.3B-Option ist fuer spaetere Erweiterung vorbereitet.",
            )
            segmentation_mode = st.selectbox(
                "Segmentierungsmodus",
                ["Auto", "Absatz", "Satz"],
                index=0,
                help="Auto arbeitet primaer absatzweise und faellt bei langen Bloecken auf Satzsegmente zurueck. Satz ist feiner, Absatz erhaelt mehr Struktur.",
            )
            max_segment_length = st.slider(
                "Maximale Segmentlaenge",
                min_value=180,
                max_value=900,
                value=420,
                step=20,
                help="420 Zeichen ist ein sinnvoller Startwert fuer NLLB 1.3B: meist genug Kontext fuer saubere Saetze und noch klein genug fuer stabile, speicherschonende Inferenz. Fuer normale Sachtexte sind etwa 300 bis 500 Zeichen meist sinnvoll. Wenn das Modell Artefakte oder Auslassungen zeigt, eher kleiner waehlen; wenn zu hart getrennt wird, etwas groesser.",
            )
            use_context_overlap = st.toggle(
                "Vorheriges Segment als Kontext verwenden",
                value=False,
                help="Gedacht fuer bessere begriffliche Konsistenz zwischen aufeinanderfolgenden Segmenten, zum Beispiel bei Namen oder wiederholten Formulierungen. In der aktuellen stabilen Umsetzung wird diese Option bewusst sehr defensiv behandelt, weil direkte Prompt-Vermischung bei NLLB schnell Dopplungen, Auslassungen oder Artefakte erzeugen kann.",
            )
            glossary_file = st.file_uploader(
                "Optionales Glossar (CSV oder JSON)",
                type=["csv", "json"],
                accept_multiple_files=False,
                help="Damit kannst du feste Terminologie vorgeben, etwa Firmennamen, Rechtsformen oder Fachbegriffe wie ООО -> GmbH. Das Glossar wird als einfache Quelle-Ziel-Ersetzung nach der Uebersetzung angewendet und ist besonders nuetzlich, wenn bestimmte Begriffe immer gleich erscheinen sollen.",
            )
        start_clicked = st.button(
            "Uebersetzung starten",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_translating,
            help="Startet die Batch-Uebersetzung fuer alle hochgeladenen TXT-Dateien.",
            on_click=request_translation_run,
        )

        st.info(
            f"Beschleunigung: `{detect_device()}`. Das Modell wird einmal geladen und danach fuer weitere Dateien wiederverwendet."
        )

    with right_col:
        render_column_heading("Output")
        status_placeholder = st.empty()
        file_progress = st.progress(0)
        segment_progress = st.progress(0)
        result_area = st.container()

    if not st.session_state.run_translation:
        return

    runtime_inputs = list(uploaded_files or [])
    if pasted_text.strip():
        runtime_inputs.append(
            {
                "name": pasted_text_name.strip() or "pasted_text.txt",
                "bytes": pasted_text.encode("utf-8"),
            }
        )

    if not runtime_inputs:
        st.session_state.is_translating = False
        st.warning("Bitte mindestens eine TXT-Datei hochladen oder Text direkt einfuegen.")
        return

    if source_label == target_label:
        st.session_state.is_translating = False
        st.warning("Quell- und Zielsprache muessen verschieden sein.")
        return

    glossary_manager = GlossaryManager()
    if glossary_file is not None:
        temp_glossary_path = BASE_DIR / "temp" / glossary_file.name
        temp_glossary_path.write_bytes(glossary_file.getvalue())
        glossary_manager = GlossaryManager(str(temp_glossary_path))
        try:
            glossary_manager.load()
        except GlossaryError as exc:
            st.session_state.is_translating = False
            st.error(f"Glossar konnte nicht geladen werden: {exc}")
            return

    session_id = build_session_id()
    output_dir = create_session_output_dir(BASE_DIR, session_id)
    translator = Translator(glossary_manager=glossary_manager)

    translated_outputs = []
    output_files = []

    try:
        for file_index, input_item in enumerate(runtime_inputs, start=1):
            input_name = input_item.name if hasattr(input_item, "name") else input_item["name"]
            input_bytes = input_item.getvalue() if hasattr(input_item, "getvalue") else input_item["bytes"]
            file_progress.progress((file_index - 1) / len(runtime_inputs))
            status_placeholder.info(f"Verarbeite Datei {file_index}/{len(runtime_inputs)}: {input_name}")

            try:
                source_text, detected_encoding = decode_uploaded_text(input_name, input_bytes)
                warnings = []
                if detected_encoding != "utf-8":
                    warnings.append(f"Datei wurde mit Fallback-Encoding gelesen: {detected_encoding}")

                def progress_callback(current: int, total: int, _source: str, _translated: str) -> None:
                    segment_progress.progress(current / total)

                request = TranslationRequest(
                    text=source_text,
                    source_language_code=get_nllb_code(source_label),
                    target_language_code=get_nllb_code(target_label),
                    model_name=resolve_model_name(model_label),
                    quality_mode="Ausgewogen",
                    segmentation_mode=segmentation_mode,
                    max_segment_length=max_segment_length,
                    use_context_overlap=use_context_overlap,
                )
                result = translator.translate_document(request, progress_callback=progress_callback)
                warnings.extend(result.warnings)

                is_uploaded_file = hasattr(input_item, "getvalue")
                should_write_pasted_outputs = (not is_uploaded_file) and save_pasted_input

                if should_write_pasted_outputs:
                    original_output_name = pasted_text_name.strip() or "pasted_text.txt"
                    if not original_output_name.lower().endswith(".txt"):
                        original_output_name = f"{original_output_name}.txt"
                    original_output_path = write_text_output(output_dir, original_output_name, source_text)
                    output_files.append(original_output_path)

                output_filename = build_output_filename(input_name, get_nllb_code(target_label).split("_")[0])
                output_path = None
                if is_uploaded_file or should_write_pasted_outputs:
                    output_path = write_text_output(output_dir, output_filename, result.translated_text)
                    output_files.append(output_path)

                translated_outputs.append(
                    {
                        "name": input_name,
                        "translation_name": output_filename,
                        "original": source_text,
                        "translated": result.translated_text,
                        "segments": len(result.segments),
                        "processing_seconds": result.processing_seconds,
                        "char_count_input": len(source_text),
                        "char_count_output": len(result.translated_text),
                        "warnings": warnings,
                        "output_path": output_path,
                    }
                )
            except (FileReadError, ValueError, RuntimeError) as exc:
                logging.exception("Fehler bei Datei %s", input_name)
                translated_outputs.append(
                    {
                        "name": input_name,
                        "error": str(exc),
                    }
                )
            finally:
                segment_progress.progress(0)

        file_progress.progress(1.0)
        status_placeholder.success(f"Session {session_id} abgeschlossen. Ausgabeordner: {output_dir}")
        zip_bytes = create_zip_archive(output_files) if output_files else None
    finally:
        st.session_state.is_translating = False
        st.session_state.run_translation = False

    with result_area:
        for item in translated_outputs:
            if item.get("error"):
                st.error(f"{item['name']}: {item['error']}")
                continue

            with st.expander(f"{item['name']} -> {item['translation_name']}", expanded=True):
                stat_cols = st.columns(3)
                stat_cols[0].metric("Zeichen", item["char_count_input"])
                stat_cols[1].metric("Segmente", item["segments"])
                stat_cols[2].metric("Laufzeit", f"{item['processing_seconds']:.2f}s")

                preview_cols = st.columns(2)
                with preview_cols[0]:
                    st.markdown("**Original**")
                    st.text_area(
                        f"original-{item['name']}",
                        item["original"],
                        height=260,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                with preview_cols[1]:
                    st.markdown("**Uebersetzung**")
                    st.text_area(
                        f"translation-{item['name']}",
                        item["translated"],
                        height=260,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                if item["warnings"]:
                    for warning in item["warnings"]:
                        st.warning(warning)

                st.download_button(
                    "TXT herunterladen",
                    data=item["translated"].encode("utf-8"),
                    file_name=item["translation_name"],
                    mime="text/plain",
                    key=f"download-{item['translation_name']}",
                )

        if zip_bytes:
            st.download_button(
                "Alle Ergebnisse als ZIP herunterladen",
                data=zip_bytes,
                file_name=f"translations_{session_id}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    render_footer()


if __name__ == "__main__":
    main()
