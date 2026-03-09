# SoftTranslate

SoftTranslate is a local Streamlit application for translating structured TXT files with Meta NLLB models.

## Features

- Local browser UI with multi-file TXT upload
- Offline translation after the initial model download
- Default support for `facebook/nllb-200-distilled-1.3B`
- Modular structure for future model upgrades
- Paragraph, sentence, and auto segmentation modes
- Optional context overlap mode
- Side-by-side preview, individual downloads, and ZIP export
- Local logs, session output folders, and optional JSON metadata
- Glossary hook for terminology control

## Requirements

- macOS on Apple Silicon recommended
- Python 3.11 or 3.12
- `venv`
- Enough free disk space for the Hugging Face model download

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Download

On first launch, the app downloads `facebook/nllb-200-distilled-1.3B` from Hugging Face and stores it in the local cache. After that, translation runs locally as long as the model is already cached.

Prepared model options:

- `facebook/nllb-200-distilled-1.3B`
- `facebook/nllb-200-3.3B`

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, typically `http://localhost:8501`.

## Logo

The app automatically loads a PNG logo from:

```text
assets/logo.png
```

Recommended asset settings:

- Filename: `logo.png`
- Location: `assets/logo.png`
- Format: PNG with transparent background
- Recommended width: 512 px to 1024 px
- Recommended aspect ratio: square or slightly portrait

If the file is present, it is rendered in the header. If not, the app falls back to the default text block.

## Typical Workflow

1. Upload one or more TXT files.
2. Select the source and target language.
3. Optionally adjust segmentation, segment length, glossary, and context settings.
4. Start the translation.
5. Review the preview and download individual files or a ZIP archive.

## Project Structure

```text
softtranslate/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── logo.png
├── config/
│   └── languages.py
├── core/
│   ├── glossary.py
│   ├── io_utils.py
│   ├── model_manager.py
│   ├── reassembler.py
│   ├── segmenter.py
│   └── translator.py
├── logs/
├── output/
├── temp/
└── tests/
```

## Notes

- If the model fails to load, check available RAM and disk space first.
- For very large files, reduce the segment length.
- If MPS is unavailable, the app falls back to CPU automatically.
- Empty or unreadable files are handled per file and do not stop the full batch.

## Privacy

- No cloud API is used for the actual translation step.
- Only the initial model download requires a network connection.
- Logs and generated outputs stay inside the local project directory.

## Extension Points

- Larger model support: `core/model_manager.py`
- Glossary logic: `core/glossary.py`
- Additional languages: `config/languages.py`
- Input/output handling: `core/io_utils.py`
