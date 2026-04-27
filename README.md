# hf-model-download

A single-file Python script that downloads a curated set of GGUF model files from HuggingFace. Re-runs skip files that are already up-to-date — freshness checking and caching are handled natively by `huggingface_hub`.

## Features

- **Native caching** — `huggingface_hub` tracks downloaded files internally, so repeat runs skip files that haven't changed upstream
- **Fast transfers** — downloads use `hf_xet` (bundled with `huggingface_hub` ≥0.32) for chunk-based deduplication
- **Hierarchical storage** — files are saved in a folder structure mirroring HuggingFace repos: `~/models/{creator}/{repo}/{filename}`
- **Multimodal projectors** — `mmproj` files are downloaded alongside models when configured; configured projector failures are treated as errors
- **Zero install friction** — inline PEP 723 metadata means `uv run` handles dependencies automatically

## Included Models

| Model | Quantization | Repo |
|-------|-------------|------|
| Qwen2.5-1.5B-Instruct | Q4_K_M | bartowski/Qwen2.5-1.5B-Instruct-GGUF |
| Qwen3.5-9B | UD-Q5_K_XL | unsloth/Qwen3.5-9B-GGUF |
| Qwen3-Coder-30B-A3B-Instruct | UD-Q4_K_XL | unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF |
| Qwen3.6-35B-A3B | UD-Q4_K_XL | unsloth/Qwen3.6-35B-A3B-GGUF |
| Qwen3.6-35B-A3B | UD-Q5_K_XL | unsloth/Qwen3.6-35B-A3B-GGUF |
| Qwen3.6-35B-A3B-uncensored-heretic | Q4_K_M | llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF |

Models with a multimodal projector also download the corresponding `mmproj` file.

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** pip

## Usage

### With uv (recommended — no manual install needed)

```bash
uv run download_models.py
```

`uv` reads the inline dependency metadata and installs `huggingface_hub`, `typer`, and `python-dotenv` into an isolated environment automatically.

### With pip

```bash
pip install "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"
python download_models.py
```

### Server wrapper without uv

Use `run.sh` on systems where `uv` is not installed. It creates `.venv` on
first run, installs dependencies once, and then reuses that environment:

```bash
./run.sh --output-dir ~/models
```

To intentionally update the Python dependencies:

```bash
./run.sh --upgrade-deps --output-dir ~/models
```

## Options

```
Usage: download_models.py [OPTIONS]

  Download GGUF model files from HuggingFace.

Options:
  --output-dir  PATH  Directory to save downloaded model files
                      [default: ~/models]
  --list              Print all configured models and exit without
                      downloading
  --help              Show this message and exit.
```

### Examples

```bash
# Download to a custom directory
uv run download_models.py --output-dir /mnt/models

# List all configured models without downloading
uv run download_models.py --list

# Server wrapper with dependency upgrade
./run.sh --upgrade-deps --list
```

## Folder Structure

Downloads are organized by repo, mirroring the HuggingFace hierarchy:

```
~/models/
├── bartowski/
│   └── Qwen2.5-1.5B-Instruct-GGUF/
│       └── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
├── unsloth/
│   ├── Qwen3.5-9B-GGUF/
│   │   ├── Qwen3.5-9B-UD-Q5_K_XL.gguf
│   │   └── mmproj-F16.gguf
│   ├── Qwen3.6-35B-A3B-GGUF/
│   │   ├── Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
│   │   ├── Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
│   │   └── mmproj-F16.gguf
│   └── Qwen3-Coder-30B-A3B-Instruct-GGUF/
│       └── Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
└── llmfan46/
    └── Qwen3.6-35B-A3B-uncensored-heretic-GGUF/
        ├── Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf
        └── Qwen3.6-35B-A3B-mmproj-BF16.gguf
```

## Customizing the Model List

Edit the `MODELS` list in `download_models.py`. Each entry is a `ModelEntry`:

```python
ModelEntry(
    repo_id="org/repo-name-GGUF",          # HuggingFace repo
    hf_filename="model.gguf",              # filename inside the repo
    mmproj_filename="mmproj-F16.gguf",     # optional multimodal projector
)
```

## HuggingFace Authentication

The script loads your HuggingFace token from a `.env` file in the project root (gitignored by default):

```bash
# Create the .env file
echo 'HF_TOKEN=hf_...' > .env

# Then run as usual
uv run download_models.py
```

Alternatively, export the token as an environment variable:

```bash
export HF_TOKEN=hf_...
uv run download_models.py
```
