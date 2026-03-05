# hf-model-download

A single-file Python script that downloads a curated set of GGUF model files from HuggingFace with parallel downloads and SHA256 freshness checks, so re-runs only pull files that have actually changed.

## Features

- **Parallel downloads** — configurable worker pool (default: 4)
- **SHA256 freshness checks** — skips files that are already up-to-date
- **Optional files** — multimodal projector (`mmproj`) files are silently skipped when not present in a repo
- **Zero install friction** — inline PEP 723 metadata means `uv run` handles dependencies automatically
- **Clean status output** — per-file `[checking]`, `[up-to-date]`, `[downloading]`, `[done]`, `[failed]` labels

## Included Models

| Model | Quantization | Repo |
|-------|-------------|------|
| GLM-4.7-Flash | UD-Q4_K_XL | unsloth/GLM-4.7-Flash-GGUF |
| gpt-oss-20b | UD-Q4_K_XL | unsloth/gpt-oss-20b-GGUF |
| Qwen3.5-2B | UD-Q4_K_XL | unsloth/Qwen3.5-2B-GGUF |
| Qwen3.5-4B | UD-Q8_K_XL | unsloth/Qwen3.5-4B-GGUF |
| Qwen3.5-9B | UD-Q4_K_XL | unsloth/Qwen3.5-9B-GGUF |
| Qwen3.5-35B-A3B | UD-Q4_K_XL | unsloth/Qwen3.5-35B-A3B-GGUF |
| Qwen3-Coder-30B-A3B-Instruct | UD-Q4_K_XL | unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF |
| Qwen3.5-35B-A3B-heretic | Q4_K_M | mradermacher/Qwen3.5-35B-A3B-heretic-GGUF |

Each model also attempts to download an optional `mmproj-F16.gguf` multimodal projector file when available.

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** pip

## Usage

### With uv (recommended — no manual install needed)

```bash
uv run download_models.py
```

`uv` reads the inline dependency metadata and installs `huggingface_hub` and `typer` into an isolated environment automatically.

### With pip

```bash
pip install "huggingface_hub>=0.24" "typer>=0.12"
python download_models.py
```

## Options

```
Usage: download_models.py [OPTIONS]

  Download GGUF model files from HuggingFace with parallel downloads and
  SHA256 freshness checks.

Options:
  --output-dir PATH   Directory to save downloaded model files [default: ~/models]
  --workers INTEGER   Number of parallel download workers [default: 4]
  --list              Print all configured models and exit without downloading
  --help              Show this message and exit.
```

### Examples

```bash
# Download to a custom directory
uv run download_models.py --output-dir /mnt/models

# Use 8 parallel workers
uv run download_models.py --workers 8

# List all configured models without downloading
uv run download_models.py --list
```

## How Freshness Checking Works

For each model file, the script:

1. Queries the HuggingFace API for the file's LFS SHA256 hash
2. If the file already exists locally, computes its SHA256 and compares
3. Skips the download if the hashes match; re-downloads if they differ or metadata is unavailable

This makes re-runs fast — only genuinely new or updated files are transferred.

## Customizing the Model List

Edit the `MODELS` list in `download_models.py`. Each entry is a `ModelEntry`:

```python
ModelEntry(
    repo_id="org/repo-name-GGUF",   # HuggingFace repo
    hf_filename="model.gguf",        # filename inside the repo
    local_filename="my-model.gguf",  # filename to save locally
    optional=False,                  # if True, silently skip when not in repo
)
```

## HuggingFace Authentication

For gated models, set your HuggingFace token before running:

```bash
export HF_TOKEN=hf_...
uv run download_models.py
```
