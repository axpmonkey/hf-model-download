# hf-model-download

A single-file Python script that downloads a curated set of GGUF model files from HuggingFace with parallel downloads and SHA256 freshness checks, so re-runs only pull files that have actually changed.

## Features

- **Parallel downloads** — configurable worker pool (default: 4)
- **Fast freshness checks** — local SHA256 cache makes repeat runs near-instant; falls back to size comparison then full hash
- **Batched metadata** — fetches repo metadata once per repo (not per file), with a progress bar
- **Optional files** — multimodal projector (`mmproj`) files are silently skipped when not present in a repo
- **Zero install friction** — inline PEP 723 metadata means `uv run` handles dependencies automatically
- **Clean status output** — per-file `[hashing]`, `[up-to-date]`, `[downloading]`, `[done]`, `[failed]` labels

## Included Models

| Model | Quantization | Repo |
|-------|-------------|------|
| GLM-4.7-Flash | UD-Q4_K_XL | unsloth/GLM-4.7-Flash-GGUF |
| gpt-oss-20b | UD-Q4_K_XL | unsloth/gpt-oss-20b-GGUF |
| Qwen3.5-4B | UD-Q8_K_XL | unsloth/Qwen3.5-4B-GGUF |
| Qwen3.5-9B | UD-Q4_K_XL | unsloth/Qwen3.5-9B-GGUF |
| Qwen3.5-35B-A3B | UD-Q4_K_XL | unsloth/Qwen3.5-35B-A3B-GGUF |
| Qwen3-Coder-30B-A3B-Instruct | UD-Q4_K_XL | unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF |
| Qwen3.5-35B-A3B-heretic | Q4_K_M | mradermacher/Qwen3.5-35B-A3B-heretic-GGUF |

Each model also attempts to download an optional `mmproj-F16.gguf` multimodal projector file when available.

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
pip install "huggingface_hub>=0.24" "typer>=0.12" "python-dotenv>=1.0"
python download_models.py
```

## Options

```
Usage: download_models.py [OPTIONS]

  Download GGUF model files from HuggingFace with parallel downloads and
  SHA256 freshness checks.

Options:
  --output-dir  PATH                Directory to save downloaded model files
                                    [default: ~/models]
  --workers     INTEGER RANGE [1<=x<=16]
                                    Number of parallel download workers
                                    [default: 4]
  --list                            Print all configured models and exit
                                    without downloading
  --no-cache                        Ignore local SHA256 cache and force
                                    re-hashing of all files
  --help                            Show this message and exit.
```

### Examples

```bash
# Download to a custom directory
uv run download_models.py --output-dir /mnt/models

# Use 8 parallel workers
uv run download_models.py --workers 8

# List all configured models without downloading
uv run download_models.py --list

# Force re-hash all files (ignore cache)
uv run download_models.py --no-cache
```

## How Freshness Checking Works

For each repo, the script fetches metadata once (batched per repo, not per file) and then checks each file through a layered strategy:

1. **Cache lookup** — if a `.hf-sha256-cache.json` entry exists with matching file size and mtime, use the cached SHA256 (no disk I/O beyond `stat`)
2. **Size check** — if the local file size differs from the remote, it's outdated; skip straight to re-download
3. **Full SHA256 hash** — if sizes match but no cache hit, compute the hash and update the cache
4. **No metadata** — if neither hash nor size is available from HuggingFace, re-download conservatively

The first run hashes every file (slow on large models), but subsequent runs are near-instant thanks to the cache. The cache is stored in the output directory and auto-updates after downloads.

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

The script loads your HuggingFace token from a `.env` file in the project root (gitignored by default):

```bash
# Create the .env file
echo 'HF_TOKEN=hf_...' > .env

# Then run as usual
uv run download_models.py
```

Alternatively, you can export the token as an environment variable:

```bash
export HF_TOKEN=hf_...
uv run download_models.py
```
