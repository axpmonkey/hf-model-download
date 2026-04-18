#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=1.5", "typer>=0.24", "python-dotenv>=1.2"]
# ///
"""
Download GGUF model files from HuggingFace.

Models are stored in a hierarchical folder structure mirroring HuggingFace:
    ~/models/{creator}/{repo}/{hf_filename}

Freshness checking and caching are handled by huggingface_hub natively — files
that are already up-to-date are skipped automatically. Downloads use hf_xet
(bundled with huggingface_hub >=0.32) for chunk-based deduplication and faster
transfers.

Usage:
    python download_models.py [--output-dir ~/models]

Dependencies:
    pip install "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    hf_filename: str
    mmproj_filename: str | None = None


MODELS: list[ModelEntry] = [
    # --- bartowski quantizations ---
    ModelEntry(
        repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        hf_filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    ),
    # --- unsloth quantizations ---
    ModelEntry(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        hf_filename="Qwen3.5-9B-UD-Q5_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        hf_filename="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        hf_filename="Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    # --- HauhauCS uncensored variant ---
    ModelEntry(
        repo_id="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        hf_filename="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf",
    ),
]


def _download_file(repo_id: str, filename: str, output_dir: Path) -> bool:
    """Download a single file. Returns True on success, False on failure."""
    label = f"{repo_id}/{filename}"
    try:
        typer.echo(f"[syncing] {label}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir / repo_id,
            token=os.getenv("HF_TOKEN"),
        )
        return True
    except Exception as exc:
        short_error = str(exc).split("\n", 1)[0]
        typer.echo(f"[failed]  {label} — {short_error}", err=True)
        return False


app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path.home() / "models",
        "--output-dir",
        help="Directory to save downloaded model files",
        show_default=True,
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        help="Print all configured models and exit without downloading",
        is_eager=True,
    ),
) -> None:
    """Download GGUF model files from HuggingFace."""
    if list_models:
        for model in MODELS:
            typer.echo(f"  {model.repo_id}/{model.hf_filename}")
            if model.mmproj_filename:
                typer.echo(f"  {model.repo_id}/{model.mmproj_filename}")
        raise typer.Exit()

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Output directory : {resolved_output_dir}")
    typer.echo(f"Models           : {len(MODELS)} models\n")

    ok = 0
    skipped = 0
    failed_labels: list[str] = []

    for model in MODELS:
        if _download_file(model.repo_id, model.hf_filename, resolved_output_dir):
            ok += 1
        else:
            failed_labels.append(f"{model.repo_id}/{model.hf_filename}")

        if model.mmproj_filename:
            if _download_file(model.repo_id, model.mmproj_filename, resolved_output_dir):
                ok += 1
            else:
                skipped += 1

    typer.echo(f"\nSummary: {ok} ok, {skipped} skipped, {len(failed_labels)} failed")

    if failed_labels:
        typer.echo(f"Failed: {', '.join(failed_labels)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
