#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=1.5", "typer>=0.24", "python-dotenv>=1.2"]
# ///
"""
Download GGUF model files from HuggingFace with parallel downloads.

Models are stored in a hierarchical folder structure mirroring HuggingFace:
    ~/models/{creator}/{repo}/{hf_filename}

Freshness checking and caching are handled by huggingface_hub natively — files
that are already up-to-date are skipped automatically. Downloads use hf_xet
(bundled with huggingface_hub >=0.32) for chunk-based deduplication and faster
transfers.

Usage:
    python download_models.py [--output-dir ~/models] [--workers 4]

Dependencies:
    pip install "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

load_dotenv()
disable_progress_bars()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    hf_filename: str
    mmproj_filename: str | None = None  # optional multimodal projector file in the same repo


@dataclass(frozen=True)
class DownloadItem:
    repo_id: str
    hf_filename: str
    optional: bool = False


MODELS: list[ModelEntry] = [
    # --- unsloth quantizations (UD-Q4_K_XL / UD-Q8_K_XL) ---
    ModelEntry(
        repo_id="unsloth/GLM-4.7-Flash-GGUF",
        hf_filename="GLM-4.7-Flash-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        hf_filename="gpt-oss-20b-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
        hf_filename="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        hf_filename="Qwen3.5-4B-UD-Q8_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        hf_filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    # --- bartowski quantizations (Q4_K_M / Q8_0) ---
    ModelEntry(
        repo_id="bartowski/zai-org_GLM-4.7-Flash-GGUF",
        hf_filename="zai-org_GLM-4.7-Flash-Q4_K_M.gguf",
        mmproj_filename="mmproj-zai-org_GLM-4.7-Flash-f16.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/openai_gpt-oss-20b-GGUF",
        hf_filename="openai_gpt-oss-20b-Q4_K_M.gguf",
        mmproj_filename="mmproj-openai_gpt-oss-20b-f16.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        hf_filename="Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
        hf_filename="Qwen_Qwen3.5-4B-Q8_0.gguf",
        mmproj_filename="mmproj-Qwen_Qwen3.5-4B-f16.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-9B-GGUF",
        hf_filename="Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen_Qwen3.5-9B-f16.gguf",
    ),
    # --- mradermacher / llmfan46 heretic variants ---
    ModelEntry(
        repo_id="mradermacher/Qwen3.5-35B-A3B-heretic-GGUF",
        hf_filename="Qwen3.5-35B-A3B-heretic.Q4_K_M.gguf",
        mmproj_filename="mmproj-F16.gguf",
    ),
    ModelEntry(
        repo_id="llmfan46/Qwen3.5-35B-A3B-heretic-v2-GGUF",
        hf_filename="Qwen3.5-35B-A3B-heretic-v2-Q4_K_M.gguf",
        mmproj_filename="Qwen3.5-35B-A3B-mmproj-BF16.gguf",
    ),
]


def _expand_models(models: list[ModelEntry]) -> list[DownloadItem]:
    """Expand ModelEntry list into flat DownloadItem list, adding mmproj as optional items."""
    items: list[DownloadItem] = []
    for m in models:
        items.append(DownloadItem(m.repo_id, m.hf_filename))
        if m.mmproj_filename:
            items.append(DownloadItem(m.repo_id, m.mmproj_filename, optional=True))
    return items


DOWNLOAD_ITEMS: list[DownloadItem] = _expand_models(MODELS)


@dataclass
class DownloadResult:
    item: DownloadItem
    status: str  # 'downloaded' | 'skipped' | 'failed'
    error: Exception | None = None


def process_download(item: DownloadItem, output_dir: Path) -> DownloadResult:
    """Download a single file via hf_hub_download. Never raises."""
    label = f"{item.repo_id}/{item.hf_filename}"
    local_path = output_dir / item.repo_id / item.hf_filename

    try:
        if local_path.exists():
            typer.echo(f"[updating]    {label}")
        else:
            typer.echo(f"[downloading] {label}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=item.repo_id,
            filename=item.hf_filename,
            local_dir=output_dir / item.repo_id,
            token=os.getenv("HF_TOKEN"),
        )
        return DownloadResult(item, "ok")

    except Exception as exc:
        if item.optional:
            typer.echo(f"[skipped]     {label} — {exc}")
            return DownloadResult(item, "skipped")
        typer.echo(f"[failed]      {label}: {exc}", err=True)
        logger.debug("Download failed for %s", label, exc_info=True)
        return DownloadResult(item, "failed", error=exc)


app = typer.Typer(add_completion=False)


def _list_models() -> None:
    """Print all configured models grouped by repo and exit."""
    from itertools import groupby

    items = DOWNLOAD_ITEMS
    typer.echo(f"{len(items)} files across {len({i.repo_id for i in items})} repos:\n")
    keyfunc = lambda i: i.repo_id  # noqa: E731
    for repo_id, group in groupby(sorted(items, key=keyfunc), key=keyfunc):
        typer.echo(f"  {repo_id}/")
        for item in group:
            typer.echo(f"    {item.hf_filename}")
    raise typer.Exit()


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path.home() / "models",
        "--output-dir",
        help="Directory to save downloaded model files",
        show_default=True,
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        help="Number of parallel download workers",
        show_default=True,
        min=1,
        max=16,
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        help="Print all configured models and exit without downloading",
        is_eager=True,
    ),
) -> None:
    """Download GGUF model files from HuggingFace with parallel downloads."""
    if list_models:
        _list_models()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    unique_repos = {item.repo_id for item in DOWNLOAD_ITEMS}
    typer.echo(f"Output directory : {resolved_output_dir}")
    typer.echo(f"Workers          : {workers}")
    typer.echo(f"Models           : {len(DOWNLOAD_ITEMS)} files across {len(unique_repos)} repos\n")

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map: dict[Future[DownloadResult], DownloadItem] = {
            executor.submit(process_download, item, resolved_output_dir): item
            for item in DOWNLOAD_ITEMS
        }
        for future in as_completed(future_map):
            results.append(future.result())

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    typer.echo(f"\nSummary: {ok} ok, {skipped} skipped, {failed} failed")

    if failed:
        failed_labels = [f"{r.item.repo_id}/{r.item.hf_filename}" for r in results if r.status == "failed"]
        typer.echo(f"Failed: {', '.join(failed_labels)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
