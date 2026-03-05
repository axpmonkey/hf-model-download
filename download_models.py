#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=0.24", "typer>=0.12"]
# ///
"""
Download GGUF model files from HuggingFace with parallel downloads and SHA256 freshness checks.

Usage:
    python download_models.py [--output-dir ~/models] [--workers 4]

Dependencies:
    pip install "huggingface_hub>=0.24" "typer>=0.12"
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    hf_filename: str
    local_filename: str
    optional: bool = False  # if True, silently skip when file is not present in the repo


@dataclass(frozen=True)
class HfFileInfo:
    exists: bool
    sha256: str | None


MODELS: list[ModelEntry] = [
    ModelEntry(
        repo_id="unsloth/GLM-4.7-Flash-GGUF",
        hf_filename="GLM-4.7-Flash-UD-Q4_K_XL.gguf",
        local_filename="GLM-4.7-Flash-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        hf_filename="gpt-oss-20b-UD-Q4_K_XL.gguf",
        local_filename="gpt-oss-20b-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-2B-GGUF",
        hf_filename="Qwen3.5-2B-UD-Q4_K_XL.gguf",
        local_filename="Qwen3.5-2B-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
        hf_filename="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        local_filename="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        hf_filename="Qwen3.5-4B-UD-Q8_K_XL.gguf",
        local_filename="Qwen3.5-4B-UD-Q8_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        hf_filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        local_filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        local_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
    ),
    ModelEntry(
        repo_id="mradermacher/Qwen3.5-35B-A3B-heretic-GGUF",
        hf_filename="Qwen3.5-35B-A3B-heretic.Q4_K_M.gguf",
        local_filename="Qwen3.5-35B-A3B-heretic-Q4_K_M.gguf",
    ),
    # mmproj (multimodal projector) files — optional, only downloaded if present in the repo
    ModelEntry(
        repo_id="unsloth/GLM-4.7-Flash-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="GLM-4.7-Flash-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="gpt-oss-20b-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-2B-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3.5-2B-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3.5-35B-A3B-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3.5-4B-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3.5-9B-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3-Coder-30B-A3B-Instruct-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="mradermacher/Qwen3.5-35B-A3B-heretic-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3.5-35B-A3B-heretic-mmproj-F16.gguf",
        optional=True,
    ),
]


def get_hf_file_info(repo_id: str, filename: str) -> HfFileInfo:
    """Fetch existence and LFS SHA256 for a specific file from the HuggingFace API."""
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        for sibling in info.siblings or []:
            if sibling.rfilename == filename:
                sha256 = sibling.lfs.sha256 if sibling.lfs is not None else None
                return HfFileInfo(exists=True, sha256=sha256)
        return HfFileInfo(exists=False, sha256=None)
    except Exception:
        logger.debug("Failed to fetch metadata for %s/%s", repo_id, filename, exc_info=True)
        return HfFileInfo(exists=False, sha256=None)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 of a local file in 8 MiB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DownloadResult:
    entry: ModelEntry
    status: str  # 'downloaded' | 'skipped' | 'failed'
    error: Exception | None = None


def process_model(entry: ModelEntry, output_dir: Path) -> DownloadResult:
    """Check freshness and download a single model file. Never raises."""
    label = entry.local_filename
    local_path = output_dir / entry.local_filename

    try:
        typer.echo(f"[checking]    {label}")
        hf_info = get_hf_file_info(entry.repo_id, entry.hf_filename)

        if not hf_info.exists:
            if entry.optional:
                typer.echo(f"[not-present] {label} — not in repo, skipping")
                return DownloadResult(entry, "skipped")
            # Non-optional file missing from repo — let hf_hub_download raise the error
            typer.echo(f"[downloading] {label}")
        elif local_path.exists():
            if hf_info.sha256 is not None:
                local_sha256 = compute_sha256(local_path)
                if local_sha256 == hf_info.sha256:
                    typer.echo(f"[up-to-date]  {label}")
                    return DownloadResult(entry, "skipped")
                typer.echo(f"[outdated]    {label} — re-downloading")
            else:
                typer.echo(f"[no-metadata] {label} — SHA256 unavailable, re-downloading")
        else:
            typer.echo(f"[downloading] {label}")

        downloaded_path = Path(
            hf_hub_download(
                repo_id=entry.repo_id,
                filename=entry.hf_filename,
                local_dir=output_dir,
            )
        )

        if entry.hf_filename != entry.local_filename:
            downloaded_path.replace(local_path)

        typer.echo(f"[done]        {label}")
        return DownloadResult(entry, "downloaded")

    except Exception as exc:
        typer.echo(f"[failed]      {label}: {exc}", err=True)
        logger.debug("Download failed for %s", label, exc_info=True)
        return DownloadResult(entry, "failed", error=exc)


app = typer.Typer(add_completion=False)


def _list_models() -> None:
    """Print all configured models grouped by repo and exit."""
    from itertools import groupby

    typer.echo(f"{len(MODELS)} entries across {len({m.repo_id for m in MODELS})} repos:\n")
    keyfunc = lambda m: m.repo_id  # noqa: E731
    for repo_id, entries in groupby(sorted(MODELS, key=keyfunc), key=keyfunc):
        typer.echo(f"  {repo_id}")
        for entry in entries:
            tag = "  [optional]" if entry.optional else ""
            typer.echo(f"    {entry.local_filename}{tag}")
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
    """Download GGUF model files from HuggingFace with parallel downloads and SHA256 freshness checks."""
    if list_models:
        _list_models()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Output directory : {resolved_output_dir}")
    typer.echo(f"Workers          : {workers}")
    typer.echo(f"Models           : {len(MODELS)}\n")

    results: list[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map: dict[Future[DownloadResult], ModelEntry] = {
            executor.submit(process_model, entry, resolved_output_dir): entry
            for entry in MODELS
        }
        for future in as_completed(future_map):
            results.append(future.result())

    downloaded = sum(1 for r in results if r.status == "downloaded")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    typer.echo(f"\nSummary: {downloaded} downloaded, {skipped} up-to-date, {failed} failed")

    if failed:
        failed_names = [r.entry.local_filename for r in results if r.status == "failed"]
        typer.echo(f"Failed: {', '.join(failed_names)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
