#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=0.24", "typer>=0.12", "python-dotenv>=1.0"]
# ///
"""
Download GGUF model files from HuggingFace with parallel downloads and freshness checks.

Freshness is checked via size comparison (instant) then SHA256 (thorough). Repo metadata
is fetched once per repo, not per file.

Usage:
    python download_models.py [--output-dir ~/models] [--workers 4]

Dependencies:
    pip install "huggingface_hub>=0.24" "typer>=0.12" "python-dotenv>=1.0"

Requires Python 3.11+ (hashlib.file_digest).
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

load_dotenv()
disable_progress_bars()

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
    size: int | None


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


def fetch_repo_file_info(repo_id: str, filenames: set[str]) -> dict[str, HfFileInfo]:
    """Fetch existence and LFS SHA256 for all requested filenames in a repo in one API call."""
    not_found = {f: HfFileInfo(exists=False, sha256=None, size=None) for f in filenames}
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=os.getenv("HF_TOKEN"))
        result: dict[str, HfFileInfo] = {}
        for sibling in info.siblings or []:
            if sibling.rfilename in filenames:
                sha256 = sibling.lfs.sha256 if sibling.lfs is not None else None
                result[sibling.rfilename] = HfFileInfo(
                    exists=True, sha256=sha256, size=sibling.size,
                )
        return {**not_found, **result}
    except Exception:
        logger.debug("Failed to fetch metadata for %s", repo_id, exc_info=True)
        return not_found


def compute_sha256(path: Path) -> str:
    """Compute SHA256 of a local file using hashlib.file_digest."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@dataclass
class DownloadResult:
    entry: ModelEntry
    status: str  # 'downloaded' | 'skipped' | 'failed'
    error: Exception | None = None


def process_model(entry: ModelEntry, output_dir: Path, hf_info: HfFileInfo) -> DownloadResult:
    """Check freshness and download a single model file. Never raises."""
    label = entry.local_filename
    local_path = output_dir / entry.local_filename

    try:
        if not hf_info.exists:
            if entry.optional:
                typer.echo(f"[not-present] {label} — not in repo, skipping")
                return DownloadResult(entry, "skipped")
            # Non-optional file missing from repo — let hf_hub_download raise the error
            typer.echo(f"[downloading] {label}")
        elif local_path.exists():
            if hf_info.sha256 is not None:
                local_size = local_path.stat().st_size
                # Quick size check before expensive hash — different size means outdated.
                if hf_info.size is not None and local_size != hf_info.size:
                    size_gib = local_size / (1024**3)
                    typer.echo(f"[outdated]    {label} — size mismatch ({size_gib:.1f} GiB local), re-downloading")
                else:
                    size_gib = local_size / (1024**3)
                    typer.echo(f"[hashing]     {label} ({size_gib:.1f} GiB)")
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
                token=os.getenv("HF_TOKEN"),
            )
        )

        if entry.hf_filename != entry.local_filename:
            downloaded_path.replace(local_path)

        size_gib = local_path.stat().st_size / (1024**3)
        typer.echo(f"[done]        {label} ({size_gib:.2f} GiB)")
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

    # Group filenames by repo so we can fetch each repo's metadata in one API call.
    repo_to_filenames: dict[str, set[str]] = defaultdict(set)
    for entry in MODELS:
        repo_to_filenames[entry.repo_id].add(entry.hf_filename)

    typer.echo(f"Output directory : {resolved_output_dir}")
    typer.echo(f"Workers          : {workers}")
    typer.echo(f"Models           : {len(MODELS)} across {len(repo_to_filenames)} repos\n")

    # Phase 1: fetch metadata for all repos in parallel (one API call per repo).
    total_repos = len(repo_to_filenames)
    file_info_cache: dict[str, dict[str, HfFileInfo]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, total_repos)) as executor:
        repo_futures: dict[Future[dict[str, HfFileInfo]], str] = {
            executor.submit(fetch_repo_file_info, repo_id, filenames): repo_id
            for repo_id, filenames in repo_to_filenames.items()
        }
        for completed_count, future in enumerate(as_completed(repo_futures), 1):
            file_info_cache[repo_futures[future]] = future.result()
            filled = int(completed_count / total_repos * 20)
            bar = "█" * filled + "░" * (20 - filled)
            typer.echo(
                f"\rFetching repo metadata  [{bar}] {completed_count}/{total_repos}",
                nl=False,
            )
    typer.echo()

    # Phase 2: check freshness and download in parallel.
    results: list[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map: dict[Future[DownloadResult], ModelEntry] = {
            executor.submit(
                process_model,
                entry,
                resolved_output_dir,
                file_info_cache[entry.repo_id][entry.hf_filename],
            ): entry
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
