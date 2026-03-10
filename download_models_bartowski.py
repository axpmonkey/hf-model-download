#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=1.5", "typer>=0.24", "python-dotenv>=1.2"]
# ///
"""
Download GGUF model files from HuggingFace with parallel downloads and freshness checks.

Uses bartowski quantizations (Q4_K_M) instead of unsloth (UD-Q4_K_XL), except for
Qwen3-Coder-30B-A3B-Instruct which stays at unsloth (no bartowski version available).

Freshness is checked via local SHA256 cache (instant), then size comparison (instant),
then full SHA256 hash (thorough). Repo metadata is fetched once per repo, not per file.
Downloads use hf_xet (bundled with huggingface_hub >=0.32) for chunk-based deduplication
and faster transfers.

Usage:
    python download_models_bartowski.py [--output-dir ~/models] [--workers 4]

Dependencies:
    pip install "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"

Requires Python 3.11+ (hashlib.file_digest).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
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

SHA256_CACHE_FILENAME = ".hf-sha256-cache.json"


def load_sha256_cache(output_dir: Path) -> dict[str, dict]:
    """Load the local SHA256 cache from disk. Returns empty dict on any error."""
    try:
        return json.loads((output_dir / SHA256_CACHE_FILENAME).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sha256_cache(output_dir: Path, cache: dict[str, dict]) -> None:
    """Persist the SHA256 cache to disk."""
    (output_dir / SHA256_CACHE_FILENAME).write_text(json.dumps(cache, indent=2) + "\n")


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
    # --- bartowski quantizations (Q4_K_M) ---
    ModelEntry(
        repo_id="bartowski/zai-org_GLM-4.7-Flash-GGUF",
        hf_filename="zai-org_GLM-4.7-Flash-Q4_K_M.gguf",
        local_filename="GLM-4.7-Flash-Q4_K_M.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/openai_gpt-oss-20b-GGUF",
        hf_filename="openai_gpt-oss-20b-Q4_K_M.gguf",
        local_filename="gpt-oss-20b-Q4_K_M.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        hf_filename="Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
        local_filename="Qwen3.5-35B-A3B-Q4_K_M.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
        hf_filename="Qwen_Qwen3.5-4B-Q8_0.gguf",
        local_filename="Qwen3.5-4B-Q8_0.gguf",
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-9B-GGUF",
        hf_filename="Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        local_filename="Qwen3.5-9B-Q4_K_M.gguf",
    ),
    # --- unsloth (no bartowski version available) ---
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        local_filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
    ),
    # --- llmfan46 heretic v2 ---
    ModelEntry(
        repo_id="llmfan46/Qwen3.5-35B-A3B-heretic-v2-GGUF",
        hf_filename="Qwen3.5-35B-A3B-heretic-v2-Q4_K_M.gguf",
        local_filename="Qwen3.5-35B-A3B-heretic-v2-Q4_K_M.gguf",
    ),
    # --- mmproj (multimodal projector) files — optional, only downloaded if present in the repo ---
    ModelEntry(
        repo_id="bartowski/zai-org_GLM-4.7-Flash-GGUF",
        hf_filename="mmproj-zai-org_GLM-4.7-Flash-f16.gguf",
        local_filename="GLM-4.7-Flash-mmproj-f16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="bartowski/openai_gpt-oss-20b-GGUF",
        hf_filename="mmproj-openai_gpt-oss-20b-f16.gguf",
        local_filename="gpt-oss-20b-mmproj-f16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        hf_filename="mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf",
        local_filename="Qwen3.5-35B-A3B-mmproj-f16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
        hf_filename="mmproj-Qwen_Qwen3.5-4B-f16.gguf",
        local_filename="Qwen3.5-4B-mmproj-f16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="bartowski/Qwen_Qwen3.5-9B-GGUF",
        hf_filename="mmproj-Qwen_Qwen3.5-9B-f16.gguf",
        local_filename="Qwen3.5-9B-mmproj-f16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        hf_filename="mmproj-F16.gguf",
        local_filename="Qwen3-Coder-30B-A3B-Instruct-mmproj-F16.gguf",
        optional=True,
    ),
    ModelEntry(
        repo_id="llmfan46/Qwen3.5-35B-A3B-heretic-v2-GGUF",
        hf_filename="Qwen3.5-35B-A3B-mmproj-BF16.gguf",
        local_filename="Qwen3.5-35B-A3B-heretic-v2-mmproj-BF16.gguf",
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


def _update_cache(
    cache: dict[str, dict], lock: threading.Lock, filename: str, sha256: str,
    size: int, mtime_ns: int,
) -> None:
    """Thread-safe update of a single cache entry."""
    with lock:
        cache[filename] = {
            "sha256": sha256,
            "size": size,
            "mtime_ns": mtime_ns,
        }


def process_model(
    entry: ModelEntry,
    output_dir: Path,
    hf_info: HfFileInfo,
    sha256_cache: dict[str, dict],
    cache_lock: threading.Lock,
    *,
    use_cache: bool = True,
) -> DownloadResult:
    """Check freshness and download a single model file. Never raises."""
    label = entry.local_filename
    local_path = output_dir / entry.local_filename

    try:
        if not hf_info.exists:
            if entry.optional:
                typer.echo(f"[not-present] {label} — not in repo, skipping")
                return DownloadResult(entry, "skipped")
            typer.echo(f"[downloading] {label}")
        elif local_path.exists():
            if hf_info.sha256 is not None:
                local_stat = local_path.stat()
                local_size = local_stat.st_size

                # Quick size check — different size means definitely outdated.
                if hf_info.size is not None and local_size != hf_info.size:
                    size_gib = local_size / (1024**3)
                    typer.echo(f"[outdated]    {label} — size mismatch ({size_gib:.1f} GiB local), re-downloading")
                else:
                    # Check local SHA256 cache before expensive hash.
                    cached = sha256_cache.get(entry.local_filename)
                    if (
                        use_cache
                        and cached
                        and cached["size"] == local_size
                        and cached["mtime_ns"] == local_stat.st_mtime_ns
                    ):
                        local_sha256 = cached["sha256"]
                    else:
                        size_gib = local_size / (1024**3)
                        typer.echo(f"[hashing]     {label} ({size_gib:.1f} GiB)")
                        local_sha256 = compute_sha256(local_path)
                        # Re-stat after hash to get consistent mtime for the content we just read.
                        post_hash_stat = local_path.stat()
                        _update_cache(
                            sha256_cache, cache_lock, entry.local_filename, local_sha256,
                            post_hash_stat.st_size, post_hash_stat.st_mtime_ns,
                        )

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

        # Cache the known remote SHA256 so next run skips hashing this file.
        if hf_info.sha256 is not None:
            post_download_stat = local_path.stat()
            _update_cache(
                sha256_cache, cache_lock, entry.local_filename, hf_info.sha256,
                post_download_stat.st_size, post_download_stat.st_mtime_ns,
            )

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
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore local SHA256 cache and force re-hashing of all files",
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
    sha256_cache = load_sha256_cache(resolved_output_dir)
    cache_lock = threading.Lock()
    results: list[DownloadResult] = []

    use_cache = not no_cache
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map: dict[Future[DownloadResult], ModelEntry] = {
            executor.submit(
                process_model,
                entry,
                resolved_output_dir,
                file_info_cache[entry.repo_id][entry.hf_filename],
                sha256_cache,
                cache_lock,
                use_cache=use_cache,
            ): entry
            for entry in MODELS
        }
        for future in as_completed(future_map):
            results.append(future.result())

    save_sha256_cache(resolved_output_dir, sha256_cache)

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
