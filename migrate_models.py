#!/usr/bin/env python3
# /// script
# dependencies = ["typer>=0.24"]
# ///
"""
Migrate existing flat model files into the new creator/repo folder structure.

Maps old flat filenames (from both the unsloth and bartowski download scripts)
to the new hierarchical paths: output_dir/{creator}/{repo}/{hf_filename}.

Usage:
    python migrate_models.py [--output-dir ~/models] [--dry-run]
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class MigrationMapping:
    old_local_filename: str
    repo_id: str
    hf_filename: str


# All old entries from both download_models.py and download_models_bartowski.py,
# mapping their old local_filename to the new repo_id/hf_filename path.
MIGRATION_MAPPINGS: list[MigrationMapping] = [
    # --- unsloth main models ---
    MigrationMapping("GLM-4.7-Flash-UD-Q4_K_XL.gguf", "unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-UD-Q4_K_XL.gguf"),
    MigrationMapping("gpt-oss-20b-UD-Q4_K_XL.gguf", "unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-UD-Q4_K_XL.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf", "unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"),
    MigrationMapping("Qwen3.5-4B-UD-Q8_K_XL.gguf", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q8_K_XL.gguf"),
    MigrationMapping("Qwen3.5-9B-UD-Q4_K_XL.gguf", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q4_K_XL.gguf"),
    MigrationMapping("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-heretic-Q4_K_M.gguf", "mradermacher/Qwen3.5-35B-A3B-heretic-GGUF", "Qwen3.5-35B-A3B-heretic.Q4_K_M.gguf"),
    # unsloth mmproj
    MigrationMapping("GLM-4.7-Flash-mmproj-F16.gguf", "unsloth/GLM-4.7-Flash-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("gpt-oss-20b-mmproj-F16.gguf", "unsloth/gpt-oss-20b-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-mmproj-F16.gguf", "unsloth/Qwen3.5-35B-A3B-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("Qwen3.5-4B-mmproj-F16.gguf", "unsloth/Qwen3.5-4B-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("Qwen3.5-9B-mmproj-F16.gguf", "unsloth/Qwen3.5-9B-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("Qwen3-Coder-30B-A3B-Instruct-mmproj-F16.gguf", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "mmproj-F16.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-heretic-mmproj-F16.gguf", "mradermacher/Qwen3.5-35B-A3B-heretic-GGUF", "mmproj-F16.gguf"),
    # --- bartowski main models ---
    MigrationMapping("GLM-4.7-Flash-Q4_K_M.gguf", "bartowski/zai-org_GLM-4.7-Flash-GGUF", "zai-org_GLM-4.7-Flash-Q4_K_M.gguf"),
    MigrationMapping("gpt-oss-20b-Q4_K_M.gguf", "bartowski/openai_gpt-oss-20b-GGUF", "openai_gpt-oss-20b-Q4_K_M.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-Q4_K_M.gguf", "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf"),
    MigrationMapping("Qwen3.5-4B-Q8_0.gguf", "bartowski/Qwen_Qwen3.5-4B-GGUF", "Qwen_Qwen3.5-4B-Q8_0.gguf"),
    MigrationMapping("Qwen3.5-9B-Q4_K_M.gguf", "bartowski/Qwen_Qwen3.5-9B-GGUF", "Qwen_Qwen3.5-9B-Q4_K_M.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-heretic-v2-Q4_K_M.gguf", "llmfan46/Qwen3.5-35B-A3B-heretic-v2-GGUF", "Qwen3.5-35B-A3B-heretic-v2-Q4_K_M.gguf"),
    # bartowski mmproj
    MigrationMapping("GLM-4.7-Flash-mmproj-f16.gguf", "bartowski/zai-org_GLM-4.7-Flash-GGUF", "mmproj-zai-org_GLM-4.7-Flash-f16.gguf"),
    MigrationMapping("gpt-oss-20b-mmproj-f16.gguf", "bartowski/openai_gpt-oss-20b-GGUF", "mmproj-openai_gpt-oss-20b-f16.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-mmproj-f16.gguf", "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf"),
    MigrationMapping("Qwen3.5-4B-mmproj-f16.gguf", "bartowski/Qwen_Qwen3.5-4B-GGUF", "mmproj-Qwen_Qwen3.5-4B-f16.gguf"),
    MigrationMapping("Qwen3.5-9B-mmproj-f16.gguf", "bartowski/Qwen_Qwen3.5-9B-GGUF", "mmproj-Qwen_Qwen3.5-9B-f16.gguf"),
    MigrationMapping("Qwen3.5-35B-A3B-heretic-v2-mmproj-BF16.gguf", "llmfan46/Qwen3.5-35B-A3B-heretic-v2-GGUF", "Qwen3.5-35B-A3B-mmproj-BF16.gguf"),
]


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path.home() / "models",
        "--output-dir",
        help="Directory containing existing flat model files",
        show_default=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview moves without executing them",
    ),
) -> None:
    """Migrate flat model files into creator/repo folder structure."""
    resolved_output_dir = output_dir.expanduser().resolve()

    if not resolved_output_dir.exists():
        typer.echo(f"Output directory does not exist: {resolved_output_dir}")
        raise typer.Exit(code=1)

    moved = 0
    skipped = 0
    not_found = 0

    for mapping in MIGRATION_MAPPINGS:
        old_path = resolved_output_dir / mapping.old_local_filename
        new_path = resolved_output_dir / mapping.repo_id / mapping.hf_filename

        if not old_path.exists():
            not_found += 1
            continue

        if new_path.exists():
            typer.echo(f"[skip]  {mapping.old_local_filename} -> {mapping.repo_id}/{mapping.hf_filename}  (destination exists)")
            skipped += 1
            continue

        if dry_run:
            typer.echo(f"[move]  {mapping.old_local_filename} -> {mapping.repo_id}/{mapping.hf_filename}")
        else:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            typer.echo(f"[moved] {mapping.old_local_filename} -> {mapping.repo_id}/{mapping.hf_filename}")
        moved += 1

    typer.echo(f"\nSummary: {moved} {'would be ' if dry_run else ''}moved, {skipped} skipped, {not_found} not found")


if __name__ == "__main__":
    app()
