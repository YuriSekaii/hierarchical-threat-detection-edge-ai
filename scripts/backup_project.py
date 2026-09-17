"""
scripts/backup_project.py

Automated Snapshot & Backup Utility for Hierarchical Threat Detection Project.
Creates a timestamped ZIP archive containing all source code, models, training scripts,
benchmarks, results, documentation, and weights (excluding large .venv and video buffers).

Usage:
    python scripts/backup_project.py [--output-dir D:\Intern AI Project\PROJECT_BACKUPS]
"""

import os
import sys
import time
import zipfile
import argparse
from pathlib import Path

DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP_ROOT = DEFAULT_WORKSPACE.parent / "PROJECT_BACKUPS"

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "temp",
    "tmp",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
}

INCLUDE_PATHS = [
    "src",
    "models",
    "training",
    "scripts",
    "weights",
    "results",
]

INCLUDE_ROOT_FILES = [
    "VERSIONS.md",
    "VERSIONS_PHASE2.md",
    "ROADMAP_AND_HANDOFF.md",
    "ROADMAP_AND_HANDOFF_PHASE2.md",
    "Stage1_SOTA_Architectures_2026.md",
    "Modern OOD Action Recognition Models.md",
    "README.md",
    "requirements.txt",
    ".gitignore",
]


def create_backup(workspace: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_name = f"backup_threat_detection_{timestamp}.zip"
    zip_path = output_dir / zip_name

    print(f"[Backup] Initiating project snapshot from: {workspace}")
    print(f"[Backup] Destination archive: {zip_path}")

    files_to_zip = []

    # 1. Collect specific root files
    for fname in INCLUDE_ROOT_FILES:
        fpath = workspace / fname
        if fpath.is_file():
            files_to_zip.append((fpath, fpath.relative_to(workspace)))

    # 2. Collect files from target directories
    for dir_name in INCLUDE_PATHS:
        dir_path = workspace / dir_name
        if not dir_path.is_dir():
            continue

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                fpath = Path(root) / file
                if fpath.suffix.lower() in EXCLUDE_EXTENSIONS:
                    continue
                files_to_zip.append((fpath, fpath.relative_to(workspace)))

    # 3. Write ZIP
    print(f"[Backup] Archiving {len(files_to_zip)} files (Code, Weights, Docs, Benchmarks)...")
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fpath, rel_path in files_to_zip:
            zipf.write(fpath, arcname=str(rel_path))

    duration = time.perf_counter() - t0
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Backup] Successfully created snapshot: {zip_name}")
    print(f"[Backup] Archive Size: {size_mb:.2f} MB | Duration: {duration:.2f}s")
    print(f"[Backup] Location: {zip_path.resolve()}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Create a project backup snapshot.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_BACKUP_ROOT),
        help="Target folder for ZIP archives",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    create_backup(DEFAULT_WORKSPACE, out_path)


if __name__ == "__main__":
    main()
