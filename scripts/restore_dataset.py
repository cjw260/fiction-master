#!/usr/bin/env python3
"""Verify and restore the complete dataset using only Python 3.12+ stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in {"data", "fiction"}
        or "\\" in name
    ):
        raise ValueError(f"Unsafe dataset path: {name}")
    return path


def restore(backup: Path, destination: Path, verify_only: bool) -> None:
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported dataset format")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate paths in manifest")
    for name in expected:
        safe_path(name)

    with tempfile.TemporaryDirectory(prefix="fiction-master-restore-") as temporary:
        staging = Path(temporary)
        archive = staging / "dataset.tar.gz"
        with archive.open("wb") as output:
            for part in manifest["parts"]:
                name = part["name"]
                if Path(name).name != name or "\\" in name:
                    raise ValueError(f"Unsafe part name: {name}")
                source = backup / name
                if (
                    source.stat().st_size != part["size"]
                    or sha256(source) != part["sha256"]
                ):
                    raise ValueError(f"Archive part failed verification: {name}")
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("Combined archive failed verification")

        extracted = staging / "extracted"
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("Archive contents do not match manifest")
            for member in members:
                safe_path(member.name)
                if not member.isfile() or member.size != expected[member.name]["size"]:
                    raise ValueError(f"Unexpected archive entry: {member.name}")
            bundle.extractall(extracted, filter="data")
        for name, entry in expected.items():
            if sha256(extracted / name) != entry["sha256"]:
                raise ValueError(f"Dataset file failed verification: {name}")
        print(
            f"Verified {len(expected)} dataset files and {len(manifest['parts'])} archive parts."
        )
        if verify_only:
            return

        destination = destination.resolve()
        # Check every target before writing; never overwrite a running/local dataset.
        for name in expected:
            target = destination / name
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing file: {target}")
            for parent in target.parents:
                if parent == destination:
                    break
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise ValueError(f"Unsafe destination directory: {parent}")
        for name in expected:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (extracted / name).open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
        print(f"Restored dataset to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup", type=Path, default=REPO_ROOT / "backups" / "dataset"
    )
    parser.add_argument("--destination", type=Path, default=REPO_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    restore(args.backup, args.destination, args.verify_only)


if __name__ == "__main__":
    main()
