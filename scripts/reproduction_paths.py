#!/usr/bin/env python3
"""Resolve generated-output paths without allowing writes into frozen inputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

def output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("Output must be a dedicated generated-results directory")
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "results"):
        raise ValueError("Output inside the repository must be under results/; frozen inputs are protected")
    if path.exists() and not path.is_dir():
        raise ValueError("Output path is not a directory")
    return path

if __name__ == "__main__":
    try:
        print(output_path(sys.argv[1]))
    except (ValueError, IndexError) as exc:
        raise SystemExit(str(exc))
