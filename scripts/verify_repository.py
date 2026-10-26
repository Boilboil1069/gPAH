#!/usr/bin/env python3
"""Verify the downloaded supplement without Git, scientific packages or reruns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = [f"fig{i}.pdf" for i in range(1, 18) if i != 13] + ["fig13a.pdf", "fig13b.pdf"]
REVISION_TABLES = {
    "phase_observation": "paper1_phase_observation_replications.csv",
    "phase_filter": "paper1_phase_filter_replications.csv",
    "infinite_buffer": "paper1_infinite_buffer_replications.csv",
    "runtime": "paper1_runtime_replications.csv",
    "rl_hyperparameters": "paper1_rl_hyperparameter_test_replications.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true", help="Check required results only; does not verify file integrity")
    args = parser.parse_args()
    errors = []
    required = [ROOT / "paper_artifacts/figs" / name for name in FIGURES]
    tables = [
        ROOT / "data/raw/final_run/policy/paper1_policy_replications.csv",
        ROOT / "data/raw/final_run/rl/paper1_rl_policy_replications.csv",
        ROOT / "data/raw/final_run/sensitivity/paper1_sensitivity_replications.csv",
        ROOT / "data/raw/final_run/buffer_robustness/paper1_buffer_robustness_replications.csv",
        ROOT / "data/raw/ablation_r50_conditional/paper1_ablation_replications.csv",
        ROOT / "data/raw/final_run/rl_topology_generalization/paper1_rl_unseen_topology_replications.csv",
        ROOT / "data/derived/family_blocked_reanalysis/paper1_family_blocked_overall.csv",
    ]
    for stage, filename in REVISION_TABLES.items():
        folder = ROOT / "data/raw/revision_experiments" / stage
        required.append(folder / "run_manifest.json")
        tables.append(folder / filename)
    required += tables
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty required file: {path.relative_to(ROOT)}")
    for path in tables:
        if path.is_file():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or next(reader, None) is None:
                        errors.append(f"Result table has no data rows: {path.relative_to(ROOT)}")
            except (OSError, UnicodeError, csv.Error) as exc:
                errors.append(f"Unreadable result table {path.relative_to(ROOT)}: {exc}")
    checkpoint_dir = ROOT / "data/raw/final_run/rl_topology_generalization/checkpoints"
    for algorithm in ("dqn", "ppo", "sac"):
        found = [p for p in checkpoint_dir.glob(f"{algorithm}_seed_*.pt") if p.is_file()]
        if len(found) != 10:
            errors.append(f"Expected 10 {algorithm} topology checkpoints, found {len(found)}")

    checked = 0
    if not args.inventory_only:
        manifest = ROOT / "CHECKSUMS.sha256"
        indexed = set()
        if not manifest.is_file():
            errors.append("Missing CHECKSUMS.sha256")
        else:
            for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
                if not match:
                    errors.append(f"Malformed checksum entry on line {line_no}")
                    continue
                expected, relative = match.groups()
                path = (ROOT / relative).resolve()
                if not path.is_relative_to(ROOT) or path == ROOT:
                    errors.append(f"Unsafe checksum path on line {line_no}")
                    continue
                if relative in indexed:
                    errors.append(f"Duplicate checksum entry: {relative}")
                indexed.add(relative)
                if not path.is_file():
                    errors.append(f"Missing indexed file: {relative}")
                elif sha256(path).lower() != expected.lower():
                    errors.append(f"SHA-256 mismatch: {relative}")
                checked += 1
            # Ensure the manifest covers every distributed code, configuration and data file.
            for folder in ("configs", "data", "docs", "experiment", "paper_artifacts", "scripts"):
                for path in (ROOT / folder).rglob("*"):
                    if not path.is_file() or any(p.startswith("._") or p in ("__pycache__", ".DS_Store") for p in path.parts):
                        continue
                    relative = path.relative_to(ROOT).as_posix()
                    if relative not in indexed:
                        errors.append(f"File missing from checksum manifest: {relative}")
    if errors:
        for error in errors[:40]:
            print("FAIL:", error)
        if len(errors) > 40:
            print(f"... and {len(errors) - 40} additional failures")
        raise SystemExit(f"Verification failed with {len(errors)} issue(s).")
    print("Required results present: main suite, conditional ablation, five revision studies, family-blocked tables,")
    print("30 held-out-topology checkpoints, and 18 manuscript figure PDFs.")
    if args.inventory_only:
        print("Inventory-only check passed; SHA-256 integrity has not been checked.")
    else:
        print(f"SHA-256 verification passed for {checked} files. No experiments were rerun.")


if __name__ == "__main__":
    main()
