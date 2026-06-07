#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

from map_gpah_paper1_common import write_csv
from map_gpah_paper1_rl_baselines import plot_rl_outputs, read_csv_rows, summarize


def read_many(shard_dirs: List[Path], filename: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for shard_dir in shard_dirs:
        path = shard_dir / filename
        if path.exists():
            rows.extend(read_csv_rows(os.fspath(path)))
    return rows


def dedupe(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Paper 1 RL shard output directories.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--shard-glob", "--shard_glob", dest="shard_glob", required=True)
    args = parser.parse_args()

    shard_dirs = sorted(Path(p) for p in glob.glob(args.shard_glob) if Path(p).is_dir())
    if not shard_dirs:
        raise SystemExit(f"no shard directories matched: {args.shard_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_rows = dedupe(read_many(shard_dirs, "paper1_rl_policy_replications.csv"))
    hist_rows = dedupe(read_many(shard_dirs, "paper1_rl_training_history.csv"))
    if not rep_rows:
        raise SystemExit("no paper1_rl_policy_replications.csv rows found in shards")

    case_summary = summarize(rep_rows, ["family", "case_id", "policy", "policy_label"])
    family_summary = summarize(rep_rows, ["family", "policy", "policy_label"])

    write_csv(os.fspath(output_dir / "paper1_rl_policy_replications.csv"), rep_rows)
    write_csv(os.fspath(output_dir / "paper1_rl_policy_case_summary.csv"), case_summary)
    write_csv(os.fspath(output_dir / "paper1_rl_policy_family_summary.csv"), family_summary)
    write_csv(os.fspath(output_dir / "paper1_rl_training_history.csv"), hist_rows)
    plot_rl_outputs(os.fspath(output_dir))

    print(f"merged {len(shard_dirs)} shard dirs")
    print(f"replications: {len(rep_rows)}")
    print(f"history: {len(hist_rows)}")
    print(output_dir)


if __name__ == "__main__":
    main()
