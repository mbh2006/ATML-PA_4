"""Publish compact, hash-linked evidence for a complete distillation condition."""

import argparse
import gzip
import json
from pathlib import Path

from scripts.audit_distillation import audit_training, sha256


def archive(root, output, kaggle_version, frozen_pool=None):
    audit = audit_training(root, expected=157, frozen_pool=frozen_pool)
    output.mkdir(parents=True, exist_ok=True)
    log = root / "training.jsonl"
    updates = []
    details = []
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "update":
                updates.append(line)
            else:
                if row["event"] == "rollout":
                    row["prompt_token_count"] = len(row.pop("prompt_ids"))
                    row["response_token_count"] = len(row.pop("response_ids"))
                details.append(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "updates.jsonl").write_text("".join(updates), encoding="utf-8", newline="\n")
    (output / "rollouts-and-segments.jsonl.gz").write_bytes(gzip.compress("".join(details).encode(), mtime=0))
    for name in ("config.json", "status.json"):
        (output / name).write_bytes((root / name).read_bytes())
    assets = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in sorted(root.iterdir()) if path.is_file() and path.name != "audit.json"}
    for name, record in (
        ("audit.json", audit),
        ("artifact-index.json", {"kaggle_version": kaggle_version, "local_directory": root.as_posix(), "artifacts": assets}),
    ):
        (output / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-version", required=True)
    parser.add_argument("--frozen-pool", type=Path)
    args = parser.parse_args()
    archive(args.training, args.output, args.kaggle_version, args.frozen_pool)


if __name__ == "__main__":
    main()
