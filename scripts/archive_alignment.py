"""Archive verified common-prefix scores for each completed bank."""

import argparse
import gzip
import json
from pathlib import Path

from scripts.audit_distillation import sha256
from scripts.summarize_alignment import DRAFTS, read


def archive(root, output, remote):
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        bank, rows = read(directory)
        if bank != directory.name or len(rows) != 600:
            raise ValueError("An alignment archive must cover all five drafts on 120 prompts")
        prompts = {row["id"] for row in rows}
        keys = {(row["id"], row["draft"]) for row in rows}
        if len(prompts) != 120 or keys != {(prompt, model) for prompt in prompts for model in DRAFTS}:
            raise ValueError(f"Incomplete or duplicated draft grid for {bank}")
        target = output / bank
        target.mkdir(parents=True, exist_ok=True)
        raw = directory / "alignment.jsonl"
        packed = target / "alignment.jsonl.gz"
        packed.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
        for name in ("status.json", "config.json"):
            (target / name).write_bytes((directory / name).read_bytes())
        evidence = {"bank": bank, "rows": len(rows), "kaggle_version": remote, "raw_sha256": sha256(raw),
                    "compressed_sha256": sha256(packed), "local_directory": directory.as_posix()}
        (target / "artifact-index.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-version", required=True)
    args = parser.parse_args()
    archive(args.root, args.output, args.kaggle_version)


if __name__ == "__main__":
    main()
