"""Archive full per-prompt draft results and link the raw Kaggle evidence."""

import argparse
import gzip
import json
from pathlib import Path

from scripts.audit_distillation import sha256
from scripts.audit_prefix_banks import audit_trained
from scripts.summarize_speculation import load_experiment


COUNTS = {"primary": 480, "quality": 120, "fixed": 48, "unmerged": 48}


def archive(root, output, remote, training, reference):
    job = json.loads((root / "job-status.json").read_bytes())
    draft = job["draft"]
    output.mkdir(parents=True, exist_ok=True)
    status = {"draft": draft, "source_revision": job["source_revision"], "kaggle_version": remote, "modes": {}}
    required = {"primary", "quality", "fixed"}
    if not required <= job["stages"].keys() or any(stage["returncode"] for stage in job["stages"].values()):
        raise ValueError(f"Required evaluation stages failed or are incomplete for {draft}")
    if draft != "original_draft":
        bank = audit_trained(root, draft, reference, training)
        (output / "bank-audit.json").write_text(json.dumps(bank, allow_nan=False, separators=(",", ":")) + "\n",
                                                encoding="utf-8")
        status["trained_bank_prompts"] = len(bank["prompts"])
    for stage in job["stages"]:
        if stage == "bank":
            continue
        original = root / stage
        config, rows = load_experiment(original)
        if len(rows) != COUNTS[stage] or (config["draft"], config["mode"], config["source_revision"]) != (
            draft, stage, job["source_revision"],
        ):
            raise ValueError(f"Unexpected evaluation coverage: {draft} {stage}")
        if draft != "original_draft":
            checkpoint_sha = config["audit"].get("checkpoint_sha256")
            if checkpoint_sha is not None and checkpoint_sha != bank["config"]["checkpoint_sha256"]:
                raise ValueError(f"Timing checkpoint differs from the bank producer: {draft} {stage}")
        target = output / stage
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "status.json"):
            (target / filename).write_bytes((original / filename).read_bytes())
        raw = original / "benchmark.jsonl"
        packed = target / "benchmark.jsonl.gz"
        packed.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
        status["modes"][stage] = {"rows": len(rows), "raw_sha256": sha256(raw), "compressed_sha256": sha256(packed)}
    asset_index = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(root.rglob("*")) if path.is_file()}
    status.update(status="passed", local_directory=root.as_posix(), files=asset_index)
    (output / "artifact-index.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-version", required=True)
    parser.add_argument("--training", type=Path, default=Path("results/task3/training"))
    parser.add_argument("--reference", type=Path, default=Path("runs/task3/prefix-banks-v1/prefix-banks/teacher"))
    args = parser.parse_args()
    archive(args.root, args.output, args.kaggle_version, args.training, args.reference)


if __name__ == "__main__":
    main()
