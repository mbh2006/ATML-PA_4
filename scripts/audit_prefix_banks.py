"""Verify private frozen prefix banks and export a token-free evidence index."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(root):
    status = json.loads((root / "job-status.json").read_bytes())
    expected = {"teacher", "original_draft", "exact_spec_original_gamma4"}
    assert set(status["banks"]) == expected
    assert all(row["exit_code"] == 0 for row in status["banks"].values())
    assert (status["start"], status["stop"]) == (0, 120)
    common_prompts = None
    results = {}
    for bank in sorted(expected):
        directory = root / bank
        config = json.loads((directory / "config.json").read_bytes())
        manifest = json.loads((directory / "manifest-0-120.json").read_bytes())
        assert config["bank"] == manifest["bank"] == bank
        assert config["source_revision"] == status["source_revision"]
        assert (manifest["start"], manifest["stop"]) == (0, 120)
        assert (config["temperature"], config["top_p"], config["max_new_tokens"]) == (1, 1, 192)
        fingerprints = manifest["prompt_sha256"]
        assert set(fingerprints) == {f"prompt-{index:03d}.json" for index in range(120)}
        prompts = {}
        groups, positions = Counter(), Counter()
        for name, sha in sorted(fingerprints.items()):
            raw = (directory / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == sha
            row = json.loads(raw)
            key = row["prompt_id"]
            assert key not in prompts
            prompts[key] = row["prompt_token_sha256"]
            groups[row["group"]] += 1
            response = row["response"]
            assert 1 <= len(response) <= 192
            assert row["eos"] == (response[-1] in config["eos_ids"])
            if not row["eos"]:
                assert len(response) == config["max_new_tokens"]
            seen = set()
            for prefix in row["prefixes"]:
                index = prefix["position"]
                assert 1 <= index <= len(response) and index not in seen
                assert prefix["tokens"] == response[:index - 1]
                seen.add(index)
                positions["short" if index <= 64 else "long"] += 1
        assert groups == {"math": 60, "familiar": 30, "shifted": 30}
        if common_prompts is None:
            common_prompts = prompts
        else:
            assert prompts == common_prompts
        results[bank] = {
            "config": config, "prompt_groups": dict(groups), "prefix_counts": dict(positions),
            "prompt_records_sha256": fingerprints,
        }
    return {"status": "verified", "source_revision": status["source_revision"], "banks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
