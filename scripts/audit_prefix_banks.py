"""Verify private frozen prefix banks and export a token-free evidence index."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pa4.prefix_banks import SEED_OFFSETS, TRAINED_BANKS, sampled_positions


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


def audit_trained(root, name, reference, training):
    bank = root / "bank"
    stage_status = json.loads((root / "job-status.json").read_bytes())
    config = json.loads((bank / "config.json").read_bytes())
    manifest = json.loads((bank / "manifest-0-120.json").read_bytes())
    merge = json.loads((bank / "merge.json").read_bytes())
    checkpoint = json.loads((training / name / "audit.json").read_bytes())
    if (
        name not in TRAINED_BANKS or stage_status["draft"] != name or stage_status["stages"]["bank"]["returncode"]
        or stage_status["source_revision"] != config["source_revision"] or config["bank"] != name
        or (manifest["bank"], manifest["start"], manifest["stop"]) != (name, 0, 120)
        or config["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or f"{merge['policy']}_{merge['seed']}" != name or len(merge["checks"]) != 3
    ):
        raise ValueError(f"{name} disagrees with the frozen training checkpoint or the bank manifest")
    if (config["temperature"], config["top_p"], config["max_new_tokens"]) != (1, 1, 192):
        raise ValueError("Frozen trained bank must use the common sampling settings")
    prompts = []
    for index in range(120):
        filename = f"prompt-{index:03d}.json"
        raw = (bank / filename).read_bytes()
        reference_raw = (reference / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["prompt_sha256"][filename]:
            raise ValueError(f"The frozen prompt file changed: {name} {filename}")
        item, expected = json.loads(raw), json.loads(reference_raw)
        if any(item[field] != expected[field] for field in ("prompt_id", "group", "prompt_token_sha256")):
            raise ValueError(f"Evaluation prompts differ: {name} {filename}")
        response = item["response"]
        if item["seed"] != 6304 + 10007 * index + SEED_OFFSETS[name] or not 1 <= len(response) <= 192:
            raise ValueError(f"Invalid generation seed or response length: {name} {filename}")
        if any(token in config["eos_ids"] for token in response[:-1]) or (
            item["eos"] != (response[-1] in config["eos_ids"])
        ) or (not item["eos"] and len(response) != 192):
            raise ValueError(f"Invalid saved EOS or cap boundary: {name} {filename}")
        positions = sampled_positions(item["prompt_id"], name, len(response))
        if item["prefixes"] != [{"position": position, "tokens": response[:position - 1]} for position in positions]:
            raise ValueError(f"Saved prefixes do not match the sampled positions: {name} {filename}")
        prompts.append({"id": item["prompt_id"], "group": item["group"], "seed": item["seed"],
                        "response_tokens": len(response), "eos": item["eos"], "positions": positions,
                        "record_sha256": manifest["prompt_sha256"][filename]})
    return {"status": "verified", "draft": name, "config": config, "merge": merge, "prompts": prompts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trained", choices=TRAINED_BANKS)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--training", type=Path, default=Path("results/task3/training"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trained and not args.reference:
        parser.error("A trained bank requires the directory of the original teacher bank")
    result = audit_trained(args.root, args.trained, args.reference, args.training) if args.trained else audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
