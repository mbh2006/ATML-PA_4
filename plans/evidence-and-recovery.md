# Evidence and recovery

## Sources and preflight

1. Use the supplied ATML PA4 manual, released 20 November 2026. Its local PDF SHA-256 is `c51a2deb4fb0bd1df2a168983b780f3b527475c455be5541167a38ae42e3ed2b`. Record staff clarifications with date and source.
2. Pin the [staff data repository](https://github.com/AbDu11aHHH/atml-pa4-datasets) to a commit. Run its asset, checksum, and setup checks. Preserve `data/local_source_revisions.json` and hashes of materialized datasets and banks. Keep the generated selections fixed for the whole assignment. The public requirements file pins Transformers 4.57.1 and huggingface_hub 0.36.2. The manual also references a course environment lock for other dependencies. Record its origin and hash when available.
3. Windows hosts source, documentation, evidence downloads, and CPU checks. Use the existing Kaggle account for private CUDA runs, subject to a GPU and quota preflight. Run a short pilot on the chosen hardware before allocating the full budget. Record the actual GPU, available memory, CUDA/PyTorch versions, throughput, session limits, disk space, and input download time. Keep each matched timing comparison on the same device and backend.
4. Materialize public data before expensive runs. Preserve the original IDs and token windows. For Task 3, smoke-test the staff bank script against the pinned generation API. If its call fails, record the exact exception and use an auditable per-prompt seeding adapter with the staff seed, prompt format, position sampling, and decoding settings. Generate the speculative bank with the validated assignment sampler.

## Run directory contract

Store local evidence at `runs/<task>/<run-id>/` and use the same relative structure on CUDA. Use an ID containing task, condition, seed, and attempt. A retry points to its predecessor so both attempts remain discoverable. Read the run through `status.json` and `artifacts.json`.

| Path in each run | Required content |
| --- | --- |
| `config.json` | Exact flags, seeds, condition grid, split hashes, model/tokenizer revisions, thresholds, and declared unit conventions |
| `provenance.json` | Git revision or source hash, dependency versions, hardware, backend, wall clock, job ID, and source asset hashes |
| `status.json` | Start and end timestamps, stage, progress, last complete shard/checkpoint, outcome, and exception or measured resource limit |
| `logs/` | Stdout/stderr, warnings, unsupported conditions, smoke-test output, profiler traces where required, and recovery attempts |
| `raw/` | Per-example/window/prefix/round measurements, source IDs, counters, seeds, paired group IDs, and measurement units |
| `checkpoints/` | Resume state and exported final model state where applicable, with step/seed and producing config hash |
| `derived/` | Reproducible tables, plots, bootstrap draws or their reproducible seed and inputs, and aligned qualitative examples |
| `artifacts.json` | Path, size, SHA-256, local and remote location, schema version, and completion/checkpoint marker |

Use append-only JSONL shards or atomic file replacement for machine-readable state. Commit a shard marker after validating its unique example keys and row count. When publishing compact results, preserve the ID mapping and hash of the underlying evidence. Keep raw inputs, full checkpoints, and detailed local evidence in workspace or private CUDA storage. Public GitHub carries code, configs, revision locks, compact results, reproduction commands, and attribution.

## Checkpoints alongside the work

- **Training.** Save at the manual’s evaluation updates, at regular resume boundaries between them, and at the end of each run. Include model or LoRA state, optimizer, scheduler and scaler where used, RNG states, data cursor, and counters. For distillation also save the token quota per category, rollout-pool cursor, producing adapter update, and any unconsumed rollout tokens. Resume at a completed optimizer boundary. Keep final checkpoints and the two latest verified resume checkpoints per run.
- **Calibration and allocation.** Save each completed block’s activation statistics, probe errors, scale search grid and winner, row mask counts, and packed-cost ledger. Flush allocation maps and the development selection record before opening final test evaluation.
- **Evaluations and timing.** Save each completed shard and per-repetition timings before aggregation. Record failed rows with their condition and inputs. Persist partial failure logs immediately. When retrying OOM with a different microbatch, retain the original result and identify comparisons whose timing shape changed.
- **Job boundaries.** Write to a private durable destination or finish bounded Kaggle job versions with committed outputs. Download evidence after each completed shard or job version. Recompute artifact hashes locally, record the private output version, and validate a resume from a copied checkpoint before relying on it for long training. Interactive scratch files alone cannot support recovery after session termination.
- **Git milestones.** Commit the plans, then tested code and config for each task milestone, then compact verified results and reproduction updates. Each run records its producing commit. Check file contents and staging before pushing. When copying externally sourced code, record the project, link, license, and the reused portions.

## Common checks before final handoff

- Compare required-condition counts with completed-condition counts. Include failed, incomplete, unsupported, and OOM statuses. Tie each summary cell to raw rows, a scoring command, and the input manifest.
- Check split isolation, paired IDs, seeds, method settings, actual sparsity and byte budgets, and units. Store both unrounded values and display conventions.
- Preserve Stage C packed bytes separately from transient GPU memory. Preserve warmed repetitions and measured elapsed times separately from analytical estimates.
- Provide raw measurements, attribution, plot source data, and reproduction commands. The student writes the report language, interpretation, and analysis and handles LMS submission.
