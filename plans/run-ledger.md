# Run ledger

Update after each verified stage. A planned run becomes complete after its artifacts are retrieved and checked.

| Date (PKT) | Stage | Status | Evidence |
| --- | --- | --- | --- |
| 2026-11-22 | Local manual | Verified | 16 pages. SHA-256 recorded in the evidence guide. |
| 2026-11-22 | GitHub destination | Verified | Public `mbh2006/ATML-PA_4`, default branch `main`, size 0 at initial check. |
| 2026-11-22 | Staff data | Source verified | Staff README and scripts specify locked public selections, bundled MQAR/retrieval data, and teacher/original-draft banks. Materialized assets are pending. |
| 2026-11-22 | Local runtime | Inspected | Python 3.11.7 on the default path, no importable PyTorch in that runtime, no NVIDIA command found, and about 49 GiB free on D:. |
| 2026-11-22 | Plans and repository | Verified | Three task plans and evidence guide pushed to `main` in `17dea92`. |
| 2026-11-22 | Kaggle CUDA smoke | Passed | Private `mbh2006/atml-pa4-gpu-preflight/1` exposed two Tesla T4 GPUs with 15,636,037,632 bytes each. A finite FP16 SDPA result and five warmed times were recovered. Both GitHub sources were reachable from Linux. |
| 2026-11-22 | Preflight evidence recovery | Verified | Local `runs/preflight/kaggle-v1/preflight.json` and kernel log. JSON SHA-256 `f844342ace48ad68b041596f23541549099bfad9137daa7db8aa1f42c4676c01`. Runtime stock Transformers 5.20.2 and Hub 1.33.0 require isolation from the course pins. |
| 2026-11-22 | Course dependency lock | Staff pins located | The supplied link is the verified staff data repository. The pinned checkout contains `requirements-public.txt` and no separate environment lock. Use its specified pins for cache experiments and record the resolved runtime for each benchmark. |
| 2026-11-22 | Staff public data preparation v1 | Failed and preserved | Private `mbh2006/atml-pa4-public-data/1`. At staff revision `6a0a9ec9da7ef3d313ee3036cfec493886723644`, the checksum manifest still lists the removed `scripts/check_release_STAFF_LEGACY.py`. Log and manifest recovered to `runs/preflight/public-data-v1/`. |
| 2026-11-22 | Staff checksum recovery | Implemented | Retry keeps the same staff revision and re-runs the staff verifier with a temporary manifest excluding the single confirmed missing legacy file. Any other discrepancy blocks preparation. |
| 2026-11-22 | Staff public data preparation v2 | Passed and recovered | Private `mbh2006/atml-pa4-public-data/2`. Remaining 22 staff checksums passed. All eight recovered artifact sizes and hashes match. Verified 128/32 disjoint C4 windows, both 50,000-token quotas, 2,000 HellaSwag and 500/60 GSM8K IDs. Raw assets and log are in `runs/preflight/public-data-v2/`. |
| 2026-11-22 | Local staff checkout | Verified | `vendor/atml-pa4-datasets/` is at the same staff commit. Local staff validation passed for 9,216 MQAR test sequences, 180 retrieval cases, 300/60 short-form prompts, and the fixed selection manifests. |
| 2026-11-22 | Task 2A exact attention | Complete | Private `mbh2006/atml-pa4-task2a-attention/1`, source `0434cd1`. Recovered all 36 conditions and 720 timings. Causality, gradients, alternative tiles, and fully-masked and extreme scores passed. SDPA dispatched efficient attention. Local JSON and job log in `runs/task2a/kaggle-v1/`. |
| 2026-11-22 | Task 3 Markov gate | Passed | 100,000 trials per method. Exact p=0.321572, incorrect residual p underflows to 0 in float64. All 27 expected counts >=10. Local evidence under `runs/task3/markov-6304-v1/`. The real-Qwen cache gate is a distinct check. |
| 2026-11-22 | Task 1 unchanged FP16 | Passed | Private `mbh2006/atml-pa4-qwen-validation/1`, source `14d780f`. Audited 196 eligible weights, 50,000 target tokens per suite, and all four candidate scores for all 2,000 HellaSwag IDs. [Baseline evidence](../results/task1/fp16/summary.json) |
| 2026-11-22 | Task 3 real-Qwen v1 | Partial | Both checkpoints passed split-prefill and natural cache replay on the fixed smoke prompts. The run failed its rollback coverage check because none of the proposals were rejected. Preserved the original log and partial diagnostics in `runs/preflight/qwen-v1/`. |

The [public data manifest](../results/setup/public-data-manifest.json) records the exact bytes and hashes of the locally recovered assets. The [source revision lock](../results/setup/local_source_revisions.json) records the upstream Hugging Face revisions. Keep the materialized files paired with this lock throughout the assignment.
