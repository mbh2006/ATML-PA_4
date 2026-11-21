# Evaluation and resource review

Reviewed against the released manual and the [staff data schemas](https://github.com/AbDu11aHHH/atml-pa4-datasets/blob/main/docs/DATA_FORMATS.md). The [run ledger](run-ledger.md) records which preflights have passed.

| Manual pages | Evaluation gate | Resource constraint and audit |
| --- | --- | --- |
| 2–5, Task 1 | Fixed C4 calibration/development, two independent 50,000-token PPL denominators, and a full or 512-ID HellaSwag evaluation according to condition. Freeze P3/Q3 before final tests. | Compare P3 with exact eligible-weight zero counts. Compare Q3 against measured Q2 packed bytes. Record metadata, padding, tied weights and compensating scales. Time 512/2048 prefill and 64-token decode with five or more warmed repeats. |
| 6–9, Task 2A | Identical Q/K/V and upstream gradients, causal and numerical edge cases, feasible dense FP32 as numerical reference. | Time forward and full forward/backward separately at ten warmed repeats. Capture actual SDPA backend, allocated/reserved memory and unsupported cases. |
| 7–10, Task 2B | Frozen K×regime test grid of 9,216 sequences. Score queries at their labeled positions. Include trained and test-only mask controls, three seeds, and prespecified checkpoint evaluations. | Train 14 conditions × 3 seeds × 4,500 updates. Fix model and optimization budgets. Report state and pair counts at T=256 plus a separate forward-only cost sweep at 256/1024/4096/8192. |
| 8–11, Task 2C | 180 retrieval cases matched across all six variants. Score unconditional and FP16-correct-only accuracy, gold-answer NLL, and logit KL. Treat ten cases per grid cell as descriptive. | Fix the measured pre-question INT4 byte cap per case. Verify residual, index, and temporary dequantization costs. Measure packed encoding, decoding, and decode steps at B=1/8 and both context lengths. |
| 11–15, Task 3 | Pass the joint-sequence sampler test and real cache equivalence. Match training quotas, prefix IDs, 120-prompt performance runs, and target-only references. Measure standalone quality at 512 and 32 tokens. | Four × 40,000 teacher-scored tokens and 2,400 primary speculation trials plus 2,400 adjacent target-only trials. Keep workload caps, target passes, draft passes, per-round stages, seed effects, and amortized cost traceable. |

## Capacity and scheduling

- The first Kaggle smoke job verified **two Tesla T4 devices with about 14.6 GiB each** and about 19.5 GiB free output space. Treat device memory per GPU as the limit for unsharded runs. Archive verified results locally between job versions. Start with one GPU per matched benchmark and use the second for independent training runs only after verifying quota and run isolation.
- A single explicit FP32 score tensor for Task 2A at B=2, H=4, L=8192 is 2 GiB before probabilities, gradients or allocator overhead. Record recognized OOM and unsupported backends with exact shapes and keep those rows in the grid. Preflight the longest quantized-cache batch and full-vocabulary distillation microbatch separately.
- Task 2B processes 6,048,000 training sequences across its 189,000 updates. Pilot each operator before reserving long sessions. Multiply each measured median update time by its remaining updates and add measured evaluation and checkpoint overhead. Capacity examples for the full matrix are 13.1 GPU-hours at 0.25 s per update and 26.3 GPU-hours at 0.50 s per update, before evaluation and checkpoint overhead.
- Use job shards that fit a verified session window. Save boundary checkpoints durably, download and hash them, and confirm restart from one recovered checkpoint. Keep latency benchmarking on the same hardware and runtime for paired conditions. When changing hardware becomes necessary, repeat the paired reference in that hardware group.

## Hold points before expensive runs

1. Apply the staff data requirements and record resolved dependencies. The supplied staff repository contains the generation and cache pins, and its data materialization checks against them passed. Record the exact PyTorch/CUDA environment for synthetic operator experiments. Apply any additional course lock if released before the affected runs. Record source-script compatibility fixes and their impact on the experiment protocol.
2. Validate the Task 1 data denominators and four-option HellaSwag scorer, operator and generator edge cases in Task 2, and exact-distribution and real-Qwen cache tests in Task 3. Store the underlying diagnostic rows and assertions before collecting large-run evidence.
3. Verify a dry-run evidence round trip, including a committed job output, local download, manifest hash and resumed run. Record experiment throughput and memory on the intended runtime so the full matrix can be scheduled against the available quota.
