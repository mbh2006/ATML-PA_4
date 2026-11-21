# ATML PA4

Experiment code and reproducibility records for EE-5102 / CS-6304, Fall 2026.

| Document | Purpose |
| --- | --- |
| [Task 1 plan](plans/task1-compression.md) | Pruning, weight quantization, and matched-budget evaluation |
| [Task 2 plan](plans/task2-attention-and-cache.md) | Exact attention, MQAR training, and packed KV experiments |
| [Task 3 plan](plans/task3-distillation-and-speculation.md) | Speculative sampling, LoRA distillation, and timed evaluation |
| [Evidence and recovery](plans/evidence-and-recovery.md) | Artifact contracts, checkpointing, validation, and recovery |
| [Evaluation and resource review](plans/protocol-review.md) | Manual cross-check, run counts, and GPU capacity gates |
| [Run ledger](plans/run-ledger.md) | Verified progress, inputs, and unresolved items |

The [course dataset repository](https://github.com/AbDu11aHHH/atml-pa4-datasets) supplies the fixed selections and preprocessing tools. The task plans identify outputs by experiment stage so each value used in the student-authored report can be traced to a saved result.

Setup evidence includes the [CUDA smoke result](results/setup/gpu-preflight.json), [validated data manifest](results/setup/public-data-manifest.json), and [locked source revisions](results/setup/local_source_revisions.json).

## Verified experiments

- Task 1 FP16 reference completed both perplexity suites and 2,000 HellaSwag examples. [Baseline summary](results/task1/fp16/summary.json)
- Task 2A ran all 36 exact attention conditions on CUDA. [Raw grid](results/task2a/operator-grid.json), [summary table](results/task2a/operator-summary.csv), [latency and memory](results/task2a/attention-latency-memory.png), [output error](results/task2a/attention-output-error.png)
- Task 3 passed the local Markov sampler gate. Reproduce with `python -m pa4.speculative_validation --output runs/task3/markov-validation.json`. [Joint sequence counts and test statistics](results/task3/markov-validation.json)
- Task 3 real-Qwen validation passed split prefill, replay after speculative rounds, and targeted rollback. [Cache validation evidence](results/task3/qwen-cache-validation.json)
