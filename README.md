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

## Reproduce completed stages

The measured CUDA runs use Linux, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, and Tesla T4. Start with that CUDA build of PyTorch and install the direct dependencies in [requirements.txt](requirements.txt). The Qwen runs use Transformers 4.57.1, huggingface_hub 0.36.2, and tokenizers 0.22.2. Staff data preparation used Datasets 4.8.5. The local result summaries and figures use NumPy 2.5.5 and Matplotlib 3.11.2. Check each result's recorded environment when comparing measurements.

From the repository root, check out the pinned staff data tools. Keep the recovered private public-data bundle, including its manifest, under `runs/preflight/public-data-v2/public-data` for the exact artifact audit and Qwen commands.

```sh
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git vendor/atml-pa4-datasets
git -C vendor/atml-pa4-datasets checkout --detach 6a0a9ec9da7ef3d313ee3036cfec493886723644
python vendor/atml-pa4-datasets/scripts/validate_assets.py
python -m scripts.audit_submission --data runs/preflight/public-data-v2/public-data --staff vendor/atml-pa4-datasets
```

To regenerate the staff materialization from the same revisions, install the dependencies above, copy `results/setup/local_source_revisions.json` to `vendor/atml-pa4-datasets/data/local_source_revisions.json`, then run the staff `scripts/setup_data.py --public` and `scripts/check_setup.py` from the staff checkout. The audit above uses the original recovered byte hashes. Preserve a separate manifest for regenerated gzip files, whose container bytes can differ.

```sh
# On the pinned CUDA environment
python -m pa4.weight_evaluation --data runs/preflight/public-data-v2/public-data --output runs/task1/fp16-reproduction
python -m pa4.weight_calibration --data runs/preflight/public-data-v2/public-data --output runs/task1/calibration-reproduction
python -m pa4.attention_benchmark --output runs/task2a/operator-grid-reproduction.json
python scripts/run_mqar_dense_controls.py --staff vendor/atml-pa4-datasets --output runs/task2b/dense-reproduction
python scripts/select_mqar_intervention.py --dense-controls results/task2b --output runs/task2b/selection-reproduction.json
python -m pa4.qwen_validation --lock results/setup/local_source_revisions.json --output runs/task3/qwen-reproduction.json
python -m pa4.kv_retrieval --lock results/setup/local_source_revisions.json --cases vendor/atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --output runs/task2c/retrieval-reproduction
python -m pa4.kv_systems --lock results/setup/local_source_revisions.json --cases vendor/atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --output runs/task2c/systems-reproduction
python -m scripts.summarize_kv --retrieval runs/task2c/retrieval-reproduction/cases.jsonl --systems runs/task2c/systems-reproduction/timings.jsonl --locked-cases vendor/atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --output runs/task2c/retrieval-reproduction/summary.json
python -m scripts.plot_kv --summary runs/task2c/retrieval-reproduction/summary.json --output runs/task2c/retrieval-reproduction/figures

# CPU checks and plots
python -m pa4.speculative_validation --output runs/task3/markov-reproduction.json
python -m pa4.kv_codec_validation --output runs/task2c/codec-reproduction.json
python scripts/plot_attention.py --input results/task2a/operator-grid.json --output runs/task2a/reproduction-plots
```

## Reproduce the compression and MQAR matrices

The adaptive compression allocations are fixed with development or calibration evidence before evaluating any held-out condition. The commands below regenerate both allocation files and use them to evaluate the same 17 compressed conditions. The earlier FP16 command supplies the reference.

```sh
DATA=runs/preflight/public-data-v2/public-data
STAFF=vendor/atml-pa4-datasets
for condition in q2-int8 q2-int4 q2-int3; do
  python -m pa4.weight_uniform --data "$DATA" --condition "$condition" --output "runs/task1/q2-reproduction/$condition"
done
python -m pa4.weight_pruning_selection --data "$DATA" --output runs/task1/p3-reproduction
python -m pa4.weight_precision_selection --data "$DATA" --q2-results runs/task1/q2-reproduction --output runs/task1/q3-reproduction
for condition in magnitude-25 magnitude-50 magnitude-75 wanda-25 wanda-50 wanda-75 wanda-2of4 q1-int8 q1-int4 q1-int3 q2-int8 q2-int4 q2-int3 p3-50 p3-75 q3-int4-budget q3-int3-budget; do
  python -m pa4.weight_final_evaluation --data "$DATA" --condition "$condition" --pruning-selection runs/task1/p3-reproduction/selection.json --precision-selection runs/task1/q3-reproduction/selection.json --output "runs/task1/heldout-reproduction/$condition"
done
```

For MQAR, the development-only top-K selection is versioned. The matrix runner uses it to train all 42 conditions and then evaluates each native checkpoint and the dense test-time local control where applicable.

```sh
python -m scripts.run_mqar_matrix --staff "$STAFF" --selection results/task2b/intervention-selection.json --output runs/task2b/matrix-reproduction
python -m pa4.mqar_cost --selection results/task2b/intervention-selection.json --output runs/task2b/cost-reproduction.json
python -m scripts.plot_mqar --raw runs/task2b/matrix-reproduction --cost runs/task2b/cost-reproduction.json --output runs/task2b/figures-reproduction
```

## Task 3 training and evaluation commands

Use the same pinned data and CUDA environment for all four training runs. Generate the teacher pool once. Each run writes its full journal and checkpoints locally.

```sh
DATA=runs/preflight/public-data-v2/public-data
STAFF=vendor/atml-pa4-datasets
python -m pa4.teacher_pool --data "$DATA" --staff "$STAFF" --output runs/task3/teacher-pool
python -m scripts.audit_distillation --pool runs/task3/teacher-pool --data "$DATA" --staff "$STAFF" --output runs/task3/teacher-pool/audit.json
for seed in 6304 6305; do
  python -m pa4.distillation --data "$DATA" --staff "$STAFF" --policy off --seed "$seed" --pool runs/task3/teacher-pool/teacher-pool.jsonl --output "runs/task3/off_${seed}"
  python -m pa4.distillation --data "$DATA" --staff "$STAFF" --policy on --seed "$seed" --output "runs/task3/on_${seed}"
done
```

To resume, use the same configuration and output directory, with `--resume <path-to-update-checkpoint>`. Run `python -m scripts.audit_distillation --training <run-directory> --output <audit-json>` on each finished run. Use only the audited final checkpoints for prefix and runtime evaluation.

```sh
# Frozen banks and a merged checkpoint, shown for one trained draft
python -m pa4.prefix_banks --data "$DATA" --staff "$STAFF" --bank off_6304 --checkpoint runs/task3/off_6304/update-0157.pt --output runs/task3/banks/off_6304 --merged-output runs/task3/merged/off_6304

# Baseline paired performance, fixed-cap diagnostic, and standalone quality
for mode in primary fixed quality; do
  python -m pa4.speculative_benchmark --data "$DATA" --staff "$STAFF" --mode "$mode" --output "runs/task3/original-${mode}"
done
```

Run the same three benchmarks with `--checkpoint` for each of the four trained drafts. Use mode `unmerged` with a completed checkpoint for the adapter overhead diagnostic. Score each of the seven frozen prefix banks with `pa4.prefix_alignment`, supplying one `--checkpoint` argument per trained draft. Use `scripts.summarize_alignment` and `scripts.summarize_speculation` to derive the token-weighted alignment and paired performance tables from those output directories.

## Attribution and submission audit

- The staff data repository supplies the MQAR generator, fixed selection manifests, retrieval files, and preprocessing scripts. The runners use the pinned staff checkout. [Course data repository](https://github.com/AbDu11aHHH/atml-pa4-datasets)
- Task 1 adapts the HellaSwag task formatting and continuation boundary convention from lm-evaluation-harness v0.4.9. [HellaSwag task code](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/tasks/hellaswag/utils.py), [continuation encoding](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/api/model.py)
- The audit records only the checks supported by completed results. The final review requires actual compressed budgets and matched timing, frozen adaptive choices, held-out result pairing, and source artifacts for each reported number. The student writes the report and its interpretation under the assignment's AI usage policy. [Evidence and recovery](plans/evidence-and-recovery.md)

## Verified experiments

- Task 1 FP16 reference completed both perplexity suites and 2,000 HellaSwag examples. [Baseline summary](results/task1/fp16/summary.json)
- Task 1 completed all 18 held-out conditions and archived paired scores, confidence intervals, and figures. Regenerate the figures with `python scripts/plot_compression.py`. [Held-out audit](results/task1/heldout-audit.json), [figures and aligned scores](results/task1/figures/)
- Task 2A ran all 36 exact attention conditions on CUDA. [Raw grid](results/task2a/operator-grid.json), [summary table](results/task2a/operator-summary.csv), [latency and memory](results/task2a/attention-latency-memory.png), [output error](results/task2a/attention-output-error.png)
- Task 3 passed the local Markov sampler gate. Reproduce with `python -m pa4.speculative_validation --output runs/task3/markov-validation.json`. [Joint sequence counts and test statistics](results/task3/markov-validation.json)
- Task 3 real-Qwen validation passed split prefill, replay after speculative rounds, and targeted rollback. [Cache validation evidence](results/task3/qwen-cache-validation.json)
- Task 3 completed the on-policy CUDA smoke and resumed from update four in a fresh process. The journal audit reconciled all six optimizer updates. [Resume audit and artifact hashes](results/task3/distillation-smoke/)
- Task 3 recovered the frozen teacher pool for all 800 locked training prompts. The raw responses are kept with the private Kaggle run and local evidence. [Audit, response-length index, and artifact hashes](results/task3/teacher-pool/)
- Both off-policy and both on-policy seeds completed their 40,000-token budgets. The archives include update losses, rollout IDs, scored-token offsets, and checkpoint hashes. [Training evidence](results/task3/training/), [loss curves](results/task3/figures/training-loss.png)
- The original-draft timing smoke covered all 12 paired trials in its three-prompt slice. [Per-round timing evidence](results/task3/original-timing-smoke/)
- Task 1 compression primitives passed exact mask, packed-byte, and mixed-allocation checks. [Compression checks](results/pilot/weight-validation.json)
- Task 2B pilots covered each operator on CUDA. The CPU recovery check reproduced model and optimizer states bit for bit. [Operator and resume checks](results/pilot/mqar-validation.json), [pilot timing](results/pilot/mqar-timing.json)
- All three dense mixed K=16 controls reached 4,500 updates. The development rule selected top-64 for the trained intervention. [Frozen selection and source scores](results/task2b/intervention-selection.json)
- Task 2B completed all 42 MQAR training runs, 54 held-out evaluations, and the 20-condition cost sweep. [Matrix audit](results/task2b/full-matrix/audit.json), [cost measurements](results/task2b/mqar-cost.json), [figures and tables](results/task2b/figures/)
- Task 2C packing and byte-budget checks passed on synthetic cache tensors. [KV codec validation](results/pilot/kv-codec-validation.json)
- Task 2C ran six cache policies on all 180 locked Qwen retrieval prompts. The archive preserves each generated answer, the per-token scores, selection positions, and measured byte budgets. [Raw paired evidence and summaries](results/task2c/retrieval-full/), [retrieval accuracy figure](results/task2c/figures/retrieval-accuracy.png)
- Task 2C completed all 24 systems conditions at batches 1 and 8 with 10 warmed measurements per condition. The benchmark uses a real prefix cache computed at batch one and replicated before timing. Packing operates on per-request payloads and the Qwen decode forward uses the full declared batch. [Timings](results/task2c/systems/), [cache figure](results/task2c/figures/cache-systems.png), [combined summary](results/task2c/summary.json)
