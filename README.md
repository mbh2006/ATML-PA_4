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

The measured CUDA runs use Linux, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, and Tesla T4. The Qwen runs use Transformers 4.57.1, huggingface_hub 0.36.2, and tokenizers 0.22.2. Staff data preparation used Datasets 4.8.5. The attention plotting script uses Matplotlib 3.11.2. Check each result's recorded environment when comparing measurements.

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

# CPU checks and plots
python -m pa4.speculative_validation --output runs/task3/markov-reproduction.json
python -m pa4.kv_codec_validation --output runs/task2c/codec-reproduction.json
python scripts/plot_attention.py --input results/task2a/operator-grid.json --output runs/task2a/reproduction-plots
```

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
- Task 1 compression primitives passed exact mask, packed-byte, and mixed-allocation checks. [Compression checks](results/pilot/weight-validation.json)
- Task 2B pilots covered each operator on CUDA. The CPU recovery check reproduced model and optimizer states bit for bit. [Operator and resume checks](results/pilot/mqar-validation.json), [pilot timing](results/pilot/mqar-timing.json)
- All three dense mixed K=16 controls reached 4,500 updates. The development rule selected top-64 for the trained intervention. [Frozen selection and source scores](results/task2b/intervention-selection.json)
- Task 2C packing and byte-budget checks passed on synthetic cache tensors. [KV codec validation](results/pilot/kv-codec-validation.json)
