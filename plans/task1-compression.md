# Task 1 · Model compression

Manual pages 2–5. Compare weight ranking and activation-aware scaling first, then compare adaptive allocation against its matched uniform method.

## Freeze inputs and baseline

- Load the revision-locked Qwen2.5-1.5B Base checkpoint in FP16. Enumerate Q/K/V/O and gate/up/down matrices by block and save each name, shape and weight count. Account for tied weights by shared storage. Restore eligible weights from the original checkpoint between conditions and hash excluded weights to verify that embeddings, biases, norms, and head are preserved.
- Use the staff 128×512 calibration tokens and 32×512 C4 development tokens. Save their row IDs, token offsets and hashes. Keep the two locked 50,000-scored-token test suites and HellaSwag out of all selection. Preserve baseline token NLL sums and per-ending scores as join keys for paired evaluation.
- Evaluate windows independently. Read `score_target_offsets`, score every target once from preceding logits, and aggregate summed NLL and scored-token counts before exponentiating. Confirm each suite has exactly 50,000 scored tokens. Implement HellaSwag context and continuation formatting and character normalization against the pinned lm-evaluation-harness convention. Verify all four options on a small hand-checked set.

## Condition matrix

| Group | Conditions | Full HellaSwag 2,000 |
| --- | --- | --- |
| Baseline | Unchanged FP16 | Yes |
| P1 magnitude | 25%, 50%, 75% | 50% |
| P2 Wanda | 25%, 50%, 75% | 50% |
| P3 nonuniform Wanda | 50%, 75% | 50% |
| Structured diagnostic | Wanda 2:4 | First 512 |
| Q1 uniform asymmetric | 8, 4, 3 bits | INT4 and INT3 |
| Q2 uniform AWQ-style | 8, 4, 3 bits | INT4 and INT3 |
| Q3 mixed AWQ-style | Q2 INT4 and INT3 packed-byte budgets | Both |

Run both perplexity suites for all 18 conditions and the first 512 HellaSwag IDs where full evaluation is unnecessary.

## Implementation and correctness

1. **Pruning.** Apply stable per-output-row sorting by score then input-column index. Wanda uses absolute weight times input-channel L2 norm from all calibration tokens. Set integer row quotas by largest remainder with a fixed projection/row ordering to get exact global 25/50/75% mask counts. Log actual zero and nonzero counts alongside mask counts. For the 2:4 diagnostic, validate every contiguous group and match 50% eligible sparsity.
2. **P3 allocator.** Compute per-block relative output MSE by replaying a Wanda-pruned block in isolation at the relevant uniform target, using the same baseline input and masks. Standardize log sensitivity over blocks with a fixed positive numerical floor. Allocate sparsity as `clip(offset − strength × standardized_sensitivity, 0.10, 0.85)`. Bisect on the parameter-weighted total, then distribute integer remainders within bounds. Try strengths 0.025, 0.05, 0.10 on development pooled NLL and break ties by smaller strength. Freeze each winning configuration before test evaluation.
3. **Q1 quantizer.** Group along input columns in blocks of 128, include zero in the range, use nearest-even rounding, stored FP16 scales, and one uint8 zero-point per group. Use the representable stored scale when encoding, with an explicit zero-group convention and underflow protection. Encode padded columns as the group zero-point so they reconstruct to zero. Pack unsigned codes in a documented bit order and verify byte accounting by serialization round trip. Replay dequantized FP16 weights in the dense path when no valid low-bit execution kernel is available.
4. **Q2 scaling.** Capture activation magnitude statistics from the same calibration manifest. Derive positive channel scales from mean absolute activations to powers {0, .25, .5, .75, 1}, with geometric normalization and a numerical floor. Select by relative output reconstruction error at 16 fixed, evenly spaced token positions per calibration sequence. Apply inverse input scaling, folding into algebraically compatible eligible producers after checking the block output before quantization. Record a compensation tensor for each other case and include its FP16 storage. Save the candidate errors and chosen exponent for every projection and width.
5. **Q3 allocator.** For each block replay its Q2-style form at 2, 3, 4, and 8 bits and measure relative block-output MSE. Choose one width per block via deterministic multiple-choice dynamic programming under each measured Q2 total-model byte cap. Require at least two distinct widths. Count all payload, group padding, metadata, uncompressed parameters, and compensation. Break ties by fuller budget use then lexicographic width assignment. Keep the exact byte cap, used bytes, and sensitivity and width maps.
6. **Gates before large runs.** Test toy masks, stable ties, allocation bounds and exact global counts. Test constant/all-zero groups, every bit width, short groups, pack/unpack and exact measured bytes. Verify FP16 baseline reload, allowed parameter set, mathematical equivalence of pre-rounding scales, PPL denominator, HellaSwag ordering, and deterministic selection before opening final results.

## Evidence saved by stage

| Stage | Evidence |
| --- | --- |
| Baseline | Eligible and tied-weight inventory, manifest audits, per-window sums and counts, four-option scores, original storage and resource timings |
| Calibration | Per-projection activation norms, compact score distributions, fixed probe token positions, output errors, scale search grids, each selected scale, runtime and peak memory |
| Allocation | All development candidates including unsuccessful ones, per-block sensitivities, mask/width maps, exact selected budgets, measured packed-byte ledger |
| Evaluation | Per-window and per-example paired rows, achieved sparsity, nominal precision, serialized or explicitly estimated storage, failures and flags |
| Resources | At least five warmed repeats for 512- and 2,048-token prefill and fixed 64-token decode, peak GPU allocation, timing units and backend. Count one mask bit per eligible weight plus 16 bits per retained weight and byte padding for the sparse-storage estimate. |
| Summaries | PPL plus mean NLL and denominator, both HellaSwag accuracies, sensitivity and quality–budget plots, paired 95% intervals, and aligned candidate-score changes |

Resample whole evaluation windows for PPL and fixed IDs for HellaSwag, preserving pairing for P3−P2 and Q3−Q2. Pool NLL by token count before exponentiating. Use a seed-fixed set of qualitative cases and log any outcome-based case selection rule before generating its examples.

Implementation references: [Wanda](https://arxiv.org/abs/2306.11695), [AWQ](https://arxiv.org/abs/2306.00978), and [BRECQ](https://arxiv.org/abs/2102.05426). Record actual reused code separately.
