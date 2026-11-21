# Task 2 · Efficient attention and context memory

Manual pages 5–11. Use standalone runners for Parts A, B, and C so operator checks, training interruption, or cache evaluation can be debugged independently.

## A. Exact attention

- Use the same seeded Q/K/V and upstream gradient for dense attention, tiled online softmax, and PyTorch SDPA. Measure lengths 256, 512, 1024, 2048, 4096, 8192 with B=2, H=4, d=64 in FP16 and FP32 where supported. Compute causal mask from actual positions. Use FP32 maxima, mass and value accumulators for tiling. Handle short tiles and fully masked rows with defined zero output.
- Test future-key/value perturbations, high-magnitude scores, nondivisible tiles, and output/gradient equivalence to dense FP32 on feasible lengths. Probe at least two tile shapes for correctness and freeze 128×128 for the grid. Save absolute and relative errors for outputs and each input gradient. Check memory before attempting long dense conditions and log unsupported/OOM cases.
- Save ten warmed repetitions per supported condition. Time forward separately from complete forward-plus-backward. Record actual SDPA backend, shape, dtype, peak allocated and reserved GPU bytes, and tiled autograd intermediates. Derive latency/error/memory curves and a backend/status table from the raw rows.

## B. MQAR

Use the staff training generator and frozen development and test files. Verify the fixed train/development/test split, unique key/value IDs in the prescribed ranges, query-slot disjointness, same-position labels and actual source distance. Validate near <64, far ≥128 with varied dilation residues, and the mixed regime. Save both source and query positions and the value used for supervision.

| Method | Conditions per seed |
| --- | --- |
| Dense causal | Mixed K=4, 8, 16 and near-only K=16 |
| Sliding window W=64 | Near K=4, 8, 16 and mixed K=16 |
| ELU+1 causal linear | Mixed K=4, 8, 16 |
| Two local plus two dilated heads | Mixed K=4 and 16, dilation offsets 0, 4, …, 252 |
| One trained intervention | Mixed K=16 content-selected top-k |

Train all 14 configurations at seeds 6304, 6305, 6306. Use two pre-norm blocks, width 128, four 32-dimensional heads, 4× MLP, sinusoidal positions and learned token embeddings, a 256-token vocabulary, sequence length 256 and batch 32. Use AdamW, peak LR 8e-4, weight decay 0.01, 200 warmup updates, linear decay to 15% of peak, clip norm 1.0, and 4,500 updates. Save final state and evaluate updates 500, 1000, 1500, 2000, 3000, and 4500. Match initialization and batch seeds across conditions where possible, and make the data cursor deterministic on resume.

1. **Correctness and learnability.** Test same-position masked cross-entropy, causal attention, accessible local and dilated offsets, and the ELU+1 recurrence against a materialized kernel reference. Run a tiny overfit test and a dense pilot to check the generator and learning trajectory. Investigate an unsuccessful dense pilot before interpreting restricted models.
2. **Intervention gate.** Use K=16 dense development checkpoints to compare causal top-32 and top-64 masks on mean far accuracy across seeds. Break ties by near accuracy then smaller k. The predicted benefit is content selection beyond local and fixed-dilation offsets. Freeze the winner and train three runs with the original parameter count. Save the development decision, number of retained pairs, full score computation cost, and state bytes. The intervention keeps standard K/V state.
3. **Evaluation.** Evaluate all trained conditions and dense checkpoints with the test-time-only W=64 mask on the same nine K×regime test cells of 1,024 sequences each. Report query-weighted accuracy, seed-level curves, 1/64 and 1/K chance references, and bins [0,32), [32,64), [64,128), [128,256). Record accessible distance and chance outcomes. A two-layer W=64 model has a theoretical 126-position maximum receptive field.
4. **Cost.** Save theoretical attended pair counts and state bytes at T=256. Run the separate forward-only sweep at T=256, 1024, 4096, 8192 with matched random embeddings. Record dense mask or intermediate allocation and measured time and memory regardless of nominal sparsity.

Retain per-update supervised token counts, clipped norm, LR, loss, elapsed time, generator cursor and seed. Save per-query prediction, distance, labels, condition and checkpoint. Preserve raw resource repetitions. Derive learning curves, distance-by-load heatmaps and mask-control comparisons.

## C. KV cache

1. **Validate the baseline.** Pin Qwen2.5-1.5B-Instruct, tokenizer, Transformers 4.57.1, and SDPA. Use the 180 staff prompts across two context lengths, three evidence positions, and three distractor counts. Use tokenizer offset mapping to convert evidence character spans into evaluation-only token indices. Verify the pre-question/suffix boundary by tokenizing the final formatted prompt and checking its actual token prefix. Save nominal and actual lengths. Verify full-forward/split equivalence, finite logits, and 100% selection equivalence before editing the cache.
2. **Fix cache representation and policies.** Store pre-question context in original temporal order. Size budgets by actual serialized group-64 channel-wise-K/token-wise-V INT4 bytes at that boundary, with FP16 scale and offset, packed codes and FP16 trailing residual. Use shared positions for heads/layers. Compare full FP16 reference, sink-plus-recent FP16, question-aware FP16, both INT4 layouts, and question-aware FP16 plus INT2. Always protect eight sinks and 64 recent positions in the selection methods. Select the maximum legal count based on real bytes, including indices and residual. In a temporary unedited cache pass, rank context positions by mean masked FP32 attention from question queries to original rotated keys. Restore the pre-question snapshot before applying the policy. Gold code and spans remain evaluation-only.
3. **Decode position and storage checks.** Preserve absolute RoPE positions of retained tokens and validate physical write pointers after eviction. Keep the identical question suffix in FP16, stage decode entries in FP16, and promote complete low-bit groups during generation where the method calls for it. Check partial and constant group behavior, distinct K/V reconstruction errors, packed sizes, and finite logits with compaction. Record pre-question packed bytes, shared suffix, live decode cache and temporary dequantized tensors. Use fixed greedy generation and first standalone five-digit code extraction, with a 32-token cap unless a staff instruction specifies a stricter rule.
4. **Collect fidelity.** Save unconditional retrieval exact match and accuracy conditional on FP16 correctness, gold answer NLL and teacher-forced logit KL including after question prefill, selection masks and evidence retention. Preserve candidate generation IDs, benchmark timing for scoring, packing and generation, and paired prompts for the main bootstrap with at least 2,000 draws. Preserve grid cells as descriptive diagnostics and report the full paired-set differences.
5. **Collect systems measurements.** At batches 1 and 8 and both context lengths, time actual pack, unpack/dequantization and decode steps versus FP16 over at least ten warmed synchronized repeats. Save peak allocated GPU bytes and packed size at each measured stage. Derive a matched-byte table, timing and memory tables, and aligned generation examples with selection masks.

References: [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [SnapKV](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html), [KIVI](https://proceedings.mlr.press/v235/liu24bz.html).
