# Task 3 · Policy distillation and speculative decoding

Manual pages 11–15. Validate exact sampling before collecting speculative prefixes or timing a real target. Match the teacher-scored training budget across prefix policies and preserve results for each seed.

## Inputs and splits

- Pin Qwen2.5-1.5B-Instruct as target and Qwen2.5-0.5B-Instruct as draft. Verify identical prompt token IDs, vocabulary mapping, chat template output, EOS handling, and temperature-1/top-p-1 sampling with top-k disabled.
- Train with the 500 locked GSM8K train IDs and 300 staff short-form training prompts. Test on 60 GSM8K prompts and 60 short-form prompts, divided equally between familiar and held-out template families. Save exact IDs, category and family, prefix token IDs or their immutable input artifact reference, scoring rules and model revisions.
- Staff materialization generates teacher and original-draft banks, as stated in the [data README](https://github.com/AbDu11aHHH/atml-pa4-datasets/blob/main/README.md). Generate the original-draft exact speculative bank with gamma 4 and each trained student's frozen bank with its named checkpoint. Use the staff matched prompt, seed and sampling conventions. Audit generation settings and the position sampling limit of 32 in positions 1–64, with separate long-position records for 65–192.

## Exact sampler

1. Define a target and draft adapter with next-token probabilities, batched verification and cache truncation. Emit each proposal with acceptance probability `min(1, p(x)/q(x))`. On rejection, sample from normalized `max(p−q,0)`. On full acceptance draw a bonus from the target next-token distribution if the EOS and cap rules permit it. Use FP32 model probabilities and log probability differences or stable residual computation. Track emitted-but-not-cached tokens explicitly.
2. Test first and later rejection, full acceptance, gamma 1, shorter final proposals, accepted and rejected EOS, bonus EOS, padding, final token truncation, all cache lengths, and equivalence to recomputing the accepted continuation. Verify real Qwen split-prefill logits and RoPE/cache positions. Count all actual forward passes, including any draft cache advancement.
3. Use a three-symbol Markov toy with target probabilities (0.60, 0.25, 0.15) and draft probabilities (0.15, 0.35, 0.50), rotating by predecessor symbol and using the unrotated probabilities at the first step. Calculate joint probabilities for all 27 triples. Over 100,000 sequences at seed 6304 verify expected category counts >=10, Pearson χ² p>=0.01 for correct sampling and p<0.01 when rejection samples directly from p. Save observed/expected counts and exact test parameters. Recheck the real-model cache before the first timed run.

## Training schedule and resume state

| Setting | Frozen value |
| --- | --- |
| Conditions | Off-policy and on-policy, each independently initialized at seeds 6304 and 6305 |
| Per-run quota | 30,000 mathematics and 10,000 extraction scored tokens |
| Response caps | 192 mathematics and 32 extraction |
| LoRA | Rank 8, alpha 16, dropout 0 on Q/K/V/O projections |
| Optimizer | AdamW 3e-4 with 0.01 weight decay, microbatch 1, one attempted update per 256 scored tokens, final short update, global clip 1.0 |
| Precision | Frozen FP16 backbones, FP32 adapters and optimizer state, FP16 autocast with dynamic scaling and unscale before clipping |

Use the same seeded round-robin prompt order in each category. Interleave categories with 768 mathematics tokens followed by 256 extraction tokens until the quotas are exhausted. For off-policy, generate one frozen teacher response per training prompt and cycle that pool. For on-policy, generate a new rollout without gradients at the current draft checkpoint. A rollout spanning updates retains its producing update, and each new rollout uses the current adapter. Score the teacher at every supervised response prefix, compute forward KL over the full common vocabulary, and normalize by actual supervised tokens per optimizer step. Include EOS when actually sampled and scored. Apply exact category caps to partial responses. Save raw response tokens, scored segment offsets, policy/checkpoint ID, seed, EOS/truncation, token counts, gradient/scaler status, loss, learning rate, and update events. The quota allows 156 full accumulation groups and one 64-token final group. Record attempted and executed updates separately.

Preserve teacher-pool materialization, teacher scoring, student generation, backward/update, and merge times separately. Checkpoint at least every 25 attempted updates and at final quota, with adapter, optimizer and scaler, counters, RNG, category/prompt cursor, and any remaining tokens from the current rollout. Merge each final adapter into a fresh FP16 draft and log merged/unmerged differences on a fixed prompt batch.

## Evaluation and evidence

| Stage | Procedure and saved evidence |
| --- | --- |
| Prefix alignment | Evaluate the original and four trained drafts on the same seven banks. Save forward KL, reverse KL, TV, greedy agreement, and expected acceptance 1−TV per prefix. Use positions [1,16], [17,32], [33,64] and a separate 65–192 bank. Match teacher and exact-speculative controls at prompt and position strata and log EOS reach and producer checkpoint. Keep the three workload groups and seed-specific estimates. |
| Primary runtime | Run original plus four merged drafts at gamma 2 and 4 on all 120 prompts with two paired repetitions and an EOS-aware 64-token cap. Save adjacent target-only results and counterbalance order. Save proposal/examined/accepted counts, accepted prefix, target and draft passes, emitted tokens, peak memory, and EOS/truncation for every run. |
| Profiling | Save synchronized draft proposal, target verification, acceptance/residual and cache time per round. Record warm setup, prefill, host overhead, end-to-end time and isolated draft/target step cost across prompt lengths. Preserve uninstrumented paired throughput separately so profiling overhead is measured. Time one unmerged adapter and run the fixed-64 ignoring-EOS diagnostic on 24 seed-selected mathematics prompts. |
| Standalone quality | Use the original and trained drafts with a 512-token mathematics cap and a 32-token extraction cap. Save output, extracted answer, extraction compliance, correctness, and truncation. Score using saved deterministic rules and retain aligned example outputs. |
| Cost and uncertainty | Compute c, actual verification cost v_gamma, projected and measured speedup, and break-even acceptance by gamma. Record target verification input lengths including a pending token when fused. Compute deployment break-even volume from the measured per-request savings and all upfront training/merge cost. Use 2,000 prompt-paired bootstrap draws for principal acceptance and throughput intervals, preserving paired repetitions and reporting seed variation. |

Gate the primary benchmarks on the toy distribution test, real cache tests, completed quotas, and merged checks. Preserve split-level metrics, prompt-level results, training curves, exception logs, and comparison tables. Report an inconclusive difference or a nonpositive measured saving as observed.

References: [Exact speculative sampling](https://proceedings.mlr.press/v202/leviathan23a.html), [DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html).
