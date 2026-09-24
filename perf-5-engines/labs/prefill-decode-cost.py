"""Prefill-decode cost lab: why one prefill stalls every decoding user (Sarathi-Serve §3)

Run: python3 labs/prefill-decode-cost.py   (stdlib only, CPU, well under a second)

Our recomputation for Llama-3.1-8B BF16 on one H100 SXM. It extends the course-2 step-time
formula (../perf-2-scaling-book-inference/step-time.html) with P prefill tokens in the same step:

  step(B, S, P) = B * S * kv_per_token / BW          decode attention: read every KV cache
                + max(2 * N * (B + P) / C,            linear layers: math for B + P tokens ...
                      weight_bytes / BW)              ... or the one shared weight read
                + attention_flops(prompts) / C        prefill attention: each prompt over itself

It's a lower bound (peak bandwidth, 100% MFU, perfect overlap, no kernel launch or scheduler
overhead). Real GPUs leave the flat region later: Sarathi-Serve reports ~500-600 tokens in
practice vs ~200 in theory on A100 (§3.1, footnote 2).
"""

N = 8.03e9                 # parameters
W = 16.06e9                # BF16 weight bytes
KV_TOK = 131072            # 2 * L * K * H * 2 bytes = 128 KiB per token
L, D = 32, 4096            # layers, model width (N_heads * head_dim = 32 * 128)
BW, C = 3.35e12, 9.89e14   # H100 SXM: HBM bytes/s, dense BF16 FLOP/s
MS = 1e3


def prefill_attn_flops(prompt_len):
    # QK^T and PV, 2 FLOPs per multiply-add, causal (half the P x P square), all heads, all layers
    return L * 2 * 2 * prompt_len * (prompt_len / 2) * D


def step(B, S, prompts=()):
    """Returns (total, kv_read, linear, linear_math, weight_read, prefill_attn) in seconds."""
    P = sum(prompts)
    t_kv = B * S * KV_TOK / BW
    t_math = 2 * N * (B + P) / C
    t_w = W / BW
    t_lin = max(t_math, t_w)
    t_att = sum(prefill_attn_flops(p) for p in prompts) / C
    return t_kv + t_lin + t_att, t_kv, t_lin, t_math, t_w, t_att


B, S = 32, 2048
base, kv, lin, math_, w, _ = step(B, S)

print("== 1. The decode-only step: B = 32 users at 2,048 tokens of context ==")
print(f"KV read  {B} x {S} x {KV_TOK:,} B / 3.35e12 = {kv * MS:.2f} ms")
print(f"linear   max(math 2 x 8.03e9 x {B} / 9.89e14 = {math_ * MS:.2f} ms, "
      f"weights 16.06e9 / 3.35e12 = {w * MS:.2f} ms) = {lin * MS:.2f} ms")
print(f"step     {base * MS:.2f} ms  -> TBT every user sees, {1 / base:.0f} tok/s each, {B / base:,.0f} tok/s total")

b_crit = W / BW / (2 * N / C)          # tokens T where 2*N*T/C = W/BW
print(f"\nlinear layers stay flat until B + P = weight read / (2N/C) = {b_crit:.0f} tokens "
      f"(the course's B_crit ~ 295)")
print(f"so the first {b_crit - B:.0f} prefill tokens ride along almost free (only their attention costs)")

print("\n== 2. Add P prefill tokens (one fresh prompt) to that same step ==")
print(f"{'P':>6} | {'step ms':>7} | {'linear':>6} | {'pf attn':>7} | {'extra ms':>8} | "
      f"{'x TBT':>5} | {'extra/token us':>14} | {'naive P x 16.2us':>16}")
rows = [(p,) for p in (0, 256, 512, 1024, 4096)] + [(4096,) * 4]
for prompts in rows:
    P = sum(prompts)
    t, _, t_lin, _, _, t_att = step(B, S, prompts)
    extra = t - base
    per = extra / P * 1e6 if P else 0
    naive = 2 * N * P / C
    label = f"{P:,}" if len(prompts) == 1 else "4x4096"
    print(f"{label:>6} | {t * MS:7.2f} | {t_lin * MS:6.2f} | {t_att * MS:7.3f} | {extra * MS:8.2f} | "
          f"{t / base:5.2f} | {per:14.2f} | {naive * MS:13.2f} ms")
print("naive = the additive rule 'P tokens x 16.2 ms per 1,000 at 100% MFU' (ignores the flat region)")
print("4x4096 = four 4k prompts admitted in one step, as an eager scheduler may do")

print("\n== 3. Marginal cost of one more prefill token ==")
print(f"below {b_crit - B:.0f} tokens: ~0 linear cost (hidden under the {w * MS:.2f} ms weight read); "
      f"only its attention")
print(f"above: 2 x 8.03e9 / 9.89e14 = {2 * N / C * 1e6:.2f} us per token (+ attention, growing with the prompt)")

print("\n== 4. vLLM-style vs Orca-style: how long the decoding users wait ==")
pf_only, *_ = step(0, 0, (4096,))
hybrid, *_ = step(B, S, (4096,))
print(f"vLLM (prefill-only iteration, then decodes): gap = {pf_only * MS:.2f} + {base * MS:.2f} "
      f"= {(pf_only + base) * MS:.2f} ms")
print(f"Orca (hybrid batch, prefill + decodes in one iteration): gap = {hybrid * MS:.2f} ms")
print(f"FasterTransformer (no new prefill until the batch drains): gap = {base * MS:.2f} ms, "
      f"but the new prompt waits for every running request to finish")

print("\n== 5. Sarathi's strict TBT SLO recipe on our H100 ==")
clean4k, *_ = step(32, 4096)
print(f"5 x a clean decode iteration at batch 32, 4k context = 5 x {clean4k * MS:.2f} = {5 * clean4k * MS:.1f} ms")
print(f"a 4,096-token prefill in the step gives {hybrid * MS:.1f} ms -> "
      f"{'violates' if hybrid > 5 * clean4k else 'meets'} it; 4x4096 gives {step(B, S, (4096,) * 4)[0] * MS:.1f} ms")

print("\n== 6. '1 decode token costs as much as X prefill tokens' (linear layers) ==")
print(f"ours, H100 theory: a batch-1 decode's linear time is the weight read, {w * MS:.2f} ms,")
print(f"  = the math for W/BW / (2N/C) = {b_crit:.0f} prefill tokens at 100% MFU  (= C/BW ops:byte, since BF16 is 2 bytes/param)")
print("paper, Mistral-7B on one A100: measured ~128 prefill tokens (Fig 4); theory ~200 on A100 (§3.1 footnote)")
print(f"at B = 32 the weight read is shared: {w / B * MS:.3f} ms per decode token = {b_crit / B:.1f} prefill tokens each")

# Try this:
# 1. Set B = 128: the flat region shrinks to 295 - 128 = 167 free tokens and the decode-only step
#    is already dominated by KV reads. Does a 512-token chunk hurt TBT more or less (as a ratio)?
# 2. Set S = 8192 (long contexts): the KV read grows 4x. The prefill spike is the same in ms but
#    a smaller multiple of the (now slower) decode step.
# 3. Switch to FP8 weights: W = 8.03e9 and C = 1.979e15. The weight read halves, B_crit stays ~295,
#    and the 4k prefill spike roughly halves (its math runs at twice the FLOP rate).
