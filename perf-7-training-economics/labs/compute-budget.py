"""Compute budgets: 6ND, Chinchilla and MFU.

Run: python3 labs/compute-budget.py
Stdlib only, CPU, well under a second.

Prints every number on compute-budgets.html:
  1. FLOPs per token: 6N plus the attention term 12*L*H*Q*T (PaLM App. B)
  2. Chinchilla: Table 3 tokens/param, the fitted loss, compute-optimal N and D for a budget
  3. MFU and HFU: PaLM's worked example, Llama 3 Table 4, Megatron PTD-P, DeepSeek-V3
  4. The worked example: 6ND -> H100-hours -> $ for 8B / 70B / 405B at MFU 30-50%
  5. Why small models get over-trained: training vs inference FLOPs
"""

H100_BF16 = 989e12          # dense BF16 FLOP/s (NVIDIA; _facts.md)
PRICE = 3.99                # $/H100-hour, Lambda 8x on-demand, seen 2026-09-24 (_facts.md default)

# Llama 3 herd Table 3 shapes; params from course 1 (8B) and course 6 (70B); 405B as named
LLAMA = {
    #  name        N         D_tokens  L    heads  head_dim  d_model
    "8B":   dict(N=8.03e9,  D=15e12,   L=32,  H=32,  Q=128, d=4096),
    "70B":  dict(N=70.55e9, D=15e12,   L=80,  H=64,  Q=128, d=8192),
    "405B": dict(N=405e9,   D=15.6e12, L=126, H=128, Q=128, d=16384),
}


def sci(x):
    return f"{x:.2e}"


def attn_per_token(L, H, Q, T):
    """PaLM App. B: dense self-attention adds 6*L*H*(2*Q*T) = 12*L*H*Q*T FLOPs per token."""
    return 12 * L * H * Q * T


print("=" * 72)
print("1. FLOPs per token: 6N and the attention correction")
print("=" * 72)
for name, m in LLAMA.items():
    six_n = 6 * m["N"]
    print(f"Llama-3.1-{name}: training 6N = {sci(six_n)} FLOPs/token, inference 2N = {sci(2*m['N'])}")
    for T in (8192, 131072):
        a = attn_per_token(m["L"], m["H"], m["Q"], T)
        print(f"    T = {T:>7,}: attention 12LHQT = {sci(a)}  -> +{100*a/six_n:5.1f}% on top of 6N"
              f"   (Scaling Book ratio T/8D = {T/(8*m['d']):.3f})")
# PaLM 540B: 118 layers, 48 heads, head dim 256, sequence 2048 (PaLM Table 1, App. B)
palm_attn = attn_per_token(118, 48, 256, 2048)
palm_6n = 6 * 540e9
print(f"PaLM 540B at T = 2048: attention adds {100*palm_attn/palm_6n:.2f}% to 6N")
print("Rule of thumb (Scaling Book Part 4): attention FLOPs dominate only when T > 8D")
for name, m in LLAMA.items():
    print(f"    Llama {name}: 8D = {8*m['d']:,} tokens")

print()
print("=" * 72)
print("2. Chinchilla (Hoffmann et al. 2022)")
print("=" * 72)
table3 = [(400e6, 1.92e19, 8.0e9), (1e9, 1.21e20, 20.2e9), (10e9, 1.23e22, 205.1e9),
          (67e9, 5.76e23, 1.5e12), (175e9, 3.85e24, 3.7e12), (280e9, 9.90e24, 5.9e12),
          (520e9, 3.43e25, 11.0e12), (1e12, 1.27e26, 21.2e12), (10e12, 1.30e28, 216.2e12)]
print("Table 3 (Approach 1): params, FLOPs, tokens, tokens/param, FLOPs/(6ND)")
for N, C, D in table3:
    print(f"    {N/1e9:>7.1f}B  {sci(C)}  {D/1e9:>9.1f}B tokens  {D/N:5.1f} tok/param  C/6ND = {C/(6*N*D):.2f}")
print(f"Chinchilla itself: 70B on 1.4T -> {1.4e12/70e9:.1f} tokens/param")

E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N, D):
    return E + A / N**alpha + B / D**beta


def opt_fit(C):
    """Minimise L(N, D) subject to 6ND = C (Chinchilla eq. 4 with the eq. 10 fit)."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)


def opt_20(C):
    """The community reading of Table 3: D = 20 N, so C = 6 * N * 20N = 120 N^2."""
    N = (C / 120) ** 0.5
    return N, 20 * N


print(f"Fitted loss L(N,D) = {E} + {A}/N^{alpha} + {B}/D^{beta}   (eq. 10)")
print(f"    exponent a = beta/(alpha+beta) = {beta/(alpha+beta):.2f} (N_opt ~ C^a),"
      f" b = {alpha/(alpha+beta):.2f} (D_opt ~ C^b)")

C8 = 6 * LLAMA["8B"]["N"] * LLAMA["8B"]["D"]
print(f"\nLlama-3.1-8B budget C = 6 * 8.03e9 * 15e12 = {sci(C8)} FLOPs. Spent compute-optimally:")
n20, d20 = opt_20(C8)
nf, df = opt_fit(C8)
print(f"    20 tokens/param reading : N = {n20/1e9:5.1f}B, D = {d20/1e12:5.2f}T")
print(f"    eq. 10 fit (Approach 3) : N = {nf/1e9:5.1f}B, D = {df/1e12:5.2f}T")
print(f"    Llama-3.1-8B actual     : N =   8.0B, D = 15.00T -> {15e12/8.03e9:,.0f} tokens/param")
l8, l20 = loss(8.03e9, 15e12), loss(n20, d20)
print(f"    loss under Chinchilla's fit: 8B/15T = {l8:.3f}, {n20/1e9:.1f}B/{d20/1e12:.2f}T = {l20:.3f}"
      f"  (gap {l8-l20:.3f}; Chinchilla's data, not Llama's: illustrative)")
print(f"    loss at the eq. 10 optimum {nf/1e9:.1f}B/{df/1e12:.2f}T = {loss(nf, df):.3f}")
print(f"    inference FLOPs/token ratio {n20/1e9:.1f}B vs 8B: {n20/8.03e9:.1f}x")
for name, m in LLAMA.items():
    print(f"    Llama {name}: {m['D']/m['N']:,.0f} tokens/param  (Chinchilla 20)")

print()
print("=" * 72)
print("3. MFU vs HFU")
print("=" * 72)
mfu_palm = 238.3e3 * 6 * 540e9 / (275e12 * 6144)
print(f"PaLM 540B: 238.3K tok/s x 6 x 540B / (275 TF x 6144 TPU v4) = {100*mfu_palm:.1f}% MFU (no attention)")
print(f"    with attention (+{100*palm_attn/palm_6n:.2f}%): {100*mfu_palm*(1+palm_attn/palm_6n):.1f}%"
      "   paper: 46.2%; HFU incl. rematerialization 57.8% (paper)")
mtnlg = 1920 * 2048 / 60.1
print(f"MT-NLG 530B: {mtnlg/1e3:.2f}K tok/s x 6 x 530B / (312 TF x 2240) = "
      f"{100*mtnlg*6*530e9/(312e12*2240):.1f}% (paper prints 29.7%)")
for gpus, tf in ((8192, 430), (16384, 400), (16384, 380)):
    print(f"Llama 3 405B Table 4, {gpus:,} GPUs: {tf} TF/GPU / 989 = {100*tf*1e12/H100_BF16:.1f}%")
C405 = 6 * 405e9 * 15.6e12
print(f"Llama 3 405B: 6 x 405e9 x 15.6e12 = {sci(C405)} (paper: 3.8e25)")
print(f"    pure compute at 400 TF on 16,384 GPUs: {C405/(16384*400e12)/86400:.0f} days")
print(f"Megatron PTD-P: 163 TF / 312 TF = {100*163/312:.0f}% of peak, counting recompute (8ND)."
      f" As MFU (x 6/8): {100*163/312*6/8:.0f}%")
C_ds = 6 * 37e9 * 14.8e12
for label, hours in (("pre-training 2,664K", 2.664e6), ("total 2,788K", 2.788e6)):
    print(f"DeepSeek-V3: 6 x 37B active x 14.8T = {sci(C_ds)} over {label} H800-hours"
          f" -> {C_ds/(hours*3600)/1e12:.0f} TF/GPU effective")
print(f"    = {100*C_ds/(2.664e6*3600)/989e12:.0f}% of dense BF16 peak, "
      f"{100*C_ds/(2.664e6*3600)/1979e12:.0f}% of dense FP8 peak (rough: ignores attention, MTP)")
print(f"    $ at $2/hr: {2.788e6*2/1e6:.3f}M; 180K hours/T x 14.8T = {180e3*14.8/1e6:.3f}M hours")

print()
print("=" * 72)
print(f"4. Worked example: 6ND -> H100-hours -> $ (at ${PRICE}/hr)")
print("=" * 72)
print(f"{'model':>6} {'tokens':>7} {'6ND FLOPs':>10}   " +
      "   ".join(f"MFU {m:.0%}: M hrs / $M" for m in (0.3, 0.4, 0.5)))
for name, m in LLAMA.items():
    C = 6 * m["N"] * m["D"]
    cells = []
    for mfu in (0.3, 0.4, 0.5):
        hrs = C / (H100_BF16 * mfu) / 3600
        cells.append(f"{hrs/1e6:6.2f} / {hrs*PRICE/1e6:6.1f}      ")
    print(f"{name:>6} {m['D']/1e12:6.1f}T {sci(C):>10}   " + "".join(cells))
C8 = 6 * 8.03e9 * 15e12
hrs8 = C8 / (H100_BF16 * 0.4) / 3600
print(f"\n8B at 40%: {sci(C8)} / (989e12 x 0.40) / 3600 = {hrs8/1e6:.3f}M H100-hours"
      f" x ${PRICE} = ${hrs8*PRICE/1e6:.2f}M")
print(f"    on 1,024 H100s: {hrs8/1024/24:.1f} days")
print("Meta's model card GPU-hours (all training time for each model) vs our 40% estimate:")
card = {"8B": 1.46e6, "70B": 7.0e6, "405B": 30.84e6}
for name, m in LLAMA.items():
    C = 6 * m["N"] * m["D"]
    est = C / (H100_BF16 * 0.4) / 3600
    implied = C / (card[name] * 3600 * H100_BF16)
    print(f"    {name:>5}: card {card[name]/1e6:5.2f}M, ours {est/1e6:5.2f}M -> ratio {card[name]/est:.1f}x,"
          f" implied average 'MFU' {100*implied:.0f}%")

print()
print("=" * 72)
print("5. Training vs inference compute")
print("=" * 72)
for name, m in LLAMA.items():
    print(f"    {name}: inference matches training FLOPs after 6ND/2N = 3D = {3*m['D']/1e12:.0f}T served tokens")
print("$/token scales with N (our framing): 8B ceiling at batch 64 = $0.17/M (_facts.md, $3.99/hr).")
print(f"    A {n20/1e9:.1f}B model reads {n20/8.03e9:.1f}x the weight bytes per step and does {n20/8.03e9:.1f}x"
      f" the FLOPs: roughly ${0.17*n20/8.03e9:.2f}/M at the same efficiency (rough scaling, ours)")

# Try this:
# 1. Change PRICE to 2.0 (DeepSeek's assumed H800 rate) or 6.88 (AWS p5) and re-read the $ column.
# 2. Put a 1B model on 15T tokens into LLAMA and compare its tokens/param and loss to the 8B.
# 3. Raise T in section 1 to 1,048,576 (1M context): for the 8B, attention now costs more than 6N.
