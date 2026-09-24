"""Brownian motion and geometric Brownian motion (GBM).

S_T = S_0 * exp((mu - sigma^2/2) T + sigma W_T),  W_T ~ N(0, T).
Median S_0 e^((mu - sigma^2/2) T); mean S_0 e^(mu T).

Run: python3 brownian-motion-gbm.py   (Python 3.10, numpy)
"""
import math
import numpy as np

S0, MU, SIGMA, T = 100.0, 0.08, 0.20, 1.0


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gbm_paths(s0, mu, sigma, t, steps, n, seed=0):
    """n GBM paths on a grid of `steps`; exact in log space."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((n, steps))
    dlog = (mu - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * z
    logs = np.cumsum(dlog, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n, 1)), logs]))


def gbm_step(s, z, mu, sigma, dt):
    """One step: multiply by exp(drift*dt + sigma*sqrt(dt)*z)."""
    return s * math.exp((mu - sigma**2 / 2) * dt
                        + sigma * math.sqrt(dt) * z)


if __name__ == "__main__":
    # 1. Closed form
    m = MU - SIGMA**2 / 2
    median = S0 * math.exp(m * T)
    mean = S0 * math.exp(MU * T)
    sd = mean * math.sqrt(math.expm1(SIGMA**2 * T))
    p_loss = norm_cdf((math.log(100 / S0) - m * T)
                      / (SIGMA * math.sqrt(T)))
    q05 = S0 * math.exp(m * T - 1.6448536 * SIGMA * math.sqrt(T))
    q95 = S0 * math.exp(m * T + 1.6448536 * SIGMA * math.sqrt(T))
    print(f"log drift mu - sigma^2/2 = {MU} - {SIGMA**2 / 2:.2f}"
          f" = {m:.2f}")
    print(f"median {median:.2f}  mean {mean:.2f}  sd {sd:.2f}")
    print(f"P(S_T < 100) = {p_loss:.4f}")
    print(f"5% / 95% quantiles {q05:.2f} / {q95:.2f}")

    # 2. Simulate 10,000 paths of 252 daily steps (seed 0)
    paths = gbm_paths(S0, MU, SIGMA, T, 252, 10_000, seed=0)
    st = paths[:, -1]
    n = len(st)
    se = st.std(ddof=1) / math.sqrt(n)
    print(f"\n10,000 paths x 252 steps, seed 0:")
    print(f"mean   {st.mean():.2f}  (+/- {se:.2f} s.e.)")
    print(f"median {np.median(st):.2f}")
    print(f"sd     {st.std(ddof=1):.2f}")
    print(f"P(S_T < 100) {np.mean(st < 100):.4f}")
    q = np.quantile(st, [0.05, 0.95])
    print(f"5% / 95% {q[0]:.2f} / {q[1]:.2f}")
    print(f"share of paths ending below the mean "
          f"{np.mean(st < mean):.4f}  (formula "
          f"{norm_cdf((math.log(mean / S0) - m * T) / SIGMA):.4f})")
    pse = math.sqrt(p_loss * (1 - p_loss) / n)
    print(f"z-scores: mean {(st.mean() - mean) / se:+.2f}, "
          f"P(loss) {(np.mean(st < 100) - p_loss) / pse:+.2f}")

    # 3. Hand trace: 4 quarterly steps with fixed z values
    zs = [0.5, -1.0, 1.2, -0.3]
    dt = 0.25
    s = S0
    print("\nquarterly trace (z chosen by hand):")
    for k, z in enumerate(zs, 1):
        x = (MU - SIGMA**2 / 2) * dt + SIGMA * math.sqrt(dt) * z
        s = gbm_step(s, z, MU, SIGMA, dt)
        print(f"q{k} z={z:+.1f} log step {x:+.4f} "
              f"factor {math.exp(x):.4f} S={s:.2f}")
    w = math.sqrt(dt) * sum(zs)
    print(f"W_1 = 0.5 * sum(z) = {w:.2f};  closed form "
          f"{S0 * math.exp(m * T + SIGMA * w):.2f}")

    # 4. Why sigma^2/2: two equally likely log returns +-0.2
    up, dn = S0 * math.exp(0.2), S0 * math.exp(-0.2)
    print(f"\n+-0.2 in log: {up:.2f} / {dn:.2f}, "
          f"median 100, mean {(up + dn) / 2:.2f}")
    print(f"E[e^(sigma Z)] = e^(sigma^2/2) = "
          f"{math.exp(SIGMA**2 / 2):.4f}")

    # 5. Quadratic variation: sum of (dW)^2 over [0,1] -> 1
    rng = np.random.default_rng(1)
    for steps in (10, 100, 10_000):
        dw = rng.standard_normal(steps) * math.sqrt(1 / steps)
        print(f"steps {steps:>6}: sum dW^2 = {np.sum(dw**2):.4f}"
              f", sum |dW| = {np.sum(np.abs(dw)):.2f}")

    # 6. sigma = 40%: the mean stays, the median drops
    for sg in (0.2, 0.4):
        md = S0 * math.exp((MU - sg**2 / 2) * T)
        pl = norm_cdf(-(MU - sg**2 / 2) * T / (sg * math.sqrt(T)))
        print(f"sigma {sg:.0%}: median {md:.2f}, mean "
              f"{S0 * math.exp(MU * T):.2f}, P(loss) {pl:.4f}")

    # 7. Arithmetic BM with the same $20/yr vol can go negative
    for yrs in (1, 10, 25):
        mu_a, sd_a = 100 + 8 * yrs, 20 * math.sqrt(yrs)
        print(f"ABM {yrs:>2} yr: P(S<0) = "
              f"{norm_cdf(-mu_a / sd_a):.2e}")
