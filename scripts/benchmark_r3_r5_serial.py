"""
R3 + R5 (serial fallback): Multiple seeds + timing error bars.
R3: 12 cases × 3 seeds (serially, no bootstrap — just critical_bandwidth + find_modes + detect_components)
R5: 3 timing iterations per function, serially, first seed only
"""

import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pola

CASES = [
    {"name": "Well-separated",       "n": 400, "components": [(0.5, -2.0, 0.3),  (0.5, 2.0, 0.3)]},
    {"name": "Moderate separation",  "n": 500, "components": [(0.5, -1.0, 0.5),  (0.5, 1.5, 0.5)]},
    {"name": "Barely separated",     "n": 600, "components": [(0.5, -0.5, 0.4),  (0.5, 0.5, 0.4)]},
    {"name": "Unequal variance",     "n": 400, "components": [(0.5, -2.0, 0.6),  (0.5, 2.0, 0.2)]},
    {"name": "Unequal weights",      "n": 500, "components": [(0.2, -2.0, 0.3),  (0.8, 2.0, 0.3)]},
    {"name": "Extreme separation",   "n": 400, "components": [(0.5, -5.0, 0.5),  (0.5, 5.0, 0.5)]},
    {"name": "Trimodal",             "n": 450, "components": [(1/3, -3.0, 0.3), (1/3, 0.0, 0.3), (1/3, 3.0, 0.3)]},
    {"name": "Skewed bimodal",       "n": 500, "components": [(0.7, -1.5, 0.4),  (0.3, 2.0, 0.6)]},
    {"name": "Heavy-tailed bimodal", "n": 400, "components": [(0.5, -3.0, 0.8),  (0.5, 3.0, 0.8)]},
    {"name": "Near unimodal",        "n": 600, "components": [(0.5, 0.0, 0.6),   (0.5, 1.5, 0.6)]},
    {"name": "Small sample",         "n": 60,  "components": [(0.5, -2.0, 0.5),  (0.5, 2.0, 0.5)]},
    {"name": "Overlapping variances","n": 500, "components": [(0.5, -0.8, 0.7),  (0.5, 0.8, 0.5)]},
]

N_SEEDS = 5  # reduced from 10 for speed
N_TIMING = 3  # reduced from 5

def gen(case, seed):
    rng = np.random.RandomState(seed)
    data = []
    for w, m, s in case["components"]:
        n_i = max(1, int(round(w * case["n"])))
        data.append(rng.normal(m, s, n_i))
    data = np.concatenate(data)[:case["n"]]
    return data

t0 = time.time()

# ============================================================
# R5 TIMING FIRST (first seed only, 3 cases)
# ============================================================
print("=" * 70)
print("R5: Timing error bars (serial, N=3)")
print("=" * 70)
timing_cases = [CASES[0], CASES[2], CASES[10]]
for name_prefix, case in [("large", CASES[0]), ("small", CASES[2]), ("tiny", CASES[10])]:
    data = gen(case, case.get("seed_offset", 42))
    h_silv = pola.silverman_bandwidth(data)
    
    for func_name, func, args in [
        ("critical_bandwidth", pola.critical_bandwidth, ((data,), {"method": "auto"})),
        ("find_modes", pola.find_modes, ((data,), {"h": h_silv})),
        ("dip_test", pola.dip_test, ((data,), {})),
    ]:
        times = []
        for _ in range(N_TIMING):
            t = time.perf_counter()
            func(*args[0], **args[1])
            times.append(time.perf_counter() - t)
        print(f"  {name_prefix:25s} {func_name:25s}: {np.mean(times):.4f} ± {np.std(times, ddof=1):.4f}s")

# ============================================================
# R3 MULTI-SEED
# ============================================================
print(f"\n{'=' * 70}")
print(f"R3: Multi-seed benchmark (serial, {N_SEEDS} seeds per case)")
print(f"{'=' * 70}")

print(f"\n{'Case':25s} {'n':>5s} {'h_crit mean':>12s} {'h_crit SD':>10s} {'CV(%)':>7s} {'modes':>6s} {'strength':>10s}")
print("-" * 75)

for case in CASES:
    name = case["name"]
    h_vals, n_modes_list, bs_list = [], [], []
    
    for s in range(N_SEEDS):
        data = gen(case, 100 + s)
        
        # critical_bandwidth
        h_crit, _ = pola.critical_bandwidth(data, k=2, method="auto")
        h_vals.append(h_crit)
        
        # find_modes
        h_silv = pola.silverman_bandwidth(data)
        mr = pola.find_modes(data, h=h_silv)
        n_modes_list.append(mr.n_modes)
        
        # bimodality_strength
        bs = pola.bimodality_strength(data)
        bs_list.append(bs)
    
    h_mean = np.mean(h_vals)
    h_std = np.std(h_vals, ddof=1)
    cv = h_std / h_mean * 100
    mode_ok = all(m == n_modes_list[0] for m in n_modes_list)
    
    print(f"{name:25s} {case['n']:>5d} {h_mean:>10.4f}  {h_std:>8.4f} {cv:>6.2f}% {n_modes_list[0]:>4d} {'✓' if mode_ok else '✗':>5s} {np.mean(bs_list):>8.2f}")

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s")
