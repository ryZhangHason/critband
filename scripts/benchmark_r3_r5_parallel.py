"""
R3 + R5: Parallel multiple-seed benchmark with timing error bars.

R3: 12 cases × 10 seeds → report h_crit mean ± SD, mode count consistency
R5: time.perf_counter() each function 5+ iterations → report mean ± SD

Uses ProcessPoolExecutor to parallelize across seeds on 8-core M2.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import critband

# ============================================================================
# Benchmark cases
# ============================================================================
CASES = [
    {"name": "Well-separated",       "n": 400, "seed_offset": 100, "components": [(0.5, -2.0, 0.3),  (0.5, 2.0, 0.3)]},
    {"name": "Moderate separation",  "n": 500, "seed_offset": 200, "components": [(0.5, -1.0, 0.5),  (0.5, 1.5, 0.5)]},
    {"name": "Barely separated",     "n": 600, "seed_offset": 300, "components": [(0.5, -0.5, 0.4),  (0.5, 0.5, 0.4)]},
    {"name": "Unequal variance",     "n": 400, "seed_offset": 400, "components": [(0.5, -2.0, 0.6),  (0.5, 2.0, 0.2)]},
    {"name": "Unequal weights",      "n": 500, "seed_offset": 500, "components": [(0.2, -2.0, 0.3),  (0.8, 2.0, 0.3)]},
    {"name": "Extreme separation",   "n": 400, "seed_offset": 600, "components": [(0.5, -5.0, 0.5),  (0.5, 5.0, 0.5)]},
    {"name": "Trimodal",             "n": 450, "seed_offset": 700, "components": [(1/3, -3.0, 0.3), (1/3, 0.0, 0.3), (1/3, 3.0, 0.3)]},
    {"name": "Skewed bimodal",       "n": 500, "seed_offset": 800, "components": [(0.7, -1.5, 0.4),  (0.3, 2.0, 0.6)]},
    {"name": "Heavy-tailed bimodal", "n": 400, "seed_offset": 900, "components": [(0.5, -3.0, 0.8),  (0.5, 3.0, 0.8)]},
    {"name": "Near unimodal",        "n": 600, "seed_offset": 1000,"components": [(0.5, 0.0, 0.6),   (0.5, 1.5, 0.6)]},
    {"name": "Small sample",         "n": 60,  "seed_offset": 1100,"components": [(0.5, -2.0, 0.5),  (0.5, 2.0, 0.5)]},
    {"name": "Overlapping variances","n": 500, "seed_offset": 1200,"components": [(0.5, -0.8, 0.7),  (0.5, 0.8, 0.5)]},
]

N_SEEDS = 10
N_TIMING = 5  # timing iterations per function


def generate_data(case, seed):
    """Generate data from mixture case with given seed."""
    rng = np.random.RandomState(seed)
    data = []
    for weight, mean, std in case["components"]:
        n_i = max(1, int(round(weight * case["n"])))
        data.append(rng.normal(mean, std, n_i))
    data = np.concatenate(data)
    # Adjust length
    if len(data) > case["n"]:
        data = data[:case["n"]]
    elif len(data) < case["n"]:
        extra_idx = rng.choice(len(data), case["n"] - len(data))
        data = np.concatenate([data, data[extra_idx]])
    return data


def run_single_case_seed(args):
    """Run all critband functions for one case+seed combination.
    
    Returns dict of results. This function runs in a worker process.
    """
    case, seed_idx = args
    seed = case["seed_offset"] + seed_idx
    data = generate_data(case, seed)
    
    result = {"case": case["name"], "seed": seed, "n": len(data)}
    
    # R3: critical bandwidth (no bootstrap — fast)
    t0 = time.perf_counter()
    h_crit, success = critband.critical_bandwidth(data, k=2, method="auto")
    result["h_crit_t"] = time.perf_counter() - t0
    result["h_crit"] = h_crit
    result["success"] = success
    
    # silverman_bandwidth (fast)
    t0 = time.perf_counter()
    h_silv = critband.silverman_bandwidth(data)
    result["h_silv_t"] = time.perf_counter() - t0
    result["h_silv"] = h_silv
    
    # bimodality_strength (fast)
    t0 = time.perf_counter()
    bs = critband.bimodality_strength(data)
    result["bs_t"] = time.perf_counter() - t0
    result["bimodality_strength"] = bs
    
    # find_modes (fast)
    t0 = time.perf_counter()
    modes_result = critband.find_modes(data, h=h_silv)
    result["nm_t"] = time.perf_counter() - t0
    result["n_modes"] = modes_result.n_modes
    
    # detect_components (fast)
    t0 = time.perf_counter()
    try:
        decomp = critband.detect_components(data)
        result["dc_t"] = time.perf_counter() - t0
        result["comp_means"] = (decomp.component1.mean, decomp.component2.mean)
        result["comp_weights"] = (decomp.component1.weight, decomp.component2.weight)
        result["separation_point"] = decomp.separation_point
    except Exception:
        result["dc_t"] = time.perf_counter() - t0
        result["comp_means"] = None
        result["comp_weights"] = None
        result["separation_point"] = None
    
    # dip_test (fast)
    t0 = time.perf_counter()
    dip_res = critband.dip_test(data)
    result["dip_t"] = time.perf_counter() - t0
    result["dip_stat"] = dip_res.statistic
    result["dip_p"] = dip_res.pvalue
    
    # silverman_test with bootstrap (slow — do only for timing stats)
    # We measure time but only run N_TIMING times on the first seed for timing
    if seed_idx == 0:
        t_times = []
        for _ in range(N_TIMING):
            t0 = time.perf_counter()
            st = critband.silverman_test(data, k=2, n_resamples=99)
            t_times.append(time.perf_counter() - t0)
        result["st_t_mean"] = np.mean(t_times)
        result["st_t_std"] = np.std(t_times, ddof=1)
        result["st_p"] = st.p_value
    else:
        # For non-primary seeds, run once only
        t0 = time.perf_counter()
        st = critband.silverman_test(data, k=2, n_resamples=99)
        result["st_t"] = time.perf_counter() - t0
        result["st_p"] = st.p_value
    
    return result


def run_timing_only(case):
    """Run timing-only benchmark for R5: N_TIMING iterations per function."""
    seed = case["seed_offset"]
    data = generate_data(case, seed)
    
    results = {}
    
    # critical_bandwidth timing
    times = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        critband.critical_bandwidth(data, k=2, method="auto")
        times.append(time.perf_counter() - t0)
    results["critical_bandwidth"] = (np.mean(times), np.std(times, ddof=1))
    
    # find_modes timing
    h_silv = critband.silverman_bandwidth(data)
    times = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        critband.find_modes(data, h=h_silv)
        times.append(time.perf_counter() - t0)
    results["find_modes"] = (np.mean(times), np.std(times, ddof=1))
    
    # dip_test timing
    times = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        critband.dip_test(data)
        times.append(time.perf_counter() - t0)
    results["dip_test"] = (np.mean(times), np.std(times, ddof=1))
    
    return case["name"], results


def main():
    print("=" * 80)
    print("R3 + R5: Parallel multiple-seed benchmark + timing error bars")
    print("=" * 80)
    print(f"\nPlatform: {os.uname().machine}, {os.cpu_count()} cores")
    print(f"Cases: {len(CASES)} × {N_SEEDS} seeds = {len(CASES) * N_SEEDS} runs")
    
    # ========================================================================
    # R5: Timing-only benchmarks (5 iterations each function)
    # ========================================================================
    print("\n\n--- R5: Performance error bars (N=5 timing iterations) ---\n")
    
    # Run timing on first 3 representative cases (well-separated, barely, small)
    timing_cases = [CASES[0], CASES[2], CASES[10]]
    timing_results = {}
    for case in timing_cases:
        name, tres = run_timing_only(case)
        timing_results[name] = tres
    
    print(f"{'Case':25s} {'Function':25s} {'Mean (s)':>10s} {'SD (s)':>10s}")
    print("-" * 70)
    for name, tres in timing_results.items():
        for func_name, (mean_t, std_t) in tres.items():
            print(f"{name:25s} {func_name:25s} {mean_t:>10.4f} {std_t:>10.4f}")
        print()
    
    # ========================================================================
    # R3: Multiple seeds — parallel across seeds
    # ========================================================================
    print("\n--- R3: Multiple seeds (N=10 per case) — parallel execution ---\n")
    
    # Build task list: (case, seed_idx) for each case × seed
    tasks = []
    for case in CASES:
        for seed_idx in range(N_SEEDS):
            tasks.append((case, seed_idx))
    
    t_start = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_case_seed, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"  FAIL: {task[0]['name']} seed_idx={task[1]}: {e}")
            if completed % 30 == 0:
                elapsed = time.time() - t_start
                print(f"  Progress: {completed}/{len(tasks)} ({elapsed:.0f}s elapsed)")
    
    total_time = time.time() - t_start
    print(f"  Completed: {completed}/{len(tasks)} in {total_time:.0f}s")
    
    # ========================================================================
    # Aggregate results by case
    # ========================================================================
    from collections import defaultdict
    by_case = defaultdict(list)
    for r in all_results:
        by_case[r["case"]].append(r)
    
    print(f"\n{'='*80}")
    print("R3 Results: Multi-seed benchmark summary")
    print(f"{'='*80}\n")
    
    # Header for multi-seed table
    print(f"{'Case':25s} {'n':>5s} {'h_crit mean':>12s} {'h_crit SD':>10s} {'CV(%)':>7s} {'modes':>6s} {'strength':>10s}")
    print("-" * 75)
    
    for case in CASES:
        name = case["name"]
        items = by_case.get(name, [])
        if not items:
            continue
        
        h_vals = np.array([r["h_crit"] for r in items])
        n_modes_vals = [r["n_modes"] for r in items]
        strength_vals = [r["bimodality_strength"] for r in items]
        
        h_mean = np.mean(h_vals)
        h_std = np.std(h_vals, ddof=1)
        cv = h_std / h_mean * 100 if h_mean > 0 else 0
        mode_consistency = all(m == n_modes_vals[0] for m in n_modes_vals)
        
        print(f"{name:25s} {case['n']:>5d} {h_mean:>10.4f}  {h_std:>8.4f} {cv:>6.2f}% {n_modes_vals[0]:>4d} {'✓' if mode_consistency else '✗':>5s} {np.mean(strength_vals):>8.2f}")
    
    print(f"\nCV(%) < 1% → h_crit is stable across seeds")
    print(f"modes ✓ → all 10 seeds return same mode count")
    
    # ========================================================================
    # Timing summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("R5 Results: Timing error bars (N=5, mean ± SD)")
    print(f"{'='*80}\n")
    
    for name, tres in timing_results.items():
        print(f"  {name}:")
        for func_name, (mean_t, std_t) in tres.items():
            cv = std_t / mean_t * 100 if mean_t > 0 else 0
            print(f"    {func_name:25s}: {mean_t:.4f} ± {std_t:.4f}s (CV={cv:.1f}%)")
    
    print(f"\nTotal execution: {total_time:.0f}s for {len(tasks)} runs ({len(CASES)} cases × {N_SEEDS} seeds)")
    print("Done.")


if __name__ == "__main__":
    main()
