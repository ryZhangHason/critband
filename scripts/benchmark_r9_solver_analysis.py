"""
R9: Trough-ratio continuity analysis + hybrid solver fallback rate.

Benchmarks across 12 cases × 50 seeds (600 runs):
1. r(h) continuity check — detect discontinuities when peak height ordering swaps
2. Hybrid solver fallback rate — count how often Brent's method fails and binary fallback is used
3. Compare 3 solver methods: 'auto', 'binary', 'brent' — convergence rate and accuracy
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import critband

# Benchmark cases (same 12 as paper)
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

N_SEEDS = 50  # More seeds for reliable fallback statistics


def generate_data(case, seed):
    """Generate data from mixture case with given seed."""
    rng = np.random.RandomState(seed)
    data = []
    for weight, mean, std in case["components"]:
        n_i = max(1, int(round(weight * case["n"])))
        data.append(rng.normal(mean, std, n_i))
    data = np.concatenate(data)
    if len(data) > case["n"]:
        data = data[:case["n"]]
    elif len(data) < case["n"]:
        extra = rng.choice(len(data), case["n"] - len(data))
        data = np.concatenate([data, data[extra]])
    return data


def check_r_h_CONTINUITY(data, h_range=None):
    """
    Check r(h) continuity by evaluating r(h) at many points and looking for jumps.
    r(h) = f_valley / max(f_peak1, f_peak2)
    
    Returns dict with:
    - n_discontinuities: number of h values where peak ordering swaps
    - r_values: list of (h, r) pairs
    """
    if h_range is None:
        h_silv = critband.silverman_bandwidth(data)
        h_range = np.linspace(h_silv * 0.1, h_silv * 5.0, 200)
    
    prev_peak_order = None
    discontinuities = 0
    r_vals = []
    
    for h in h_range:
        # Evaluate KDE at grid points
        grid = np.linspace(data.min() - 3*h, data.max() + 3*h, 500)
        kde = critband.gaussian_kde(data, grid, h)
        
        # Find peaks and valley
        from scipy.signal import find_peaks
        peaks, peak_props = find_peaks(kde, prominence=1e-6)
        
        if len(peaks) < 2:
            r_vals.append((h, 1.0))  # unimodal → r=1
            prev_peak_order = None
            continue
        
        # Sort peaks by height
        peak_heights = kde[peaks]
        sorted_idx = np.argsort(peak_heights)[::-1]
        top_two = sorted_idx[:2]
        current_order = tuple(peaks[top_two])
        
        # Check if peak order swapped
        if prev_peak_order is not None and current_order != prev_peak_order:
            discontinuities += 1
        
        # Find valley between the two highest peaks
        p1, p2 = sorted(peaks[top_two[0]], peaks[top_two[1]])
        valley_region = kde[p1:p2+1]
        if len(valley_region) > 2:
            valley_min = valley_region.min()
            peak_max = max(kde[peaks[top_two[0]]], kde[peaks[top_two[1]]])
            r = valley_min / peak_max if peak_max > 0 else 1.0
        else:
            r = 1.0
        
        r_vals.append((h, r))
        prev_peak_order = current_order
    
    return {"n_discontinuities": discontinuities, "r_values": r_vals}


def run_single_case_seed_R9(args):
    """Run solver comparison for one case+seed. No bootstrap — all fast."""
    case, seed_idx = args
    seed = 1000 + seed_idx  # avoid collision with R3 seeds
    data = generate_data(case, seed)
    
    result = {
        "case": case["name"],
        "seed": seed,
        "n": len(data),
    }
    
    # 1. Try all 3 solver methods
    for method in ["auto", "binary", "brent"]:
        try:
            t0 = time.perf_counter()
            h_crit, success = critband.critical_bandwidth(
                data, k=2, method=method, tol=1e-6, max_iter=100
            )
            elapsed = time.perf_counter() - t0
            result[f"{method}_h"] = h_crit
            result[f"{method}_ok"] = success
            result[f"{method}_t"] = elapsed
        except Exception:
            result[f"{method}_h"] = np.nan
            result[f"{method}_ok"] = False
            result[f"{method}_t"] = np.nan
    
    # 2. For 'auto' solver, check if fallback was used
    # The 'auto' method returns success=True even if it falls back to binary.
    # We detect fallback by checking if auto_h ≈ binary_h (fallback) vs auto_h ≈ brent_h (Brent converged)
    auto_h = result.get("auto_h", np.nan)
    binary_h = result.get("binary_h", np.nan)
    brent_h = result.get("brent_h", np.nan)
    
    if not np.isnan(auto_h) and not np.isnan(binary_h) and not np.isnan(brent_h):
        # If auto result matches binary more closely than brent, likely fell back
        diff_auto_binary = abs(auto_h - binary_h)
        diff_auto_brent = abs(auto_h - brent_h)
        if brent_h > 0 and not np.isnan(brent_h):
            result["likely_fallback"] = diff_auto_binary < diff_auto_brent
        else:
            result["likely_fallback"] = False
    else:
        result["likely_fallback"] = False
    
    # 3. Check r(h) continuity (sample at 100 points for speed)
    try:
        cont_info = check_r_h_CONTINUITY(data, h_range=np.linspace(
            result.get("auto_h", critband.silverman_bandwidth(data)) * 0.5,
            result.get("auto_h", critband.silverman_bandwidth(data)) * 3.0,
            100
        ))
        result["n_discontinuities"] = cont_info["n_discontinuities"]
    except Exception:
        result["n_discontinuities"] = -1
    
    return result


def main():
    print("=" * 80)
    print("R9: Trough-ratio continuity analysis + solver fallback rate")
    print("=" * 80)
    print(f"\nPlatform: {os.uname().machine}, {os.cpu_count()} cores")
    print(f"Cases: {len(CASES)} × {N_SEEDS} seeds = {len(CASES) * N_SEEDS} runs")
    print(f"(No bootstrap — fast runs only)\n")
    
    # Build task list: all case×seed combinations
    tasks = [(case, seed_idx) for case in CASES for seed_idx in range(N_SEEDS)]
    N_TOTAL = len(tasks)
    
    t_start = time.time()
    all_results = []
    
    print("Running parallel solver benchmark...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_case_seed_R9, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"  FAIL: {task[0]['name']} seed={task[1]}: {e}")
            if completed % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  Progress: {completed}/{N_TOTAL} ({elapsed:.0f}s elapsed)")
    
    total_time = time.time() - t_start
    print(f"\nCompleted {completed}/{N_TOTAL} in {total_time:.0f}s")
    
    # ========================================================================
    # Aggregate results
    # ========================================================================
    from collections import defaultdict
    by_case = defaultdict(list)
    for r in all_results:
        by_case[r["case"]].append(r)
    
    print(f"\n{'='*80}")
    print("R9 Results: Solver comparison and r(h) analysis")
    print(f"{'='*80}\n")
    
    # --- Solver accuracy comparison ---
    print("### Solver accuracy: agreement between methods\n")
    print(f"{'Case':25s} {'auto h':>10s} {'binary h':>10s} {'brent h':>10s} "
          f"{'auto-binary Δ%':>14s} {'auto-brent Δ%':>14s} {'fallback%':>10s}")
    print("-" * 95)
    
    total_fallbacks = 0
    total_runs = 0
    
    for case in CASES:
        name = case["name"]
        items = by_case.get(name, [])
        if not items:
            continue
        
        auto_h = np.array([r.get("auto_h", np.nan) for r in items])
        binary_h = np.array([r.get("binary_h", np.nan) for r in items])
        brent_h = np.array([r.get("brent_h", np.nan) for r in items])
        fallbacks = sum(1 for r in items if r.get("likely_fallback", False))
        
        auto_mean = np.nanmean(auto_h)
        binary_mean = np.nanmean(binary_h)
        brent_mean = np.nanmean(brent_h)
        
        diff_auto_binary = np.nanmean(np.abs(auto_h - binary_h) / binary_h * 100) if binary_mean > 0 else 0
        diff_auto_brent = np.nanmean(np.abs(auto_h - brent_h) / brent_h * 100) if brent_mean > 0 else 0
        
        n_valid = np.sum(~np.isnan(auto_h))
        fallback_pct = fallbacks / n_valid * 100 if n_valid > 0 else 0
        
        total_fallbacks += fallbacks
        total_runs += n_valid
        
        print(f"{name:25s} {auto_mean:>10.4f} {binary_mean:>10.4f} {brent_mean:>10.4f} "
              f"{diff_auto_binary:>13.2f}% {diff_auto_brent:>13.2f}% {fallback_pct:>8.1f}%")
    
    print(f"\nOverall fallback rate: {total_fallbacks}/{total_runs} = {total_fallbacks/total_runs*100:.2f}%" if total_runs > 0 else "")
    
    # --- Solver timing ---
    print(f"\n### Solver timing (mean ± SD, seeds)\n")
    print(f"{'Case':25s} {'auto time (s)':>14s} {'binary time (s)':>14s} {'brent time (s)':>14s}")
    print("-" * 70)
    
    for case in CASES:
        name = case["name"]
        items = by_case.get(name, [])
        if not items:
            continue
        
        auto_t = np.array([r.get("auto_t", np.nan) for r in items])
        binary_t = np.array([r.get("binary_t", np.nan) for r in items])
        brent_t = np.array([r.get("brent_t", np.nan) for r in items])
        
        auto_m, auto_s = np.nanmean(auto_t), np.nanstd(auto_t)
        binary_m, binary_s = np.nanmean(binary_t), np.nanstd(binary_t)
        brent_m, brent_s = np.nanmean(brent_t), np.nanstd(brent_t)
        
        print(f"{name:25s} {auto_m:>7.4f} ± {auto_s:<.4f} {binary_m:>7.4f} ± {binary_s:<.4f} {brent_m:>7.4f} ± {brent_s:<.4f}")
    
    # --- r(h) discontinuity analysis ---
    print(f"\n### r(h) discontinuity analysis\n")
    print(f"{'Case':25s} {'avg discontinuities':>20s} {'max discontinuities':>20s} {'% runs with jumps':>18s}")
    print("-" * 85)
    
    for case in CASES:
        name = case["name"]
        items = by_case.get(name, [])
        if not items:
            continue
        
        disconts = np.array([r.get("n_discontinuities", -1) for r in items])
        valid = disconts[disconts >= 0]
        
        if len(valid) > 0:
            avg_d = np.mean(valid)
            max_d = np.max(valid)
            pct_with_jumps = np.mean(valid > 0) * 100
            print(f"{name:25s} {avg_d:>18.2f} {max_d:>18.0f} {pct_with_jumps:>16.1f}%")
        else:
            print(f"{name:25s} {'N/A':>18s} {'N/A':>18s} {'N/A':>16s}")
    
    print(f"\nTotal execution: {total_time:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
