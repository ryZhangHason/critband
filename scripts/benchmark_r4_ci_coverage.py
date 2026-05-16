"""
R4: Bootstrap CI coverage verification (parallel Monte Carlo).

For 3 representative cases, run 200 Monte Carlo simulations:
1. Generate data from known distribution
2. Compute h_crit (ground truth = h_crit at large n approx)
3. Compute 95% bootstrap CI (n_resamples=99)
4. Check if ground truth h_crit falls in CI → coverage rate

Uses ProcessPoolExecutor for parallel MC iterations.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pola

# Representative cases for CI coverage
CASES = [
    {
        "name": "Well-separated (n=400)",
        "n": 400,
        "components": [(0.5, -2.0, 0.3), (0.5, 2.0, 0.3)],
    },
    {
        "name": "Barely separated (n=600)",
        "n": 600,
        "components": [(0.5, -0.5, 0.4), (0.5, 0.5, 0.4)],
    },
    {
        "name": "Small sample (n=60)",
        "n": 60,
        "components": [(0.5, -2.0, 0.5), (0.5, 2.0, 0.5)],
    },
]

N_MC = 200          # Monte Carlo iterations per case
N_BOOT = 99         # Bootstrap resamples for CI
ALPHA = 0.05        # Nominal significance level
N_GT = 100000       # Use large sample for ground truth h_crit


def estimate_ground_truth(case, seed=42):
    """Estimate 'true' h_crit from a very large sample."""
    rng = np.random.RandomState(seed)
    data = []
    for weight, mean, std in case["components"]:
        n_i = int(round(weight * N_GT))
        data.append(rng.normal(mean, std, n_i))
    data = np.concatenate(data)[:N_GT]
    h_gt, _ = pola.critical_bandwidth(data, k=2, method="auto")
    return h_gt


def one_mc_iteration(args):
    """One MC iteration: generate data, compute h_crit + CI, check coverage.
    
    Returns dict: {covered: bool, h_crit: float, ci_low: float, ci_high: float}
    """
    case, mc_seed, h_gt = args
    
    # Generate data
    rng = np.random.RandomState(mc_seed)
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
    
    # Compute h_crit with bootstrap CI
    try:
        h_crit, success, ci_low, ci_high, se = pola.critical_bandwidth(
            data, k=2, return_ci=True, ci_resamples=N_BOOT, method="auto"
        )
        covered = (ci_low <= h_gt <= ci_high) if success else False
        ci_width = ci_high - ci_low
    except Exception:
        covered = False
        h_crit = np.nan
        ci_low = np.nan
        ci_high = np.nan
        ci_width = np.nan
    
    return {
        "covered": covered,
        "h_crit": h_crit,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
    }


def main():
    print("=" * 70)
    print("R4: Bootstrap CI Coverage Verification")
    print("=" * 70)
    print(f"Platform: {os.uname().machine}, {os.cpu_count()} cores")
    print(f"MC iterations per case: {N_MC}")
    print(f"Bootstrap resamples: {N_BOOT}")
    print(f"Nominal coverage: {1 - ALPHA:.0%}")
    
    for case in CASES:
        print(f"\n--- {case['name']} ---")
        
        # Step 1: Estimate ground truth h_crit from large sample
        print(f"  Estimating ground truth h_crit (n={N_GT})...", end=" ", flush=True)
        h_gt = estimate_ground_truth(case)
        print(f"h_crit(∞) = {h_gt:.4f}")
        
        # Step 2: Run Monte Carlo in parallel
        args_list = [(case, i, h_gt) for i in range(N_MC)]
        
        results = []
        t0 = time.time()
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(one_mc_iteration, a): a for a in args_list}
            completed = 0
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"    FAIL iteration: {e}")
                completed += 1
                if completed % 50 == 0:
                    print(f"    Progress: {completed}/{N_MC} ({time.time()-t0:.0f}s)")
        
        elapsed = time.time() - t0
        
        # Step 3: Compute actual coverage
        h_crits = np.array([r["h_crit"] for r in results])
        coverages = np.array([r["covered"] for r in results])
        ci_widths = np.array([r["ci_width"] for r in results if not np.isnan(r["ci_width"])])
        
        n_valid = np.sum(~np.isnan(h_crits))
        actual_coverage = np.mean(coverages)
        h_crit_mean = np.mean(h_crits[~np.isnan(h_crits)])
        h_crit_std = np.std(h_crits[~np.isnan(h_crits)], ddof=1)
        ci_width_mean = np.mean(ci_widths) if len(ci_widths) > 0 else np.nan
        
        print(f"\n  Results ({elapsed:.0f}s, {n_valid}/{N_MC} valid):")
        print(f"    Ground truth h_crit(∞):    {h_gt:.4f}")
        print(f"    Mean estimated h_crit:      {h_crit_mean:.4f} ± {h_crit_std:.4f}")
        print(f"    Actual coverage rate:       {actual_coverage:.1%}  (nominal: {1-ALPHA:.0%})")
        print(f"    Mean CI width:              {ci_width_mean:.4f}")
        print(f"    Coverage gap:               {actual_coverage - (1-ALPHA):+.1%}")
        
        if actual_coverage < 0.90:
            print(f"    ⚠ WARNING: Coverage well below nominal. CI may be anti-conservative.")
        elif actual_coverage < 0.93:
            print(f"    ⚠ NOTE: Coverage slightly below nominal. Acceptable for bootstrap.")
        else:
            print(f"    ✓ Coverage acceptable.")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
