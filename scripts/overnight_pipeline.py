#!/usr/bin/env python3
"""
Overnight pipeline: R3 → R5 → R4 → R9 (serial, with tqdm progress).

Serial execution plan (no ProcessPoolExecutor — deadlocks in Hermes env):
  1. R5: Timing benchmarks (3 functions × 3 cases × 3 iterations) [~1 min]
  2. R3: Multi-seed benchmark (12 cases × 5 seeds, no bootstrap) [~2 min]
  3. R9: Solver comparison (12 cases × 20 seeds, no bootstrap) [~3 min]
  4. R4: CI coverage (1 case × 200 MC × 99 bootstrap) [~30 min]
  5. Update .tex tables from results
  6. Compile final PDF

Usage: uv run python scripts/overnight_pipeline.py
"""

import time
import numpy as np
import sys, os, subprocess, json, textwrap
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import critband

PROJECT_ROOT = "/Users/qadz/Downloads/Polarization-CBW"
TEX_DIR = f"{PROJECT_ROOT}/02_projects/01_polarization_manuscript/05_writing/arxiv_v1"
TEX_PATH = f"{TEX_DIR}/pola_arxiv.tex"
RESULTS_FILE = f"{TEX_DIR}/benchmark_results.json"

# ========================================================================== #
# Benchmark cases
# ========================================================================== #
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

N_R3_SEEDS = 3
N_R9_SEEDS = 10
N_R4_CASES = 1  # just well-separated
N_TIMING_ITERS = 3


def gen(case, seed):
    rng = np.random.RandomState(seed)
    data = []
    for w, m, s in case["components"]:
        n_i = max(1, int(round(w * case["n"])))
        data.append(rng.normal(m, s, n_i))
    data = np.concatenate(data)[:case["n"]]
    return data


def eta_str(elapsed, fraction):
    """Return ETA string given elapsed seconds and fraction complete."""
    if fraction <= 0:
        return "?"
    remaining = elapsed / fraction * (1 - fraction)
    m, s = divmod(int(remaining), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ========================================================================== #
# R5: Timing benchmarks
# ========================================================================== #
def run_r5():
    print("\n" + "=" * 60)
    print("R5: Timing benchmarks (serial, N=3)")
    print("=" * 60)
    
    results = {}
    t0 = time.time()
    
    for ci, case in enumerate(CASES[:3]):  # first 3 cases
        data = gen(case, 42)
        h_silv = critband.silverman_bandwidth(data)
        
        for func_name, func, args in [
            ("critical_bandwidth", critband.critical_bandwidth, ((data,), {"method": "auto"})),
            ("find_modes", critband.find_modes, ((data,), {"h": h_silv})),
            # dip_test excluded: O(n²) Hartigan's dip, takes ~75 min per call on n=400-600 data
        ]:
            times = []
            for _ in range(N_TIMING_ITERS):
                t = time.perf_counter()
                func(*args[0], **args[1])
                times.append(time.perf_counter() - t)
            mean_t, std_t = np.mean(times), np.std(times, ddof=1)
            print(f"  {case['name']:25s} {func_name:25s}: {mean_t:.4f} ± {std_t:.4f}s")
            results[f"{case['name']}_{func_name}"] = {"mean": mean_t, "std": std_t}
        
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s elapsed, ETA: {eta_str(elapsed, (ci+1)*3/(len(CASES[:3])*3))}]")
    
    print(f"  ✓ R5 done in {time.time()-t0:.0f}s")
    return results


# ========================================================================== #
# R3: Multi-seed benchmark
# ========================================================================== #
def run_r3():
    print("\n" + "=" * 60)
    print(f"R3: Multi-seed benchmark ({len(CASES)} cases × {N_R3_SEEDS} seeds)")
    print("=" * 60)
    
    results = {}
    t0 = time.time()
    total = len(CASES) * N_R3_SEEDS
    done = 0
    
    print(f"{'Case':25s} {'n':>5s} {'h_crit mean':>12s} {'h_crit SD':>10s} {'CV(%)':>7s} {'modes':>6s} {'strength':>10s}")
    print("-" * 75)
    
    for ci, case in enumerate(CASES):
        h_vals, n_modes_list, bs_list = [], [], []
        for si in range(N_R3_SEEDS):
            data = gen(case, 100 + si)
            h_crit, _ = critband.critical_bandwidth(data, k=2, method="auto")
            h_vals.append(h_crit)
            h_silv = critband.silverman_bandwidth(data)
            mr = critband.find_modes(data, h=h_silv)
            n_modes_list.append(mr.n_modes)
            bs = critband.bimodality_strength(data)
            bs_list.append(bs.strength_score)
            done += 1
        
        h_mean = float(np.mean(h_vals))
        h_std = float(np.std(h_vals, ddof=1))
        cv = h_std / h_mean * 100 if h_mean > 0 else 0
        mode_ok = all(m == n_modes_list[0] for m in n_modes_list)
        
        print(f"{case['name']:25s} {case['n']:>5d} {h_mean:>10.4f}  {h_std:>8.4f} {cv:>6.2f}% {n_modes_list[0]:>4d} {'✓' if mode_ok else '✗':>5s} {float(np.mean(bs_list)):>8.2f}")
        
        results[case['name']] = {
            "h_crit_mean": round(h_mean, 4),
            "h_crit_std": round(h_std, 4),
            "cv_pct": round(cv, 2),
            "n_modes": n_modes_list[0],
            "mode_consistent": mode_ok,
            "strength": round(float(np.mean(bs_list)), 2),
        }
        
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s elapsed, {done}/{total}, ETA: {eta_str(elapsed, done/total)}]")
    
    print(f"  ✓ R3 done in {time.time()-t0:.0f}s")
    return results


# ========================================================================== #
# R9: Solver comparison
# ========================================================================== #
def run_r9():
    print("\n" + "=" * 60)
    print(f"R9: Solver comparison ({len(CASES)} cases × {N_R9_SEEDS} seeds)")
    print("=" * 60)
    
    results = {}
    t0 = time.time()
    total = len(CASES) * N_R9_SEEDS
    done = 0
    
    print(f"{'Case':25s} {'auto h':>10s} {'binary h':>10s} {'brent h':>10s} {'fallback%':>10s}")
    print("-" * 75)
    
    for case in CASES:
        name = case["name"]
        auto_h_list, binary_h_list, brent_h_list = [], [], []
        fallbacks = 0
        
        for si in range(N_R9_SEEDS):
            data = gen(case, 2000 + si)  # different seed range
            
            auto_h, _ = critband.critical_bandwidth(data, k=2, method="auto")
            auto_h_list.append(auto_h)
            
            binary_h, _ = critband.critical_bandwidth(data, k=2, method="binary")
            binary_h_list.append(binary_h)
            
            brent_h, _ = critband.critical_bandwidth(data, k=2, method="brent")
            brent_h_list.append(brent_h)
            
            # Detect fallback: auto ≈ binary (not auto ≈ brent)
            if abs(auto_h - binary_h) < abs(auto_h - brent_h):
                fallbacks += 1
            
            done += 1
        
        ah = float(np.mean(auto_h_list))
        bh = float(np.mean(binary_h_list))
        brh = float(np.mean(brent_h_list))
        fb_pct = fallbacks / N_R9_SEEDS * 100
        
        print(f"{name:25s} {ah:>10.4f} {bh:>10.4f} {brh:>10.4f} {fb_pct:>8.1f}%")
        
        results[name] = {
            "auto_h_mean": round(ah, 4),
            "binary_h_mean": round(bh, 4),
            "brent_h_mean": round(brh, 4),
            "fallback_pct": round(fb_pct, 1),
        }
        
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s elapsed, {done}/{total}, ETA: {eta_str(elapsed, done/total)}]")
    
    print(f"  ✓ R9 done in {time.time()-t0:.0f}s")
    return results


# ========================================================================== #
# R4: CI coverage (well-separated only, 200 MC × 99 bootstrap)
# ========================================================================== #
def run_r4():
    N_MC = 200
    N_BOOT = 99
    print(f"\n{'=' * 60}\nR4: CI coverage (well-separated, {N_MC} MC × {N_BOOT} bootstrap)\n{'=' * 60}")
    case = CASES[0]  # well-separated
    ALPHA = 0.05
    
    t0 = time.time()
    
    # Ground truth from large sample
    h_gt, _ = critband.critical_bandwidth(gen(case, 42), k=2, method="auto")
    # Large-n ground truth — use existing data
    rng = np.random.RandomState(42)
    big_data = np.concatenate([rng.normal(m, s, 10000) for _, m, s in case["components"]])
    h_gt, _ = critband.critical_bandwidth(big_data, k=2, method="auto")
    print(f"  Ground truth h_crit (n=20000): {h_gt:.4f}")
    
    covered = 0
    n_valid = 0
    
    for mc_i in range(N_MC):
        data = gen(case, 10000 + mc_i)
        try:
            h, success, cl, ch, se = critband.critical_bandwidth(
                data, k=2, return_ci=True, ci_resamples=N_BOOT, method="auto"
            )
            if success and not np.isnan(cl):
                n_valid += 1
                if cl <= h_gt <= ch:
                    covered += 1
        except Exception:
            pass
        
        if (mc_i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (mc_i + 1) / elapsed if elapsed > 0 else 0
            remaining = (N_MC - mc_i - 1) / rate if rate > 0 else 0
            cur_cov = covered / n_valid * 100 if n_valid > 0 else 0
            print(f"  [{mc_i+1}/{N_MC}] {elapsed:.0f}s elapsed, "
                  f"{remaining:.0f}s remaining, "
                  f"current coverage: {cur_cov:.1f}% "
                  f"(valid: {n_valid})")
    
    actual_coverage = covered / n_valid * 100 if n_valid > 0 else 0
    print(f"\n  Results ({n_valid}/{N_MC} valid):")
    print(f"    Actual coverage: {actual_coverage:.1f}% (nominal: {95}%)")
    print(f"    Coverage gap: {actual_coverage - 95:+.1f}%")
    
    print(f"  ✓ R4 done in {time.time()-t0:.0f}s")
    return {"coverage_pct": round(actual_coverage, 1), "n_valid": n_valid, "h_gt": round(h_gt, 4)}


# ========================================================================== #
# Compile PDF
# ========================================================================== #
def compile_pdf():
    print("\n" + "=" * 60)
    print("Compiling final PDF")
    print("=" * 60)
    
    env = os.environ.copy()
    env.update({
        "TEXMFROOT": os.path.expanduser("~/tinytex_install"),
        "TEXMFCNF": os.path.expanduser("~/tinytex_install/texmf-dist/web2c"),
        "TEXMFDIST": os.path.expanduser("~/tinytex_install/texmf-dist"),
        "TEXMFLOCAL": os.path.expanduser("~/tinytex_install/texmf-local"),
        "TEXMFSYSCONFIG": os.path.expanduser("~/tinytex_install/texmf-config"),
        "TEXMFSYSVAR": os.path.expanduser("~/tinytex_install/texmf-var"),
        "PATH": f"{os.path.expanduser('~/tinytex_install/bin/universal-darwin')}:{env.get('PATH', '')}"
    })
    
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "pola_arxiv"],
                       cwd=TEX_DIR, env=env, capture_output=True, timeout=60)
    subprocess.run(["bibtex", "pola_arxiv"],
                   cwd=TEX_DIR, env=env, capture_output=True, timeout=30)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "pola_arxiv"],
                       cwd=TEX_DIR, env=env, capture_output=True, timeout=60)
    
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "pola_arxiv"],
                       cwd=TEX_DIR, env=env, capture_output=True, text=True, timeout=60)
    
    if "Output written" in r.stdout:
        out_line = [l for l in r.stdout.split("\n") if "Output written" in l][0]
        print(f"  ✓ {out_line}")
    else:
        print("  ✗ Compilation may have issues")
    
    return r.stdout


# ========================================================================== #
# Main
# ========================================================================== #
def main():
    print("=" * 60)
    print("critband overnight pipeline: R5 → R3 → R9 → R4")
    print("=" * 60)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Platform: arm64, serial execution")
    
    all_results = {}
    pipeline_t0 = time.time()
    
    # R5: timing (~1 min)
    all_results["R5"] = run_r5()
    
    # R3: multi-seed (~2 min)
    all_results["R3"] = run_r3()
    
    # R9: solver comparison (~3 min)
    all_results["R9"] = run_r9()
    
    # R4: CI coverage (~30 min)
    all_results["R4"] = run_r4()
    
    # Save all results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Compile
    compile_pdf()
    
    total = time.time() - pipeline_t0
    m, s = divmod(int(total), 60)
    h, m = divmod(m, 60)
    print(f"\nPipeline complete in {h:02d}:{m:02d}:{s:02d}" if h else f"\nPipeline complete in {m:02d}:{s:02d}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
