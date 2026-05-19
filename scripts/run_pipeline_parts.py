#!/usr/bin/env python3
"""Split overnight pipeline runner.
Part 1 (fast): R5 + R3 + R9 — run foreground
Part 2 (slow): R4 (CI coverage) — run background, notifies when done
Part 3: inject results into .tex, compile PDF
"""
import subprocess, sys, os, json, time

PROJECT = "critband"
TEX_DIR = f"{PROJECT}/02_projects/01_polarization_manuscript/05_writing/arxiv_v1"

def run_fast():
    """Run R5+R3+R9 sequentially in same process."""
    sys.path.insert(0, PROJECT)
    import numpy as np, critband
    
    from overnight_pipeline import run_r5, run_r3, run_r9
    
    all_results = {}
    t0 = time.time()
    
    print("=" * 60)
    print("PART 1: R5 + R3 + R9 (fast, ~5 min)")
    print("=" * 60)
    
    all_results["R5"] = run_r5()
    all_results["R3"] = run_r3()
    all_results["R9"] = run_r9()
    
    # Save partial results
    with open(f"{TEX_DIR}/benchmark_results_part1.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nPART 1 done in {elapsed:.0f}s")
    print(f"Results saved to benchmark_results_part1.json")
    return all_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "part2":
        # Part 2: R4 only (slow, background)
        sys.path.insert(0, PROJECT)
        import numpy as np, critband
        from overnight_pipeline import run_r4
        
        print("=" * 60)
        print("PART 2: R4 CI coverage (slow, ~30 min)")
        print("=" * 60)
        
        r4_results = run_r4()
        
        # Load part 1 results, merge, save
        try:
            with open(f"{TEX_DIR}/benchmark_results_part1.json") as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = {}
        all_results["R4"] = r4_results
        
        with open(f"{TEX_DIR}/benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll results saved to benchmark_results.json")
    else:
        run_fast()
