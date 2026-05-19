"""Fast critband benchmark — no excess_mass (too slow), minimal bootstrap."""
import time
import numpy as np
from critband import critical_bandwidth, dip_test, find_modes, bimodality_strength, silverman_bandwidth
from critband.benchmark import BENCHMARK_CASES

print("case,desc,n,h_ref,h_crit,time_crit,ok,dip_stat,dip_time,n_modes,strength,score")
for name, case in BENCHMARK_CASES.items():
    x = case.generator(42)
    n = len(x)

    # critical_bandwidth (binary, high precision)
    t0 = time.perf_counter()
    h_pola, ok = critical_bandwidth(x, method="binary", tol=1e-8, max_iter=500)
    tc = time.perf_counter() - t0

    # dip_test (49 bootstrap, fast)
    t0 = time.perf_counter()
    dt = dip_test(x, n_boot=49)
    td = time.perf_counter() - t0

    # find_modes at Silverman h
    hs = silverman_bandwidth(x)
    fm = find_modes(x, hs)

    # bimodality_strength (no bootstrap)
    bs = bimodality_strength(x)

    print(f"{name},{case.description},{n},{case.h_crit_expected:.6f},"
          f"{h_pola:.6f},{tc:.4f},{ok},{dt.dip:.6f},{td:.4f},"
          f"{fm.n_modes},{bs.strength},{bs.strength_score:.4f}")
