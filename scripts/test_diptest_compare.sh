#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

cat <<'PY' | "$PYTHON_BIN"
import importlib.util
import json
import sys

from critband.benchmark import get_all_benchmark_cases
from critband.bandwidth import _compute_dip_statistic

print(f"python={sys.version.split()[0]}")
print(f"diptest_installed={importlib.util.find_spec('diptest') is not None}")

from diptest import diptest as ext_diptest

rows = []
for case in get_all_benchmark_cases():
    x = case.generator(42)
    ours_dip = float(_compute_dip_statistic(x))
    ext_dip = ext_diptest(x, boot_pval=False)
    ext_dip = float(ext_dip[0] if isinstance(ext_dip, tuple) else getattr(ext_dip, "dip"))
    rows.append({
        "case": case.name,
        "ours_dip": ours_dip,
        "ext_dip": ext_dip,
        "abs_diff": abs(ours_dip - ext_dip),
    })

print(json.dumps(rows, indent=2))
print(f"max_abs_dip_diff={max(r['abs_diff'] for r in rows):.6g}")

spot = []
for name in ["well_separated_equal_var", "barely_separated", "trimodal"]:
    case = next(c for c in get_all_benchmark_cases() if c.name == name)
    x = case.generator(42)
    # Cheap p-value spot check: use a small number of bootstrap samples
    # in our implementation and the external backend.
    from critband import dip_test
    ours = dip_test(x, n_boot=19, random_state=42)
    ext = ext_diptest(x, boot_pval=True, n_boot=999, seed=42)
    if isinstance(ext, tuple):
        ext_dip, ext_p = ext[0], ext[1]
    else:
        ext_dip = getattr(ext, "dip")
        ext_p = getattr(ext, "pval", getattr(ext, "p_value"))
    spot.append({
        "case": name,
        "ours_p": float(getattr(ours, "p_value", getattr(ours, "pvalue"))),
        "ext_p": float(ext_p),
        "abs_p_diff": abs(float(getattr(ours, "p_value", getattr(ours, "pvalue"))) - float(ext_p)),
        "ext_dip": float(ext_dip),
    })

print(json.dumps(spot, indent=2))
print(f"max_abs_p_diff_spot={max(s['abs_p_diff'] for s in spot):.6g}")
PY
