#!/usr/bin/env python3
"""Check result files (finite values, no local paths, key invariants) and that the publication figures are present."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FIGURES = tuple(f"Figure{i}" for i in range(1, 7)) + ("FigureS1",)


def _check_finite(obj: object, path: str, bad: list) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}", bad)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]", bad)


def main() -> None:
    res_dir = REPO_ROOT / "results"
    required_jsons = (
        "unified_benchmark_results.json",
        "homeodomain_familycode_repro.json",
        "homeodomain_hgbdt_b49.json",
        "homeodomain_hgbdt_fullk2000_bridge.json",
        "homeodomain_hgbdt_fullk2000_b19.json",
        "bhlh_cross_family_results.json",
        "hoxc10_and_upbm_surrogates.json",
        "deep_dnashape_comparison.json",
        "verified_references.json",
    )
    for name in required_jsons:
        assert (res_dir / name).exists(), f"Missing required result file: {name}"

    for p in sorted(res_dir.glob("*.json")):
        raw = p.read_text(encoding="utf-8")
        assert not re.search(r"/(?:mnt/[a-z]|home/[^/\s\"']+|Users/[^/\s\"']+)/", raw), f"Leaked local path in {p.name}"
        bad: list = []
        _check_finite(json.loads(raw), p.name, bad)
        # Two known missing confidence values in the recognition-code reproduction are expected.
        unexpected = [b for b in bad if "pred_confidence" not in b]
        assert not unexpected, f"Non-finite values in {p.name}: {unexpected[:5]}"
        print(f"[PASS] {p.name}")

    # Key invariants of the 19-surrogate tree analysis and the Deep DNAshape comparison
    b19 = json.loads((res_dir / "homeodomain_hgbdt_fullk2000_b19.json").read_text(encoding="utf-8"))
    assert b19["parity_check_vs_b9_bridge"]["passed"] is True
    assert b19["B_per_family"] == 19 and abs(b19["p_min_possible"] - 0.05) < 1e-12
    print("[PASS] 19-surrogate analysis: parity with the 9-surrogate run and minimum P = 0.05")

    deep = json.loads((res_dir / "deep_dnashape_comparison.json").read_text(encoding="utf-8"))
    assert deep["deep_dnashape_verification"]["rc_equivariance_max_overall"] == 0.0
    assert deep["deep_dnashape_verification"]["pentamer_L2_mean_pearson_7ch"] > 0.96
    print("[PASS] Deep DNAshape: reverse-complement equivariance and agreement with DNAshapeR")

    fig_dir = REPO_ROOT / "paper" / "figures"
    for stem in EXPECTED_FIGURES:
        for ext in ("png", "pdf"):
            f = fig_dir / f"{stem}.{ext}"
            assert f.exists() and f.stat().st_size > 1000, f"Missing or empty figure: {f}"
    print("[PASS] Figures 1-6 and S1 (PNG and PDF) present")
    print("All checks passed.")


if __name__ == "__main__":
    main()
