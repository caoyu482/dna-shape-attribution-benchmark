#!/usr/bin/env python3
"""
Experiment 1a: High-Resolution HGBDT Matched Surrogate Control (`biophys_inductive_gfm/v7_1`).

Addresses Reviewer Critique #1 from v7 Stage 3b:
  - Evaluates matched pure-sequence `HGBDT_Seq_Only`, `F_true`, and high-resolution
    surrogate null banks on BOTH `dz_f70` AND `dz_spec_f70`:
      * `ShiftGC_ExactCRT_WS_G2` (`B=49`, `p_min=0.020`)
      * `Sparse4mer_K12` (`B=49`, `p_min=0.020`)
      * Sparsity ladder `Sparse4mer K in {4, 8, 24, 48}` (`B=19`, `p_min=0.050` on `dz_f70`)
  - Uses v7's deterministic WT-rank 500/mut training subsampling (`CFG_HGB_SURR60`)
    evaluated across ALL finite `K=2,000` test probes per mutant under 30-fold LOGO.
  - Enforces strict startup parity gate against v7 `F_true == 0.086020` (`|diff| <= 1e-4`).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V7_1_SRC = ROOT / "v7_1" / "src"
V7_ART = ROOT / "v7" / "artifacts"
V7_1_ART = ROOT / "v7_1" / "artifacts"
if str(V7_1_SRC) not in sys.path:
    sys.path.insert(0, str(V7_1_SRC))

import hgbdt_surrogate_core as hsc  # noqa: E402
import kock_pbm_v7 as kv7  # noqa: E402


def main() -> None:
    t_all = time.time()
    V7_1_ART.mkdir(parents=True, exist_ok=True)
    ckpt_root = V7_1_ART / ".hgb_parts_exp1a"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    avail = hsc._avail_ram_mb()
    print(f"[Exp 1a] Building fold cache for dz_f70 (avail_ram={avail:.0f}MB)...", flush=True)
    cache_dz = hsc.build_v7_hgbdt_fold_cache(target_key="dz_f70", n_sub_per_mut=500)

    print("[Exp 1a] Building shared unlabeled surrogate bank (seed=2026071, B=49 CRT/K12 + B=19 K4/8/24/48)...", flush=True)
    bank, slices = hsc.build_hgbdt_surrogate_bank(
        cache_dz["c5_unlabeled_wt"],
        seed=2026071,
        b_crt=49,
        b_k12=49,
        b_kladder=19,
        k_other=(4, 8, 24, 48),
    )
    rc_err = float(np.max(np.abs(bank - bank[:, hsc.ng4.RC_5, :])))
    assert rc_err <= 1e-12, f"Surrogate bank RC asymmetry error too large: {rc_err}"
    print(f"  Bank shape={bank.shape}, max RC asym err={rc_err:.2e}", flush=True)

    # Startup Parity Gate against v7 Stage 3b surrogate `F_true = 0.08601983`
    v7_3b = json.loads((V7_ART / "ckpt_v7_step3b_hgbdt.json").read_text(encoding="utf-8"))
    v7_true_gm = float(v7_3b["primary_logo_hgbdt_matched_surrogate_check"]["true_gene_mean_spearman"])

    n_workers = 6 if hsc._avail_ram_mb() >= 7000.0 else 4
    print(f"[Exp 1a] Running Startup Parity Gate on dz_f70 (`Seq_Only` + `F_true`, n_workers={n_workers})...", flush=True)
    seq_dz, ftrue_dz_arr, t_gate = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_dz,
        bank[0:1],
        hsc.CFG_HGB_SURR60,
        include_seq_only=True,
        n_workers=n_workers,
        checkpoint_dir=ckpt_root / "dz_f70_gate",
        checkpoint_prefix="gate",
        progress_label="Exp1a-ParityGate",
    )
    assert seq_dz is not None
    true_dz_rhos = ftrue_dz_arr[0]
    n_genes = int(cache_dz["n_genes"])
    mut_g = cache_dz["mut_gene_idx"]
    gate_gm = float(np.mean([np.mean(true_dz_rhos[mut_g == g]) for g in range(n_genes)]))
    gate_diff = abs(gate_gm - v7_true_gm)
    print(f"  [Parity Gate] Exp1a F_true={gate_gm:.6f} vs v7={v7_true_gm:.6f} (|diff|={gate_diff:.2e}, t={t_gate:.1f}s)", flush=True)
    if gate_diff > 1e-4:
        raise RuntimeError(f"PARITY GATE FAILED: |{gate_gm:.6f} - {v7_true_gm:.6f}| = {gate_diff:.2e} > 1e-4")

    # Full dz_f70 null bank evaluation (indices 1..end -> B=49 CRT + B=49 K12 + 4*19 ladder = 174 null tables)
    print("[Exp 1a] Running dz_f70 174 surrogate tables (B=49 CRT/K12 + B=19 K4/8/24/48)...", flush=True)
    _, null_dz_arr, t_dz_null = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_dz,
        bank[1:],
        hsc.CFG_HGB_SURR60,
        include_seq_only=False,
        n_workers=n_workers,
        checkpoint_dir=ckpt_root / "dz_f70_nulls",
        checkpoint_prefix="dz",
        progress_label="Exp1a-dz_f70",
    )
    null_dict_dz = {}
    for fk, sl in slices.items():
        # slices start at index 1 in `bank`, so offset by -1 into `null_dz_arr`
        null_dict_dz[fk] = null_dz_arr[slice(sl.start - 1, sl.stop - 1)]

    sum_dz = hsc.summarize_hgbdt_surrogate_family(
        seq_dz,
        true_dz_rhos,
        null_dict_dz,
        mut_g,
        cache_dz["mut_segments"],
        cache_dz["mut_kock_strata"],
        n_boot=5000,
        boot_seed=42,
    )
    sum_dz["runtime_seconds"] = float(t_gate + t_dz_null)

    # Free dz_f70 cache before building dz_spec_f70 cache to keep RSS flat
    c5_unlabeled = cache_dz["c5_unlabeled_wt"]
    del cache_dz
    gc.collect()

    # Secondary endpoint: dz_spec_f70 (Seq_Only + F_true + B=49 ShiftGC + B=49 Sparse4mer_K12)
    print(f"[Exp 1a] Building fold cache for dz_spec_f70 (avail_ram={hsc._avail_ram_mb():.0f}MB)...", flush=True)
    cache_sp = hsc.build_v7_hgbdt_fold_cache(target_key="dz_spec_f70", n_sub_per_mut=500)
    sp_indices = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.arange(slices["ShiftGC_ExactCRT_WS_G2"].start, slices["ShiftGC_ExactCRT_WS_G2"].stop),
            np.arange(slices["Sparse4mer_K12"].start, slices["Sparse4mer_K12"].stop),
        ]
    )
    seq_sp, sp_arr, t_sp = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_sp,
        bank[sp_indices],
        hsc.CFG_HGB_SURR60,
        include_seq_only=True,
        n_workers=n_workers,
        checkpoint_dir=ckpt_root / "dz_spec_f70",
        checkpoint_prefix="sp",
        progress_label="Exp1a-dz_spec_f70",
    )
    assert seq_sp is not None
    sum_sp = hsc.summarize_hgbdt_surrogate_family(
        seq_sp,
        sp_arr[0],
        {
            "ShiftGC_ExactCRT_WS_G2": sp_arr[1:50],
            "Sparse4mer_K12": sp_arr[50:99],
        },
        cache_sp["mut_gene_idx"],
        cache_sp["mut_segments"],
        cache_sp["mut_kock_strata"],
        n_boot=5000,
        boot_seed=42,
    )
    sum_sp["runtime_seconds"] = float(t_sp)

    # Check zero NaNs
    for arr_name, arr in [
        ("seq_dz", seq_dz),
        ("true_dz", true_dz_rhos),
        ("null_dz", null_dz_arr),
        ("seq_sp", seq_sp),
        ("sp_arr", sp_arr),
    ]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values detected in {arr_name}")

    out = {
        "stage": "v7_1_exp1a_hgbdt_matched_surrogate_b49",
        "config": hsc.CFG_HGB_SURR60,
        "subsampling": cache_sp["subsampling"],
        "seed_bank": 2026071,
        "shared_unlabeled_surrogate_bank_across_dz_and_dz_spec": True,
        "surrogate_bank_rc_max_err": rc_err,
        "parity_gate_dz_f70": {
            "v7_true_gene_mean_spearman": v7_true_gm,
            "v7_1_true_gene_mean_spearman": gate_gm,
            "abs_diff": gate_diff,
            "passed": True,
        },
        "dz_f70": sum_dz,
        "dz_spec_f70": sum_sp,
        "per_mutant_arrays": {
            "mut_alleles": [str(x) for x in cache_sp["mut_alleles"]],
            "mut_gene_idx": [int(x) for x in cache_sp["mut_gene_idx"]],
            "dz_f70_seq_only": [float(x) for x in seq_dz],
            "dz_f70_true": [float(x) for x in true_dz_rhos],
            "dz_spec_f70_seq_only": [float(x) for x in seq_sp],
            "dz_spec_f70_true": [float(x) for x in sp_arr[0]],
        },
        "total_runtime_seconds": float(time.time() - t_all),
        "n_workers": int(n_workers),
    }

    p_out = V7_1_ART / "ckpt_v7_1_exp1a_hgbdt_b49.json"
    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE Exp 1a] Saved {p_out} in {out['total_runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
