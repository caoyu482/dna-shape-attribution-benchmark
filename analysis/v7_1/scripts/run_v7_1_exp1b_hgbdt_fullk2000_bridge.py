#!/usr/bin/env python3
"""
Experiment 1b: Full K=2,000 / 180k-Row HGBDT Direct Bridge (`biophys_inductive_gfm/v7_1`).

Tests whether the exact full-dataset `CFG_HGB_100` lift observed in v7
(`dz_f70`: `0.1006 -> 0.1177`; `dz_spec_f70`: `0.1268 -> 0.1429`) also collapses
under matched `B=9` `ShiftGC_ExactCRT_WS_G2` and `Sparse4mer_K12` surrogates
using the first 9 draws of the shared `seed_bank = 2026071` surrogate bank.
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


def main() -> None:
    t_all = time.time()
    V7_1_ART.mkdir(parents=True, exist_ok=True)
    ckpt_root = V7_1_ART / ".hgb_parts_exp1b"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    v7_3b = json.loads((V7_ART / "ckpt_v7_step3b_hgbdt.json").read_text(encoding="utf-8"))
    t_dz_seq_v7 = float(v7_3b["primary_dz_f70_logo_hgbdt"]["models"]["HGBDT_Mut_x_Seq"]["gene_mean_spearman"])
    t_dz_tru_v7 = float(v7_3b["primary_dz_f70_logo_hgbdt"]["models"]["HGBDT_Mut_x_(Seq + Shape_N1N4)"]["gene_mean_spearman"])
    t_sp_seq_v7 = float(v7_3b["secondary_dz_spec_f70_logo_hgbdt"]["models"]["HGBDT_Mut_x_Seq"]["gene_mean_spearman"])
    t_sp_tru_v7 = float(v7_3b["secondary_dz_spec_f70_logo_hgbdt"]["models"]["HGBDT_Mut_x_(Seq + Shape_N1N4)"]["gene_mean_spearman"])

    print(f"[Exp 1b] Building FULL K=2,000 cache for dz_f70 (avail_ram={hsc._avail_ram_mb():.0f}MB)...", flush=True)
    cache_dz = hsc.build_v7_hgbdt_fold_cache(target_key="dz_f70", n_sub_per_mut=None)
    # Draw slice(0,9) from identical B=49 bank (seed=2026071) for strict paired consistency across Exp 1a and Exp 1b
    bank_full, slices_full = hsc.build_hgbdt_surrogate_bank(
        cache_dz["c5_unlabeled_wt"],
        seed=2026071,
        b_crt=49,
        b_k12=49,
        b_kladder=0,
        k_other=(),
    )
    pick_idx = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.arange(slices_full["ShiftGC_ExactCRT_WS_G2"].start, slices_full["ShiftGC_ExactCRT_WS_G2"].start + 9),
            np.arange(slices_full["Sparse4mer_K12"].start, slices_full["Sparse4mer_K12"].start + 9),
        ]
    )
    bank_19 = bank_full[pick_idx]
    del bank_full, slices_full

    n_workers = 4 if hsc._avail_ram_mb() >= 7000.0 else 3
    print(f"[Exp 1b] Running dz_f70 Startup Parity Gate (`Seq_Only` + `F_true`, n_workers={n_workers})...", flush=True)
    seq_dz, tru_dz, t_dz_g = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_dz,
        bank_19[0:1],
        hsc.CFG_HGB_100,
        include_seq_only=True,
        n_workers=n_workers,
        checkpoint_dir=ckpt_root / "dz_f70_gate",
        checkpoint_prefix="gate",
        progress_label="Exp1b-dz_f70-Gate",
    )
    assert seq_dz is not None
    mut_g = cache_dz["mut_gene_idx"]
    n_genes = int(cache_dz["n_genes"])
    gm_dz_seq = float(np.mean([np.mean(seq_dz[mut_g == g]) for g in range(n_genes)]))
    gm_dz_tru = float(np.mean([np.mean(tru_dz[0, mut_g == g]) for g in range(n_genes)]))
    print(
        f"  [Parity dz_f70] Seq={gm_dz_seq:.6f} vs v7={t_dz_seq_v7:.6f} (|d|={abs(gm_dz_seq - t_dz_seq_v7):.2e}) | "
        f"True={gm_dz_tru:.6f} vs v7={t_dz_tru_v7:.6f} (|d|={abs(gm_dz_tru - t_dz_tru_v7):.2e})",
        flush=True,
    )
    if abs(gm_dz_seq - t_dz_seq_v7) > 5e-4 or abs(gm_dz_tru - t_dz_tru_v7) > 5e-4:
        raise RuntimeError("Exp 1b dz_f70 PARITY GATE FAILED!")

    _, null_dz, t_dz_n = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_dz,
        bank_19[1:],
        hsc.CFG_HGB_100,
        include_seq_only=False,
        n_workers=n_workers,
        checkpoint_dir=ckpt_root / "dz_f70_nulls",
        checkpoint_prefix="dz",
        progress_label="Exp1b-dz_f70-Nulls",
    )
    sum_dz = hsc.summarize_hgbdt_surrogate_family(
        seq_dz,
        tru_dz[0],
        {
            "ShiftGC_ExactCRT_WS_G2": null_dz[:9],
            "Sparse4mer_K12": null_dz[9:18],
        },
        mut_g,
        cache_dz["mut_segments"],
        cache_dz["mut_kock_strata"],
        n_boot=5000,
        boot_seed=42,
    )
    sum_dz["runtime_seconds"] = float(t_dz_g + t_dz_n)

    del cache_dz
    gc.collect()

    print(f"[Exp 1b] Building FULL K=2,000 cache for dz_spec_f70 (avail_ram={hsc._avail_ram_mb():.0f}MB)...", flush=True)
    cache_sp = hsc.build_v7_hgbdt_fold_cache(target_key="dz_spec_f70", n_sub_per_mut=None)
    n_workers_sp = 4 if hsc._avail_ram_mb() >= 7000.0 else 3
    seq_sp, tru_sp, t_sp_g = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_sp,
        bank_19[0:1],
        hsc.CFG_HGB_100,
        include_seq_only=True,
        n_workers=n_workers_sp,
        checkpoint_dir=ckpt_root / "dz_spec_f70_gate",
        checkpoint_prefix="gate",
        progress_label="Exp1b-dz_spec-Gate",
    )
    assert seq_sp is not None
    mut_g_sp = cache_sp["mut_gene_idx"]
    gm_sp_seq = float(np.mean([np.mean(seq_sp[mut_g_sp == g]) for g in range(n_genes)]))
    gm_sp_tru = float(np.mean([np.mean(tru_sp[0, mut_g_sp == g]) for g in range(n_genes)]))
    print(
        f"  [Parity dz_spec_f70] Seq={gm_sp_seq:.6f} vs v7={t_sp_seq_v7:.6f} (|d|={abs(gm_sp_seq - t_sp_seq_v7):.2e}) | "
        f"True={gm_sp_tru:.6f} vs v7={t_sp_tru_v7:.6f} (|d|={abs(gm_sp_tru - t_sp_tru_v7):.2e})",
        flush=True,
    )
    if abs(gm_sp_seq - t_sp_seq_v7) > 5e-4 or abs(gm_sp_tru - t_sp_tru_v7) > 5e-4:
        raise RuntimeError("Exp 1b dz_spec_f70 PARITY GATE FAILED!")

    _, null_sp, t_sp_n = hsc.evaluate_hgbdt_surrogate_table_logo(
        cache_sp,
        bank_19[1:],
        hsc.CFG_HGB_100,
        include_seq_only=False,
        n_workers=n_workers_sp,
        checkpoint_dir=ckpt_root / "dz_spec_f70_nulls",
        checkpoint_prefix="sp",
        progress_label="Exp1b-dz_spec-Nulls",
    )
    sum_sp = hsc.summarize_hgbdt_surrogate_family(
        seq_sp,
        tru_sp[0],
        {
            "ShiftGC_ExactCRT_WS_G2": null_sp[:9],
            "Sparse4mer_K12": null_sp[9:18],
        },
        mut_g_sp,
        cache_sp["mut_segments"],
        cache_sp["mut_kock_strata"],
        n_boot=5000,
        boot_seed=42,
    )
    sum_sp["runtime_seconds"] = float(t_sp_g + t_sp_n)

    for nm, a in [("null_dz", null_dz), ("null_sp", null_sp), ("seq_dz", seq_dz), ("seq_sp", seq_sp)]:
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Non-finite in {nm}")

    out = {
        "stage": "v7_1_exp1b_hgbdt_fullk2000_bridge",
        "config": hsc.CFG_HGB_100,
        "subsampling": cache_sp["subsampling"],
        "seed_bank": 2026071,
        "draws_used": "first_9_draws_from_Exp1a_bank",
        "B_per_family": 9,
        "parity_gate": {
            "dz_f70_seq_diff": float(abs(gm_dz_seq - t_dz_seq_v7)),
            "dz_f70_true_diff": float(abs(gm_dz_tru - t_dz_tru_v7)),
            "dz_spec_f70_seq_diff": float(abs(gm_sp_seq - t_sp_seq_v7)),
            "dz_spec_f70_true_diff": float(abs(gm_sp_tru - t_sp_tru_v7)),
            "passed": True,
        },
        "dz_f70": sum_dz,
        "dz_spec_f70": sum_sp,
        "total_runtime_seconds": float(time.time() - t_all),
    }
    p_out = V7_1_ART / "ckpt_v7_1_exp1b_hgbdt_fullk2000_bridge.json"
    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE Exp 1b] Saved {p_out} in {out['total_runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
