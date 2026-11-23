#!/usr/bin/env python3
"""
Experiment 1b B=19 Extension (`biophys_inductive_gfm/v7_1`):
Full K=2,000 (180k train rows, `CFG_HGB_100`, 30-fold LOGO) `GSE233827`
`dz_spec_f70` HGBDT matched surrogate control extended from B=9 (`p_min=0.10`)
to B=19 (`p_min=0.05`, formal alpha=0.05 power) for BOTH:
  - `ShiftGC_ExactCRT_WS_G2` (draws 0..18 from seed=2026071 B=49 bank)
  - `Sparse4mer_K12`         (draws 0..18 from seed=2026071 B=49 bank)

CPU-only (strictly 0 GPU). Reuses already-saved `.npy` checkpoints for
`gate_seqonly`, `gate_0000`, and draws 0..8 (`sp_0000..sp_0017`), verifies
exact `<= 1e-8` parity against `ckpt_v7_1_exp1b_hgbdt_fullk2000_bridge.json`
AND exact single-fold worker parity against saved `.npy` outputs, then
computes ONLY the 20 new draws (ShiftGC draws 9..18, Sparse4mer_K12 draws 9..18)
using a low-RAM per-mutant worker pool (`n_workers=3`, <1 GiB total RSS).

Offline / zero network imports for CoE Pillar 2 compliance.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V7_SRC = ROOT / "v7" / "src"
V7_1_SRC = ROOT / "v7_1" / "src"
V7_PROC_DIR = ROOT / "v7" / "data" / "processed"
V7_1_ART = ROOT / "v7_1" / "artifacts"

for _p in (V7_1_SRC, V7_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hgbdt_surrogate_core as hsc  # noqa: E402
import kock_pbm_v7 as kv7  # noqa: E402

SEED_BANK = 2026071
B_TARGET = 19
B_REUSED = 9
N_EXTRA_PER_FAM = B_TARGET - B_REUSED  # 10
DEFAULT_WORKERS = 3
RAM_GUARD_MIN_MB = 3072.0  # >= 3 GiB host available RAM

STRATUM_KEYS = [
    "all_92",
    "N_arm_1_9",
    "H1_H2_linkers",
    "H3_42_58",
    "specificity_yes",
    "affinity_only",
    "no_change",
]
SUB_STRATUM_KEYS = [
    "N_arm_1_9",
    "H1_H2_linkers",
    "H3_42_58",
    "specificity_yes",
    "affinity_only",
    "no_change",
]

_LOWRAM_CTX: Dict[str, Any] = {}


def _build_lowram_mutant_cache(
    target_key: str = "dz_spec_f70",
    npz_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Build the exact same per-mutant `X_seq_full`, `c5_full`, `y_full` arrays as
    `hsc.build_v7_hgbdt_fold_cache(..., n_sub_per_mut=None)`, WITHOUT pre-stacking
    30 duplicated copies (~75 MB instead of ~2.25 GB).
    """
    p_npz = npz_path or (V7_PROC_DIR / "v7_kock_dataset.npz")
    ds = dict(np.load(p_npz, allow_pickle=False))

    n_genes = int(len(ds["gene_names"]))
    n_mut = int(len(ds["mut_alleles"]))
    mut_gene_idx = ds["mut_gene_idx"].astype(np.int32)
    phi_prot = ds["phi_prot"].astype(np.float32)
    win_codes_g = ds["win12_codes_f70"]
    y_mat = ds[target_key]

    gene_seq_dna: List[np.ndarray] = []
    gene_c5_n1n4: List[np.ndarray] = []
    for g in range(n_genes):
        c12 = win_codes_g[g].astype(np.uint8)
        g1, gc2 = hsc._encode_g1_g2_gc_fast(c12)
        c5_n1n4, _, _ = kv7.extract_pentamer_indices_12bp(c12)
        gene_seq_dna.append(np.hstack([c12.astype(np.float32), g1, gc2]))
        gene_c5_n1n4.append(c5_n1n4.astype(np.int32))

    X_seq_full: List[np.ndarray] = []
    c5_full: List[np.ndarray] = []
    y_full_f32: List[np.ndarray] = []
    y_full_f64: List[np.ndarray] = []
    for m in range(n_mut):
        mk = np.isfinite(y_mat[m])
        y_m = y_mat[m, mk].astype(np.float32)
        y_full_f32.append(np.ascontiguousarray(y_m, dtype=np.float32))
        y_full_f64.append(np.ascontiguousarray(y_m, dtype=np.float64))
        g = int(mut_gene_idx[m])
        dna_m = gene_seq_dna[g][mk]
        pm = np.broadcast_to(phi_prot[m], (len(dna_m), phi_prot.shape[1]))
        X_seq_full.append(np.ascontiguousarray(np.hstack([pm, dna_m]), dtype=np.float32))
        c5_full.append(np.ascontiguousarray(gene_c5_n1n4[g][mk], dtype=np.int32))

    fold_splits: List[Tuple[List[int], np.ndarray]] = []
    for gh in range(n_genes):
        tr_m = [m for m in range(n_mut) if int(mut_gene_idx[m]) != gh]
        te_m = np.array([m for m in range(n_mut) if int(mut_gene_idx[m]) == gh], dtype=np.int32)
        fold_splits.append((tr_m, te_m))

    c5_unlabeled_wt = np.vstack(gene_c5_n1n4).astype(np.int32)
    sub_meta = {
        "rule": "full_all_finite_K2000_training_probes",
        "target_probes_per_mutant": 2000,
        "total_probes_across_92_mutants": int(sum(len(y) for y in y_full_f32)),
        "evaluation_probes": "ALL_finite_K2000_test_probes",
        "mutant_label_leakage": False,
    }
    return {
        "target_key": target_key,
        "n_genes": n_genes,
        "n_mut": n_mut,
        "gene_names": ds["gene_names"],
        "mut_alleles": ds["mut_alleles"],
        "mut_gene_idx": mut_gene_idx,
        "mut_segments": ds["mut_segments"],
        "mut_kock_strata": ds["mut_kock_strata"],
        "c5_unlabeled_wt": c5_unlabeled_wt,
        "X_seq_full": X_seq_full,
        "c5_full": c5_full,
        "y_full_f32": y_full_f32,
        "y_full_f64": y_full_f64,
        "fold_splits": fold_splits,
        "subsampling": sub_meta,
    }


def _init_lowram_worker(ctx: Dict[str, Any]) -> None:
    global _LOWRAM_CTX
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _LOWRAM_CTX = ctx


def _fit_single_fold_lowram(args: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
    """
    Bit-identical to `hsc._worker_fit_fold_task` while constructing fold arrays
    on the fly from the ~75 MB per-mutant lists.
    """
    tab_idx, gh = args
    c = _LOWRAM_CTX
    tr_m, te_m = c["fold_splits"][gh]
    X_seq_full = c["X_seq_full"]
    c5_full = c["c5_full"]
    y_full_f32 = c["y_full_f32"]
    y_full_f64 = c["y_full_f64"]

    X_seq_tr = np.ascontiguousarray(np.vstack([X_seq_full[m] for m in tr_m]), dtype=np.float32)
    c5_tr = np.ascontiguousarray(np.vstack([c5_full[m] for m in tr_m]), dtype=np.int32)
    y_tr = np.ascontiguousarray(np.concatenate([y_full_f32[m] for m in tr_m]), dtype=np.float32)

    X_seq_te = np.ascontiguousarray(np.vstack([X_seq_full[m] for m in te_m]), dtype=np.float32)
    c5_te = np.ascontiguousarray(np.vstack([c5_full[m] for m in te_m]), dtype=np.int32)

    tab = c["tables"][tab_idx]
    sh_tr = tab[c5_tr].reshape(len(X_seq_tr), 28)
    sh_te = tab[c5_te].reshape(len(X_seq_te), 28)
    X_tr = np.hstack([X_seq_tr, sh_tr])
    X_te = np.hstack([X_seq_te, sh_te])

    model = HistGradientBoostingRegressor(**c["cfg"])
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rhos = np.zeros(len(te_m), dtype=np.float64)
    cur = 0
    for i, m in enumerate(te_m):
        yt = y_full_f64[int(m)]
        nm = len(yt)
        rhos[i] = kv7.safe_spearman_pearson(yt, preds[cur : cur + nm])[0]
        cur += nm
    return tab_idx, gh, rhos


def _compare_nested_floats(
    ref: Any,
    got: Any,
    path: str = "root",
    diffs: List[Tuple[str, float]] | None = None,
) -> List[Tuple[str, float]]:
    if diffs is None:
        diffs = []
    if isinstance(ref, bool) or isinstance(got, bool):
        if ref != got:
            raise ValueError(f"Boolean mismatch at {path}: ref={ref} vs got={got}")
        return diffs
    if isinstance(ref, (int, float)) and isinstance(got, (int, float)):
        d = abs(float(ref) - float(got))
        diffs.append((path, d))
        return diffs
    if isinstance(ref, dict) and isinstance(got, dict):
        for k in ref:
            if k == "runtime_seconds":
                continue
            if k not in got:
                raise KeyError(f"Missing key {path}.{k} in reconstructed summary")
            _compare_nested_floats(ref[k], got[k], f"{path}.{k}", diffs)
        return diffs
    if isinstance(ref, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(ref) != len(got):
            raise ValueError(f"Length mismatch at {path}: {len(ref)} vs {len(got)}")
        for i, (r_i, g_i) in enumerate(zip(ref, got)):
            _compare_nested_floats(r_i, g_i, f"{path}[{i}]", diffs)
        return diffs
    if ref != got:
        raise ValueError(f"Value mismatch at {path}: ref={ref} vs got={got}")
    return diffs


def _gene_balanced_mean(vec: np.ndarray, g_idx: np.ndarray, mask: np.ndarray) -> float:
    gs = np.unique(g_idx[mask])
    return float(np.mean([np.mean(vec[mask & (g_idx == g)]) for g in gs]))


def _gene_balanced_mat(mat: np.ndarray, g_idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gs = np.unique(g_idx[mask])
    cols = [np.mean(mat[:, mask & (g_idx == g)], axis=1) for g in gs]
    return np.mean(np.column_stack(cols), axis=1)


def _compute_stratified_and_hoxc10_analysis(
    seq: np.ndarray,
    tru: np.ndarray,
    crt: np.ndarray,
    k12: np.ndarray,
    mut_g: np.ndarray,
    mut_seg: np.ndarray,
    mut_ks: np.ndarray,
    mut_all: np.ndarray,
    gene_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Compute BOTH gene-balanced AND mutant-weighted stratum p_mc across all 7
    strata, Holm-Bonferroni across strata per null family (both 6 non-all strata
    and all 7 strata), and HOXC10-excluded 29-gene sensitivity.
    """
    gene_list = [str(g) for g in gene_names]
    hoxc10_idx = gene_list.index("HOXC10")
    no_hoxc10_mask = mut_g != hoxc10_idx
    hoxc10_alleles = [str(a) for a in mut_all[mut_g == hoxc10_idx]]

    base_masks = {
        "all_92": np.ones(len(mut_g), dtype=bool),
        "N_arm_1_9": mut_seg == "N-arm",
        "H1_H2_linkers": np.isin(mut_seg, ["H1", "L12", "H2", "L23"]),
        "H3_42_58": mut_seg == "H3",
        "specificity_yes": mut_ks == "specificity_yes",
        "affinity_only": mut_ks == "affinity_only",
        "no_change": mut_ks == "no_change",
    }

    h_idx = np.where(mut_g == hoxc10_idx)[0]
    B_crt = int(crt.shape[0])
    B_k12 = int(k12.shape[0])
    hoxc10_per_allele = {
        str(mut_all[i]): {
            "seq_only": float(seq[i]),
            "true": float(tru[i]),
            "shiftgc_median": float(np.median(crt[:, i])),
            "shiftgc_range": [float(np.min(crt[:, i])), float(np.max(crt[:, i]))],
            "shiftgc_p_mc": float((1.0 + np.sum(crt[:, i] >= tru[i])) / float(B_crt + 1)),
            "k12_median": float(np.median(k12[:, i])),
            "k12_range": [float(np.min(k12[:, i])), float(np.max(k12[:, i]))],
            "k12_p_mc": float((1.0 + np.sum(k12[:, i] >= tru[i])) / float(B_k12 + 1)),
        }
        for i in h_idx
    }

    out: Dict[str, Any] = {
        "B_per_family": B_crt,
        "p_min_possible": float(1.0 / float(B_crt + 1)),
        "hoxc10_gene_index": int(hoxc10_idx),
        "hoxc10_alleles": hoxc10_alleles,
        "hoxc10_per_allele": hoxc10_per_allele,
    }

    for mode_lbl, extra_m in [
        ("all_genes_30", np.ones(len(mut_g), dtype=bool)),
        ("excl_HOXC10_29", no_hoxc10_mask),
    ]:
        mode_out: Dict[str, Any] = {}
        for fam_lbl, mat in [("ShiftGC_ExactCRT_WS_G2", crt), ("Sparse4mer_K12", k12)]:
            B = int(mat.shape[0])
            s_out: Dict[str, Any] = {}
            for sk in STRATUM_KEYS:
                mk = base_masks[sk] & extra_m
                # Gene-balanced
                t_g = _gene_balanced_mean(tru, mut_g, mk)
                s_g = _gene_balanced_mean(seq, mut_g, mk)
                n_g = _gene_balanced_mat(mat, mut_g, mk)
                k_g = int(np.sum(n_g >= t_g))
                p_g = float((1.0 + k_g) / float(B + 1))

                # Mutant-weighted
                t_m = float(np.mean(tru[mk]))
                s_m = float(np.mean(seq[mk]))
                n_m = np.mean(mat[:, mk], axis=1)
                k_m = int(np.sum(n_m >= t_m))
                p_m = float((1.0 + k_m) / float(B + 1))

                s_out[sk] = {
                    "n_genes": int(len(np.unique(mut_g[mk]))),
                    "n_mutants": int(np.sum(mk)),
                    "gene_balanced": {
                        "seq_only": s_g,
                        "true": t_g,
                        "null_median": float(np.median(n_g)),
                        "null_q95": float(np.quantile(n_g, 0.95)),
                        "null_min": float(np.min(n_g)),
                        "null_max": float(np.max(n_g)),
                        "delta_true_vs_null": float(t_g - np.median(n_g)),
                        "true_lift_over_seq": float(t_g - s_g),
                        "null_median_lift_over_seq": float(np.median(n_g) - s_g),
                        "n_null_ge_true": k_g,
                        "p_mc": p_g,
                    },
                    "mutant_weighted": {
                        "seq_only": s_m,
                        "true": t_m,
                        "null_median": float(np.median(n_m)),
                        "null_q95": float(np.quantile(n_m, 0.95)),
                        "null_min": float(np.min(n_m)),
                        "null_max": float(np.max(n_m)),
                        "delta_true_vs_null": float(t_m - np.median(n_m)),
                        "true_lift_over_seq": float(t_m - s_m),
                        "null_median_lift_over_seq": float(np.median(n_m) - s_m),
                        "n_null_ge_true": k_m,
                        "p_mc": p_m,
                    },
                }

            # Holm-Bonferroni across 6 non-all strata (matches scholarpeer sprint addendum)
            p_hol6_g = hsc._holm_bonferroni([s_out[k]["gene_balanced"]["p_mc"] for k in SUB_STRATUM_KEYS])
            p_hol6_m = hsc._holm_bonferroni([s_out[k]["mutant_weighted"]["p_mc"] for k in SUB_STRATUM_KEYS])
            for k, phg, phm in zip(SUB_STRATUM_KEYS, p_hol6_g, p_hol6_m):
                s_out[k]["gene_balanced"]["p_holm_6strata"] = float(phg)
                s_out[k]["mutant_weighted"]["p_holm_6strata"] = float(phm)

            # Holm-Bonferroni across all 7 strata
            p_hol7_g = hsc._holm_bonferroni([s_out[k]["gene_balanced"]["p_mc"] for k in STRATUM_KEYS])
            p_hol7_m = hsc._holm_bonferroni([s_out[k]["mutant_weighted"]["p_mc"] for k in STRATUM_KEYS])
            for k, phg, phm in zip(STRATUM_KEYS, p_hol7_g, p_hol7_m):
                s_out[k]["gene_balanced"]["p_holm_7strata"] = float(phg)
                s_out[k]["mutant_weighted"]["p_holm_7strata"] = float(phm)

            mode_out[fam_lbl] = s_out
        out[mode_lbl] = mode_out
    return out


def _verify_all_finite(obj: Any, path: str = "root") -> int:
    """Recursively verify 0 NaNs/Infs and return total float count checked."""
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj}")
        return 1
    if isinstance(obj, dict):
        cnt = 0
        for k, v in obj.items():
            cnt += _verify_all_finite(v, f"{path}.{k}")
        return cnt
    if isinstance(obj, (list, tuple)):
        cnt = 0
        for i, v in enumerate(obj):
            cnt += _verify_all_finite(v, f"{path}[{i}]")
        return cnt
    return 0


def _sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    t_all = time.time()
    start_avail_mb = hsc._avail_ram_mb()
    min_avail_mb = start_avail_mb

    parts_root = V7_1_ART / ".hgb_parts_exp1b"
    gate_dir = parts_root / "dz_spec_f70_gate"
    reused_dir = parts_root / "dz_spec_f70_nulls"
    extra_dir = parts_root / "dz_spec_f70_b19_extra"
    extra_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = V7_1_ART / "ckpt_v7_1_exp1b_hgbdt_fullk2000_bridge.json"
    out_path = V7_1_ART / "ckpt_v7_1_exp1b_hgbdt_fullk2000_b19.json"

    print(
        f"[Exp 1b B=19] Starting dz_spec_f70 extension (CPU-only, n_workers={DEFAULT_WORKERS}, "
        f"avail_ram={start_avail_mb:.0f}MB)...",
        flush=True,
    )

    # 1. Build low-RAM exact per-mutant cache (~75 MB)
    cache = _build_lowram_mutant_cache(target_key="dz_spec_f70")
    mut_g = cache["mut_gene_idx"]
    mut_seg = cache["mut_segments"]
    mut_ks = cache["mut_kock_strata"]
    mut_all = cache["mut_alleles"]
    gene_names = cache["gene_names"]
    n_genes = int(cache["n_genes"])
    n_mut = int(cache["n_mut"])
    print(
        f"  [Cache] Built per-mutant cache (n_genes={n_genes}, n_mut={n_mut}, "
        f"total_probes={cache['subsampling']['total_probes_across_92_mutants']}, "
        f"rss={hsc._rss_mb():.0f}MB, avail={hsc._avail_ram_mb():.0f}MB)",
        flush=True,
    )

    # 2. Construct exact same B=49 bank (seed=2026071) and slice draws 0..18
    bank_full, slices_full = hsc.build_hgbdt_surrogate_bank(
        cache["c5_unlabeled_wt"],
        seed=SEED_BANK,
        b_crt=49,
        b_k12=49,
        b_kladder=0,
        k_other=(),
    )
    crt_start = slices_full["ShiftGC_ExactCRT_WS_G2"].start
    k12_start = slices_full["Sparse4mer_K12"].start
    assert crt_start == 1 and k12_start == 50, f"Unexpected bank slices: {slices_full}"

    F_true_tab = bank_full[0:1].astype(np.float32)
    crt_all19_tabs = bank_full[crt_start : crt_start + B_TARGET]
    k12_all19_tabs = bank_full[k12_start : k12_start + B_TARGET]

    # Extra 20 draws: indices 9..18 for ShiftGC, indices 9..18 for Sparse4mer_K12
    extra_specs: List[Tuple[str, str, int, np.ndarray]] = []
    for d in range(B_REUSED, B_TARGET):
        extra_specs.append(("ShiftGC_ExactCRT_WS_G2", f"crt_draw_{d:02d}.npy", d, crt_all19_tabs[d]))
    for d in range(B_REUSED, B_TARGET):
        extra_specs.append(("Sparse4mer_K12", f"k12_draw_{d:02d}.npy", d, k12_all19_tabs[d]))

    bank_hashes = {
        "F_true_sha256": _sha256_arr(bank_full[0]),
        "ShiftGC_draws_0_8_sha256": _sha256_arr(crt_all19_tabs[:B_REUSED]),
        "ShiftGC_draws_9_18_sha256": _sha256_arr(crt_all19_tabs[B_REUSED:B_TARGET]),
        "Sparse4mer_K12_draws_0_8_sha256": _sha256_arr(k12_all19_tabs[:B_REUSED]),
        "Sparse4mer_K12_draws_9_18_sha256": _sha256_arr(k12_all19_tabs[B_REUSED:B_TARGET]),
    }

    # 3. Load reused B=9 .npy checkpoints and verify exact <= 1e-8 parity vs bridge JSON
    seq_sp = np.load(gate_dir / "gate_seqonly.npy").astype(np.float64)
    tru_sp = np.load(gate_dir / "gate_0000.npy").astype(np.float64)
    crt_0_8 = np.vstack([np.load(reused_dir / f"sp_{i:04d}.npy") for i in range(0, B_REUSED)]).astype(np.float64)
    k12_0_8 = np.vstack([np.load(reused_dir / f"sp_{i:04d}.npy") for i in range(B_REUSED, 2 * B_REUSED)]).astype(
        np.float64
    )

    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    sum_b9_check = hsc.summarize_hgbdt_surrogate_family(
        seq_sp,
        tru_sp,
        {
            "ShiftGC_ExactCRT_WS_G2": crt_0_8,
            "Sparse4mer_K12": k12_0_8,
        },
        mut_g,
        mut_seg,
        mut_ks,
        n_boot=5000,
        boot_seed=42,
    )
    diffs = _compare_nested_floats(bridge["dz_spec_f70"], sum_b9_check, path="dz_spec_f70")
    max_diff_name, max_diff_val = max(diffs, key=lambda kv: kv[1])
    crt_diffs = [d for p, d in diffs if "ShiftGC_ExactCRT_WS_G2" in p]
    k12_diffs = [d for p, d in diffs if "Sparse4mer_K12" in p]
    print(
        f"  [Parity B=9 vs bridge JSON] checked {len(diffs)} scalars | "
        f"max_abs_diff={max_diff_val:.3e} at {max_diff_name} "
        f"(CRT max={max(crt_diffs):.3e}, K12 max={max(k12_diffs):.3e})",
        flush=True,
    )
    if max_diff_val > 1e-8:
        raise RuntimeError(f"B=9 bridge parity check FAILED: max_diff={max_diff_val:.3e} at {max_diff_name}")

    # 4. Verify exact low-RAM worker parity on fold 0 for F_true & ShiftGC draw 0
    #    against saved gate_0000.npy and sp_0000.npy BEFORE running the 20 new draws
    parity_tables = np.stack([bank_full[0], crt_all19_tabs[0]], axis=0).astype(np.float32)
    parity_ctx = {
        "fold_splits": cache["fold_splits"],
        "X_seq_full": cache["X_seq_full"],
        "c5_full": cache["c5_full"],
        "y_full_f32": cache["y_full_f32"],
        "y_full_f64": cache["y_full_f64"],
        "tables": parity_tables,
        "cfg": dict(hsc.CFG_HGB_100),
    }
    _init_lowram_worker(parity_ctx)
    _, _, r_f0_true = _fit_single_fold_lowram((0, 0))
    _, _, r_f0_crt0 = _fit_single_fold_lowram((1, 0))
    te_m_f0 = cache["fold_splits"][0][1]
    w_diff_true = float(np.max(np.abs(r_f0_true - tru_sp[te_m_f0])))
    w_diff_crt0 = float(np.max(np.abs(r_f0_crt0 - crt_0_8[0, te_m_f0])))
    w_diff_max = max(w_diff_true, w_diff_crt0)
    print(
        f"  [Parity LowRAM Worker Fold 0] F_true max|d|={w_diff_true:.3e} | "
        f"ShiftGC[0] max|d|={w_diff_crt0:.3e} (avail={hsc._avail_ram_mb():.0f}MB)",
        flush=True,
    )
    if w_diff_max > 1e-8:
        raise RuntimeError(f"Low-RAM worker parity FAILED: max|d|={w_diff_max:.3e}")
    _LOWRAM_CTX.clear()

    # Release full bank now that extra_specs holds the 20 tables
    extra_tables_f32 = np.stack([spec[3] for spec in extra_specs], axis=0).astype(np.float32)
    del bank_full, parity_tables, parity_ctx

    # 5. Compute ONLY the 20 new draws (or resume from existing extra_dir .npy files)
    extra_rhos = np.zeros((len(extra_specs), n_mut), dtype=np.float64)
    pending_indices: List[int] = []
    for idx, (fam, fname, draw_i, _) in enumerate(extra_specs):
        p_f = extra_dir / fname
        if p_f.exists():
            arr = np.load(p_f).astype(np.float64)
            if arr.shape != (n_mut,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"Corrupt cached file {p_f}")
            extra_rhos[idx] = arr
        else:
            pending_indices.append(idx)

    per_draw_sec: Dict[str, float] = {}
    if not pending_indices:
        print(f"  [Extra 20 draws] 100% cache hit in {extra_dir}.", flush=True)
    else:
        avail_now = hsc._avail_ram_mb()
        min_avail_mb = min(min_avail_mb, avail_now)
        n_workers = DEFAULT_WORKERS
        if avail_now < RAM_GUARD_MIN_MB:
            n_workers = 1
        elif avail_now < 4500.0:
            n_workers = 2
        print(
            f"  [Extra 20 draws] Computing {len(pending_indices)}/20 pending draws x {n_genes} LOGO folds "
            f"(n_workers={n_workers}, avail_ram={avail_now:.0f}MB)...",
            flush=True,
        )

        worker_ctx = {
            "fold_splits": cache["fold_splits"],
            "X_seq_full": cache["X_seq_full"],
            "c5_full": cache["c5_full"],
            "y_full_f32": cache["y_full_f32"],
            "y_full_f64": cache["y_full_f64"],
            "tables": extra_tables_f32,
            "cfg": dict(hsc.CFG_HGB_100),
        }
        mp_ctx = mp.get_context("spawn")
        with mp_ctx.Pool(
            processes=n_workers,
            initializer=_init_lowram_worker,
            initargs=(worker_ctx,),
        ) as pool:
            for done_i, idx in enumerate(pending_indices, start=1):
                fam, fname, draw_i, _ = extra_specs[idx]
                t_d0 = time.time()
                avail_pre = hsc._avail_ram_mb()
                min_avail_mb = min(min_avail_mb, avail_pre)

                # RAM guard pause if host RAM ever dips below 3 GiB
                wait_loops = 0
                while avail_pre < RAM_GUARD_MIN_MB and wait_loops < 60:
                    print(
                        f"    [RAM GUARD] avail={avail_pre:.0f}MB < {RAM_GUARD_MIN_MB:.0f}MB; sleeping 10s...",
                        flush=True,
                    )
                    time.sleep(10.0)
                    avail_pre = hsc._avail_ram_mb()
                    min_avail_mb = min(min_avail_mb, avail_pre)
                    wait_loops += 1

                tasks = [(idx, gh) for gh in range(n_genes)]
                vec = np.zeros(n_mut, dtype=np.float64)
                for _, gh, r_fold in pool.map(_fit_single_fold_lowram, tasks):
                    te_m = cache["fold_splits"][gh][1]
                    vec[te_m] = r_fold

                if not np.all(np.isfinite(vec)):
                    raise ValueError(f"Non-finite Spearman values in {fam} draw {draw_i}")

                # Atomic write via tmp + replace
                p_f = extra_dir / fname
                p_tmp = extra_dir / f".tmp_{fname}"
                np.save(p_tmp, vec)
                os.replace(p_tmp, p_f)
                extra_rhos[idx] = vec

                dt = float(time.time() - t_d0)
                per_draw_sec[f"{fam}_draw_{draw_i:02d}"] = dt
                avail_post = hsc._avail_ram_mb()
                min_avail_mb = min(min_avail_mb, avail_post)
                gm_d = float(np.mean([np.mean(vec[mut_g == g]) for g in range(n_genes)]))
                mm_d = float(np.mean(vec))
                print(
                    f"    [{done_i:2d}/{len(pending_indices)}] {fam} draw {draw_i:02d} -> "
                    f"gene_mean={gm_d:.5f}, mut_mean={mm_d:.5f} in {dt:.1f}s "
                    f"(rss={hsc._rss_mb():.0f}MB, avail={avail_post:.0f}MB)",
                    flush=True,
                )

    # 6. Assemble full B=19 arrays (reused 0..8 + newly computed 9..18)
    crt_9_18 = extra_rhos[:N_EXTRA_PER_FAM]
    k12_9_18 = extra_rhos[N_EXTRA_PER_FAM : 2 * N_EXTRA_PER_FAM]
    crt_19 = np.vstack([crt_0_8, crt_9_18])
    k12_19 = np.vstack([k12_0_8, k12_9_18])
    assert crt_19.shape == (B_TARGET, n_mut) and k12_19.shape == (B_TARGET, n_mut)

    for nm, arr in [("seq_sp", seq_sp), ("tru_sp", tru_sp), ("crt_19", crt_19), ("k12_19", k12_19)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite value detected in {nm}")

    # 7. Canonical summarize_hgbdt_surrogate_family (B=19)
    sum_b19 = hsc.summarize_hgbdt_surrogate_family(
        seq_sp,
        tru_sp,
        {
            "ShiftGC_ExactCRT_WS_G2": crt_19,
            "Sparse4mer_K12": k12_19,
        },
        mut_g,
        mut_seg,
        mut_ks,
        n_boot=5000,
        boot_seed=42,
    )

    # 8. Full gene-balanced + mutant-weighted + HOXC10-excluded 29-gene + Holm-Bonferroni analysis
    strat_hoxc10_b19 = _compute_stratified_and_hoxc10_analysis(
        seq_sp,
        tru_sp,
        crt_19,
        k12_19,
        mut_g,
        mut_seg,
        mut_ks,
        mut_all,
        gene_names,
    )
    strat_hoxc10_b9_ref = _compute_stratified_and_hoxc10_analysis(
        seq_sp,
        tru_sp,
        crt_0_8,
        k12_0_8,
        mut_g,
        mut_seg,
        mut_ks,
        mut_all,
        gene_names,
    )

    # Per-draw gene-mean & mut-mean trace for full auditability
    def _draw_trace(mat: np.ndarray) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for d in range(mat.shape[0]):
            v = mat[d]
            rows.append(
                {
                    "draw_index": int(d),
                    "source": "reused_b9" if d < B_REUSED else "computed_b19_extra",
                    "gene_mean_all30": float(np.mean([np.mean(v[mut_g == g]) for g in range(n_genes)])),
                    "mut_mean_all92": float(np.mean(v)),
                }
            )
        return rows

    end_avail_mb = hsc._avail_ram_mb()
    min_avail_mb = min(min_avail_mb, end_avail_mb)
    total_runtime = float(time.time() - t_all)

    out: Dict[str, Any] = {
        "stage": "v7_1_exp1b_hgbdt_fullk2000_b19",
        "target_key": "dz_spec_f70",
        "config": hsc.CFG_HGB_100,
        "subsampling": cache["subsampling"],
        "seed_bank": SEED_BANK,
        "draws_used": "draws_0_to_18_from_shared_B49_bank_reusing_0_to_8",
        "B_per_family": B_TARGET,
        "p_min_possible": float(1.0 / float(B_TARGET + 1)),
        "bank_sha256": bank_hashes,
        "checkpoint_sources": {
            "gate_dir": str(gate_dir),
            "reused_b9_dir": str(reused_dir),
            "extra_b19_dir": str(extra_dir),
        },
        "parity_check_vs_b9_bridge": {
            "bridge_json": str(bridge_path),
            "tolerance": 1e-8,
            "n_scalars_compared": int(len(diffs)),
            "max_abs_diff": float(max_diff_val),
            "max_abs_diff_field": str(max_diff_name),
            "shiftgc_b9_max_abs_diff": float(max(crt_diffs)),
            "sparse4mer_k12_b9_max_abs_diff": float(max(k12_diffs)),
            "lowram_worker_fold0_f_true_max_abs_diff": float(w_diff_true),
            "lowram_worker_fold0_shiftgc0_max_abs_diff": float(w_diff_crt0),
            "passed": bool(max_diff_val <= 1e-8 and w_diff_max <= 1e-8),
        },
        "dz_spec_f70_b19_summary": sum_b19,
        "stratified_and_hoxc10_sensitivity_b19": strat_hoxc10_b19,
        "stratified_and_hoxc10_sensitivity_b9_reused_ref": strat_hoxc10_b9_ref,
        "per_draw_summary": {
            "ShiftGC_ExactCRT_WS_G2": _draw_trace(crt_19),
            "Sparse4mer_K12": _draw_trace(k12_19),
        },
        "compute_telemetry": {
            "cpu_only": True,
            "gpu_used": False,
            "n_workers_configured": DEFAULT_WORKERS,
            "reused_null_draws": int(2 * B_REUSED),
            "newly_computed_null_draws": int(2 * N_EXTRA_PER_FAM),
            "computed_in_final_session_draws": int(len(pending_indices)),
            "resumed_from_extra_cache_draws": int(2 * N_EXTRA_PER_FAM - len(pending_indices)),
            "total_null_draws_in_b19": int(2 * B_TARGET),
            "peak_parent_rss_mb": float(hsc._rss_mb()),
            "start_avail_ram_mb": float(start_avail_mb),
            "min_avail_ram_mb": float(min_avail_mb),
            "end_avail_ram_mb": float(end_avail_mb),
            "per_extra_draw_runtime_seconds": per_draw_sec,
            "total_runtime_seconds": total_runtime,
        },
    }

    n_floats_checked = _verify_all_finite(out)
    out["finite_verification"] = {
        "passed": True,
        "n_floats_checked": int(n_floats_checked),
        "nan_inf_count": 0,
    }
    # Re-verify including finite_verification block
    _verify_all_finite(out)

    p_tmp_out = out_path.with_suffix(".json.tmp")
    p_tmp_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(p_tmp_out, out_path)

    # Print concise summary
    crt_s = sum_b19["null_families"]["ShiftGC_ExactCRT_WS_G2"]
    k12_s = sum_b19["null_families"]["Sparse4mer_K12"]
    a30_crt = strat_hoxc10_b19["all_genes_30"]["ShiftGC_ExactCRT_WS_G2"]
    a30_k12 = strat_hoxc10_b19["all_genes_30"]["Sparse4mer_K12"]
    e29_crt = strat_hoxc10_b19["excl_HOXC10_29"]["ShiftGC_ExactCRT_WS_G2"]
    e29_k12 = strat_hoxc10_b19["excl_HOXC10_29"]["Sparse4mer_K12"]
    print(
        f"[DONE Exp 1b B=19] Saved {out_path} in {total_runtime:.1f}s "
        f"(0 NaNs across {n_floats_checked} floats, parity={max_diff_val:.2e}, "
        f"min_avail_ram={min_avail_mb:.0f}MB)\n"
        f"  all_30 gene-mean: True={sum_b19['true_gene_mean_spearman']:.4f} (Seq={sum_b19['seq_only_comparison']['seq_only_gene_mean_spearman']:.4f}) | "
        f"CRT med={crt_s['null_gene_mean_median']:.4f} (p={crt_s['p_mc_gene_mean']:.3f}) | "
        f"K12 med={k12_s['null_gene_mean_median']:.4f} (p={k12_s['p_mc_gene_mean']:.3f})\n"
        f"  all_30 mut-mean:  True={sum_b19['true_mut_mean_spearman']:.4f} (Seq={sum_b19['seq_only_comparison']['seq_only_mut_mean_spearman']:.4f}) | "
        f"CRT med={crt_s['null_mut_mean_median']:.4f} (p={crt_s['p_mc_mut_mean']:.3f}) | "
        f"K12 med={k12_s['null_mut_mean_median']:.4f} (p={k12_s['p_mc_mut_mean']:.3f})\n"
        f"  all_30 spec_yes (gb/mw): CRT p={a30_crt['specificity_yes']['gene_balanced']['p_mc']:.3f}/{a30_crt['specificity_yes']['mutant_weighted']['p_mc']:.3f} | "
        f"K12 p={a30_k12['specificity_yes']['gene_balanced']['p_mc']:.3f}/{a30_k12['specificity_yes']['mutant_weighted']['p_mc']:.3f}\n"
        f"  excl_HOXC10_29 all_92 (gb/mw): CRT p={e29_crt['all_92']['gene_balanced']['p_mc']:.3f}/{e29_crt['all_92']['mutant_weighted']['p_mc']:.3f} | "
        f"K12 p={e29_k12['all_92']['gene_balanced']['p_mc']:.3f}/{e29_k12['all_92']['mutant_weighted']['p_mc']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
