"""
High-Resolution HGBDT Matched Surrogate Evaluation Core (`biophys_inductive_gfm/v7_1`).

Supports both:
  1. Exp 1a: High-resolution matched surrogate control (B=49 for ShiftGC_ExactCRT &
     Sparse4mer_K12; B=19 across K={4,8,24,48}) on both `dz_f70` and `dz_spec_f70`
     using deterministic WT-rank 500/mut training subsampling (`CFG_HGB_SURR60`)
     evaluated on ALL finite K=2,000 test probes, PLUS matched `HGBDT_Seq_Only`.
  2. Exp 1b: Direct full-K=2,000 (180k train rows, `CFG_HGB_100`) matched surrogate
     bridge (B=9) on both `dz_f70` and `dz_spec_f70`.

Offline / zero network imports for ScientistPro CoE Pillar 2 AST compliance.
"""
from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V3_SRC = ROOT / "v3" / "src"
V4_SRC = ROOT / "v4" / "src"
V5_SRC = ROOT / "v5" / "src"
V7_SRC = ROOT / "v7" / "src"
for _p in (V7_SRC, V5_SRC, V4_SRC, V3_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kock_pbm_v7 as kv7  # noqa: E402
import null_generators_v4 as ng4  # noqa: E402
import paired_perturbation_v5 as pp5  # noqa: E402

V4_DATA_DIR = ROOT / "v4" / "data"
V7_PROC_DIR = ROOT / "v7" / "data" / "processed"

CFG_HGB_100: Dict[str, object] = {
    "max_iter": 100,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 3.0,
    "max_bins": 64,
    "early_stopping": True,
    "n_iter_no_change": 10,
    "random_state": 42,
}

CFG_HGB_SURR60: Dict[str, object] = {
    "max_iter": 60,
    "learning_rate": 0.10,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 3.0,
    "max_bins": 64,
    "early_stopping": True,
    "n_iter_no_change": 10,
    "random_state": 42,
}

_WORKER_CTX: Dict[str, object] = {}


def _rss_mb() -> float:
    """Return current process peak RSS (VmHWM) in MiB."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _avail_ram_mb() -> float:
    """Return host available RAM in MiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values in original order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: float(pvals[i]))
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(pvals[idx]) * float(m - rank))
        if val > running_max:
            running_max = val
        adj[idx] = running_max
    return adj


def deterministic_wt_rank_subsamples(
    ds: Dict[str, np.ndarray],
    fin_masks: Sequence[np.ndarray],
    n_per_mut: int = 500,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """
    Reproduce exact v7 deterministic WT-Z-score linspace-stratified subsampling
    per mutant (zero mutant label leakage).
    """
    mut_gene_idx = ds["mut_gene_idx"]
    z_wt = ds["z_wt_f70"]
    sub_idx: List[np.ndarray] = []
    counts: List[int] = []
    for m in range(len(fin_masks)):
        g = int(mut_gene_idx[m])
        mk = fin_masks[m]
        zw = z_wt[g][mk]
        order = np.lexsort((np.arange(len(zw)), zw))
        k = min(n_per_mut, len(order))
        sel = order[np.linspace(0, len(order) - 1, k, dtype=int)]
        sub_idx.append(sel)
        counts.append(int(k))
    meta = {
        "rule": "deterministic_wt_z_score_sorted_linspace_stratified_per_mutant",
        "target_probes_per_mutant": int(n_per_mut),
        "actual_min_max_per_mutant": [int(min(counts)), int(max(counts))],
        "total_subset_probes_across_92_mutants": int(sum(counts)),
        "mean_training_rows_per_logo_fold": float(sum(counts) * 29.0 / 30.0),
        "evaluation_probes": "ALL_finite_K2000_test_probes",
        "mutant_label_leakage": False,
    }
    return sub_idx, meta


def _encode_g1_g2_gc_fast(c12: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match v7 `_build_hgb_feature_lists` exact columns `[pm, c12, g1, gc2]`."""
    g1 = kv7.encode_seq_g1_12bp(c12).astype(np.float32)
    g2 = kv7.encode_seq_g2_12bp(c12).astype(np.float32)
    gc2 = g2[:, -12:-10].astype(np.float32)
    return g1, gc2


def build_v7_hgbdt_fold_cache(
    target_key: str = "dz_f70",
    n_sub_per_mut: Optional[int] = 500,
    npz_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Build pre-stacked per-fold (30 LOGO folds) HGBDT sequence & c5 index cache
    so surrogate evaluations avoid repeated per-task `np.vstack` overhead while
    matching v7 feature construction to float32 exactness.
    """
    if target_key not in ("dz_f70", "dz_spec_f70"):
        raise ValueError(f"Unsupported target_key: {target_key}")
    p_npz = npz_path or (V7_PROC_DIR / "v7_kock_dataset.npz")
    ds = dict(np.load(p_npz, allow_pickle=False))

    n_genes = len(ds["gene_names"])
    n_mut = len(ds["mut_alleles"])
    mut_gene_idx = ds["mut_gene_idx"].astype(np.int32)
    phi_prot = ds["phi_prot"].astype(np.float32)
    win_codes_g = ds["win12_codes_f70"]
    y_mat = ds[target_key]

    # Per-gene 12-bp DNA sequence & c5_n1n4 indices
    gene_c12: List[np.ndarray] = []
    gene_seq_dna: List[np.ndarray] = []
    gene_c5_n1n4: List[np.ndarray] = []
    for g in range(n_genes):
        c12 = win_codes_g[g].astype(np.uint8)
        g1, gc2 = _encode_g1_g2_gc_fast(c12)
        c5_n1n4, _, _ = kv7.extract_pentamer_indices_12bp(c12)
        gene_c12.append(c12)
        gene_seq_dna.append(np.hstack([c12.astype(np.float32), g1, gc2]))
        gene_c5_n1n4.append(c5_n1n4.astype(np.int32))

    # Per-mutant full finite slices & deterministic WT-rank subsample
    fin_masks: List[np.ndarray] = []
    y_full: List[np.ndarray] = []
    X_seq_full: List[np.ndarray] = []
    c5_full: List[np.ndarray] = []
    for m in range(n_mut):
        mk = np.isfinite(y_mat[m])
        fin_masks.append(mk)
        y_m = y_mat[m, mk].astype(np.float32)
        y_full.append(y_m)
        g = int(mut_gene_idx[m])
        dna_m = gene_seq_dna[g][mk]
        pm = np.broadcast_to(phi_prot[m], (len(dna_m), phi_prot.shape[1]))
        X_seq_full.append(np.ascontiguousarray(np.hstack([pm, dna_m]), dtype=np.float32))
        c5_full.append(np.ascontiguousarray(gene_c5_n1n4[g][mk], dtype=np.int32))

    sub_meta: Dict[str, object]
    if n_sub_per_mut is not None:
        sub_idx, sub_meta = deterministic_wt_rank_subsamples(ds, fin_masks, n_sub_per_mut)
        X_seq_sub = [np.ascontiguousarray(X_seq_full[m][sub_idx[m]], dtype=np.float32) for m in range(n_mut)]
        c5_sub = [np.ascontiguousarray(c5_full[m][sub_idx[m]], dtype=np.int32) for m in range(n_mut)]
        y_sub = [np.ascontiguousarray(y_full[m][sub_idx[m]], dtype=np.float32) for m in range(n_mut)]
    else:
        sub_meta = {
            "rule": "full_all_finite_K2000_training_probes",
            "target_probes_per_mutant": 2000,
            "total_probes_across_92_mutants": int(sum(len(y) for y in y_full)),
            "evaluation_probes": "ALL_finite_K2000_test_probes",
            "mutant_label_leakage": False,
        }
        X_seq_sub = X_seq_full
        c5_sub = c5_full
        y_sub = y_full

    # Pre-stack 30 LOGO folds ONCE to avoid per-draw vstack
    folds: List[Dict[str, object]] = []
    for gh in range(n_genes):
        tr_m = [m for m in range(n_mut) if int(mut_gene_idx[m]) != gh]
        te_m = [m for m in range(n_mut) if int(mut_gene_idx[m]) == gh]
        folds.append(
            {
                "g_hold": int(gh),
                "te_muts": np.array(te_m, dtype=np.int32),
                "X_seq_tr": np.ascontiguousarray(np.vstack([X_seq_sub[m] for m in tr_m]), dtype=np.float32),
                "c5_tr": np.ascontiguousarray(np.vstack([c5_sub[m] for m in tr_m]), dtype=np.int32),
                "y_tr": np.ascontiguousarray(np.concatenate([y_sub[m] for m in tr_m]), dtype=np.float32),
                "X_seq_te": np.ascontiguousarray(np.vstack([X_seq_full[m] for m in te_m]), dtype=np.float32),
                "c5_te": np.ascontiguousarray(np.vstack([c5_full[m] for m in te_m]), dtype=np.int32),
                "y_te_list": [y_full[m].astype(np.float64) for m in te_m],
            }
        )

    c5_unlabeled_wt = np.vstack(gene_c5_n1n4).astype(np.int32)
    return {
        "target_key": target_key,
        "n_genes": int(n_genes),
        "n_mut": int(n_mut),
        "gene_names": ds["gene_names"],
        "mut_alleles": ds["mut_alleles"],
        "mut_gene_idx": mut_gene_idx,
        "mut_segments": ds["mut_segments"],
        "mut_kock_strata": ds["mut_kock_strata"],
        "c5_unlabeled_wt": c5_unlabeled_wt,
        "folds": folds,
        "subsampling": sub_meta,
    }


def _init_hgb_surr_worker(ctx: Dict[str, object]) -> None:
    global _WORKER_CTX
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_CTX = ctx


def _worker_fit_fold_task(args: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
    """
    Fit one (table_idx, g_hold) LOGO fold.
    `table_idx == -1` evaluates matched pure-sequence `HGBDT_Seq_Only`.
    `table_idx >= 0` appends `tables[table_idx]` pentamer shape lookup (4 slots * 7 ch = 28 cols).
    """
    tab_idx, gh = args
    c = _WORKER_CTX
    fld = c["folds"][gh]
    cfg: Dict[str, object] = c["cfg"]
    X_seq_tr = fld["X_seq_tr"]
    X_seq_te = fld["X_seq_te"]
    y_tr = fld["y_tr"]

    if tab_idx < 0:
        X_tr = X_seq_tr
        X_te = X_seq_te
    else:
        tab = c["tables"][tab_idx]
        sh_tr = tab[fld["c5_tr"]].reshape(len(X_seq_tr), 28)
        sh_te = tab[fld["c5_te"]].reshape(len(X_seq_te), 28)
        X_tr = np.hstack([X_seq_tr, sh_tr])
        X_te = np.hstack([X_seq_te, sh_te])

    model = HistGradientBoostingRegressor(**cfg)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    y_te_list = fld["y_te_list"]
    rhos = np.zeros(len(y_te_list), dtype=np.float64)
    cur = 0
    for i, yt in enumerate(y_te_list):
        nm = len(yt)
        rhos[i] = kv7.safe_spearman_pearson(yt, preds[cur : cur + nm])[0]
        cur += nm
    return tab_idx, gh, rhos


def evaluate_hgbdt_surrogate_table_logo(
    cache: Dict[str, object],
    tables: np.ndarray,
    cfg: Dict[str, object],
    include_seq_only: bool = False,
    n_workers: int = 6,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: str = "tab",
    progress_label: str = "HGBDT",
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Evaluate `tables` of shape (M, 1024, 7) (plus optional `Seq_Only`) across all
    30 LOGO folds on ALL finite K=2,000 test probes. Supports per-table checkpointing.
    Returns `(seq_only_rhos_92_or_None, table_rhos_M_x_92, elapsed_sec)`.
    """
    t0 = time.time()
    n_genes = int(cache["n_genes"])
    n_mut = int(cache["n_mut"])
    folds = cache["folds"]
    tabs_f32 = np.asarray(tables, dtype=np.float32)
    M = tabs_f32.shape[0]

    seq_only_rhos: Optional[np.ndarray] = np.zeros(n_mut, dtype=np.float64) if include_seq_only else None
    table_rhos = np.zeros((M, n_mut), dtype=np.float64)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pending_ids: List[int] = []
    if include_seq_only:
        p_sq = (checkpoint_dir / f"{checkpoint_prefix}_seqonly.npy") if checkpoint_dir else None
        if p_sq is not None and p_sq.exists():
            seq_only_rhos[:] = np.load(p_sq)
        else:
            pending_ids.append(-1)

    for m in range(M):
        p_m = (checkpoint_dir / f"{checkpoint_prefix}_{m:04d}.npy") if checkpoint_dir else None
        if p_m is not None and p_m.exists():
            table_rhos[m] = np.load(p_m)
        else:
            pending_ids.append(m)

    if not pending_ids:
        print(f"  [{progress_label}] 100% cache hit for M={M} tables (include_seq={include_seq_only}).", flush=True)
        return seq_only_rhos, table_rhos, float(time.time() - t0)

    print(
        f"  [{progress_label}] Running {len(pending_ids)} table entries x {n_genes} LOGO folds "
        f"(n_workers={n_workers}, avail_ram={_avail_ram_mb():.0f}MB)...",
        flush=True,
    )

    # Batch pending tables in chunks of <= 12 so checkpoints hit disk frequently & RSS stays flat
    batch_sz = 12 if n_workers <= 6 else 16
    ctx_base = {
        "folds": folds,
        "tables": tabs_f32,
        "cfg": dict(cfg),
    }
    mp_ctx = mp.get_context("spawn")
    done_cnt = 0
    with mp_ctx.Pool(processes=n_workers, initializer=_init_hgb_surr_worker, initargs=(ctx_base,)) as pool:
        for b0 in range(0, len(pending_ids), batch_sz):
            b_ids = pending_ids[b0 : b0 + batch_sz]
            tasks = [(tid, gh) for tid in b_ids for gh in range(n_genes)]
            buf: Dict[int, np.ndarray] = {tid: np.zeros(n_mut, dtype=np.float64) for tid in b_ids}
            for tid, gh, r_fold in pool.map(_worker_fit_fold_task, tasks):
                te_muts = folds[gh]["te_muts"]
                buf[tid][te_muts] = r_fold
            for tid in b_ids:
                if tid < 0:
                    assert seq_only_rhos is not None
                    seq_only_rhos[:] = buf[tid]
                    if checkpoint_dir is not None:
                        np.save(checkpoint_dir / f"{checkpoint_prefix}_seqonly.npy", seq_only_rhos)
                else:
                    table_rhos[tid] = buf[tid]
                    if checkpoint_dir is not None:
                        np.save(checkpoint_dir / f"{checkpoint_prefix}_{tid:04d}.npy", table_rhos[tid])
            done_cnt += len(b_ids)
            el = time.time() - t0
            print(
                f"    [{progress_label}] {done_cnt}/{len(pending_ids)} pending done in {el:.1f}s "
                f"(rss={_rss_mb():.0f}MB, avail={_avail_ram_mb():.0f}MB)",
                flush=True,
            )

    return seq_only_rhos, table_rhos, float(time.time() - t0)


def build_hgbdt_surrogate_bank(
    c5_unlabeled: np.ndarray,
    seed: int = 2026071,
    b_crt: int = 49,
    b_k12: int = 49,
    b_kladder: int = 19,
    k_other: Sequence[int] = (4, 8, 24, 48),
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """
    Construct the reproducible unlabeled surrogate table bank (`F_true` at index 0)
    matching v7 RC-even symmetry invariants (`error <= 1e-13`).
    """
    F_true, _ = kv7.load_biophysical_table_matrix(include_odd_step_diff=False)
    p_raw = np.bincount(c5_unlabeled.ravel(), minlength=1024).astype(np.float64)
    n_obs = float(c5_unlabeled.size)
    rng = np.random.default_rng(seed)

    parts: List[np.ndarray] = [F_true[None, :, :]]
    slices: Dict[str, slice] = {}
    cur = 1

    crt_g2 = ng4.ShiftGCExactCRTNull(F_true, p_raw, conditioning="WS_G2", n_total_obs=n_obs)
    parts.append(np.stack([crt_g2.sample(rng) for _ in range(b_crt)], axis=0))
    slices["ShiftGC_ExactCRT_WS_G2"] = slice(cur, cur + b_crt)
    cur += b_crt

    sp12 = pp5.Sparse4merSurrogateNull(F_true, p_raw, n_total_obs=n_obs, n_pockets_per_ch=12)
    parts.append(np.stack([sp12.sample(rng) for _ in range(b_k12)], axis=0))
    slices["Sparse4mer_K12"] = slice(cur, cur + b_k12)
    cur += b_k12

    for k in k_other:
        spk = pp5.Sparse4merSurrogateNull(F_true, p_raw, n_total_obs=n_obs, n_pockets_per_ch=int(k))
        parts.append(np.stack([spk.sample(rng) for _ in range(b_kladder)], axis=0))
        slices[f"Sparse4mer_K{int(k)}"] = slice(cur, cur + b_kladder)
        cur += b_kladder

    bank = np.concatenate(parts, axis=0).astype(np.float64)
    return bank, slices


def summarize_hgbdt_surrogate_family(
    seq_only_rhos: Optional[np.ndarray],
    true_rhos: np.ndarray,
    null_dict: Dict[str, np.ndarray],
    mut_gene_idx: np.ndarray,
    mut_segments: np.ndarray,
    mut_kock_strata: np.ndarray,
    n_boot: int = 5000,
    boot_seed: int = 42,
) -> Dict[str, object]:
    """
    Comprehensive 7-stratum Monte Carlo surrogate summary with paired lift vs
    matched `HGBDT_Seq_Only`, exact `p_mc = (1+k)/(B+1)`, and Holm-Bonferroni `p_holm`.
    """
    n_genes = int(np.max(mut_gene_idx) + 1)

    def _stratum_masks() -> Dict[str, np.ndarray]:
        return {
            "all_92": np.ones(len(true_rhos), dtype=bool),
            "N_arm_1_9": mut_segments == "N-arm",
            "H1_H2_linkers": np.isin(mut_segments, ["H1", "L12", "H2", "L23"]),
            "H3_42_58": mut_segments == "H3",
            "specificity_yes": mut_kock_strata == "specificity_yes",
            "affinity_only": mut_kock_strata == "affinity_only",
            "no_change": mut_kock_strata == "no_change",
        }

    def _gene_mean(vec: np.ndarray) -> float:
        return float(np.mean([np.mean(vec[mut_gene_idx == g]) for g in range(n_genes)]))

    def _gene_mean_mat(mat: np.ndarray) -> np.ndarray:
        cols = [np.mean(mat[:, mut_gene_idx == g], axis=1) for g in range(n_genes)]
        return np.mean(np.column_stack(cols), axis=1)

    masks = _stratum_masks()
    true_gm = _gene_mean(true_rhos)
    true_mm = float(np.mean(true_rhos))

    seq_summary: Dict[str, object] = {}
    if seq_only_rhos is not None:
        seq_gm = _gene_mean(seq_only_rhos)
        seq_mm = float(np.mean(seq_only_rhos))
        lift_stats = kv7.compute_stratified_metrics(
            true_rhos,
            seq_only_rhos,
            mut_gene_idx,
            mut_segments,
            mut_kock_strata,
            n_boot=n_boot,
            seed=boot_seed,
        )
        seq_summary = {
            "seq_only_gene_mean_spearman": seq_gm,
            "seq_only_mut_mean_spearman": seq_mm,
            "lift_true_minus_seq_gene_mean": float(true_gm - seq_gm),
            "lift_true_minus_seq_mut_mean": float(true_mm - seq_mm),
            "stratified_lift_true_vs_seq": lift_stats,
        }

    fam_out: Dict[str, Dict[str, object]] = {}
    fam_names = list(null_dict.keys())
    raw_gm_pvals: List[float] = []
    raw_mm_pvals: List[float] = []
    raw_narm_pvals: List[float] = []
    raw_spec_pvals: List[float] = []

    for nk in fam_names:
        mat = np.asarray(null_dict[nk], dtype=np.float64)
        B = mat.shape[0]
        null_gms = _gene_mean_mat(mat)
        null_mms = np.mean(mat, axis=1)
        p_gm = float((1.0 + np.sum(null_gms >= true_gm)) / float(B + 1))
        p_mm = float((1.0 + np.sum(null_mms >= true_mm)) / float(B + 1))
        raw_gm_pvals.append(p_gm)
        raw_mm_pvals.append(p_mm)

        strata_res: Dict[str, Dict[str, object]] = {}
        for sk, smask in masks.items():
            if int(np.sum(smask)) == 0:
                continue
            t_s = float(np.mean(true_rhos[smask]))
            n_s = np.mean(mat[:, smask], axis=1)
            p_s = float((1.0 + np.sum(n_s >= t_s)) / float(B + 1))
            s_seq = float(np.mean(seq_only_rhos[smask])) if seq_only_rhos is not None else None
            strata_res[sk] = {
                "n_mutants": int(np.sum(smask)),
                "true_mut_mean": t_s,
                "seq_only_mut_mean": s_seq,
                "null_median": float(np.median(n_s)),
                "null_q95": float(np.quantile(n_s, 0.95)),
                "null_min": float(np.min(n_s)),
                "null_max": float(np.max(n_s)),
                "delta_true_minus_null_median": float(t_s - np.median(n_s)),
                "null_median_lift_over_seq": float(np.median(n_s) - s_seq) if s_seq is not None else None,
                "true_lift_over_seq": float(t_s - s_seq) if s_seq is not None else None,
                "p_mc": p_s,
            }
        raw_narm_pvals.append(strata_res["N_arm_1_9"]["p_mc"])
        raw_spec_pvals.append(strata_res["specificity_yes"]["p_mc"])

        seq_gm = _gene_mean(seq_only_rhos) if seq_only_rhos is not None else None
        fam_out[nk] = {
            "B": int(B),
            "p_min_possible": float(1.0 / float(B + 1)),
            "true_gene_mean_rho": true_gm,
            "null_gene_mean_median": float(np.median(null_gms)),
            "null_gene_mean_q95": float(np.quantile(null_gms, 0.95)),
            "null_gene_mean_range": [float(np.min(null_gms)), float(np.max(null_gms))],
            "delta_gene_mean_minus_null_median": float(true_gm - np.median(null_gms)),
            "null_gene_lift_over_seq_median": float(np.median(null_gms) - seq_gm) if seq_gm is not None else None,
            "p_mc_gene_mean": p_gm,
            "true_mut_mean_rho": true_mm,
            "null_mut_mean_median": float(np.median(null_mms)),
            "null_mut_mean_q95": float(np.quantile(null_mms, 0.95)),
            "delta_mut_mean_minus_null_median": float(true_mm - np.median(null_mms)),
            "p_mc_mut_mean": p_mm,
            "strata": strata_res,
        }

    holm_gm = _holm_bonferroni(raw_gm_pvals)
    holm_mm = _holm_bonferroni(raw_mm_pvals)
    holm_narm = _holm_bonferroni(raw_narm_pvals)
    holm_spec = _holm_bonferroni(raw_spec_pvals)
    for i, nk in enumerate(fam_names):
        fam_out[nk]["p_holm_gene_mean"] = float(holm_gm[i])
        fam_out[nk]["p_holm_mut_mean"] = float(holm_mm[i])
        fam_out[nk]["p_holm_narm"] = float(holm_narm[i])
        fam_out[nk]["p_holm_spec_yes"] = float(holm_spec[i])

    return {
        "true_gene_mean_spearman": true_gm,
        "true_mut_mean_spearman": true_mm,
        "seq_only_comparison": seq_summary,
        "null_families": fam_out,
    }
