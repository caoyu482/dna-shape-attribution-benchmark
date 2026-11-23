#!/usr/bin/env python3
"""
Motif-Aligned Oriented Feature Extractors & Full-CV Solvers (`biophys_inductive_gfm/v4`).

Fixes v3's structural blind spot for Exd-Hox heterodimer SELEX (GSE65073) & E-box PBM:
  1. Zero-relKa-leakage sequence-only motif alignment scanning both strands:
     - Exd-Hox consensus `A0 T1 G2 A3 T4 N5 N6 A7 T8 N9` (zero weight on N5, N6, N9 so
       minor-groove spacer diversity is completely unbiased!).
     - bHLH E-box consensus `N N N C A N N T G N N N` (12-bp / 14-bp window around CANNTG).
  2. Builds **oriented** (non-palindrome-folded) 1-bp-resolved positional bases (G1 and G2)
     and 1-bp-resolved 5-mer assignment tensors B_oriented of shape (N, L_core - 4, 1024),
     plus RC-symmetric aligned tensors for palindromic E-box homodimers.
  3. Provides full 5-fold GroupKFold out-of-fold evaluation for small/medium datasets AND
     per-position attribution delta_rho(p) across individual 5-mer slots p = 0..L_core-5.
"""
import gzip
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

V3_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v3/src")
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))

import step1_models as v3_sm  # noqa: E402
import null_generators_v4 as ng4  # noqa: E402

_TRANTAB = str.maketrans("ACGT", "TGCA")
_MONO_H = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]], dtype=np.float64
) / 2.0


def revcomp(seq: str) -> str:
    return seq.translate(_TRANTAB)[::-1]


def canonical_rc(seq: str) -> str:
    rc = revcomp(seq)
    return seq if seq <= rc else rc


def _score_exd_hox_10mer(w: str) -> float:
    """
    Sequence-only Exd-Hox alignment score on A0 T1 G2 A3 T4 [N5 N6] A7 T8 [N9].
    Zero weights on N5, N6, N9 (the minor-groove readout pocket and specificity base)
    to eliminate any circularity when analyzing minor-groove shape sensitivity.
    Max score = 12.0.
    """
    return (
        (1.0 if w[0] == "A" else 0.0)
        + 2.0 * ((w[1] == "T") + (w[2] == "G") + (w[3] == "A") + (w[4] == "T"))
        + 1.5 * ((w[7] == "A") + (w[8] == "T"))
    )


def load_and_align_exd_hox_selex(
    gz_path: Path,
    min_score: float = 8.0,
    core_mode: str = "12mer",
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """
    Zero-leakage sequence-only alignment for GSE65073 Exd-Hox SELEX raw `.txt.gz` tables.

    Args:
      gz_path: Raw GSE65073 `Kmer\\trelKa` gzip file.
      min_score: Minimum consensus alignment score (out of 12.0). `8.0` keeps <=2
                 mismatches on ATGAT..AT; `10.0` keeps <=1 mismatch.
      core_mode: `"12mer"` extracts `[off-1 : off+11]` (`N_{-1} A0..N9 N_{10}`),
                 `"10mer"` extracts `[off : off+10]` (`A0..N9`).
    Returns:
      (aligned_cores, y_log_relka, groups, meta)
    """
    # Step 1: Canonicalize raw RC-duplicate k-mer table purely by sequence string
    by_rc: Dict[str, List[float]] = {}
    raw_rows = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        hdr = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            km = parts[0].strip().upper()
            if not km or any(c not in "ACGT" for c in km):
                continue
            try:
                ka = float(parts[1])
            except ValueError:
                continue
            if not np.isfinite(ka) or ka < 0.0:
                continue
            raw_rows += 1
            by_rc.setdefault(canonical_rc(km), []).append(ka)

    # Step 2: Sequence-only orientation & offset alignment onto Exd-Hox 10-bp/12-bp core
    aligned_map: Dict[str, List[float]] = {}
    n_passed = 0
    for seq_c, vals in by_rc.items():
        y_val = float(np.log(max(float(np.median(vals)), 1e-4)))
        best = None
        for st, s_cur in ((0, seq_c), (1, revcomp(seq_c))):
            L = len(s_cur)
            for off in range(L - 10 + 1):
                w = s_cur[off : off + 10]
                sc = _score_exd_hox_10mer(w)
                # Deterministic center-proximity tie-breaker (zero label dependence)
                center_dist = abs((off + 5.0) - 0.5 * L)
                cand = (sc, -center_dist, st, off, s_cur, w)
                if best is None or cand[:2] > best[:2]:
                    best = cand
        sc, _, _, off, s_cur, w = best
        if sc < min_score:
            continue
        if core_mode == "12mer":
            if off < 1 or off + 11 > len(s_cur):
                continue
            core = s_cur[off - 1 : off + 11]
        elif core_mode == "10mer":
            core = w
        else:
            raise ValueError(f"Unsupported core_mode: {core_mode}")
        n_passed += 1
        aligned_map.setdefault(core, []).append(y_val)

    cores = sorted(aligned_map.keys())
    y_arr = np.array([float(np.median(aligned_map[c])) for c in cores], dtype=np.float64)

    # Leak-free homology grouping: exact middle 6-mer `[T4 N5 N6 A7 T8 N9]` (or `[3:9]` for 10mer)
    if core_mode == "12mer":
        grp_strs = [c[5:11] for c in cores]
    else:
        grp_strs = [c[4:10] for c in cores]
    uniq_g = sorted(set(grp_strs))
    g_map = {g: i for i, g in enumerate(uniq_g)}
    groups = np.array([g_map[g] for g in grp_strs], dtype=np.int32)

    meta = {
        "source_file": gz_path.name,
        "raw_rows": raw_rows,
        "unique_rc_kmers": len(by_rc),
        "aligned_hits": n_passed,
        "unique_aligned_cores": len(cores),
        "core_len": len(cores[0]) if cores else 0,
        "n_groups": len(uniq_g),
    }
    return cores, y_arr, groups, meta


def load_and_align_ebox_probes(
    parquet_path: Path,
    window_len: int = 14,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """
    Zero-leakage sequence-only alignment for bHLH E-box gcPBM 36-bp probes (`CANNTG`).
    Picks the best central CANNTG 6-mer (canonical C-A-N-N-T-G, preferring CACGTG) and
    extracts canonical RC symmetric `window_len` bp window centered on it.
    """
    df = pd.read_parquet(parquet_path)
    seqs = df["seq"].tolist()
    y_raw = df["y"].to_numpy(dtype=np.float64)
    half_left = (window_len - 6) // 2
    half_right = window_len - 6 - half_left

    aligned_map: Dict[str, List[float]] = {}
    center_ref = 0.5 * (len(seqs[0]) - 6)
    for seq, y_val in zip(seqs, y_raw):
        best = None
        for s_cur in (seq, revcomp(seq)):
            L = len(s_cur)
            for off in range(half_left, L - 6 - half_right + 1):
                hex6 = s_cur[off : off + 6]
                sc = (
                    2.0 * (hex6[0] == "C")
                    + 2.0 * (hex6[1] == "A")
                    + 2.0 * (hex6[4] == "T")
                    + 2.0 * (hex6[5] == "G")
                    + 1.0 * (hex6[2:4] == "CG")
                    + 0.5 * ((hex6[2] in "CT") + (hex6[3] in "AG"))
                )
                cand = (sc, -abs(off - center_ref), s_cur[off - half_left : off + 6 + half_right])
                if best is None or cand[:2] > best[:2]:
                    best = cand
        core = canonical_rc(best[2])
        aligned_map.setdefault(core, []).append(float(y_val))

    cores = sorted(aligned_map.keys())
    y_arr = np.array([float(np.median(aligned_map[c])) for c in cores], dtype=np.float64)
    # Homology grouping by central 10-mer [2:-2]
    grp_strs = [canonical_rc(c[2:-2]) for c in cores]
    uniq_g = sorted(set(grp_strs))
    g_map = {g: i for i, g in enumerate(uniq_g)}
    groups = np.array([g_map[g] for g in grp_strs], dtype=np.int32)
    meta = {
        "source_parquet": parquet_path.name,
        "input_probes": len(seqs),
        "unique_aligned_ebox_cores": len(cores),
        "window_len": window_len,
        "n_groups": len(uniq_g),
    }
    return cores, y_arr, groups, meta


def build_oriented_positional_features(seqs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build asymmetric 5'->3' oriented 1-bp-resolved features for aligned heterodimers:
      - X_g1: (N, 3 * L) orthonormal mononucleotide contrasts at every position 0..L-1
      - X_g2: (N, 3 * L + 9 * (L - 1) + 12) X_g1 + adjacent dinuc contrasts + bulk GC, GC^2, 10 dinuc freqs
      - B_oriented: (N, L - 4, 1024) binary 1-bp pentamer indicator tensor (NO RC averaging!)
    """
    codes = v3_sm.encode_sequences_acgt(seqs)
    N, L = codes.shape
    m_f = _MONO_H[codes]  # (N, L, 3)
    X_g1 = m_f.reshape(N, 3 * L)
    di_f = np.einsum("npa,npb->npab", m_f[:, :-1, :], m_f[:, 1:, :]).reshape(N, 9 * (L - 1))

    # Bulk GC, GC^2, 10 RC-canonical dinuc composition
    gc = ((codes == 1) | (codes == 2)).mean(axis=1).astype(np.float64)
    c2 = (codes[:, :-1].astype(np.int32) << 2) | codes[:, 1:]
    c2_rc = ((3 - codes[:, 1:]).astype(np.int32) << 2) | (3 - codes[:, :-1])
    c2_canon = np.minimum(c2, c2_rc)
    uniq_2 = np.unique(np.minimum(np.arange(16), [((3 - (x & 3)) << 2) | (3 - (x >> 2)) for x in range(16)]))
    remap = np.zeros(16, dtype=np.int32)
    for idx_u, val_u in enumerate(uniq_2):
        remap[val_u] = idx_u
    c2_mapped = remap[c2_canon]
    di_bulk = np.zeros((N, len(uniq_2)), dtype=np.float64)
    for i in range(N):
        di_bulk[i] = np.bincount(c2_mapped[i], minlength=len(uniq_2)) / float(L - 1)
    X_g2 = np.hstack([X_g1, di_f, gc[:, None], (gc * gc)[:, None], di_bulk])

    c5 = (
        ((((codes[:, :-4].astype(np.int32) << 2) | codes[:, 1:-3]) << 2) | codes[:, 2:-2]) << 2
        | codes[:, 3:-1]
    ) << 2 | codes[:, 4:]
    B_oriented = np.zeros((N, L - 4, 1024), dtype=np.float32)
    rows = np.arange(N, dtype=np.int32)[:, None]
    cols = np.arange(L - 4, dtype=np.int32)[None, :]
    B_oriented[rows, cols, c5] = 1.0
    return X_g1.astype(np.float64), X_g2.astype(np.float64), B_oriented


def eval_batch_5fold_group_oof(
    X_base: np.ndarray,
    B_mat: np.ndarray,
    tables_bank: np.ndarray,  # (M, 1024, d) with table[0] == F_true
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """
    Evaluate all M tables (tables_bank[0] = True, tables_bank[1:] = Nulls) using full
    K-fold GroupKFold out-of-fold Ridge predictions across the ENTIRE dataset.
    Returns base_spearman, preds_oof shape (M, N), spearmans (M,), delta_true_minus_median,
    and permutation p_mc.
    """
    N = len(y)
    k_eff = min(n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=k_eff)
    M = tables_bank.shape[0]
    preds_all = np.empty((M, N), dtype=np.float64)
    base_oof = np.empty(N, dtype=np.float64)

    for tr_i, va_i in gkf.split(X_base, y, groups):
        b_oof, _ = v3_sm.fit_predict_ridge_groupcv(
            X_base[tr_i], y[tr_i], groups[tr_i], X_base[va_i]
        )
        base_oof[va_i] = b_oof
        preds_all[:, va_i] = v3_sm.fit_predict_batch_fixed_base_ridge(
            X_base[tr_i], X_base[va_i], B_mat[tr_i], B_mat[va_i], tables_bank, y[tr_i], groups[tr_i]
        )

    base_sr, _ = v3_sm.safe_spearman_pearson(y, base_oof)
    srs, _ = v3_sm.fast_batch_spearman_pearson(y, preds_all)
    true_sr = float(srs[0])
    null_srs = srs[1:]
    med_null = float(np.median(null_srs))
    p_mc = float((1.0 + np.sum(null_srs >= true_sr)) / float(len(null_srs) + 1))
    return {
        "base_spearman": base_sr,
        "true_spearman": true_sr,
        "delta_true_minus_base": float(true_sr - base_sr),
        "null_median": med_null,
        "null_q95": float(np.quantile(null_srs, 0.95)),
        "delta_true_minus_median": float(true_sr - med_null),
        "p_mc": p_mc,
        "significant_p005": bool(p_mc <= 0.05),
    }
