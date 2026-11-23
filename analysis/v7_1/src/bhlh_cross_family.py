"""
Cross-Family bHLH (`Max`, `Mad1`, `cMyc`) & Replicate-Null Core Module (`biophys_inductive_gfm/v7_1`).

Evaluates aligned 14-bp E-box cohorts (`NNNNCANNTGNNNN`, zero-leakage sequence-only
CANNTG alignment + central-10mer GroupKFold CV) across:
  - 3 Single-TF affinity endpoints: `Max`, `Mad1`, `cMyc`
  - 3 Paired between-paralog specificity differentials: `cMyc - Max`, `Mad1 - Max`, `cMyc - Mad1`
  - 9 Within-array same-protein technical replicate-difference negative controls (`GSE59845`)
  - Orthogonal universal PBM (`GSE97794`: `Max`, `MadMax`, `MycMax`, `MycMax - Max`, `MadMax - Max`)

Model ladder: `Seq_G1`, `Seq_G2`, `Seq_Pos4mer`, `Poly3_Kernel`, `HGBDT_Seq` vs `+DNAshape (7ch, MGW)`.
Null calibration battery: `Naive_RC_Perm`, `Shapelizer_2018_Std`, `ShiftGC_ExactCRT_WS_G2`, and
`Sparse4mer K in {4, 8, 12, 24, 48}` (`B=199`) + matched `HGBDT` surrogates (`B=49`).

Offline / zero network imports for ScientistPro CoE Pillar 2 AST compliance.
"""
from __future__ import annotations

import gzip
import math
import multiprocessing as mp
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh
from scipy.stats import kstest, pearsonr, rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V4_SRC = ROOT / "v4" / "src"
V5_SRC = ROOT / "v5" / "src"
V7_SRC = ROOT / "v7" / "src"
for _p in (V7_SRC, V5_SRC, V4_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import null_generators_v4 as ng4  # noqa: E402

_TRANTAB = str.maketrans("ACGT", "TGCA")
K_LADDER_BHLH: Tuple[int, ...] = (4, 8, 12, 24, 48)

def _lazy_v5_v7():
    import kock_pbm_v7 as kv7
    import paired_perturbation_v5 as pp5
    return kv7, pp5

_ASCII_LUT = np.full(256, -1, dtype=np.int8)
for _c, _v in (("A", 0), ("a", 0), ("C", 1), ("c", 1), ("G", 2), ("g", 2), ("T", 3), ("t", 3)):
    _ASCII_LUT[ord(_c)] = _v

_MONO_H = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]], dtype=np.float64
) / 2.0


def revcomp_str(seq: str) -> str:
    return seq.translate(_TRANTAB)[::-1]


def canonical_rc_str(seq: str) -> str:
    rc = revcomp_str(seq)
    return seq if seq <= rc else rc


def encode_sequences_acgt(seqs: Sequence[str]) -> np.ndarray:
    """Encode equal-length ACGT strings to (N, L) int8 cleanly in pure NumPy."""
    N = len(seqs)
    L = len(seqs[0])
    buf = "".join(seqs).encode("ascii")
    arr = _ASCII_LUT[np.frombuffer(buf, dtype=np.uint8)].reshape(N, L).copy()
    if np.any(arr < 0):
        raise ValueError("Invalid non-ACGT base encountered in sequence array")
    return arr


def safe_spearman_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if len(y_true) < 3 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return 0.0, 0.0
    sr = spearmanr(y_true, y_pred).statistic
    pr = pearsonr(y_true, y_pred).statistic
    return (0.0 if np.isnan(sr) else float(sr)), (0.0 if np.isnan(pr) else float(pr))


def fast_batch_spearman_pearson(y_true: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact rankdata-based batched Spearman & Pearson over preds shape `(M, N)`."""
    M, N = preds.shape
    yt = np.asarray(y_true, dtype=np.float64)
    yt_c = yt - yt.mean()
    yn = float(np.linalg.norm(yt_c))
    ry = rankdata(yt, method="average")
    ry_c = ry - ry.mean()
    ryn = float(np.linalg.norm(ry_c))
    if yn < 1e-12 or ryn < 1e-12:
        return np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64)

    pr = np.zeros(M, dtype=np.float64)
    sr = np.zeros(M, dtype=np.float64)
    for m in range(M):
        p = preds[m]
        pc = p - p.mean()
        pn = float(np.linalg.norm(pc))
        if pn > 1e-12:
            pr[m] = float(np.dot(pc, yt_c) / (pn * yn))
            rp = rankdata(p, method="average")
            rpc = rp - rp.mean()
            rpn = float(np.linalg.norm(rpc))
            if rpn > 1e-12:
                sr[m] = float(np.dot(rpc, ry_c) / (rpn * ryn))
    return sr, pr


def extract_c5_pair(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(c5_fwd, c5_rc)` shape `(N, L-4)`."""
    c5 = (((((codes[:, :-4].astype(np.int32) << 2) | codes[:, 1:-3]) << 2) | codes[:, 2:-2]) << 2 | codes[:, 3:-1]) << 2 | codes[:, 4:]
    c5_rc = ng4.RC_5[c5][:, ::-1]
    return c5, c5_rc


def build_spatial_bin_tensor(c5: np.ndarray, c5_rc: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Build RC-symmetrized spatial pentamer tensor `(N, n_bins, 1024)` float32."""
    N, L5 = c5.shape
    bin_ids = np.arange(L5, dtype=np.int32) if n_bins >= L5 else np.floor(np.arange(L5) * n_bins / float(L5)).astype(np.int32)
    n_eff = L5 if n_bins >= L5 else n_bins
    B_mat = np.zeros((N, n_eff, 1024), dtype=np.float32)
    rows = np.arange(N, dtype=np.int32)[:, None]
    np.add.at(B_mat, (rows, bin_ids[None, :], c5), 0.5)
    np.add.at(B_mat, (rows, bin_ids[None, :], c5_rc), 0.5)
    counts = np.bincount(bin_ids, minlength=n_eff).astype(np.float32)[None, :, None]
    B_mat /= np.maximum(counts, 1.0)
    return B_mat


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values in original index order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    out = [0.0] * m
    run_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * float(m - rank))
        if val > run_max:
            run_max = val
        out[idx] = run_max
    return out


def align_ebox_14bp_cores(
    seqs: Sequence[str],
    y_map: Dict[str, np.ndarray],
    window_len: int = 14,
) -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray, Dict[str, object]]:
    """
    Zero-leakage sequence-only CANNTG alignment matching `v4::load_and_align_ebox_probes`
    across paired target vectors `y_map` simultaneously.
    Returns `(cores, y_agg_dict, groups, meta)` with RC-canonical 14-bp `cores`
    and central-10mer homology groups (`canonical_rc(c[2:-2])`).
    """
    half_left = (window_len - 6) // 2
    half_right = window_len - 6 - half_left
    center_ref = 0.5 * (len(seqs[0]) - 6)

    keys = list(y_map.keys())
    aligned_vals: Dict[str, Dict[str, List[float]]] = {k: defaultdict(list) for k in keys}
    core_probe_counts: Dict[str, int] = defaultdict(int)

    for idx, seq in enumerate(seqs):
        best = None
        for s_cur in (seq, revcomp_str(seq)):
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
        core = canonical_rc_str(best[2])
        core_probe_counts[core] += 1
        for k in keys:
            val = float(y_map[k][idx])
            if np.isfinite(val):
                aligned_vals[k][core].append(val)

    cores = sorted(core_probe_counts.keys())
    y_agg: Dict[str, np.ndarray] = {}
    for k in keys:
        y_agg[k] = np.array(
            [float(np.median(aligned_vals[k][c])) if aligned_vals[k][c] else np.nan for c in cores],
            dtype=np.float64,
        )

    grp_strs = [canonical_rc_str(c[2:-2]) for c in cores]
    uniq_g = sorted(set(grp_strs))
    g_map = {g: i for i, g in enumerate(uniq_g)}
    groups = np.array([g_map[g] for g in grp_strs], dtype=np.int32)
    meta = {
        "input_probes": len(seqs),
        "unique_aligned_ebox_cores": len(cores),
        "window_len": int(window_len),
        "n_groups": len(uniq_g),
    }
    return cores, y_agg, groups, meta


def _build_rc4_table() -> Tuple[np.ndarray, np.ndarray]:
    """Return `(rc4_map (256,), canon4_unique)` for 4-mers."""
    d4 = np.array(np.unravel_index(np.arange(256), (4,) * 4)).T
    rc4 = np.ravel_multi_index((3 - d4[:, ::-1]).T, (4,) * 4).astype(np.int32)
    canon_ids = np.unique(np.minimum(np.arange(256, dtype=np.int32), rc4))
    return rc4, canon_ids


_RC4_MAP, _CANON4_IDS = _build_rc4_table()


def build_rc_sym_ebox_features(
    cores: Sequence[str],
    max_pos_bins_seq: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Build strictly RC-invariant (`phi(s) == phi(rc(s))`) pure-sequence & 5-mer features:
      - `codes`: `(N, 14)` int8
      - `X_g1`: non-redundant RC-symmetrized mononucleotide contrasts (`21` cols for `L=14`)
      - `X_g2`: `X_g1` + RC-symmetrized adj-dinucleotide contrasts + GC + GC^2 + 10 bulk dinuc (`96` cols pre-prune -> `93` full rank)
      - `X_pos4mer`: RC-symmetrized 11-slot positional 4-mers (`1,416` cols, RC-exact!)
      - `B_rc`: `(N, 10, 1024)` float32 1-bp-resolved RC-symmetrized pentamer tensor
      - `c5_all`: `(N, 20)` raw fwd+rc pentamer indices (for p_raw estimation)
    """
    _ = max_pos_bins_seq
    codes = encode_sequences_acgt(cores)
    rc_codes = (3 - codes[:, ::-1]).copy()
    N, L = codes.shape

    m_f = _MONO_H[codes]
    m_r = _MONO_H[rc_codes]
    m_sym = 0.5 * (m_f + m_r)
    half_m = (L + 1) // 2
    X_g1 = m_sym[:, :half_m, :].reshape(N, half_m * 3)

    di_f = np.einsum("npa,npb->npab", m_f[:, :-1, :], m_f[:, 1:, :]).reshape(N, L - 1, 9)
    di_r = np.einsum("npa,npb->npab", m_r[:, :-1, :], m_r[:, 1:, :]).reshape(N, L - 1, 9)
    di_sym = 0.5 * (di_f + di_r)
    half_d = (L - 1 + 1) // 2
    X_di = di_sym[:, :half_d, :].reshape(N, half_d * 9)

    gc = ((codes == 1) | (codes == 2)).mean(axis=1).astype(np.float64)
    c2 = (codes[:, :-1].astype(np.int32) << 2) | codes[:, 1:]
    c2_rc = ((3 - codes[:, 1:]).astype(np.int32) << 2) | (3 - codes[:, :-1])
    c2_canon = np.minimum(c2, c2_rc)
    uniq_2 = np.unique(np.minimum(np.arange(16), [((3 - (x & 3)) << 2) | (3 - (x >> 2)) for x in range(16)]))
    remap2 = np.zeros(16, dtype=np.int32)
    for iu, vu in enumerate(uniq_2):
        remap2[vu] = iu
    c2_mapped = remap2[c2_canon]
    di_bulk = np.zeros((N, len(uniq_2)), dtype=np.float64)
    for i in range(N):
        di_bulk[i] = np.bincount(c2_mapped[i], minlength=len(uniq_2)) / float(L - 1)
    X_g2 = np.hstack([X_g1, X_di, gc[:, None], (gc * gc)[:, None], di_bulk])

    c5, c5rc = extract_c5_pair(codes)
    n_bins = L - 4  # 10 for 14-bp
    B_rc = build_spatial_bin_tensor(c5, c5rc, n_bins=n_bins)

    # Positional 4-mers across L - 3 = 11 slots, folded strictly under RC: (p, k) <-> (10 - p, rc4(k))
    c4 = (
        ((((codes[:, :-3].astype(np.int32) << 2) | codes[:, 1:-2]) << 2) | codes[:, 2:-1]) << 2
        | codes[:, 3:]
    )  # (N, L - 3)
    L4 = L - 3
    half = L4 // 2
    sym_blocks: List[np.ndarray] = []
    rows = np.arange(N, dtype=np.int32)
    for p in range(half):
        q = L4 - 1 - p
        blk = np.zeros((N, 256), dtype=np.float32)
        np.add.at(blk, (rows, c4[:, p]), 0.5)
        np.add.at(blk, (rows, _RC4_MAP[c4[:, q]]), 0.5)
        sym_blocks.append(blk)
    if L4 % 2 == 1:
        mid = half
        c_mid = np.minimum(c4[:, mid], _RC4_MAP[c4[:, mid]])
        remap_mid = np.full(256, -1, dtype=np.int32)
        remap_mid[_CANON4_IDS] = np.arange(len(_CANON4_IDS), dtype=np.int32)
        blk_m = np.zeros((N, len(_CANON4_IDS)), dtype=np.float32)
        blk_m[rows, remap_mid[c_mid]] = 1.0
        sym_blocks.append(blk_m)
    X_pos4mer = np.hstack(sym_blocks).astype(np.float64)

    return {
        "codes": codes,
        "X_g1": X_g1,
        "X_g2": X_g2,
        "X_pos4mer": X_pos4mer,
        "B_rc": B_rc,
        "c5_all": np.hstack([c5, c5rc]),
    }


def _contract_b_rc_single(B_rc: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Contract (N, n_bins, 1024) B_rc with (1024, d) table into (N, n_bins * d)."""
    N, nb, _ = B_rc.shape
    return (B_rc.astype(np.float64) @ np.asarray(table, dtype=np.float64)).reshape(N, nb * table.shape[1])


def fit_predict_poly3_kernel_ridge(
    X_seq_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_seq_te: np.ndarray,
    X_mod_tr: Optional[np.ndarray] = None,
    X_mod_te: Optional[np.ndarray] = None,
    gammas: Sequence[float] = (0.5, 1.0, 2.0),
    lams_seq: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    lams_mod: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_inner_folds: int = 2,
) -> Dict[str, object]:
    """
    Exact degree-3 inhomogeneous polynomial kernel Ridge on `X_seq` with optional
    additive two-kernel linear block on `X_mod` (`(1/ls)*K_poly3 + (1/lm)*Z_mod@Z_mod.T`),
    using exact Woodbury reduction over `p_m <= 70` in `K_poly3`'s eigenbasis.
    """
    ms = X_seq_tr.mean(axis=0, keepdims=True)
    ss = np.maximum(X_seq_tr.std(axis=0, keepdims=True), 1e-8)
    Zs_tr = (X_seq_tr - ms) / ss
    Zs_te = (X_seq_te - ms) / ss
    p_s = float(Zs_tr.shape[1])
    D_tr = (Zs_tr @ Zs_tr.T) / p_s
    D_te = (Zs_te @ Zs_tr.T) / p_s

    if X_mod_tr is None:
        gkf = GroupKFold(n_splits=min(max(3, n_inner_folds), len(np.unique(groups_tr))))
        splits = list(gkf.split(Zs_tr, y_tr, groups_tr))[:n_inner_folds]
        lam_arr = np.asarray(lams_seq, dtype=np.float64)
        y_mean = float(y_tr.mean())
        yc = y_tr - y_mean
        best_mse, best_g, best_l = np.inf, float(gammas[0]), float(lam_arr[min(1, len(lam_arr) - 1)])
        for g in gammas:
            K_all = np.power(1.0 + float(g) * D_tr, 3)
            mse_g = np.zeros(len(lam_arr), dtype=np.float64)
            for i_tr, i_va in splits:
                ym_f = float(y_tr[i_tr].mean())
                Kaa = K_all[np.ix_(i_tr, i_tr)]
                Kva = K_all[np.ix_(i_va, i_tr)]
                ev_d, Qd = sp_eigh(Kaa, check_finite=False, driver="evd")
                ev_d = np.maximum(ev_d, 0.0)
                proj = Qd.T @ (y_tr[i_tr] - ym_f)
                preds = (Kva @ Qd) @ (proj[:, None] / (ev_d[:, None] + lam_arr[None, :]))
                diff = preds - (y_tr[i_va] - ym_f)[:, None]
                mse_g += np.sum(diff * diff, axis=0)
            ib = int(np.argmin(mse_g))
            if mse_g[ib] < best_mse:
                best_mse = float(mse_g[ib])
                best_g, best_l = float(g), float(lam_arr[ib])
        K_tr = np.power(1.0 + best_g * D_tr, 3)
        K_tr[np.diag_indices(len(Zs_tr))] += best_l
        al = cho_solve(cho_factor(K_tr, lower=True, check_finite=False), yc, check_finite=False)
        pred_te = y_mean + np.power(1.0 + best_g * D_te, 3) @ al
        return {"pred_te": pred_te, "best_gamma": best_g, "best_lambda": best_l}

    assert X_mod_te is not None
    mm = X_mod_tr.mean(axis=0, keepdims=True)
    sm = np.maximum(X_mod_tr.std(axis=0, keepdims=True), 1e-8)
    Zm_tr = (np.asarray(X_mod_tr, dtype=np.float64) - mm) / sm
    Zm_te = (np.asarray(X_mod_te, dtype=np.float64) - mm) / sm
    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)

    gkf = GroupKFold(n_splits=min(max(3, n_inner_folds), len(np.unique(groups_tr))))
    splits = list(gkf.split(Zs_tr, yc, groups_tr))[:n_inner_folds]

    best_mse = np.inf
    best_g = float(gammas[0])
    best_ls = float(ls_arr[min(1, len(ls_arr) - 1)])
    best_lm = float(lm_arr[min(1, len(lm_arr) - 1)])

    for g in gammas:
        K_all = np.power(1.0 + float(g) * D_tr, 3)
        mse_grid = np.zeros((len(ls_arr), len(lm_arr)), dtype=np.float64)
        for tr_i, va_i in splits:
            ym_f = float(y_tr[tr_i].mean())
            ya, yv = y_tr[tr_i] - ym_f, y_tr[va_i] - ym_f
            Zma, Zmv = Zm_tr[tr_i], Zm_tr[va_i]
            Kaa = K_all[np.ix_(tr_i, tr_i)]
            Kva = K_all[np.ix_(va_i, tr_i)]
            ev_d, Qd = sp_eigh(Kaa, check_finite=False, driver="evd")
            ev_d = np.maximum(ev_d, 0.0)
            qd = Qd.T @ ya
            Zm_q = Qd.T @ Zma          # (na, p_m)
            Kva_Qd = Kva @ Qd          # (nv, na)
            for i_s, ls in enumerate(ls_arr):
                w_d = ls / (ev_d + ls)
                S = (Zm_q * w_d[:, None]).T @ Zm_q
                S = 0.5 * (S + S.T)
                rhs_m = Zm_q.T @ (w_d * qd)
                ev_t, Qt = sp_eigh(S, check_finite=False, driver="evd")
                ev_t = np.maximum(ev_t, 0.0)
                proj_t = Qt.T @ rhs_m
                for i_m, lm in enumerate(lm_arr):
                    wm = Qt @ (proj_t / (ev_t + lm))
                    res_q = qd - Zm_q @ wm
                    pred_v = Kva_Qd @ ((w_d * res_q) / ls) + Zmv @ wm
                    diff = pred_v - yv
                    mse_grid[i_s, i_m] += float(np.sum(diff * diff))
        bi, bj = np.unravel_index(int(np.argmin(mse_grid)), mse_grid.shape)
        if mse_grid[bi, bj] < best_mse:
            best_mse = float(mse_grid[bi, bj])
            best_g = float(g)
            best_ls = float(ls_arr[bi])
            best_lm = float(lm_arr[bj])

    # Exact full-training fit with Woodbury over Zm_tr
    K_tr = np.power(1.0 + best_g * D_tr, 3)
    K_te = np.power(1.0 + best_g * D_te, 3)
    ev_f, Qf = sp_eigh(K_tr, check_finite=False, driver="evd")
    ev_f = np.maximum(ev_f, 0.0)
    w_df = best_ls / (ev_f + best_ls)
    qf = Qf.T @ yc
    Zmf_q = Qf.T @ Zm_tr
    Sf = (Zmf_q * w_df[:, None]).T @ Zmf_q
    Sf[np.diag_indices(Zm_tr.shape[1])] += best_lm
    cf_s = cho_factor(Sf, lower=True, check_finite=False)
    wm_f = cho_solve(cf_s, Zmf_q.T @ (w_df * qf), check_finite=False)
    res_qf = qf - Zmf_q @ wm_f
    al_f = Qf @ ((w_df * res_qf) / best_ls)
    pred_te = y_mean + K_te @ al_f + Zm_te @ wm_f
    return {"pred_te": pred_te, "best_gamma": best_g, "best_lam_seq": best_ls, "best_lam_mod": best_lm}


def generate_multisparsity_null_bank(
    c5_unlabeled: np.ndarray,
    shap_std: np.ndarray,
    B: int = 199,
    seed: int = 2026072,
    k_list: Sequence[int] = K_LADDER_BHLH,
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """Build combined `(F_true + all 8 null families)` table bank using v7 helper."""
    kv7, _ = _lazy_v5_v7()
    F_true, _ = kv7.load_biophysical_table_matrix(include_odd_step_diff=False)
    return kv7.build_multi_sparsity_null_banks(
        F_true, c5_unlabeled, shap_std, B=B, seed=seed, k_list=k_list, include_naive=True
    )


def verify_table_rc_invariants(tables: np.ndarray) -> float:
    """Verify max RC asymmetry residual across a bank `(M, 1024, 7)` (channels 0..6 are RC-even)."""
    diffs = np.abs(tables - tables[:, ng4.RC_5, :])
    return float(np.max(diffs))


def cluster_bootstrap_spearman_diff_exact(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Exact tie-safe core-cluster paired bootstrap for Spearman(pred_a) - Spearman(pred_b)."""
    rho_a, pearson_a = safe_spearman_pearson(y_true, pred_a)
    rho_b, pearson_b = safe_spearman_pearson(y_true, pred_b)
    diff_obs = float(rho_a - rho_b)

    uniq_g = np.unique(groups)
    g_arrs = [np.flatnonzero(groups == g) for g in uniq_g]
    G = len(g_arrs)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([g_arrs[p] for p in pick])
        ry = rankdata(y_true[idx], method="average")
        ra = rankdata(pred_a[idx], method="average")
        rb = rankdata(pred_b[idx], method="average")
        ry -= ry.mean()
        ra -= ra.mean()
        rb -= rb.mean()
        yn = np.linalg.norm(ry)
        if yn < 1e-12:
            diffs[b] = 0.0
            continue
        r_ab = float(np.dot(ra, ry) / (max(np.linalg.norm(ra), 1e-12) * yn))
        r_bb = float(np.dot(rb, ry) / (max(np.linalg.norm(rb), 1e-12) * yn))
        diffs[b] = r_ab - r_bb

    ci_lo = float(np.quantile(diffs, 0.025))
    ci_hi = float(np.quantile(diffs, 0.975))
    # Two-sided paired bootstrap p-value AND one-sided P(diff <= 0)
    p_gt0 = float((1.0 + np.sum(diffs <= 0.0)) / float(n_boot + 1))
    p_two = min(1.0, 2.0 * float((1.0 + min(np.sum(diffs <= 0.0), np.sum(diffs >= 0.0))) / float(n_boot + 1)))
    return {
        "n_groups": int(G),
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "pearson_a": float(pearson_a),
        "pearson_b": float(pearson_b),
        "diff": diff_obs,
        "ci95": [ci_lo, ci_hi],
        "p_boot_gt0": p_gt0,
        "p_boot_two_sided": p_two,
    }


def _eval_null_bank_chunked_multibin(
    X_seq: np.ndarray,
    B_rc: np.ndarray,
    tables_bank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lams_seq: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_mod: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_folds: int = 5,
    chunk_size: int = 40,
) -> np.ndarray:
    """
    Exact batched 5-fold GroupKFold OOF two-penalty Schur solver supporting ANY number
    of spatial bins `n_bins` (e.g. `10` bins -> `p_m=70` for 7ch) and arbitrary `M`
    tables in `O(chunk_size)` memory (< 350 MiB). Returns Spearman vector shape `(M,)`.
    """
    M, _, d = tables_bank.shape
    N, n_bins, _ = B_rc.shape
    p_m = n_bins * d
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)
    outer_gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    outer_splits = list(outer_gkf.split(X_seq, y, groups))

    # Precompute outer-fold sequence standardization + inner eigendecomposition ONCE per fold
    fold_seq_pre: List[Dict[str, object]] = []
    for f_id, (tr_i, va_i) in enumerate(outer_splits):
        Xtr_s = X_seq[tr_i]
        std_raw = Xtr_s.std(axis=0)
        keep_cols = std_raw > 1e-12
        Xtr_s = Xtr_s[:, keep_cols]
        Xva_s = X_seq[va_i][:, keep_cols]
        ms = Xtr_s.mean(axis=0, keepdims=True)
        ss = np.maximum(Xtr_s.std(axis=0, keepdims=True), 1e-8)
        Zs_tr = (Xtr_s - ms) / ss
        Zs_va = (Xva_s - ms) / ss
        p_s = Zs_tr.shape[1]
        y_tr = y[tr_i]
        ym = float(y_tr.mean())
        yc = y_tr - ym

        inner_gkf = GroupKFold(n_splits=min(3, len(np.unique(groups[tr_i]))))
        inner_splits = list(inner_gkf.split(Zs_tr, yc, groups[tr_i]))[:2]
        in_pre = []
        for i_tr, i_va in inner_splits:
            Zsa, Zsv = Zs_tr[i_tr], Zs_tr[i_va]
            ym_i = float(y_tr[i_tr].mean())
            ya, yv = y_tr[i_tr] - ym_i, y_tr[i_va] - ym_i
            if len(i_tr) >= p_s:
                ev_b, Qb = sp_eigh(Zsa.T @ Zsa, check_finite=False, driver="evd")
                ev_b = np.maximum(ev_b, 0.0)
                qb = Qb.T @ (Zsa.T @ ya)
                Zsv_Qb = Zsv @ Qb
                in_pre.append({"mode": "primal", "i_tr": i_tr, "i_va": i_va, "ya": ya, "yv": yv, "ev_b": ev_b, "Qb": Qb, "qb": qb, "Zsv_Qb": Zsv_Qb})
            else:
                ev_d, Qd = sp_eigh(Zsa @ Zsa.T, check_finite=False, driver="evd")
                ev_d = np.maximum(ev_d, 0.0)
                qd = Qd.T @ ya
                Ks_va = Zsv @ Zsa.T
                in_pre.append({"mode": "dual", "i_tr": i_tr, "i_va": i_va, "ya": ya, "yv": yv, "ev_d": ev_d, "Qd": Qd, "qd": qd, "Ks_va": Ks_va})

        if len(tr_i) >= p_s:
            ev_bf, Qbf = sp_eigh(Zs_tr.T @ Zs_tr, check_finite=False, driver="evd")
            ev_bf = np.maximum(ev_bf, 0.0)
            qbf = Qbf.T @ (Zs_tr.T @ yc)
            Zva_Qbf = Zs_va @ Qbf
            out_mode = {"mode": "primal", "ev_b": ev_bf, "Qb": Qbf, "qb": qbf, "Zva_Qb": Zva_Qbf}
        else:
            ev_df, Qdf = sp_eigh(Zs_tr @ Zs_tr.T, check_finite=False, driver="evd")
            ev_df = np.maximum(ev_df, 0.0)
            qdf = Qdf.T @ yc
            Kva_f = Zs_va @ Zs_tr.T
            out_mode = {"mode": "dual", "ev_d": ev_df, "Qd": Qdf, "qd": qdf, "Kva": Kva_f}

        fold_seq_pre.append(
            {
                "tr_i": tr_i,
                "va_i": va_i,
                "Zs_tr": Zs_tr,
                "Zs_va": Zs_va,
                "p_s": p_s,
                "ym": ym,
                "yc": yc,
                "in_pre": in_pre,
                "out_mode": out_mode,
            }
        )

    B_flat = B_rc.reshape(N * n_bins, 1024).astype(np.float64)
    all_srs = np.zeros(M, dtype=np.float64)
    eye_pm = np.eye(p_m, dtype=np.float64)

    for c0 in range(0, M, chunk_size):
        c1 = min(c0 + chunk_size, M)
        Mc = c1 - c0
        T_flat = np.ascontiguousarray(tables_bank[c0:c1].transpose(1, 0, 2).reshape(1024, Mc * d), dtype=np.float64)
        Xm_all = (B_flat @ T_flat).reshape(N, n_bins, Mc, d).transpose(2, 0, 1, 3).reshape(Mc, N, p_m)
        preds_oof = np.empty((Mc, N), dtype=np.float64)

        for fp in fold_seq_pre:
            tr_i, va_i = fp["tr_i"], fp["va_i"]
            Zs_tr, p_s = fp["Zs_tr"], fp["p_s"]
            ym, yc = fp["ym"], fp["yc"]

            Xm_tr_raw = Xm_all[:, tr_i, :]
            mm = Xm_tr_raw.mean(axis=1, keepdims=True)
            sm = np.maximum(Xm_tr_raw.std(axis=1, keepdims=True), 1e-8)
            Zm_tr = (Xm_tr_raw - mm) / sm
            Zm_va = (Xm_all[:, va_i, :] - mm) / sm

            # Tune (lam_seq, lam_mod) on inner CV simultaneously across Mc tables
            cv_mse = np.zeros((Mc, len(ls_arr), len(lm_arr)), dtype=np.float64)
            for inp in fp["in_pre"]:
                i_tr, i_va = inp["i_tr"], inp["i_va"]
                Zma = Zm_tr[:, i_tr, :]
                Zmv = Zm_tr[:, i_va, :]
                ya, yv = inp["ya"], inp["yv"]
                if inp["mode"] == "primal":
                    ev_b, Qb, qb, Zsv_Qb = inp["ev_b"], inp["Qb"], inp["qb"], inp["Zsv_Qb"]
                    gsm = (Zs_tr[i_tr].T @ Zma.transpose(1, 0, 2).reshape(len(i_tr), Mc * p_m)).reshape(p_s, Mc, p_m).transpose(1, 0, 2)
                    Zsa_Qb_T_Zma = np.matmul(Qb.T[None, :, :], gsm)  # (Mc, p_s, p_m)
                    U_T = np.swapaxes(Zsa_Qb_T_Zma, 1, 2)             # (Mc, p_m, p_s)
                    G_mm = np.matmul(np.swapaxes(Zma, 1, 2), Zma)     # (Mc, p_m, p_m)
                    zm_y = np.matmul(np.swapaxes(Zma, 1, 2), ya[:, None])[:, :, 0]
                    for i_s, ls in enumerate(ls_arr):
                        inv_d = 1.0 / (ev_b + ls)
                        U_sc = Zsa_Qb_T_Zma * inv_d[None, :, None]
                        S = G_mm - np.matmul(U_T, U_sc)
                        S = 0.5 * (S + S.swapaxes(-1, -2))
                        rhs_m = zm_y - np.matmul(np.swapaxes(U_sc, 1, 2), qb[None, :, None])[:, :, 0]
                        ev_t, Qt = np.linalg.eigh(S)
                        ev_t = np.maximum(ev_t, 0.0)
                        Qt_T = np.swapaxes(Qt, 1, 2)
                        proj_t = np.matmul(Qt_T, rhs_m[:, :, None])[:, :, 0]
                        for i_m, lm in enumerate(lm_arr):
                            wm = np.matmul(Qt, (proj_t / (ev_t + lm))[:, :, None])[:, :, 0]
                            ws_q = (qb[None, :] - np.matmul(Zsa_Qb_T_Zma, wm[:, :, None])[:, :, 0]) * inv_d[None, :]
                            pred_v = ws_q @ Zsv_Qb.T + np.matmul(Zmv, wm[:, :, None])[:, :, 0]
                            diff = pred_v - yv[None, :]
                            cv_mse[:, i_s, i_m] += np.sum(diff * diff, axis=1)
                else:
                    ev_d, Qd, qd, Ks_va = inp["ev_d"], inp["Qd"], inp["qd"], inp["Ks_va"]
                    Zm_q = np.matmul(Qd.T[None, :, :], Zma)
                    Zm_q_T = np.swapaxes(Zm_q, 1, 2)
                    Ksv_Qd = Ks_va @ Qd
                    for i_s, ls in enumerate(ls_arr):
                        w_d = ls / (ev_d + ls)
                        S = np.matmul(Zm_q_T * w_d[None, None, :], Zm_q)
                        S = 0.5 * (S + S.swapaxes(-1, -2))
                        rhs_m = np.matmul(Zm_q_T, (w_d * qd)[None, :, None])[:, :, 0]
                        ev_t, Qt = np.linalg.eigh(S)
                        ev_t = np.maximum(ev_t, 0.0)
                        Qt_T = np.swapaxes(Qt, 1, 2)
                        proj_t = np.matmul(Qt_T, rhs_m[:, :, None])[:, :, 0]
                        for i_m, lm in enumerate(lm_arr):
                            wm = np.matmul(Qt, (proj_t / (ev_t + lm))[:, :, None])[:, :, 0]
                            res_q = qd[None, :] - np.matmul(Zm_q, wm[:, :, None])[:, :, 0]
                            pred_v = ((res_q * w_d[None, :]) / ls) @ Ksv_Qd.T + np.matmul(Zmv, wm[:, :, None])[:, :, 0]
                            diff = pred_v - yv[None, :]
                            cv_mse[:, i_s, i_m] += np.sum(diff * diff, axis=1)

            flat_b = np.argmin(cv_mse.reshape(Mc, -1), axis=1)
            best_ls = ls_arr[flat_b // len(lm_arr)]
            best_lm = lm_arr[flat_b % len(lm_arr)]

            om = fp["out_mode"]
            if om["mode"] == "primal":
                ev_bf, Qbf, qbf, Zva_Qbf = om["ev_b"], om["Qb"], om["qb"], om["Zva_Qb"]
                G_sm = (Zs_tr.T @ Zm_tr.transpose(1, 0, 2).reshape(len(tr_i), Mc * p_m)).reshape(p_s, Mc, p_m).transpose(1, 0, 2)
                Uf = np.matmul(Qbf.T[None, :, :], G_sm)
                Uf_T = np.swapaxes(Uf, 1, 2)
                G_mm_f = np.matmul(np.swapaxes(Zm_tr, 1, 2), Zm_tr)
                zm_yf = np.matmul(np.swapaxes(Zm_tr, 1, 2), yc[:, None])[:, :, 0]
                inv_df = 1.0 / (ev_bf[None, :] + best_ls[:, None])
                Uf_sc = Uf * inv_df[:, :, None]
                Sf = G_mm_f - np.matmul(Uf_T, Uf_sc)
                Sf += best_lm[:, None, None] * eye_pm[None, :, :]
                rhs_mf = zm_yf - np.matmul(np.swapaxes(Uf_sc, 1, 2), qbf[None, :, None])[:, :, 0]
                wmf = np.linalg.solve(Sf, rhs_mf[:, :, None])[:, :, 0]
                wsf_q = (qbf[None, :] - np.matmul(Uf, wmf[:, :, None])[:, :, 0]) * inv_df
                preds_oof[:, va_i] = ym + wsf_q @ Zva_Qbf.T + np.matmul(Zm_va, wmf[:, :, None])[:, :, 0]
            else:
                ev_df, Qdf, qdf, Kva_f = om["ev_d"], om["Qd"], om["qd"], om["Kva"]
                Zmf_q = np.matmul(Qdf.T[None, :, :], Zm_tr)
                Zmf_q_T = np.swapaxes(Zmf_q, 1, 2)
                Kva_Qd = Kva_f @ Qdf
                w_df = best_ls[:, None] / (ev_df[None, :] + best_ls[:, None])  # (Mc, na)
                Sf = np.matmul(Zmf_q_T * w_df[:, None, :], Zmf_q)
                Sf += best_lm[:, None, None] * eye_pm[None, :, :]
                rhs_mf = np.matmul(Zmf_q_T, (w_df * qdf[None, :])[:, :, None])[:, :, 0]
                wmf = np.linalg.solve(Sf, rhs_mf[:, :, None])[:, :, 0]
                res_qf = qdf[None, :] - np.matmul(Zmf_q, wmf[:, :, None])[:, :, 0]
                preds_oof[:, va_i] = ym + ((res_qf * w_df) / best_ls[:, None]) @ Kva_Qd.T + np.matmul(Zm_va, wmf[:, :, None])[:, :, 0]

        srs_c, _ = fast_batch_spearman_pearson(y, preds_oof)
        all_srs[c0:c1] = srs_c
    return all_srs


_NULL_POOL_CTX: Dict[str, object] = {}


def _eval_null_chunk_multi_target(args: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
    """Worker solving `(c0, c1)` table chunk across ALL `K_tgt` targets sharing `(X_seq, B_rc, groups)`."""
    c0, c1 = args
    ctx = _NULL_POOL_CTX
    B_flat = ctx["B_flat"]
    tables_bank = ctx["tables_bank"]
    Y_mat = ctx["Y_mat"]
    fold_seq_pre = ctx["fold_seq_pre"]
    ls_arr = ctx["ls_arr"]
    lm_arr = ctx["lm_arr"]
    N = int(ctx["N"])
    n_bins = int(ctx["n_bins"])
    d = int(ctx["d"])
    p_m = int(ctx["p_m"])
    K_tgt = int(ctx["K_tgt"])
    eye_pm = ctx["eye_pm"]

    Mc = c1 - c0
    n_ls = len(ls_arr)
    n_lm = len(lm_arr)
    m_idx = np.arange(Mc)

    T_flat = np.ascontiguousarray(tables_bank[c0:c1].transpose(1, 0, 2).reshape(1024, Mc * d), dtype=np.float64)
    Xm_all = (B_flat @ T_flat).reshape(N, n_bins, Mc, d).transpose(2, 0, 1, 3).reshape(Mc, N, p_m)
    preds_oof = np.empty((K_tgt, Mc, N), dtype=np.float64)

    for fp in fold_seq_pre:
        tr_i, va_i = fp["tr_i"], fp["va_i"]
        Zs_tr, p_s = fp["Zs_tr"], fp["p_s"]
        ym, Yc = fp["ym"], fp["Yc"]

        Xm_tr_raw = Xm_all[:, tr_i, :]
        mm = Xm_tr_raw.mean(axis=1, keepdims=True)
        sm = np.maximum(Xm_tr_raw.std(axis=1, keepdims=True), 1e-8)
        Zm_tr = (Xm_tr_raw - mm) / sm
        Zm_va = (Xm_all[:, va_i, :] - mm) / sm

        cv_mse = np.zeros((Mc, K_tgt, n_ls, n_lm), dtype=np.float64)
        for inp in fp["in_pre"]:
            i_tr, i_va = inp["i_tr"], inp["i_va"]
            Zma = Zm_tr[:, i_tr, :]
            Zmv = Zm_tr[:, i_va, :]
            Ya, Yv = inp["Ya"], inp["Yv"]
            if inp["mode"] == "primal":
                ev_b, Qb, qb, Zsv_Qb = inp["ev_b"], inp["Qb"], inp["qb"], inp["Zsv_Qb"]
                gsm = (Zs_tr[i_tr].T @ Zma.transpose(1, 0, 2).reshape(len(i_tr), Mc * p_m)).reshape(p_s, Mc, p_m).transpose(1, 0, 2)
                U = np.matmul(Qb.T[None, :, :], gsm)                 # (Mc, p_s, p_m)
                U_T = np.swapaxes(U, 1, 2)                           # (Mc, p_m, p_s)
                G_mm = np.matmul(np.swapaxes(Zma, 1, 2), Zma)        # (Mc, p_m, p_m)
                zm_y = np.matmul(np.swapaxes(Zma, 1, 2), Ya[None, :, :])  # (Mc, p_m, K_tgt)
                for i_s, ls in enumerate(ls_arr):
                    inv_d = 1.0 / (ev_b + ls)
                    U_sc = U * inv_d[None, :, None]
                    S = G_mm - np.matmul(U_T, U_sc)
                    S = 0.5 * (S + S.swapaxes(-1, -2))
                    rhs_m = zm_y - np.matmul(np.swapaxes(U_sc, 1, 2), qb[None, :, :])
                    ev_t, Qt = np.linalg.eigh(S)
                    ev_t = np.maximum(ev_t, 0.0)
                    proj_t = np.matmul(np.swapaxes(Qt, 1, 2), rhs_m)  # (Mc, p_m, K_tgt)
                    for i_m, lm in enumerate(lm_arr):
                        wm = np.matmul(Qt, proj_t / (ev_t[:, :, None] + lm))
                        ws_q = (qb[None, :, :] - np.matmul(U, wm)) * inv_d[None, :, None]
                        pred_v = np.matmul(Zsv_Qb[None, :, :], ws_q) + np.matmul(Zmv, wm)
                        diff = pred_v - Yv[None, :, :]
                        cv_mse[:, :, i_s, i_m] += np.sum(diff * diff, axis=1)
            else:
                ev_d, Qd, qd, Ks_va = inp["ev_d"], inp["Qd"], inp["qd"], inp["Ks_va"]
                Zm_q = np.matmul(Qd.T[None, :, :], Zma)
                Zm_q_T = np.swapaxes(Zm_q, 1, 2)
                Ksv_Qd = Ks_va @ Qd
                for i_s, ls in enumerate(ls_arr):
                    w_d = ls / (ev_d + ls)
                    S = np.matmul(Zm_q_T * w_d[None, None, :], Zm_q)
                    S = 0.5 * (S + S.swapaxes(-1, -2))
                    rhs_m = np.matmul(Zm_q_T, (w_d[:, None] * qd)[None, :, :])
                    ev_t, Qt = np.linalg.eigh(S)
                    ev_t = np.maximum(ev_t, 0.0)
                    proj_t = np.matmul(np.swapaxes(Qt, 1, 2), rhs_m)
                    for i_m, lm in enumerate(lm_arr):
                        wm = np.matmul(Qt, proj_t / (ev_t[:, :, None] + lm))
                        res_q = qd[None, :, :] - np.matmul(Zm_q, wm)
                        pred_v = np.matmul(Ksv_Qd[None, :, :], (res_q * w_d[None, :, None]) / ls) + np.matmul(Zmv, wm)
                        diff = pred_v - Yv[None, :, :]
                        cv_mse[:, :, i_s, i_m] += np.sum(diff * diff, axis=1)

        flat_b = np.argmin(cv_mse.reshape(Mc, K_tgt, -1), axis=2)  # (Mc, K_tgt)
        bi_s_all = flat_b // n_lm
        bi_m_all = flat_b % n_lm

        om = fp["out_mode"]
        if om["mode"] == "primal":
            ev_bf, Qbf, qbf, Zva_Qbf = om["ev_b"], om["Qb"], om["qb"], om["Zva_Qb"]
            G_sm = (Zs_tr.T @ Zm_tr.transpose(1, 0, 2).reshape(len(tr_i), Mc * p_m)).reshape(p_s, Mc, p_m).transpose(1, 0, 2)
            Uf = np.matmul(Qbf.T[None, :, :], G_sm)
            Uf_T = np.swapaxes(Uf, 1, 2)
            G_mm_f = np.matmul(np.swapaxes(Zm_tr, 1, 2), Zm_tr)
            zm_yf = np.matmul(np.swapaxes(Zm_tr, 1, 2), Yc[None, :, :])
            S_by_ls = np.empty((n_ls, Mc, p_m, p_m), dtype=np.float64)
            rhs_by_ls = np.empty((n_ls, Mc, p_m, K_tgt), dtype=np.float64)
            for i_s, ls in enumerate(ls_arr):
                inv_d = 1.0 / (ev_bf + ls)
                Uf_sc = Uf * inv_d[None, :, None]
                S_by_ls[i_s] = G_mm_f - np.matmul(Uf_T, Uf_sc)
                rhs_by_ls[i_s] = zm_yf - np.matmul(np.swapaxes(Uf_sc, 1, 2), qbf[None, :, :])
            for k in range(K_tgt):
                bi_s = bi_s_all[:, k]
                bi_m = bi_m_all[:, k]
                Sf = S_by_ls[bi_s, m_idx] + lm_arr[bi_m, None, None] * eye_pm[None, :, :]
                rhs_k = rhs_by_ls[bi_s, m_idx, :, k]
                wmf = np.linalg.solve(Sf, rhs_k[:, :, None])[:, :, 0]
                inv_df = 1.0 / (ev_bf[None, :] + ls_arr[bi_s, None])
                wsf_q = (qbf[None, :, k] - np.matmul(Uf, wmf[:, :, None])[:, :, 0]) * inv_df
                preds_oof[k][:, va_i] = ym[k] + wsf_q @ Zva_Qbf.T + np.matmul(Zm_va, wmf[:, :, None])[:, :, 0]
        else:
            ev_df, Qdf, qdf, Kva_f = om["ev_d"], om["Qd"], om["qd"], om["Kva"]
            Zmf_q = np.matmul(Qdf.T[None, :, :], Zm_tr)
            Zmf_q_T = np.swapaxes(Zmf_q, 1, 2)
            Kva_Qd = Kva_f @ Qdf
            S_by_ls = np.empty((n_ls, Mc, p_m, p_m), dtype=np.float64)
            rhs_by_ls = np.empty((n_ls, Mc, p_m, K_tgt), dtype=np.float64)
            for i_s, ls in enumerate(ls_arr):
                w_d = ls / (ev_df + ls)
                S_by_ls[i_s] = np.matmul(Zmf_q_T * w_d[None, None, :], Zmf_q)
                rhs_by_ls[i_s] = np.matmul(Zmf_q_T, (w_d[:, None] * qdf)[None, :, :])
            for k in range(K_tgt):
                bi_s = bi_s_all[:, k]
                bi_m = bi_m_all[:, k]
                Sf = S_by_ls[bi_s, m_idx] + lm_arr[bi_m, None, None] * eye_pm[None, :, :]
                rhs_k = rhs_by_ls[bi_s, m_idx, :, k]
                wmf = np.linalg.solve(Sf, rhs_k[:, :, None])[:, :, 0]
                best_ls = ls_arr[bi_s]
                w_df = best_ls[:, None] / (ev_df[None, :] + best_ls[:, None])
                res_qf = qdf[None, :, k] - np.matmul(Zmf_q, wmf[:, :, None])[:, :, 0]
                preds_oof[k][:, va_i] = ym[k] + ((res_qf * w_df) / best_ls[:, None]) @ Kva_Qd.T + np.matmul(Zm_va, wmf[:, :, None])[:, :, 0]

    srs_chunk = np.empty((K_tgt, Mc), dtype=np.float64)
    for k in range(K_tgt):
        srs_chunk[k], _ = fast_batch_spearman_pearson(Y_mat[:, k], preds_oof[k])
    return c0, c1, srs_chunk


def eval_null_bank_multi_target_parallel(
    X_seq: np.ndarray,
    B_rc: np.ndarray,
    tables_bank: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    groups: np.ndarray,
    lams_seq: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_mod: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_folds: int = 5,
    chunk_size: int = 20,
    n_workers: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Exact multi-target + multi-core (`fork` CoW) batched 5-fold GroupKFold OOF two-penalty Schur solver.
    Reuses sequence-modality cross-products (`G_sm`, `U`, `G_mm`, `S`, `eigh(S)`) ONCE across all `K_tgt`
    paired targets sharing `(X_seq, B_rc, groups)`, delivering exact `<= 1e-12` parity with
    `_eval_null_bank_chunked_multibin` at ~30-60x wall-clock speedup.
    """
    t_names = list(y_dict.keys())
    Y_mat = np.column_stack([np.asarray(y_dict[t], dtype=np.float64) for t in t_names])  # (N, K_tgt)
    N, K_tgt = Y_mat.shape
    M, _, d = tables_bank.shape
    _, n_bins, _ = B_rc.shape
    p_m = n_bins * d
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)
    outer_gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    outer_splits = list(outer_gkf.split(X_seq, Y_mat[:, 0], groups))

    fold_seq_pre: List[Dict[str, object]] = []
    for tr_i, va_i in outer_splits:
        Xtr_s = X_seq[tr_i]
        keep_cols = Xtr_s.std(axis=0) > 1e-12
        Xtr_s = Xtr_s[:, keep_cols]
        Xva_s = X_seq[va_i][:, keep_cols]
        ms = Xtr_s.mean(axis=0, keepdims=True)
        ss = np.maximum(Xtr_s.std(axis=0, keepdims=True), 1e-8)
        Zs_tr = (Xtr_s - ms) / ss
        Zs_va = (Xva_s - ms) / ss
        p_s = Zs_tr.shape[1]
        Y_tr = Y_mat[tr_i]
        ym = Y_tr.mean(axis=0)        # (K_tgt,)
        Yc = Y_tr - ym[None, :]       # (n_tr, K_tgt)

        inner_gkf = GroupKFold(n_splits=min(3, len(np.unique(groups[tr_i]))))
        inner_splits = list(inner_gkf.split(Zs_tr, Yc[:, 0], groups[tr_i]))[:2]
        in_pre = []
        for i_tr, i_va in inner_splits:
            Zsa, Zsv = Zs_tr[i_tr], Zs_tr[i_va]
            ym_i = Y_tr[i_tr].mean(axis=0)
            Ya = Y_tr[i_tr] - ym_i[None, :]
            Yv = Y_tr[i_va] - ym_i[None, :]
            if len(i_tr) >= p_s:
                ev_b, Qb = sp_eigh(Zsa.T @ Zsa, check_finite=False, driver="evd")
                ev_b = np.maximum(ev_b, 0.0)
                qb = Qb.T @ (Zsa.T @ Ya)
                Zsv_Qb = Zsv @ Qb
                in_pre.append({"mode": "primal", "i_tr": i_tr, "i_va": i_va, "Ya": Ya, "Yv": Yv, "ev_b": ev_b, "Qb": Qb, "qb": qb, "Zsv_Qb": Zsv_Qb})
            else:
                ev_d, Qd = sp_eigh(Zsa @ Zsa.T, check_finite=False, driver="evd")
                ev_d = np.maximum(ev_d, 0.0)
                qd = Qd.T @ Ya
                Ks_va = Zsv @ Zsa.T
                in_pre.append({"mode": "dual", "i_tr": i_tr, "i_va": i_va, "Ya": Ya, "Yv": Yv, "ev_d": ev_d, "Qd": Qd, "qd": qd, "Ks_va": Ks_va})

        if len(tr_i) >= p_s:
            ev_bf, Qbf = sp_eigh(Zs_tr.T @ Zs_tr, check_finite=False, driver="evd")
            ev_bf = np.maximum(ev_bf, 0.0)
            qbf = Qbf.T @ (Zs_tr.T @ Yc)
            Zva_Qbf = Zs_va @ Qbf
            out_mode = {"mode": "primal", "ev_b": ev_bf, "Qb": Qbf, "qb": qbf, "Zva_Qb": Zva_Qbf}
        else:
            ev_df, Qdf = sp_eigh(Zs_tr @ Zs_tr.T, check_finite=False, driver="evd")
            ev_df = np.maximum(ev_df, 0.0)
            qdf = Qdf.T @ Yc
            Kva_f = Zs_va @ Zs_tr.T
            out_mode = {"mode": "dual", "ev_d": ev_df, "Qd": Qdf, "qd": qdf, "Kva": Kva_f}

        fold_seq_pre.append(
            {
                "tr_i": tr_i,
                "va_i": va_i,
                "Zs_tr": Zs_tr,
                "Zs_va": Zs_va,
                "p_s": p_s,
                "ym": ym,
                "Yc": Yc,
                "in_pre": in_pre,
                "out_mode": out_mode,
            }
        )

    B_flat = B_rc.reshape(N * n_bins, 1024).astype(np.float64)
    chunks = [(c0, min(c0 + chunk_size, M)) for c0 in range(0, M, chunk_size)]
    all_srs = np.zeros((K_tgt, M), dtype=np.float64)

    _NULL_POOL_CTX.clear()
    _NULL_POOL_CTX.update(
        {
            "B_flat": B_flat,
            "tables_bank": tables_bank,
            "Y_mat": Y_mat,
            "fold_seq_pre": fold_seq_pre,
            "ls_arr": ls_arr,
            "lm_arr": lm_arr,
            "N": N,
            "n_bins": n_bins,
            "d": d,
            "p_m": p_m,
            "K_tgt": K_tgt,
            "eye_pm": np.eye(p_m, dtype=np.float64),
        }
    )
    try:
        eff_workers = min(max(1, int(n_workers)), len(chunks))
        if eff_workers <= 1:
            for ch in chunks:
                c0, c1, srs_c = _eval_null_chunk_multi_target(ch)
                all_srs[:, c0:c1] = srs_c
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=eff_workers, mp_context=ctx) as ex:
                for c0, c1, srs_c in ex.map(_eval_null_chunk_multi_target, chunks):
                    all_srs[:, c0:c1] = srs_c
    finally:
        _NULL_POOL_CTX.clear()

    return {t: all_srs[k] for k, t in enumerate(t_names)}


def _fit_predict_ridge_pruned_5fold_oof(
    X_seq: np.ndarray,
    X_mod: Optional[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """Exact 5-fold GroupKFold OOF prediction (zero leakage pruning & tuning inside each fold)."""
    _, pp5 = _lazy_v5_v7()
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    oof = np.empty(len(y), dtype=np.float64)
    for tr_i, va_i in gkf.split(X_seq, y, groups):
        keep = X_seq[tr_i].std(axis=0) > 1e-12
        Xs_tr, Xs_va = X_seq[tr_i][:, keep], X_seq[va_i][:, keep]
        if X_mod is None:
            res = pp5.fit_predict_ridge_groupcv(Xs_tr, y[tr_i], groups[tr_i], Xs_va, n_folds=3)
        else:
            res = pp5.fit_predict_two_penalty_ridge_groupcv(
                Xs_tr, X_mod[tr_i], y[tr_i], groups[tr_i], Xs_va, X_mod[va_i], n_folds=3
            )
        oof[va_i] = res["pred_te"]
    return oof


def _fit_predict_poly3_5fold_oof(
    X_g1: np.ndarray,
    X_mod: Optional[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    oof = np.empty(len(y), dtype=np.float64)
    for tr_i, va_i in gkf.split(X_g1, y, groups):
        res = fit_predict_poly3_kernel_ridge(
            X_g1[tr_i], y[tr_i], groups[tr_i], X_g1[va_i],
            X_mod_tr=None if X_mod is None else X_mod[tr_i],
            X_mod_te=None if X_mod is None else X_mod[va_i],
            n_inner_folds=2,
        )
        oof[va_i] = res["pred_te"]
    return oof


def _fit_predict_hgbdt_5fold_oof(
    codes: np.ndarray,
    X_g1: np.ndarray,
    X_g2: np.ndarray,
    X_mod: Optional[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """5-fold GroupKFold OOF for HGBDT on sequence `[codes, X_g1, gc2]` vs `+Shape`."""
    gc2 = X_g2[:, -12:-10]
    X_base = np.hstack([codes.astype(np.float32), X_g1.astype(np.float32), gc2.astype(np.float32)])
    X_in = X_base if X_mod is None else np.hstack([X_base, X_mod.astype(np.float32)])
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    splits = list(gkf.split(X_in, y, groups))

    configs = [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 3.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 3.0},
    ]
    oof = np.empty(len(y), dtype=np.float64)
    chosen_cfgs = []
    for f_id, (tr_i, va_i) in enumerate(splits):
        # Inner GroupKFold single-fold config selection inside tr_i
        in_gkf = GroupKFold(n_splits=min(3, len(np.unique(groups[tr_i]))))
        itr, iva = next(in_gkf.split(X_in[tr_i], y[tr_i], groups[tr_i]))
        best_cfg = configs[0]
        best_mse = np.inf
        for cfg in configs:
            m = HistGradientBoostingRegressor(max_iter=200, max_bins=64, early_stopping=True, random_state=42, **cfg)
            m.fit(X_in[tr_i][itr], y[tr_i][itr])
            diff = m.predict(X_in[tr_i][iva]) - y[tr_i][iva]
            mse = float(np.sum(diff * diff))
            if mse < best_mse:
                best_mse = mse
                best_cfg = cfg
        chosen_cfgs.append(best_cfg)
        m_full = HistGradientBoostingRegressor(max_iter=250, max_bins=64, early_stopping=True, random_state=42, **best_cfg)
        m_full.fit(X_in[tr_i], y[tr_i])
        oof[va_i] = m_full.predict(X_in[va_i])
    return oof, {"fold_configs": chosen_cfgs, "n_features": int(X_in.shape[1])}


def eval_paired_poly3_multi_targets(
    X_g1: np.ndarray,
    X_sh7: np.ndarray,
    X_mgw: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    groups: np.ndarray,
    gammas: Sequence[float] = (0.5, 1.0, 2.0),
    lams_seq: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    lams_mod: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_folds: int = 5,
    n_inner_folds: int = 2,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Exact low-RAM multi-target Poly3_Kernel (+Shape_7ch, +MGW) solver that computes the
    (fold, gamma) kernel eigendecomposition ONCE across all paired endpoints sharing
    `(X_g1, groups)`, cutting eigendecomps ~18x with exact numerical equivalence (`<=1e-13`).
    """
    t_names = list(y_dict.keys())
    Y_full = np.column_stack([np.asarray(y_dict[t], dtype=np.float64) for t in t_names])  # (N, K_tgt)
    N, K_tgt = Y_full.shape
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)

    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    outer_splits = list(gkf.split(X_g1, Y_full[:, 0], groups))

    out_oofs: Dict[str, Dict[str, np.ndarray]] = {
        t: {
            "Poly3_Kernel": np.empty(N, dtype=np.float64),
            "Poly3_Kernel+Shape_7ch": np.empty(N, dtype=np.float64),
            "Poly3_Kernel+MGW": np.empty(N, dtype=np.float64),
        }
        for t in t_names
    }

    for f_id, (tr_i, va_i) in enumerate(outer_splits):
        ms = X_g1[tr_i].mean(axis=0, keepdims=True)
        ss = np.maximum(X_g1[tr_i].std(axis=0, keepdims=True), 1e-8)
        Zs_tr = (X_g1[tr_i] - ms) / ss
        Zs_va = (X_g1[va_i] - ms) / ss
        p_s = float(Zs_tr.shape[1])
        D_tr = (Zs_tr @ Zs_tr.T) / p_s
        D_te = (Zs_va @ Zs_tr.T) / p_s

        mods = {}
        for mn, Xm in [("Poly3_Kernel+Shape_7ch", X_sh7), ("Poly3_Kernel+MGW", X_mgw)]:
            mm = Xm[tr_i].mean(axis=0, keepdims=True)
            sm = np.maximum(Xm[tr_i].std(axis=0, keepdims=True), 1e-8)
            mods[mn] = ((Xm[tr_i] - mm) / sm, (Xm[va_i] - mm) / sm)

        Y_tr = Y_full[tr_i]
        ym_tr = Y_tr.mean(axis=0, keepdims=True)
        Yc_tr = Y_tr - ym_tr

        in_gkf = GroupKFold(n_splits=min(max(3, n_inner_folds), len(np.unique(groups[tr_i]))))
        in_splits = list(in_gkf.split(Zs_tr, Y_tr[:, 0], groups[tr_i]))[:n_inner_folds]

        mse_pure = np.zeros((K_tgt, len(gammas), len(ls_arr)), dtype=np.float64)
        mse_mod = {mn: np.zeros((K_tgt, len(gammas), len(ls_arr), len(lm_arr)), dtype=np.float64) for mn in mods}

        for ig, g in enumerate(gammas):
            K_all = np.power(1.0 + float(g) * D_tr, 3)
            for i_tr, i_va in in_splits:
                ym_i = Y_tr[i_tr].mean(axis=0, keepdims=True)
                Ya = Y_tr[i_tr] - ym_i
                Yv = Y_tr[i_va] - ym_i
                Kaa = K_all[np.ix_(i_tr, i_tr)]
                Kva = K_all[np.ix_(i_va, i_tr)]
                ev_d, Qd = sp_eigh(Kaa, check_finite=False, driver="evd")
                ev_d = np.maximum(ev_d, 0.0)
                Qd_T = Qd.T
                qd = Qd_T @ Ya               # (na, K_tgt)
                Kva_Qd = Kva @ Qd            # (nv, na)

                # Pure Poly3_Kernel across all K_tgt
                for i_s, ls in enumerate(ls_arr):
                    al_q = qd / (ev_d[:, None] + ls)
                    diff = (Kva_Qd @ al_q) - Yv
                    mse_pure[:, ig, i_s] += np.sum(diff * diff, axis=0)

                # +Shape_7ch and +MGW using same (ev_d, Qd, qd, Kva_Qd)
                for mn, (Zmt, _) in mods.items():
                    Zma = Zmt[i_tr]
                    Zmv = Zmt[i_va]
                    Zm_q = Qd_T @ Zma        # (na, p_m)
                    Zm_q_T = Zm_q.T
                    for i_s, ls in enumerate(ls_arr):
                        w_d = ls / (ev_d + ls)
                        S = (Zm_q * w_d[:, None]).T @ Zm_q
                        S = 0.5 * (S + S.T)
                        rhs_m = Zm_q_T @ (w_d[:, None] * qd)  # (p_m, K_tgt)
                        ev_t, Qt = sp_eigh(S, check_finite=False, driver="evd")
                        ev_t = np.maximum(ev_t, 0.0)
                        proj_t = Qt.T @ rhs_m
                        for i_m, lm in enumerate(lm_arr):
                            wm = Qt @ (proj_t / (ev_t[:, None] + lm))
                            res_q = qd - Zm_q @ wm
                            pred_v = Kva_Qd @ ((w_d[:, None] * res_q) / ls) + Zmv @ wm
                            diff = pred_v - Yv
                            mse_mod[mn][:, ig, i_s, i_m] += np.sum(diff * diff, axis=0)

        # Predict outer test fold using each target's best (gamma, ls, lm)
        # Solve by grouping across selected gammas so outer K_tr is formed <= 3 times per fold!
        best_pure = [
            np.unravel_index(int(np.argmin(mse_pure[k])), (len(gammas), len(ls_arr)))
            for k in range(K_tgt)
        ]
        best_mod = {
            mn: [
                np.unravel_index(int(np.argmin(mse_mod[mn][k])), (len(gammas), len(ls_arr), len(lm_arr)))
                for k in range(K_tgt)
            ]
            for mn in mods
        }
        needed_g = {bp[0] for bp in best_pure}
        for mn in mods:
            needed_g.update(bm[0] for bm in best_mod[mn])

        for ig in sorted(needed_g):
            g = float(gammas[ig])
            K_tr = np.power(1.0 + g * D_tr, 3)
            K_te = np.power(1.0 + g * D_te, 3)
            ev_f, Qf = sp_eigh(K_tr, check_finite=False, driver="evd")
            ev_f = np.maximum(ev_f, 0.0)
            qf = Qf.T @ Yc_tr
            Kte_Qf = K_te @ Qf

            for k, tname in enumerate(t_names):
                g_p, i_ls_p = best_pure[k]
                if g_p == ig:
                    ls = float(ls_arr[i_ls_p])
                    out_oofs[tname]["Poly3_Kernel"][va_i] = ym_tr[0, k] + Kte_Qf @ (qf[:, k] / (ev_f + ls))

            for mn, (Zmt, Zmv) in mods.items():
                Zmf_q = Qf.T @ Zmt
                for k, tname in enumerate(t_names):
                    g_m, i_ls_m, i_lm_m = best_mod[mn][k]
                    if g_m == ig:
                        ls = float(ls_arr[i_ls_m])
                        lm = float(lm_arr[i_lm_m])
                        w_df = ls / (ev_f + ls)
                        Sf = (Zmf_q * w_df[:, None]).T @ Zmf_q
                        Sf[np.diag_indices(Zmt.shape[1])] += lm
                        cf_s = cho_factor(Sf, lower=True, check_finite=False)
                        wm_f = cho_solve(cf_s, Zmf_q.T @ (w_df * qf[:, k]), check_finite=False)
                        res_qf = qf[:, k] - Zmf_q @ wm_f
                        out_oofs[tname][mn][va_i] = ym_tr[0, k] + Kte_Qf @ ((w_df * res_qf) / ls) + Zmv @ wm_f

        print(f"    [Poly3 multi-target] outer fold {f_id + 1}/{len(outer_splits)} complete", flush=True)
    return out_oofs


def eval_bhlh_ladder_and_nulls(
    name: str,
    feats: Dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    F_true: np.ndarray,
    shap_std: np.ndarray,
    B_null: int = 199,
    null_seed: int = 2026072,
    n_boot: int = 2000,
    run_full_ladder: bool = True,
    shared_bank: Optional[Tuple[np.ndarray, Dict[str, slice]]] = None,
    precomputed_oofs: Optional[Dict[str, np.ndarray]] = None,
    precomputed_null_srs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, object]:
    """
    Evaluate a bHLH aligned cohort across:
      - Pure-sequence vs `+DNAshape (7ch & MGW)` ladder (`Seq_G1`, `Seq_G2`, `Seq_Pos4mer`,
        and optionally `Poly3_Kernel` & `HGBDT_Seq`) with core-cluster 95% bootstrap CIs;
      - `B=199` multi-sparsity null controls (`Naive_RC`, `Shapelizer`, `ShiftGC_ExactCRT_WS_G2`,
        `Sparse4mer K={4,8,12,24,48}`) on BOTH `Seq_G2` and `Seq_Pos4mer`.
    """
    codes = feats["codes"]
    X_g1 = feats["X_g1"]
    X_g2 = feats["X_g2"]
    X_pos4 = feats["X_pos4mer"]
    B_rc = feats["B_rc"]

    X_sh7 = _contract_b_rc_single(B_rc, F_true)
    X_mgw = _contract_b_rc_single(B_rc, F_true[:, :1])

    models_spec: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = [
        ("Seq_G1", X_g1, None),
        ("Seq_G1+Shape_7ch", X_g1, X_sh7),
        ("Seq_G1+MGW", X_g1, X_mgw),
        ("Seq_G2", X_g2, None),
        ("Seq_G2+Shape_7ch", X_g2, X_sh7),
        ("Seq_G2+MGW", X_g2, X_mgw),
        ("Seq_Pos4mer", X_pos4, None),
        ("Seq_Pos4mer+Shape_7ch", X_pos4, X_sh7),
        ("Seq_Pos4mer+MGW", X_pos4, X_mgw),
    ]

    oof_map: Dict[str, np.ndarray] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}

    for mname, xs, xm in models_spec:
        oof = _fit_predict_ridge_pruned_5fold_oof(xs, xm, y, groups, n_folds=5)
        oof_map[mname] = oof
        sr, pr = safe_spearman_pearson(y, oof)
        metrics_map[mname] = {"spearman": float(sr), "pearson": float(pr)}

    hgb_oof_cache: Dict[str, np.ndarray] = {}
    if run_full_ladder:
        if precomputed_oofs is not None and all(k in precomputed_oofs for k in ("Poly3_Kernel", "Poly3_Kernel+Shape_7ch", "Poly3_Kernel+MGW")):
            for mname in ("Poly3_Kernel", "Poly3_Kernel+Shape_7ch", "Poly3_Kernel+MGW"):
                oof = precomputed_oofs[mname]
                oof_map[mname] = oof
                sr, pr = safe_spearman_pearson(y, oof)
                metrics_map[mname] = {"spearman": float(sr), "pearson": float(pr)}
        else:
            poly_res = eval_paired_poly3_multi_targets(X_g1, X_sh7, X_mgw, {"_single": y}, groups, n_folds=5, n_inner_folds=2)["_single"]
            for mname in ("Poly3_Kernel", "Poly3_Kernel+Shape_7ch", "Poly3_Kernel+MGW"):
                oof = poly_res[mname]
                oof_map[mname] = oof
                sr, pr = safe_spearman_pearson(y, oof)
                metrics_map[mname] = {"spearman": float(sr), "pearson": float(pr)}

        for mname, xm in [
            ("HGBDT_Seq", None),
            ("HGBDT_Seq+Shape_7ch", X_sh7),
            ("HGBDT_Seq+MGW", X_mgw),
        ]:
            oof, hmeta = _fit_predict_hgbdt_5fold_oof(codes, X_g1, X_g2, xm, y, groups, n_folds=5)
            oof_map[mname] = oof
            hgb_oof_cache[mname] = oof
            sr, pr = safe_spearman_pearson(y, oof)
            metrics_map[mname] = {"spearman": float(sr), "pearson": float(pr), "n_features": float(hmeta["n_features"])}

    # Core-cluster paired bootstrap contrasts
    contrast_pairs = [
        ("G1_7ch_vs_G1", "Seq_G1+Shape_7ch", "Seq_G1"),
        ("G2_7ch_vs_G2", "Seq_G2+Shape_7ch", "Seq_G2"),
        ("Pos4mer_7ch_vs_Pos4mer", "Seq_Pos4mer+Shape_7ch", "Seq_Pos4mer"),
        ("G1_MGW_vs_G1", "Seq_G1+MGW", "Seq_G1"),
        ("G2_MGW_vs_G2", "Seq_G2+MGW", "Seq_G2"),
        ("Pos4mer_MGW_vs_Pos4mer", "Seq_Pos4mer+MGW", "Seq_Pos4mer"),
    ]
    if run_full_ladder:
        contrast_pairs.extend(
            [
                ("Poly3_7ch_vs_Poly3", "Poly3_Kernel+Shape_7ch", "Poly3_Kernel"),
                ("Poly3_MGW_vs_Poly3", "Poly3_Kernel+MGW", "Poly3_Kernel"),
                ("HGBDT_7ch_vs_HGBDT", "HGBDT_Seq+Shape_7ch", "HGBDT_Seq"),
                ("HGBDT_MGW_vs_HGBDT", "HGBDT_Seq+MGW", "HGBDT_Seq"),
                ("Pos4mer_vs_G2", "Seq_Pos4mer", "Seq_G2"),
            ]
        )

    contrasts: Dict[str, Dict[str, object]] = {}
    for cid, ma, mb in contrast_pairs:
        contrasts[cid] = cluster_bootstrap_spearman_diff_exact(
            y, oof_map[ma], oof_map[mb], groups, n_boot=n_boot, seed=42
        )

    # B=199 Multi-sparsity Null Calibration on Seq_G2 and Seq_Pos4mer (both d7 and MGW)
    if shared_bank is None:
        bank_all, bank_slices = generate_multisparsity_null_bank(
            feats["c5_all"], shap_std, B=B_null, seed=null_seed, k_list=K_LADDER_BHLH
        )
    else:
        bank_all, bank_slices = shared_bank

    null_results: Dict[str, Dict[str, object]] = {}
    for base_lbl, xs in [("Seq_G2", X_g2), ("Seq_Pos4mer", X_pos4)]:
        null_results[base_lbl] = {}
        for ch_lbl, sl_c in [("7ch", slice(0, 7)), ("MGW", slice(0, 1))]:
            if precomputed_null_srs is not None and base_lbl in precomputed_null_srs and ch_lbl in precomputed_null_srs[base_lbl]:
                srs_all = np.asarray(precomputed_null_srs[base_lbl][ch_lbl], dtype=np.float64)
            else:
                srs_all = _eval_null_bank_chunked_multibin(xs, B_rc, bank_all[:, :, sl_c], y, groups, n_folds=5, chunk_size=40)
            true_sr = float(srs_all[0])
            base_sr = float(metrics_map[base_lbl]["spearman"])
            fams = {}
            raw_ps = []
            f_keys = list(bank_slices.keys())
            for nk in f_keys:
                arr = srs_all[bank_slices[nk]]
                med = float(np.median(arr))
                p_mc = float((1.0 + np.sum(arr >= true_sr)) / float(len(arr) + 1))
                raw_ps.append(p_mc)
                fams[nk] = {
                    "B": int(len(arr)),
                    "null_median": med,
                    "null_q95": float(np.quantile(arr, 0.95)),
                    "delta_true_minus_median": float(true_sr - med),
                    "p_mc": p_mc,
                    "significant_p005": bool(p_mc <= 0.05),
                }
            holm_ps = holm_bonferroni(raw_ps)
            for i, nk in enumerate(f_keys):
                fams[nk]["p_holm_within_track"] = float(holm_ps[i])
            null_results[base_lbl][ch_lbl] = {
                "base_spearman": base_sr,
                "true_spearman": true_sr,
                "delta_true_minus_base": float(true_sr - base_sr),
                "null_families": fams,
            }

    return {
        "name": name,
        "n_cores": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "models": metrics_map,
        "contrasts": contrasts,
        "null_calibration": null_results,
    }


_HGB_POOL_CTX: Dict[str, object] = {}


def _eval_hgb_table_worker(tab_idx: int) -> Tuple[int, float]:
    ctx = _HGB_POOL_CTX
    X_seq = ctx["X_seq"]
    B_rc = ctx["B_rc"]
    y = ctx["y"]
    splits = ctx["splits"]
    best_cfg = ctx["best_cfg"]
    tables_bank = ctx["tables_bank"]
    if tab_idx < 0:
        X_in = X_seq
    else:
        X_in = np.hstack([X_seq, _contract_b_rc_single(B_rc, tables_bank[tab_idx]).astype(np.float32)])
    oof = np.empty(len(y), dtype=np.float64)
    for tr_i, va_i in splits:
        m = HistGradientBoostingRegressor(max_iter=250, max_bins=64, early_stopping=True, random_state=42, **best_cfg)
        m.fit(X_in[tr_i], y[tr_i])
        oof[va_i] = m.predict(X_in[va_i])
    sr, _ = safe_spearman_pearson(y, oof)
    return int(tab_idx), float(sr)


def eval_bhlh_hgbdt_surrogate(
    name: str,
    feats: Dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    tables_bank: np.ndarray,
    bank_slices: Dict[str, slice],
    B_surr: int = 49,
    n_workers: int = 6,
) -> Dict[str, object]:
    """
    Matched HGBDT surrogate control (F_true vs B=49 ShiftGC_ExactCRT_WS_G2 & Sparse4mer_K12)
    on aligned 14-bp bHLH cores under exact 5-fold GroupKFold CV.
    """
    codes = feats["codes"]
    X_g1 = feats["X_g1"]
    X_g2 = feats["X_g2"]
    B_rc = feats["B_rc"]
    gc2 = X_g2[:, -12:-10]
    X_seq = np.hstack([codes.astype(np.float32), X_g1.astype(np.float32), gc2.astype(np.float32)])

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    splits = list(gkf.split(X_seq, y, groups))
    # Lock deterministic HGBDT config from Seq_Only fold-0 inner GroupCV (matched across true & nulls!)
    in_gkf = GroupKFold(n_splits=min(3, len(np.unique(groups[splits[0][0]]))))
    itr, iva = next(in_gkf.split(X_seq[splits[0][0]], y[splits[0][0]], groups[splits[0][0]]))
    X0 = X_seq[splits[0][0]]
    y0 = y[splits[0][0]]
    cfgs = [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 3.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 3.0},
    ]
    best_cfg, best_mse = cfgs[0], np.inf
    for cfg in cfgs:
        m = HistGradientBoostingRegressor(max_iter=200, max_bins=64, early_stopping=True, random_state=42, **cfg)
        m.fit(X0[itr], y0[itr])
        d = m.predict(X0[iva]) - y0[iva]
        mse = float(np.sum(d * d))
        if mse < best_mse:
            best_mse = mse
            best_cfg = cfg

    fam_idx_map: Dict[str, List[int]] = {}
    all_jobs: List[int] = [-1, 0]
    for fk in ("ShiftGC_ExactCRT_WS_G2", "Sparse4mer_K12"):
        sl = bank_slices[fk]
        idxs = list(range(sl.start, min(sl.stop, sl.start + B_surr)))
        fam_idx_map[fk] = idxs
        all_jobs.extend(idxs)

    _HGB_POOL_CTX.clear()
    _HGB_POOL_CTX.update(
        {
            "X_seq": X_seq,
            "B_rc": B_rc,
            "y": y,
            "splits": splits,
            "best_cfg": best_cfg,
            "tables_bank": tables_bank,
        }
    )
    res_map: Dict[int, float] = {}
    try:
        eff_w = min(max(1, int(n_workers)), len(all_jobs))
        if eff_w <= 1:
            for tid in all_jobs:
                k, v = _eval_hgb_table_worker(tid)
                res_map[k] = v
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=eff_w, mp_context=ctx) as ex:
                for k, v in ex.map(_eval_hgb_table_worker, all_jobs):
                    res_map[k] = v
    finally:
        _HGB_POOL_CTX.clear()

    seq_only_sr = float(res_map[-1])
    true_sr = float(res_map[0])

    fams: Dict[str, Dict[str, object]] = {}
    for fk, idxs in fam_idx_map.items():
        arr = np.array([res_map[i] for i in idxs], dtype=np.float64)
        p_mc = float((1.0 + np.sum(arr >= true_sr)) / float(len(arr) + 1))
        med = float(np.median(arr))
        fams[fk] = {
            "B": int(len(arr)),
            "null_median": med,
            "null_q95": float(np.quantile(arr, 0.95)),
            "null_range": [float(np.min(arr)), float(np.max(arr))],
            "delta_true_minus_median": float(true_sr - med),
            "null_lift_over_seq_median": float(med - seq_only_sr),
            "p_mc": p_mc,
        }

    return {
        "name": name,
        "locked_cfg": best_cfg,
        "seq_only_spearman": seq_only_sr,
        "true_spearman": true_sr,
        "lift_true_minus_seq": float(true_sr - seq_only_sr),
        "null_families": fams,
    }


def parse_gse59845_replicate_cohorts(
    raw_dir: Path,
    paired_36mers: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Parse GSE59845 raw spot replicates (`GSM1447392_Max`, `GSM1447393_cMyc`, `GSM1447394_Mad1`)
    strictly matched onto the `paired_36mers` (`8,000` cohort probes -> canonical 36-mer), computing
    WITHIN-ORIENTATION replicate differences `r1-r2`, `r1-r3`, `r2-r3` on log(max(Alexa488, 1))
    plus `o1-o2` orientation diagnostic.
    """
    target_canon = [canonical_rc_str(s) for s in paired_36mers]
    canon_to_pos: Dict[str, int] = {cs: i for i, cs in enumerate(target_canon)}
    files = {
        "Max": raw_dir / "GSM1447392_Max_rawdata.txt.gz",
        "cMyc": raw_dir / "GSM1447393_cMyc_rawdata.txt.gz",
        "Mad1": raw_dir / "GSM1447394_Mad1_rawdata.txt.gz",
    }
    diff_map: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {}
    pairs = (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))

    for tf, fp in files.items():
        by_ori: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                nm, sq, val = parts[2], parts[4].strip().upper(), parts[5]
                if len(sq) < 36:
                    continue
                cs = canonical_rc_str(sq[:36])
                pos = canon_to_pos.get(cs)
                if pos is None:
                    continue
                toks = nm.split("_")
                if len(toks) < 2:
                    continue
                ori, rep = toks[-2].lower(), toks[-1].lower()
                if ori not in ("o1", "o2") or rep not in ("r1", "r2", "r3"):
                    continue
                try:
                    v = math.log(max(float(val), 1.0))
                except ValueError:
                    continue
                by_ori[(pos, ori)][rep] = v

        # Build probe-aligned vectors (averaged across available orientations o1/o2 for each replicate pair)
        tf_stats = {}
        for ra, rb in pairs:
            arr = np.full(len(paired_36mers), np.nan, dtype=np.float64)
            per_probe = defaultdict(list)
            for (pos, _ori), d in by_ori.items():
                if ra in d and rb in d:
                    per_probe[pos].append(d[ra] - d[rb])
            for pos, vals in per_probe.items():
                arr[pos] = float(np.mean(vals))
            key = f"{tf}_rep_{ra}_minus_{rb}"
            diff_map[key] = arr
            fin = arr[np.isfinite(arr)]
            tf_stats[key] = {
                "n_matched_probes": int(len(fin)),
                "std_log_diff": float(np.std(fin)),
            }

        # Secondary diagnostic: orientation difference within same replicate index
        o_arr = np.full(len(paired_36mers), np.nan, dtype=np.float64)
        o_vals = defaultdict(list)
        for pos in range(len(paired_36mers)):
            d1 = by_ori.get((pos, "o1"), {})
            d2 = by_ori.get((pos, "o2"), {})
            for r in ("r1", "r2", "r3"):
                if r in d1 and r in d2:
                    o_vals[pos].append(d1[r] - d2[r])
        for pos, vals in o_vals.items():
            o_arr[pos] = float(np.mean(vals))
        diff_map[f"{tf}_diag_o1_minus_o2"] = o_arr
        fin_o = o_arr[np.isfinite(o_arr)]
        tf_stats[f"{tf}_diag_o1_minus_o2"] = {
            "n_matched_probes": int(len(fin_o)),
            "std_log_diff": float(np.std(fin_o)) if len(fin_o) else 0.0,
        }
        meta[tf] = tf_stats

    return diff_map, meta


def parse_gse97794_upbm_cohorts(
    raw_dir: Path,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, object]]:
    """
    Parse orthogonal universal PBM arrays (`GSE97794`: `GSM2746660_Max`, `GSM2746659_MadMax`,
    `GSM2746661_MycMax`), filtering flagged probes and selecting CANNTG-bearing 36-mers.
    Returns `(seqs_36bp, y_map, meta)` ready for `align_ebox_14bp_cores`.
    """
    files = {
        "Max_uPBM": raw_dir / "GSM2746660_Max_uPBM_rawdata.txt.gz",
        "MadMax_uPBM": raw_dir / "GSM2746659_MadMax_uPBM_rawdata.txt.gz",
        "MycMax_uPBM": raw_dir / "GSM2746661_MycMax_uPBM_rawdata.txt.gz",
    }
    raw_maps: Dict[str, Dict[str, List[float]]] = {k: defaultdict(list) for k in files}
    for k, fp in files.items():
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                nm, sq = parts[2], parts[4].strip().upper()
                cy3f, alf = parts[6].strip(), parts[8].strip()
                if len(sq) < 36 or nm.startswith(("GE_", "Dark")):
                    continue
                if cy3f or alf:
                    continue
                s36 = sq[:36]
                if any(c not in "ACGT" for c in s36):
                    continue
                # Restrict strictly to probes containing a central CANNTG 6-mer (offsets 4..26 so 14-bp frame fits)
                has_ebox = False
                for off in range(4, 27):
                    hx = s36[off : off + 6]
                    if hx[:2] == "CA" and hx[4:] == "TG":
                        has_ebox = True
                        break
                if not has_ebox:
                    continue
                try:
                    al = float(parts[7])
                except ValueError:
                    continue
                if np.isfinite(al):
                    raw_maps[k][s36].append(math.log(max(al, 1.0)))

    common_seqs = sorted(set(raw_maps["Max_uPBM"]) & set(raw_maps["MadMax_uPBM"]) & set(raw_maps["MycMax_uPBM"]))
    y_max = np.array([float(np.median(raw_maps["Max_uPBM"][s])) for s in common_seqs], dtype=np.float64)
    y_mad = np.array([float(np.median(raw_maps["MadMax_uPBM"][s])) for s in common_seqs], dtype=np.float64)
    y_myc = np.array([float(np.median(raw_maps["MycMax_uPBM"][s])) for s in common_seqs], dtype=np.float64)
    # Z-normalize each array globally across shared CANNTG probes before taking cross-array differential
    z_max = (y_max - y_max.mean()) / max(y_max.std(), 1e-8)
    z_mad = (y_mad - y_mad.mean()) / max(y_mad.std(), 1e-8)
    z_myc = (y_myc - y_myc.mean()) / max(y_myc.std(), 1e-8)

    y_map = {
        "Max_uPBM": z_max,
        "MadMax_uPBM": z_mad,
        "MycMax_uPBM": z_myc,
        "MycMax_minus_Max_uPBM": z_myc - z_max,
        "MadMax_minus_Max_uPBM": z_mad - z_max,
    }
    meta = {
        "n_common_ebox_36bp_probes": len(common_seqs),
        "normalization": "unflagged_log_Alexa488_zscored_across_shared_ebox_probes",
    }
    return common_seqs, y_map, meta


def ks_uniformity_summary(p_values: Sequence[float]) -> Dict[str, object]:
    """Summarize empirical null FPR@0.05 and two-sided KS test vs Uniform(0,1)."""
    arr = np.asarray(p_values, dtype=np.float64)
    ks_res = kstest(arr, "uniform")
    return {
        "n_trials": int(len(arr)),
        "fpr_at_005": float(np.mean(arr <= 0.05)),
        "median_p": float(np.median(arr)),
        "ks_stat": float(ks_res.statistic),
        "ks_p": float(ks_res.pvalue),
    }


def run_inductive_pos4mer_ridge(
    train_seqs: Sequence[str],
    train_y: np.ndarray,
    eval_seqs: Sequence[str],
    seed: int = 42,
) -> np.ndarray:
    """
    Strictly inductive `Pos4mer + Ridge` predictor on aligned 14-bp DNA cores:
      - Zero eval-row statistics used in column pruning, scaling, or Ridge fitting.
      - Zero network calls; invariant to eval row permutation (`diff <= 1e-12`).
    """
    _ = seed  # deterministic solver
    tr_f = build_rc_sym_ebox_features(train_seqs, max_pos_bins_seq=14)
    ev_f = build_rc_sym_ebox_features(eval_seqs, max_pos_bins_seq=14)

    X_tr_raw = tr_f["X_pos4mer"]
    X_ev_raw = ev_f["X_pos4mer"]
    keep = X_tr_raw.std(axis=0) > 1e-12
    X_tr = X_tr_raw[:, keep]
    X_ev = X_ev_raw[:, keep]

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-8)
    Z_tr = (X_tr - mu) / sd
    Z_ev = (X_ev - mu) / sd
    y_mean = float(np.mean(train_y))
    yc = train_y - y_mean

    groups_tr = np.array([canonical_rc_str(s[2:-2]) for s in train_seqs])
    uniq_g = np.unique(groups_tr)
    alphas = np.logspace(0, 4, 9)
    k_folds = min(5, len(uniq_g))
    best_alpha = 100.0
    if k_folds >= 2 and len(yc) >= 30:
        gkf = GroupKFold(n_splits=k_folds)
        splits = list(gkf.split(Z_tr, yc, groups_tr))[:2]
        cv_mse = np.zeros(len(alphas), dtype=np.float64)
        for tr_i, va_i in splits:
            Za, Zv = Z_tr[tr_i], Z_tr[va_i]
            ym_f = float(train_y[tr_i].mean())
            ya, yv = train_y[tr_i] - ym_f, train_y[va_i] - ym_f
            ev, ec = sp_eigh(Za.T @ Za, check_finite=False, driver="evd")
            ev = np.maximum(ev, 0.0)
            proj = ec.T @ (Za.T @ ya)
            preds = (Zv @ ec) @ (proj[:, None] / (ev[:, None] + alphas[None, :]))
            diff = preds - yv[:, None]
            cv_mse += np.sum(diff * diff, axis=0)
        best_alpha = float(alphas[int(np.argmin(cv_mse))])

    p = Z_tr.shape[1]
    K = Z_tr.T @ Z_tr
    K[np.diag_indices(p)] += best_alpha
    cf = cho_factor(K, lower=True, check_finite=False)
    w = cho_solve(cf, Z_tr.T @ yc, check_finite=False)
    return y_mean + Z_ev @ w
