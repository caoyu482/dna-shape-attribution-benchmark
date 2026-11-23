#!/usr/bin/env python3
"""
Step 3C: Fast Batched GroupKFold Solvers & Cross-Architecture 1D RC-CNN Checker
for the 7+1 Group Biological Alignment Ladder.

All feature scalers, residualizers, hyperparameters (lambda), and learnable 5-mer
tables are fitted STRICTLY inside the outer training split via GroupKFold.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

import null_generators as ng

_ASCII_LUT = np.full(256, -1, dtype=np.int8)
for _c, _v in (("A", 0), ("a", 0), ("C", 1), ("c", 1), ("G", 2), ("g", 2), ("T", 3), ("t", 3)):
    _ASCII_LUT[ord(_c)] = _v

# Orthonormal 3-column zero-sum contrast per nucleotide position (A,C,G,T)
_MONO_H = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]], dtype=np.float64
) / 2.0  # columns: MK, WS, RY

# RC sign of each contrast: MK (-1), WS (+1), RY (-1)
_MONO_RC_SIGN = np.array([-1.0, 1.0, -1.0], dtype=np.float64)


def encode_sequences_acgt(seqs: Sequence[str]) -> np.ndarray:
    """Encode equal-length uppercase/lowercase ACGT strings to (N, L) int8."""
    N = len(seqs)
    L = len(seqs[0])
    buf = "".join(seqs).encode("ascii")
    arr = _ASCII_LUT[np.frombuffer(buf, dtype=np.uint8)].reshape(N, L).copy()
    if np.any(arr < 0):
        raise ValueError("Invalid non-ACGT base encountered in sequence array")
    return arr


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    return (3 - codes[:, ::-1]).copy()


def extract_c5_pair(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (c5_fwd, c5_rc) both of shape (N, L-4) aligned in forward orientation."""
    c5 = (((((codes[:, :-4].astype(np.int32) << 2) | codes[:, 1:-3]) << 2) | codes[:, 2:-2]) << 2 | codes[:, 3:-1]) << 2 | codes[:, 4:]
    c5_rc = ng.RC_5[c5][:, ::-1]
    return c5, c5_rc


def build_spatial_bin_tensor(c5: np.ndarray, c5_rc: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
    """
    Build RC-symmetrized spatial pentamer assignment matrix B_mat of shape (N, n_bins, 1024)
    so that ANY table lookup S in R^{1024 x d} projects via simple einsum:
        X_tab(S) = (B_mat @ S).reshape(N, n_bins * d)
    If n_bins is None or n_bins >= L5, uses exact 1-bp positional bins (L5 bins).
    Otherwise pools evenly across n_bins linear spatial bins (zero nonlinearities!).
    """
    N, L5 = c5.shape
    if n_bins is None or n_bins >= L5:
        n_bins_eff = L5
        bin_ids = np.arange(L5, dtype=np.int32)
    else:
        n_bins_eff = n_bins
        bin_ids = np.floor(np.arange(L5) * n_bins_eff / float(L5)).astype(np.int32)

    B_mat = np.zeros((N, n_bins_eff, 1024), dtype=np.float32)
    rows = np.arange(N, dtype=np.int32)[:, None]
    np.add.at(B_mat, (rows, bin_ids[None, :], c5), 0.5)
    np.add.at(B_mat, (rows, bin_ids[None, :], c5_rc), 0.5)
    counts = np.bincount(bin_ids, minlength=n_bins_eff).astype(np.float32)[None, :, None]
    B_mat /= np.maximum(counts, 1.0)
    return B_mat


def build_base_sequence_features(codes: np.ndarray, mode: str = "mono_dinuc", max_pos_bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build strictly RC-symmetrized sequence baselines:
      - X_mono: RC-symmetrized positional (or binned) mononucleotide contrasts (G1).
      - X_mono_dinuc: X_mono + RC-symmetrized positional (or binned) adjacent dinucleotide
                      contrasts + bulk GC, GC^2, 10 canonical dinuc freqs (G2).
    """
    N, L = codes.shape
    rc_codes = reverse_complement_codes(codes)

    # Mono contrasts (N, L, 3)
    m_f = _MONO_H[codes]
    m_r = _MONO_H[rc_codes]
    m_sym = 0.5 * (m_f + m_r[:, ::-1, :] * _MONO_RC_SIGN[None, None, :])
    if L > max_pos_bins:
        b_idx = np.floor(np.arange(L) * max_pos_bins / float(L)).astype(np.int32)
        m_pool = np.zeros((N, max_pos_bins, 3), dtype=np.float64)
        for b in range(max_pos_bins):
            m_pool[:, b, :] = m_sym[:, b_idx == b, :].mean(axis=1)
        X_mono = m_pool.reshape(N, -1)
    else:
        X_mono = m_sym.reshape(N, -1)

    if mode == "mono":
        return X_mono.astype(np.float64), X_mono.astype(np.float64)

    # Pure 9-dim adjacent dinucleotide contrasts (outer product of 3-dim mono contrasts)
    di_f = np.einsum("npa,npb->npab", m_f[:, :-1, :], m_f[:, 1:, :]).reshape(N, L - 1, 9)
    # Under RC, position (p, p+1) maps to (L-2-p, L-1-p) with swapped bases and product sign
    sign_9 = np.outer(_MONO_RC_SIGN, _MONO_RC_SIGN).ravel()
    di_r = np.einsum("npa,npb->npab", m_r[:, :-1, :], m_r[:, 1:, :])
    di_r_swap = np.transpose(di_r, (0, 1, 3, 2)).reshape(N, L - 1, 9)
    di_sym = 0.5 * (di_f + di_r_swap[:, ::-1, :] * sign_9[None, None, :])

    n_di_bins = min(max_pos_bins, L - 1)
    if (L - 1) > n_di_bins:
        b_di = np.floor(np.arange(L - 1) * n_di_bins / float(L - 1)).astype(np.int32)
        di_pool = np.zeros((N, n_di_bins, 9), dtype=np.float64)
        for b in range(n_di_bins):
            di_pool[:, b, :] = di_sym[:, b_di == b, :].mean(axis=1)
        X_di_pos = di_pool.reshape(N, -1)
    else:
        X_di_pos = di_sym.reshape(N, -1)

    # Bulk GC, GC^2, and 10 RC-canonical dinucleotide frequencies
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

    X_g2 = np.hstack([X_mono, X_di_pos, gc[:, None], (gc * gc)[:, None], di_bulk])
    return X_mono.astype(np.float64), X_g2.astype(np.float64)


def fit_predict_ridge_groupcv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: np.ndarray = np.logspace(-2, 4, 13),
    n_folds: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Ultra-fast GroupKFold Ridge via dual/primal eigendecomposition.
    Standardizes features inside training folds only (zero leakage).
    """
    n_tr, p = X_tr.shape
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-8)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean

    uniq_g = np.unique(groups_tr)
    k_folds = min(n_folds, len(uniq_g))
    if k_folds < 2 or n_tr < 15:
        alpha_best = 10.0
        w = np.linalg.solve(Z_tr.T @ Z_tr + alpha_best * np.eye(p), Z_tr.T @ yc)
        return y_mean + Z_te @ w, alpha_best

    gkf = GroupKFold(n_splits=k_folds)
    all_splits = list(gkf.split(Z_tr, yc, groups_tr))
    # Use fold 0 (20% inner group validation) for N_tr > 300, and 3-fold GroupKFold for N_tr <= 300
    splits = all_splits[:1] if n_tr > 300 else all_splits[:min(3, len(all_splits))]

    # Choose dual (n_tr < p) or primal (p <= n_tr) eigendecomposition per fold
    cv_mse = np.zeros(len(alphas), dtype=np.float64)
    for tr_idx, va_idx in splits:
        Za, Zv = Z_tr[tr_idx], Z_tr[va_idx]
        ya, yv = yc[tr_idx], yc[va_idx]
        if Za.shape[0] >= p:
            K = Za.T @ Za
            evals, evecs = eigh(K)
            evals = np.maximum(evals, 0.0)
            Zt_y = Za.T @ ya
            proj = evecs.T @ Zt_y  # (p,)
            Zv_e = Zv @ evecs      # (n_va, p)
            # All alphas simultaneously: (p, n_alpha)
            weights = proj[:, None] / (evals[:, None] + alphas[None, :])
            preds = Zv_e @ weights  # (n_va, n_alpha)
        else:
            G = Za @ Za.T
            evals, evecs = eigh(G)
            evals = np.maximum(evals, 0.0)
            proj = evecs.T @ ya
            Zv_Za = Zv @ Za.T
            Zv_e = Zv_Za @ evecs
            weights = proj[:, None] / (evals[:, None] + alphas[None, :])
            preds = Zv_e @ weights
        err = preds - yv[:, None]
        cv_mse += np.sum(err * err, axis=0)

    best_alpha = float(alphas[int(np.argmin(cv_mse))])
    if n_tr >= p:
        w = np.linalg.solve(Z_tr.T @ Z_tr + best_alpha * np.eye(p), Z_tr.T @ yc)
        y_pred = y_mean + Z_te @ w
    else:
        dual_w = np.linalg.solve(Z_tr @ Z_tr.T + best_alpha * np.eye(n_tr), yc)
        y_pred = y_mean + (Z_te @ Z_tr.T) @ dual_w
    return y_pred, best_alpha


def fit_predict_oof_ridge_groupcv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: np.ndarray = np.logspace(-2, 4, 13),
    n_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit GroupKFold Ridge on (X_tr, y_tr) to predict X_te, AND compute genuine
    out-of-fold predictions y_oof_tr with scalers and alphas tuned inside each fold.
    """
    y_pred_te, best_alpha = fit_predict_ridge_groupcv(
        X_tr, y_tr, groups_tr, X_te, alphas=alphas, n_folds=n_folds
    )
    n_tr = len(y_tr)
    uniq_g = np.unique(groups_tr)
    k_folds = min(n_folds, len(uniq_g))
    if k_folds < 2 or n_tr < 15:
        y_oof_tr = np.full(n_tr, float(y_tr.mean()), dtype=np.float64)
        return y_pred_te, y_oof_tr, best_alpha

    gkf = GroupKFold(n_splits=k_folds)
    y_oof_tr = np.empty(n_tr, dtype=np.float64)
    for tr_i, va_i in gkf.split(X_tr, y_tr, groups_tr):
        inner_folds = max(2, min(n_folds - 1, len(np.unique(groups_tr[tr_i]))))
        pred_va, _ = fit_predict_ridge_groupcv(
            X_tr[tr_i], y_tr[tr_i], groups_tr[tr_i], X_tr[va_i],
            alphas=alphas, n_folds=inner_folds,
        )
        y_oof_tr[va_i] = pred_va
    return y_pred_te, y_oof_tr, best_alpha


def fit_predict_batch_fixed_base_ridge(
    X_base_tr: np.ndarray,
    X_base_te: np.ndarray,
    B_tr: np.ndarray,  # (n_tr, n_bins, 1024)
    B_te: np.ndarray,  # (n_te, n_bins, 1024)
    tables_bank: np.ndarray,  # (M, 1024, d)
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    alphas: np.ndarray = np.logspace(-2, 4, 13),
) -> np.ndarray:
    """
    Evaluate M candidate tables (e.g., 199 nulls + 1 real) with IDENTICAL base features
    and GroupKFold Ridge pipeline using precomputed base Gram blocks. Returns (M, n_te).
    """
    M, _, d = tables_bank.shape
    n_tr, n_bins, _ = B_tr.shape
    n_te = B_te.shape[0]
    p_b = X_base_tr.shape[1]
    p_t = n_bins * d
    p = p_b + p_t
    preds = np.empty((M, n_te), dtype=np.float64)

    # Chunked DGEMM contraction in float64 across all M tables (matches einsum float64 to < 1e-13)
    T_flat = np.ascontiguousarray(tables_bank.transpose(1, 0, 2).reshape(1024, M * d), dtype=np.float64)

    def _contract_bins_f64(B_arr: np.ndarray) -> np.ndarray:
        N_cur = B_arr.shape[0]
        out = np.empty((M, N_cur, p_t), dtype=np.float64)
        chunk = max(256, 16384 // max(n_bins, 1))
        for s0 in range(0, N_cur, chunk):
            s1 = min(s0 + chunk, N_cur)
            blk = B_arr[s0:s1].reshape((s1 - s0) * n_bins, 1024).astype(np.float64)
            prod = (blk @ T_flat).reshape(s1 - s0, n_bins, M, d).transpose(2, 0, 1, 3).reshape(M, s1 - s0, p_t)
            out[:, s0:s1, :] = prod
        return out

    Xt_tr_all = _contract_bins_f64(B_tr)
    Xt_te_all = _contract_bins_f64(B_te)

    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    uniq_g = np.unique(groups_tr)
    k_folds = min(5, len(uniq_g))
    if k_folds < 2 or n_tr < max(15, p):
        for m in range(M):
            X_tr = np.hstack([X_base_tr, Xt_tr_all[m]])
            X_te = np.hstack([X_base_te, Xt_te_all[m]])
            preds[m], _ = fit_predict_ridge_groupcv(X_tr, y_tr, groups_tr, X_te, alphas=alphas)
        return preds

    gkf = GroupKFold(n_splits=k_folds)
    all_splits = list(gkf.split(X_base_tr, yc, groups_tr))
    splits = all_splits[:1] if n_tr > 300 else all_splits[:min(3, len(all_splits))]
    if not all(len(tr_i) >= p for tr_i, _ in splits):
        for m in range(M):
            X_tr = np.hstack([X_base_tr, Xt_tr_all[m]])
            X_te = np.hstack([X_base_te, Xt_te_all[m]])
            preds[m], _ = fit_predict_ridge_groupcv(X_tr, y_tr, groups_tr, X_te, alphas=alphas)
        return preds

    mu_b = X_base_tr.mean(axis=0, keepdims=True)
    sd_b = np.maximum(X_base_tr.std(axis=0, keepdims=True), 1e-8)
    Zb_tr = (X_base_tr - mu_b) / sd_b
    Zb_te = (X_base_te - mu_b) / sd_b

    mu_t = Xt_tr_all.mean(axis=1, keepdims=True)
    sd_t = np.maximum(Xt_tr_all.std(axis=1, keepdims=True), 1e-8)
    Zt_tr_all = (Xt_tr_all - mu_t) / sd_t
    Zt_te_all = (Xt_te_all - mu_t) / sd_t
    del Xt_tr_all, Xt_te_all

    # Precompute base & cross-Gram matrices ONCE outside the M loop
    G_bb_full = Zb_tr.T @ Zb_tr
    Zb_yc = Zb_tr.T @ yc
    G_bt_full = (Zb_tr.T @ Zt_tr_all.transpose(1, 0, 2).reshape(n_tr, M * p_t)).reshape(p_b, M, p_t).transpose(1, 0, 2)
    Zt_yc_all = np.einsum("mni,n->mi", Zt_tr_all, yc)

    split_cache = []
    for tr_i, va_i in splits:
        Zb_a, Zb_v = Zb_tr[tr_i], Zb_tr[va_i]
        Zt_a = Zt_tr_all[:, tr_i, :]
        Zt_v = Zt_tr_all[:, va_i, :]
        ya, yv = yc[tr_i], yc[va_i]
        G_bb_a = Zb_a.T @ Zb_a
        Zb_ya = Zb_a.T @ ya
        G_bt_a = (Zb_a.T @ Zt_a.transpose(1, 0, 2).reshape(len(tr_i), M * p_t)).reshape(p_b, M, p_t).transpose(1, 0, 2)
        Zt_ya_all = np.einsum("mni,n->mi", Zt_a, ya)
        split_cache.append((Zb_v, Zt_a, Zt_v, yv, G_bb_a, Zb_ya, G_bt_a, Zt_ya_all))

    diag_idx = np.diag_indices(p)
    K_a = np.empty((p, p), dtype=np.float64)
    K_full = np.empty((p, p), dtype=np.float64)
    Zt_y_a = np.empty(p, dtype=np.float64)
    Zt_y_f = np.empty(p, dtype=np.float64)

    for m in range(M):
        cv_mse = np.zeros(len(alphas), dtype=np.float64)
        for Zb_v, Zt_a, Zt_v, yv, G_bb_a, Zb_ya, G_bt_a, Zt_ya_all in split_cache:
            K_a[:p_b, :p_b] = G_bb_a
            K_a[:p_b, p_b:] = G_bt_a[m]
            K_a[p_b:, :p_b] = G_bt_a[m].T
            Zt_am = Zt_a[m]
            K_a[p_b:, p_b:] = Zt_am.T @ Zt_am
            evals, evecs = eigh(K_a)
            evals = np.maximum(evals, 0.0)
            Zt_y_a[:p_b] = Zb_ya
            Zt_y_a[p_b:] = Zt_ya_all[m]
            proj = evecs.T @ Zt_y_a
            Zv_e = Zb_v @ evecs[:p_b, :] + Zt_v[m] @ evecs[p_b:, :]
            weights = proj[:, None] / (evals[:, None] + alphas[None, :])
            preds_va = Zv_e @ weights
            err = preds_va - yv[:, None]
            cv_mse += np.sum(err * err, axis=0)

        best_alpha = float(alphas[int(np.argmin(cv_mse))])
        K_full[:p_b, :p_b] = G_bb_full
        K_full[:p_b, p_b:] = G_bt_full[m]
        K_full[p_b:, :p_b] = G_bt_full[m].T
        Zt_fm = Zt_tr_all[m]
        K_full[p_b:, p_b:] = Zt_fm.T @ Zt_fm
        K_full[diag_idx] += best_alpha
        Zt_y_f[:p_b] = Zb_yc
        Zt_y_f[p_b:] = Zt_yc_all[m]
        w = np.linalg.solve(K_full, Zt_y_f)
        preds[m] = y_mean + Zb_te @ w[:p_b] + Zt_te_all[m] @ w[p_b:]
    return preds


def fit_learnable_5mer_control(
    X_base_tr: np.ndarray,
    X_base_te: np.ndarray,
    B_tr: np.ndarray,
    B_te: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    n_channels: int = 7,
    alphas: np.ndarray = np.logspace(-2, 4, 13),
    alphas_5mer: np.ndarray = np.logspace(-3, 5, 17),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group 7 Capacity Control (G7_Learnable_5mer):
    Two-stage leak-free residual Ridge on canonical 512 RC-even pentamer bin spectrum:
      Stage 1: Fit X_base via GroupKFold OOF Ridge to obtain out-of-fold residual r_oof.
      Stage 2: Fit scalar-RMS-normalized multi-resolution canonical 512-mer weights on r_oof
               via GroupKFold eigendecomposition over wide alphas_5mer, preventing rare-kmer
               inverse-std dilution and ensuring smooth shrinkage to X_base at low N while
               converging to physical lookup capacity at high N.
    Returns (y_pred_learn_d7, y_pred_full_5mer).
    """
    p_base_te, p_base_oof, _ = fit_predict_oof_ridge_groupcv(
        X_base_tr, y_tr, groups_tr, X_base_te, alphas=alphas
    )
    r_oof = y_tr - p_base_oof

    N_tr, n_bins, _ = B_tr.shape
    canon = ng.CANON_5
    # Sum both RC partners onto canonical 512 pentamers: B_canon in R^{N x n_bins x 512}
    B_tr_c = (B_tr[:, :, canon] + B_tr[:, :, ng.RC_5[canon]]).astype(np.float64)
    B_te_c = (B_te[:, :, canon] + B_te[:, :, ng.RC_5[canon]]).astype(np.float64)
    mu_c = B_tr_c.mean(axis=0, keepdims=True)
    Bc_tr = B_tr_c - mu_c
    Bc_te = B_te_c - mu_c

    if n_bins == 1:
        P_tr = Bc_tr[:, 0, :]
        inv_sqb = 1.0
    else:
        inv_sqb = 1.0 / np.sqrt(float(n_bins))
        P_tr = np.hstack([Bc_tr.mean(axis=1), Bc_tr.reshape(N_tr, n_bins * 512) * inv_sqb])

    rms = np.sqrt(max(float(np.mean(np.sum(P_tr * P_tr, axis=1))), 1e-12))
    Z_tr = P_tr / rms
    p_z = Z_tr.shape[1]

    def _w_to_mat(w_vec: np.ndarray) -> np.ndarray:
        w_eff = w_vec / rms
        if n_bins == 1:
            return w_eff.reshape(1, 512)
        return (w_eff[:512] / float(n_bins))[None, :] + w_eff[512:].reshape(n_bins, 512) * inv_sqb

    def _solve_alpha_path_matrix(Za: np.ndarray, ya: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
        """
        Return W_z_all of shape (p_z, len(alpha_grid)) solving the exact primal/dual Ridge
        normal equations. Uses fast divide-and-conquer sp_eigh for min(n_a, p_z) <= 2800,
        and precomputed Gram + warm-started descending-alpha CG above 2800.
        """
        n_a = Za.shape[0]
        if min(n_a, p_z) <= 2800:
            if n_a >= p_z:
                evals, evecs = eigh(Za.T @ Za, check_finite=False, driver="evd")
                evals = np.maximum(evals, 0.0)
                proj = evecs.T @ (Za.T @ ya)
                return evecs @ (proj[:, None] / (evals[:, None] + alpha_grid[None, :]))
            evals, evecs = eigh(Za @ Za.T, check_finite=False, driver="evd")
            evals = np.maximum(evals, 0.0)
            proj = evecs.T @ ya
            dual_all = evecs @ (proj[:, None] / (evals[:, None] + alpha_grid[None, :]))
            return Za.T @ dual_all

        W_out = np.empty((p_z, len(alpha_grid)), dtype=np.float64)
        desc_idx = np.argsort(alpha_grid)[::-1]
        if n_a >= p_z:
            K = Za.T @ Za
            rhs = Za.T @ ya
            x_prev = np.zeros(p_z, dtype=np.float64)
            for idx_a in desc_idx:
                lam = float(alpha_grid[idx_a])
                op = LinearOperator((p_z, p_z), matvec=lambda v, l=lam: K @ v + l * v, dtype=np.float64)
                x_prev, _ = cg(op, rhs, x0=x_prev, rtol=1e-5, maxiter=150)
                W_out[:, idx_a] = x_prev
        else:
            G = Za @ Za.T
            x_prev = np.zeros(n_a, dtype=np.float64)
            for idx_a in desc_idx:
                lam = float(alpha_grid[idx_a])
                op = LinearOperator((n_a, n_a), matvec=lambda v, l=lam: G @ v + l * v, dtype=np.float64)
                x_prev, _ = cg(op, ya, x0=x_prev, rtol=1e-5, maxiter=150)
                W_out[:, idx_a] = Za.T @ x_prev
        return W_out

    k_use = min(n_channels, n_bins)
    uniq_g = np.unique(groups_tr)
    k_folds = min(5, len(uniq_g))
    if k_folds < 2 or N_tr < 15:
        best_alpha_full = 100.0
        best_alpha_d7 = 100.0
    else:
        gkf = GroupKFold(n_splits=k_folds)
        all_splits = list(gkf.split(Z_tr, r_oof, groups_tr))
        splits = all_splits[:1] if N_tr > 300 else all_splits[:min(3, len(all_splits))]
        cv_mse_full = np.zeros(len(alphas_5mer), dtype=np.float64)
        cv_mse_d7 = np.zeros(len(alphas_5mer), dtype=np.float64)
        for tr_i, va_i in splits:
            Za = Z_tr[tr_i]
            ra_mean = float(r_oof[tr_i].mean())
            ya = r_oof[tr_i] - ra_mean
            yv = r_oof[va_i] - ra_mean
            Bc_va = Bc_tr[va_i]
            W_z_all = _solve_alpha_path_matrix(Za, ya, alphas_5mer)

            for a_idx in range(len(alphas_5mer)):
                W_lam = _w_to_mat(W_z_all[:, a_idx])
                pred_full_va = np.einsum("nbk,bk->n", Bc_va, W_lam)
                diff_f = pred_full_va - yv
                cv_mse_full[a_idx] += float(np.sum(diff_f * diff_f))
                if k_use < n_bins:
                    u, s, vt = np.linalg.svd(W_lam, full_matrices=False)
                    W_d7 = (u[:, :k_use] * s[:k_use]) @ vt[:k_use, :]
                    pred_d7_va = np.einsum("nbk,bk->n", Bc_va, W_d7)
                    diff_d = pred_d7_va - yv
                    cv_mse_d7[a_idx] += float(np.sum(diff_d * diff_d))
                else:
                    cv_mse_d7[a_idx] += float(np.sum(diff_f * diff_f))

        best_alpha_full = float(alphas_5mer[int(np.argmin(cv_mse_full))])
        best_alpha_d7 = float(alphas_5mer[int(np.argmin(cv_mse_d7))])

    r_mean = float(r_oof.mean())
    yc_r = r_oof - r_mean

    def _solve_full_z(lam: float) -> np.ndarray:
        if N_tr >= p_z:
            K = Z_tr.T @ Z_tr
            K[np.diag_indices(p_z)] += lam
            cf = cho_factor(K, lower=True, check_finite=False)
            wz = cho_solve(cf, Z_tr.T @ yc_r, check_finite=False)
        else:
            G = Z_tr @ Z_tr.T
            G[np.diag_indices(N_tr)] += lam
            cf = cho_factor(G, lower=True, check_finite=False)
            dual = cho_solve(cf, yc_r, check_finite=False)
            wz = Z_tr.T @ dual
        return _w_to_mat(wz)

    W_full = _solve_full_z(best_alpha_full)
    y_pred_full = p_base_te + r_mean + np.einsum("nbk,bk->n", Bc_te, W_full)

    W_raw_d7 = W_full if best_alpha_d7 == best_alpha_full else _solve_full_z(best_alpha_d7)
    if k_use < n_bins:
        u, s, vt = np.linalg.svd(W_raw_d7, full_matrices=False)
        W_d7 = (u[:, :k_use] * s[:k_use]) @ vt[:k_use, :]
    else:
        W_d7 = W_raw_d7
    y_pred_d7 = p_base_te + r_mean + np.einsum("nbk,bk->n", Bc_te, W_d7)
    return y_pred_d7, y_pred_full


def safe_spearman_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return 0.0, 0.0
    sr = float(spearmanr(y_true, y_pred).statistic)
    pr = float(pearsonr(y_true, y_pred).statistic)
    return (0.0 if not np.isfinite(sr) else sr), (0.0 if not np.isfinite(pr) else pr)


def fast_batch_spearman_pearson(y_true: np.ndarray, Y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch Spearman & Pearson correlations across M models (Y_preds shape (M, N_te)).
    Uses scipy.stats.rankdata once on y_true (handling ties) and pure C argsort-twice on Y_preds.
    """
    from scipy.stats import rankdata
    M, N = Y_preds.shape
    if np.std(y_true) < 1e-12 or N < 3:
        return np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64)

    yt_c = y_true - y_true.mean()
    yt_norm = float(np.linalg.norm(yt_c))
    yp_c = Y_preds - Y_preds.mean(axis=1, keepdims=True)
    yp_norm = np.linalg.norm(yp_c, axis=1)
    valid = yp_norm > 1e-12
    pearsons = np.zeros(M, dtype=np.float64)
    pearsons[valid] = (yp_c[valid] @ yt_c) / (yp_norm[valid] * yt_norm)

    ry = rankdata(y_true).astype(np.float64)
    ry -= ry.mean()
    ry_norm = float(np.linalg.norm(ry))
    rp = np.argsort(np.argsort(Y_preds, axis=1), axis=1).astype(np.float64)
    rp -= rp.mean(axis=1, keepdims=True)
    rp_norm = np.linalg.norm(rp, axis=1)
    spearmans = np.zeros(M, dtype=np.float64)
    spearmans[valid] = (rp[valid] @ ry) / (np.maximum(rp_norm[valid], 1e-12) * ry_norm)
    return spearmans, pearsons


class TinyRCCNN(nn.Module):
    """Compact RC-equivariant 1D CNN for cross-architecture consistency check."""

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.head = nn.Linear(hidden * 2, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)

    def forward(self, x_fwd: torch.Tensor, x_rc: torch.Tensor) -> torch.Tensor:
        e = 0.5 * (self._encode(x_fwd) + self._encode(x_rc))
        return self.head(e).squeeze(-1)


def _seq_to_cnn_tensor(codes: np.ndarray, table: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    N, L = codes.shape
    rc_codes = reverse_complement_codes(codes)
    oh_f = np.eye(4, dtype=np.float32)[codes].transpose(0, 2, 1)  # (N, 4, L)
    oh_r = np.eye(4, dtype=np.float32)[rc_codes].transpose(0, 2, 1)
    if table is None:
        return torch.from_numpy(oh_f), torch.from_numpy(oh_r)
    c5_f, c5_r_aligned = extract_c5_pair(codes)
    # Pad edges by 2 bases so shape tracks align to length L
    pad_f = np.zeros((N, L, table.shape[1]), dtype=np.float32)
    pad_r = np.zeros((N, L, table.shape[1]), dtype=np.float32)
    pad_f[:, 2:L - 2, :] = table[c5_f].astype(np.float32)
    c5_rc_raw = (((((rc_codes[:, :-4].astype(np.int32) << 2) | rc_codes[:, 1:-3]) << 2) | rc_codes[:, 2:-2]) << 2 | rc_codes[:, 3:-1]) << 2 | rc_codes[:, 4:]
    pad_r[:, 2:L - 2, :] = table[c5_rc_raw].astype(np.float32)
    xf = np.concatenate([oh_f, pad_f.transpose(0, 2, 1)], axis=1)
    xr = np.concatenate([oh_r, pad_r.transpose(0, 2, 1)], axis=1)
    return torch.from_numpy(xf), torch.from_numpy(xr)


def eval_rc_cnn_one(
    codes_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    codes_te: np.ndarray,
    y_te: np.ndarray,
    table: Optional[np.ndarray],
    seed: int = 11,
    epochs: int = 30,
) -> float:
    """Train and evaluate TinyRCCNN with inner GroupKFold validation early stopping; returns test Spearman."""
    torch.manual_seed(seed)
    xf_tr, xr_tr = _seq_to_cnn_tensor(codes_tr, table)
    xf_te, xr_te = _seq_to_cnn_tensor(codes_te, table)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_tr))))
    tr_i, va_i = next(gkf.split(codes_tr, y_tr, groups_tr))

    ym, ys = float(y_tr[tr_i].mean()), max(float(y_tr[tr_i].std()), 1e-6)
    yt = torch.from_numpy(((y_tr - ym) / ys).astype(np.float32))

    model = TinyRCCNN(in_ch=xf_tr.shape[1], hidden=32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_val = 1e18
    best_state = None
    bs = 128
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train()
        perm = rng.permutation(tr_i)
        for s0 in range(0, len(perm), bs):
            b = perm[s0:s0 + bs]
            opt.zero_grad()
            pred = model(xf_tr[b], xr_tr[b])
            loss = torch.mean((pred - yt[b]) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(xf_tr[va_i], xr_tr[va_i])
            vloss = float(torch.mean((vpred - yt[va_i]) ** 2))
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = (model(xf_te, xr_te).numpy() * ys) + ym
    sr, _ = safe_spearman_pearson(y_te, te_pred)
    return sr
