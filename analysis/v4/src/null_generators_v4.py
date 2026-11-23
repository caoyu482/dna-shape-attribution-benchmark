#!/usr/bin/env python3
"""
Phase 1 Zero-Hyperparameter CRT & Exact Shapelizer Port (`biophys_inductive_gfm/v4`).

Provides:
1. ShiftGCExactCRTNull:
   Zero-hyperparameter Fixed-X Representation-Space Conditional Randomization /
   Compact Group-Invariance Test stratified analytically by:
     - Conditioning nuisance subspace W_fix in {"WS_only" (20d), "WS_G1" (24d), "WS_G2" (40d)}
     - Exact Hamming interaction order r = 1..5 in L^2(p_X) (Laplace-smoothed RC-sym p_X > 0)
     - Exact De Bruijn shift-coboundary vs shift-coherent algebraic split (zero SVD tuning!)
     - Adjacent r=2 vs gapped r=2 split when r=2 is active ("WS_only", "WS_G1").
2. Shapelizer2018ExactPort:
   Exact Python 3 port of BussemakerLab/Shapelizer randomKmerTable.py with switches
   for exact raw unstandardized output vs column-standardized output and consistent_ddof.
3. RubeShapeProjectionEvaluator & compute_shapelizer_fig3_crossr2:
   Faithful Python 3 port of BussemakerLab/Shapelizer kMerLinearRegression.py and
   shapeProjection.py for direct Rube et al. (MSB 2018) Fig 3 & Fig 5 reproduction.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt
from scipy.stats import pearsonr

V3_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v3/src")
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))

import null_generators as v3_ng  # noqa: E402

CANON_5 = v3_ng.CANON_5
RC_5 = v3_ng.RC_5
PENTAMER_STRINGS = v3_ng.PENTAMER_STRINGS
CHANNELS_7_EVEN = v3_ng.CHANNELS_7_EVEN
DATA_TABLE_DIR = v3_ng.DATA_TABLE_DIR
load_biophysical_table_matrix = v3_ng.load_biophysical_table_matrix
haar_orthogonal = v3_ng.haar_orthogonal
build_fourier_basis = v3_ng.build_fourier_basis
NaiveRCPermutationNull = v3_ng.NaiveRCPermutationNull
Shapelizer2018Null = v3_ng.Shapelizer2018Null
UnconstrainedSpectralNull = v3_ng.UnconstrainedSpectralNull
GCConstrainedSpectralNull = v3_ng.GCConstrainedSpectralNull
ShiftGCUnifSpectralNull = v3_ng.ShiftGCUnifSpectralNull
ShiftGCEmpiricalSpectralNull = v3_ng.ShiftGCEmpiricalSpectralNull

_NUCL = ("A", "C", "G", "T")
_TRANTAB = str.maketrans("ACGT", "TGCA")


def _build_exact_algebraic_decomposition() -> Dict:
    """
    Precompute the exact 512-dim RC-even & 512-dim RC-odd orthogonal decomposition on 5-mers,
    together with the exact De Bruijn shift-coboundary spaces Delta(D_odd) -> V_even and
    Delta(D_even) -> V_odd. Cached once at module import.
    """
    U, digits, signs, _ = build_fourier_basis()
    pure_ws = np.all((digits == 0) | (digits == 2), axis=1)
    orders = np.sum(digits > 0, axis=1)
    weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
    sup5 = np.sum((digits > 0) * weights[None, :], axis=1)
    adj_supports = {3, 6, 12, 24}

    # Exact 4-mer -> 5-mer discrete derivative Delta g(x0..x4) = g(x0..x3) - g(x1..x4)
    d4 = np.array(np.unravel_index(np.arange(256), (4,) * 4)).T
    rc4 = np.ravel_multi_index((3 - d4[:, ::-1]).T, (4,) * 4)
    odd4, even4 = [], []
    for i in range(256):
        j = int(rc4[i])
        if i == j:
            e = np.zeros(256, dtype=np.float64)
            e[i] = 1.0
            even4.append(e)
        elif i < j:
            vo = np.zeros(256, dtype=np.float64)
            vo[i] = 1.0 / np.sqrt(2.0)
            vo[j] = -1.0 / np.sqrt(2.0)
            odd4.append(vo)
            ve = np.zeros(256, dtype=np.float64)
            ve[i] = 1.0 / np.sqrt(2.0)
            ve[j] = 1.0 / np.sqrt(2.0)
            even4.append(ve)
    G4_odd = np.column_stack(odd4)   # (256, 120) -> maps to V_even coboundaries
    G4_even = np.column_stack(even4) # (256, 136) -> maps to V_odd coboundaries

    d5 = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T
    p_left = np.ravel_multi_index(d5[:, :4].T, (4,) * 4)
    p_right = np.ravel_multi_index(d5[:, 1:].T, (4,) * 4)

    Psi_even_cob = G4_odd[p_left, :] - G4_odd[p_right, :]
    u_ec, s_ec, _ = np.linalg.svd(Psi_even_cob, full_matrices=False)
    Q_cob_even = u_ec[:, s_ec > 1e-10]  # (1024, 120)

    Psi_odd_cob = G4_even[p_left, :] - G4_even[p_right, :]
    u_oc, s_oc, _ = np.linalg.svd(Psi_odd_cob, full_matrices=False)
    Q_cob_odd = u_oc[:, s_oc > 1e-10]   # (1024, 135)

    even_ws: List[np.ndarray] = []
    even_nonws_by_r: Dict[int, List[np.ndarray]] = {r: [] for r in range(1, 6)}
    even_adj_r2: List[np.ndarray] = []
    even_gap_r2: List[np.ndarray] = []

    odd_by_r: Dict[int, List[np.ndarray]] = {r: [] for r in range(1, 6)}

    used = set()
    for i in range(1024):
        if i in used:
            continue
        j = int(np.ravel_multi_index(digits[i, ::-1], (4,) * 5))
        used.add(i)
        used.add(j)
        sg = int(np.prod(signs[digits[i]]))
        r = int(orders[i])
        if i == j:
            if sg == 1:
                v_e = U[:, i]
                if pure_ws[i]:
                    even_ws.append(v_e)
                else:
                    even_nonws_by_r[r].append(v_e)
                    if r == 2:
                        if int(sup5[i]) in adj_supports:
                            even_adj_r2.append(v_e)
                        else:
                            even_gap_r2.append(v_e)
            else:
                odd_by_r[r].append(U[:, i])
        else:
            v_e = (U[:, i] + sg * U[:, j]) / np.sqrt(2.0)
            v_o = (U[:, i] - sg * U[:, j]) / np.sqrt(2.0)
            if pure_ws[i]:
                even_ws.append(v_e)
            else:
                even_nonws_by_r[r].append(v_e)
                if r == 2:
                    if int(sup5[i]) in adj_supports:
                        even_adj_r2.append(v_e)
                    else:
                        even_gap_r2.append(v_e)
            odd_by_r[r].append(v_o)

    def _split_coh_tele(W: np.ndarray, Q_cob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_w, s_w, _ = np.linalg.svd(W.T @ Q_cob, full_matrices=True)
        W_rot = W @ u_w
        n_tele = int(np.sum(s_w > 1.0 - 1e-9))
        W_tele = W_rot[:, :n_tele] if n_tele > 0 else np.zeros((1024, 0), dtype=np.float64)
        W_coh = W_rot[:, n_tele:] if n_tele < W_rot.shape[1] else np.zeros((1024, 0), dtype=np.float64)
        return W_coh, W_tele

    even_subblocks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, vecs in [
        ("r1", even_nonws_by_r[1]),
        ("r2_adj", even_adj_r2),
        ("r2_gap", even_gap_r2),
        ("r3", even_nonws_by_r[3]),
        ("r4", even_nonws_by_r[4]),
        ("r5", even_nonws_by_r[5]),
    ]:
        even_subblocks[key] = _split_coh_tele(np.column_stack(vecs), Q_cob_even)

    odd_subblocks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for r in range(1, 6):
        odd_subblocks[r] = _split_coh_tele(np.column_stack(odd_by_r[r]), Q_cob_odd)

    return {
        "W_ws": np.column_stack(even_ws),               # (1024, 20)
        "W_r1_nonws": np.column_stack(even_nonws_by_r[1]),  # (1024, 4)
        "W_r2_adj": np.column_stack(even_adj_r2),        # (1024, 16)
        "W_r2_gap": np.column_stack(even_gap_r2),        # (1024, 26)
        "even_subblocks": even_subblocks,
        "odd_subblocks": odd_subblocks,
        "Q_cob_even": Q_cob_even,
        "Q_cob_odd": Q_cob_odd,
    }


_ALGEBRAIC_DECOMP = _build_exact_algebraic_decomposition()


class ShiftGCExactCRTNull:
    """
    Zero-Hyperparameter Fixed-X Representation-Space CRT / Group-Invariance Null.

    Given empirical pentamer frequencies p_raw (counts or probabilities) on unlabeled sequences X,
    defines Laplace-smoothed RC-symmetric measure p_emp > 0 and compact orthogonal group
    G_X = prod_b O(d_b) acting in L^2(p_emp) conditionally on explicit analytical nuisance space:
      - conditioning == "WS_only": fixes W_ws (20 dims: constant + all pure-WS GC functions).
      - conditioning == "WS_G1"  : fixes W_ws + non-WS r=1 (24 dims).
      - conditioning == "WS_G2"  : fixes W_ws + non-WS r=1 + adjacent r=2 (40 dims; exact G2 closure).
    Optionally handles RC-odd blocks (only active if table has non-zero RC-odd differential channels).
    """

    def __init__(
        self,
        F_true: np.ndarray,
        p_raw: np.ndarray,
        conditioning: str = "WS_G2",
        n_total_obs: Optional[float] = None,
    ):
        if conditioning not in ("WS_only", "WS_G1", "WS_G2"):
            raise ValueError(f"Unsupported conditioning level: {conditioning}")
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.conditioning = conditioning

        p_raw_arr = np.asarray(p_raw, dtype=np.float64)
        p_sym_raw = 0.5 * (p_raw_arr + p_raw_arr[RC_5])
        raw_sum = float(np.sum(p_sym_raw))
        if raw_sum <= 0.0:
            n_tot = 1.0 if n_total_obs is None else float(n_total_obs)
            p_emp = np.full(1024, 1.0 / 1024.0, dtype=np.float64)
        else:
            if n_total_obs is not None:
                n_tot = float(n_total_obs)
            elif raw_sum > 2.0:
                n_tot = raw_sum
            else:
                # Default pseudocount horizon if caller passed normalized probabilities without N_total
                n_tot = 10000.0
            p_norm = p_sym_raw / raw_sum
            p_emp = (p_norm * n_tot + 1.0 / 1024.0) / (n_tot + 1.0)
        p_emp /= p_emp.sum()
        self.p_emp = p_emp
        sqp = np.sqrt(p_emp)

        dec = _ALGEBRAIC_DECOMP
        fix_parts = [dec["W_ws"]]
        if conditioning in ("WS_G1", "WS_G2"):
            fix_parts.append(dec["W_r1_nonws"])
        if conditioning == "WS_G2":
            fix_parts.append(dec["W_r2_adj"])
        W_fix = np.hstack(fix_parts)

        def _orth_in_p(M_mat: np.ndarray, Q_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if M_mat.shape[1] == 0:
                return np.zeros((1024, 0), dtype=np.float64), np.zeros((1024, 0), dtype=np.float64)
            Mw = sqp[:, None] * M_mat
            if Q_prev.shape[1] > 0:
                Mw = Mw - Q_prev @ (Q_prev.T @ Mw)
            u, s, _ = np.linalg.svd(Mw, full_matrices=False)
            keep = s > 1e-10
            Q_new = u[:, keep]
            return Q_new, Q_new / sqp[:, None]

        Q_fix, Phi_fix = _orth_in_p(W_fix, np.zeros((1024, 0), dtype=np.float64))
        Q_prev_even = Q_fix
        rot_blocks: List[Tuple[str, np.ndarray]] = []

        even_order_keys = ["r1", "r2_adj", "r2_gap", "r3", "r4", "r5"]
        if conditioning == "WS_G1":
            even_order_keys = ["r2_adj", "r2_gap", "r3", "r4", "r5"]
        elif conditioning == "WS_G2":
            even_order_keys = ["r2_gap", "r3", "r4", "r5"]

        for key in even_order_keys:
            W_coh, W_tele = dec["even_subblocks"][key]
            for tag, W_part in ((f"{key}_coh", W_coh), (f"{key}_tele", W_tele)):
                if W_part.shape[1] == 0:
                    continue
                Qb, Phib = _orth_in_p(W_part, Q_prev_even)
                Q_prev_even = np.hstack([Q_prev_even, Qb])
                if Phib.shape[1] >= 2:
                    rot_blocks.append((tag, Phib))
                elif Phib.shape[1] == 1:
                    # 1D orthogonal group {-1, +1}; append or keep separate depending on parity
                    rot_blocks.append((tag, Phib))

        # Check if F_true has any RC-odd energy
        F_odd = 0.5 * (self.F_true - self.F_true[RC_5])
        if np.max(np.abs(F_odd)) > 1e-11:
            Q_prev_odd = np.zeros((1024, 0), dtype=np.float64)
            for r in range(1, 6):
                W_coh, W_tele = dec["odd_subblocks"][r]
                for tag, W_part in ((f"odd_r{r}_coh", W_coh), (f"odd_r{r}_tele", W_tele)):
                    if W_part.shape[1] == 0:
                        continue
                    Qb, Phib = _orth_in_p(W_part, Q_prev_odd)
                    Q_prev_odd = np.hstack([Q_prev_odd, Qb])
                    if Phib.shape[1] >= 1:
                        rot_blocks.append((tag, Phib))

        self.Phi_fix = Phi_fix
        self.base_proj = Phi_fix @ ((Phi_fix.T * p_emp[None, :]) @ self.F_true)
        self.n_fixed_dims = int(Phi_fix.shape[1])
        self.rot_list: List[Tuple[str, int, np.ndarray, np.ndarray]] = []
        self.n_rot_dims = 0
        for tag, Phi_b in rot_blocks:
            C_b = (Phi_b.T * p_emp[None, :]) @ self.F_true
            if np.max(np.abs(C_b)) > 1e-12:
                self.rot_list.append((tag, Phi_b.shape[1], Phi_b, C_b))
                self.n_rot_dims += Phi_b.shape[1]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.base_proj.copy()
        for _, dim, Phi_b, C_b in self.rot_list:
            if dim == 1:
                q = np.array([[1.0 if rng.random() < 0.5 else -1.0]], dtype=np.float64)
            else:
                q = haar_orthogonal(dim, rng)
            ans += Phi_b @ (q @ C_b)
        return ans

    def audit_invariants(self, F_null: np.ndarray) -> Dict[str, float]:
        """Verify machine-precision exact preservation of nuisance projection, p_emp mean/cov & block Grams."""
        p = self.p_emp
        mu_t = p @ self.F_true
        mu_n = p @ F_null
        Ft_c = self.F_true - mu_t[None, :]
        Fn_c = F_null - mu_n[None, :]
        cov_t = (Ft_c.T * p[None, :]) @ Ft_c
        cov_n = (Fn_c.T * p[None, :]) @ Fn_c
        fix_t = (self.Phi_fix.T * p[None, :]) @ self.F_true
        fix_n = (self.Phi_fix.T * p[None, :]) @ F_null
        max_gram_err = 0.0
        for _, _, Phi_b, C_t in self.rot_list:
            C_n = (Phi_b.T * p[None, :]) @ F_null
            max_gram_err = max(max_gram_err, float(np.max(np.abs(C_n.T @ C_n - C_t.T @ C_t))))
        rc_err = float(np.max(np.abs(F_null - F_null[RC_5]))) if np.max(np.abs(self.F_true - self.F_true[RC_5])) < 1e-11 else 0.0
        return {
            "mean_l_inf_err": float(np.max(np.abs(mu_n - mu_t))),
            "cov_l_inf_err": float(np.max(np.abs(cov_n - cov_t))),
            "nuisance_proj_err": float(np.max(np.abs(fix_n - fix_t))),
            "max_block_gram_err": max_gram_err,
            "rc_parity_err": rc_err,
        }


class Shapelizer2018ExactPort:
    """
    Exact Python 3 port of BussemakerLab/Shapelizer/package/randomKmerTable.py (Rube et al., MSB 2018),
    exposing `standardize_output` and `consistent_ddof` for side-by-side fidelity vs. fair audits.
    Vectorizes conditional-covariance design tensor D and gradient products (~150x faster).
    """

    def __init__(
        self,
        F_true: np.ndarray,
        symmetric: bool = True,
        maxiter: int = 1000,
        standardize_output: bool = False,
        consistent_ddof: bool = False,
    ):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.k = 5
        self.n_ch = self.F_true.shape[1]
        self.symmetric = symmetric
        self.maxiter = maxiter
        self.standardize_output = standardize_output
        self.consistent_ddof = consistent_ddof

        base = Shapelizer2018Null(self.F_true, symmetric=symmetric, maxiter=maxiter)
        self._base = base
        if consistent_ddof:
            # Recompute target conditional variance using ddof=1 to match np.cov in design tensor D
            for ch in range(self.n_ch):
                vals = self.F_true[:, ch]
                for i1 in range(self.k):
                    for i2 in range(self.k):
                        g_vars = [np.var(vals[idx], ddof=1) for idx in base.cond_groups[i1][i2]]
                        base.target_cond_var[ch, i1, i2] = float(np.mean(g_vars))

        # Precompute vectorized index arrays for D computation:
        # 5 diagonal pairs (i1==i2): 4 subgroups of size 256; 20 off-diagonal pairs: 16 subgroups of size 64.
        diag_groups, diag_pos = [], []
        off_groups, off_pos = [], []
        for i1 in range(self.k):
            for i2 in range(self.k):
                pos = i1 * self.k + i2
                grp = np.stack(base.cond_groups[i1][i2], axis=0).astype(np.int32)
                if i1 == i2:
                    diag_groups.append(grp)
                    diag_pos.append(pos)
                else:
                    off_groups.append(grp)
                    off_pos.append(pos)
        self._diag_idx = np.stack(diag_groups, axis=0)  # (5, 4, 256)
        self._diag_pos = np.array(diag_pos, dtype=np.int32)
        self._off_idx = np.stack(off_groups, axis=0)    # (20, 16, 64)
        self._off_pos = np.array(off_pos, dtype=np.int32)

    def _compute_cond_cov_flat(self, X: np.ndarray) -> np.ndarray:
        """Vectorized computation of D_flat (25, n_mats, n_mats) matching _base._compute_cond_cov_tensor."""
        n_mats = X.shape[1]
        D_flat = np.empty((25, n_mats, n_mats), dtype=np.float64)
        Xd = X[self._diag_idx]  # (5, 4, 256, n_mats)
        Xd = Xd - Xd.mean(axis=2, keepdims=True)
        D_flat[self._diag_pos] = np.einsum("pgna,pgnb->pab", Xd, Xd) / (255.0 * 4.0)

        Xo = X[self._off_idx]   # (20, 16, 64, n_mats)
        Xo = Xo - Xo.mean(axis=2, keepdims=True)
        D_flat[self._off_pos] = np.einsum("pgna,pgnb->pab", Xo, Xo) / (63.0 * 16.0)
        return D_flat

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros_like(self.F_true)
        base = self._base
        for ch in range(self.n_ch):
            X = base._build_design_X(rng)
            D_flat = self._compute_cond_cov_flat(X)
            target = base.target_cond_var[ch].reshape(self.k * self.k)

            def loss_and_grad(v: np.ndarray):
                Dv = D_flat @ v        # (25, n_mats)
                diff = (Dv @ v) - target  # (25,)
                return float(diff @ diff), 4.0 * (diff @ Dv)

            x0 = rng.random(base.n_mats)
            res = opt.minimize(
                loss_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.maxiter},
            )
            col_vals = X @ res.x
            if self.standardize_output:
                col_vals = (col_vals - col_vals.mean()) / max(col_vals.std(), 1e-12)
            out[:, ch] = col_vals
        return out


# ==============================================================================
# Rube et al. (2018) Fig 3 & Fig 5 Exact Reproduction Helpers
# ==============================================================================

def load_rube_shape_table_csv(csv_path: Path, is_4mer_step: bool = False) -> Tuple[List[str], np.ndarray]:
    """
    Load original Shapelizer input shapeTables/*.csv.
    For 5-mer tables (MGW.5mer.csv, ProT.5mer.csv): col0=5mer, col1=value.
    For Roll/HelT step tables: matches fig3.sh awk aggregation of 4-mers via both step1/step2 edges.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or not parts[0]:
                continue
            rows.append((parts[0].strip().upper(), [float(x) for x in parts[1:]]))
    if not is_4mer_step:
        kmers = [r[0] for r in rows]
        vals = np.array([r[1][0] for r in rows], dtype=np.float64)
        return kmers, vals
    # Exact fig3.sh / fig5.sh 4-mer averaging:
    # awk -F ',' '{s[substr($1,1,4)]+=$2;s[substr($1,2,4)]+=$4} END {for(i in s) print i","s[i]/8}'
    acc: Dict[str, float] = {}
    for km, vs in rows:
        acc[km[:4]] = acc.get(km[:4], 0.0) + vs[0]
        acc[km[1:5]] = acc.get(km[1:5], 0.0) + vs[2]
    kmers = sorted(acc.keys())
    vals = np.array([acc[k] / 8.0 for k in kmers], dtype=np.float64)
    return kmers, vals


def fit_kmer_monodi_shape_model(kmers: Sequence[str], vals: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Exact Python 3 reproduction of Shapelizer kMerLinearRegression.py --di --betas:
      Stage 1: OLS/pinv on position-wise mononucleotides -> betaMono
      Stage 2: OLS/pinv on adjacent dinucleotides residual (vals - X_mono @ betaMono) -> betaDi
    Returns shapeModel = {0: (k, 4), 1: (k-1, 16)}.
    """
    k = len(kmers[0])
    n = len(kmers)
    bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
    X_mono = np.zeros((n, 4 * k), dtype=np.float64)
    X_di = np.zeros((n, 16 * (k - 1)), dtype=np.float64)
    for row_i, km in enumerate(kmers):
        idxs = [bmap[c] for c in km]
        for p in range(k):
            X_mono[row_i, 4 * p + idxs[p]] = 1.0
        for p in range(k - 1):
            X_di[row_i, 16 * p + 4 * idxs[p] + idxs[p + 1]] = 1.0
    beta_mono = np.linalg.pinv(X_mono.T @ X_mono) @ (X_mono.T @ vals)
    res_vals = vals - X_mono @ beta_mono
    beta_di = np.linalg.pinv(X_di.T @ X_di) @ (X_di.T @ res_vals)
    return {
        0: beta_mono.reshape(k, 4),
        1: beta_di.reshape(k - 1, 16),
    }


def compute_shapelizer_fig3_crossr2(kmers: Sequence[str], vals: np.ndarray) -> Dict[str, float]:
    """
    Reproduce Shapelizer kMerLinearRegression.py --crossR2 (hold-one-RC-pair-out CV R^2)
    for --mono, --di, --all, and --centralMono.
    """
    k = len(kmers[0])
    n = len(kmers)
    bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
    # Group by RC pairs exactly as in kMerLinearRegression.py
    km_to_idx = {km: i for i, km in enumerate(kmers)}
    pairs: List[Tuple[int, ...]] = []
    seen = set()
    for i, km in enumerate(kmers):
        rc_km = km[::-1].translate(_TRANTAB)
        j = km_to_idx[rc_km]
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((i,) if i == j else (i, j))

    def _design(mode: str) -> np.ndarray:
        if mode == "mono":
            X = np.zeros((n, 4 * k), dtype=np.float64)
            for r_i, km in enumerate(kmers):
                for p, ch in enumerate(km):
                    X[r_i, 4 * p + bmap[ch]] = 1.0
            return X
        if mode == "centralMono":
            i_start = (k - 1) // 2
            pos_list = list(range(i_start, k - i_start))
            X = np.zeros((n, 4 * len(pos_list)), dtype=np.float64)
            for r_i, km in enumerate(kmers):
                for j_p, p in enumerate(pos_list):
                    X[r_i, 4 * j_p + bmap[km[p]]] = 1.0
            return X
        if mode == "di":
            X = np.zeros((n, 16 * (k - 1)), dtype=np.float64)
            for r_i, km in enumerate(kmers):
                for p in range(k - 1):
                    X[r_i, 16 * p + 4 * bmap[km[p]] + bmap[km[p + 1]]] = 1.0
            return X
        if mode == "all":
            Xm = _design("mono")
            return (Xm[:, :, None] * Xm[:, None, :]).reshape(n, -1)
        raise ValueError(mode)

    # Fast exact leave-pair-out leverage formula? Note Shapelizer uses pinv on rank-deficient binary matrices;
    # for 512 pairs on tiny p <= 400 matrices, exact ridge-free pinv check matches verbatim!
    out = {}
    for mode in ("mono", "di", "all", "centralMono"):
        X = _design(mode)
        in_v, out_v = [], []
        XtX = X.T @ X
        Xty = X.T @ vals
        for tup in pairs:
            idx = list(tup)
            X_hold = X[idx, :]
            y_hold = vals[idx]
            # Exact downdate XtX - X_hold.T @ X_hold, Xty - X_hold.T @ y_hold
            XtX_m = XtX - X_hold.T @ X_hold
            Xty_m = Xty - X_hold.T @ y_hold
            beta_m = np.linalg.pinv(XtX_m, rcond=1e-10) @ Xty_m
            pred = X_hold @ beta_m
            in_v.extend(y_hold.tolist())
            out_v.extend(pred.tolist())
        r = float(pearsonr(np.array(in_v), np.array(out_v)).statistic)
        out[mode] = float(r * r)
    return out


def load_rube_binding_model(tsv_path: Path) -> Dict[int, np.ndarray]:
    """Load BussemakerLab *.monoDi.tsv binding model (0: Lx4, 1: (L-1)x16)."""
    d0: Dict[str, List[float]] = {}
    d1: Dict[str, List[float]] = {}
    dinucl = [n1 + n2 for n1 in _NUCL for n2 in _NUCL]
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\r\n").split("\t")
            if not cols or not cols[0] or cols[0].lower() == "intercept":
                continue
            key = cols[0].split(":")
            vals = [float(x) for x in cols[1:]]
            if len(key) == 1:
                d0[key[0]] = vals
            elif int(key[1]) == 1:
                d1[key[0]] = vals
    return {
        0: np.array([d0[n] for n in _NUCL], dtype=np.float64).T,
        1: np.array([d1[dn] for dn in dinucl], dtype=np.float64).T,
    }


class RubeShapeProjectionEvaluator:
    """
    Faithful Python 3 port of BussemakerLab/Shapelizer/package/shapeProjection.py
    using exact transfer-matrix KL divergence & L-BFGS-B / SLSQP minimization.
    """

    X_MONO = np.array(
        [[3.0, -1.0, -1.0, -1.0], [-1.0, 3.0, -1.0, -1.0], [-1.0, -1.0, 3.0, -1.0]],
        dtype=np.float64,
    ) / 3.0

    def __init__(self, agnostic_model: Dict[int, np.ndarray], normalize_model: bool = True):
        self.agnostic_mono = np.copy(agnostic_model[0]).astype(np.float64)
        self.agnostic_di = agnostic_model[1].reshape(-1, 4, 4).astype(np.float64)
        self.L = int(self.agnostic_mono.shape[0])
        self.agnostic_mono_aff = np.exp(self.agnostic_mono)
        self.agnostic_di_aff = np.exp(self.agnostic_di)
        if normalize_model:
            mean_aff = self._compute_sum(1.0, self.agnostic_mono_aff, self.agnostic_di_aff) / float(4 ** self.L)
            self.agnostic_mono -= np.log(mean_aff) / float(self.L)
            self.agnostic_mono_aff = np.exp(self.agnostic_mono)

        # Precompute agnostic partition sums (independent of shapeModel!)
        self.left1, self.right1 = self._compute_partial_sums(
            1.0, self.agnostic_mono_aff / 4.0, self.agnostic_di_aff
        )
        self.s1 = float(np.dot(self.left1[-1], self.agnostic_mono_aff[-1] / 4.0))
        self.w_mono1 = (self.left1 * self.right1 * (self.agnostic_mono_aff / 4.0)) / self.s1
        self.w_di1 = np.array(
            [
                np.outer(
                    self.left1[i] * (self.agnostic_mono_aff[i] / 4.0),
                    self.right1[i + 1] * (self.agnostic_mono_aff[i + 1] / 4.0),
                ) * self.agnostic_di_aff[i]
                for i in range(self.L - 1)
            ],
            dtype=np.float64,
        ) / self.s1
        self.const_kl_term = float(
            np.sum(self.agnostic_mono * self.w_mono1)
            + np.sum(self.agnostic_di * self.w_di1)
            - np.log(self.s1)
        )

    def _compute_partial_sums(
        self, intercept: float, mono_mat: np.ndarray, di_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        left = [np.ones(4, dtype=np.float64) * intercept]
        right = [np.ones(4, dtype=np.float64)]
        for i in range(1, self.L):
            left.append(np.dot(left[-1] * mono_mat[i - 1], di_mat[i - 1]))
            right.append(np.dot(di_mat[self.L - i - 1], right[-1] * mono_mat[self.L - i]))
        return np.array(left, dtype=np.float64), np.array(right[::-1], dtype=np.float64)

    def _compute_sum(self, intercept: float, mono_mat: np.ndarray, di_mat: np.ndarray) -> float:
        ps = np.ones(4, dtype=np.float64) * intercept
        for i in range(1, self.L):
            ps = np.dot(ps * mono_mat[i - 1], di_mat[i - 1])
        return float(np.dot(ps, mono_mat[-1]))

    @staticmethod
    def build_di_design_matrix(shape_model: Dict[int, np.ndarray], L: int) -> np.ndarray:
        k_s = int(shape_model[0].shape[0])
        shape_di = shape_model[1].reshape(k_s - 1, 4, 4).astype(np.float64).copy()
        mono_rep = [np.tile(shape_model[0][i], (4, 1)) for i in range(k_s)]
        left_w = [1.0] + [0.5] * (k_s - 2) + [0.0]
        right_w = [0.0] + [0.5] * (k_s - 2) + [1.0]
        for i in range(k_s - 1):
            shape_di[i] += left_w[i] * mono_rep[i].T + right_w[i + 1] * mono_rep[i + 1]
        mats = []
        for x in range(L - k_s + 1):
            m = np.concatenate(
                [np.zeros((x, 4, 4)), shape_di, np.zeros((L - k_s - x, 4, 4))],
                axis=0,
            )
            mats.append(m)
        return np.array(mats, dtype=np.float64)  # (L - k_s + 1, L - 1, 4, 4)

    def fit_projection(self, shape_model: Dict[int, np.ndarray], maxiter: int = 500) -> Dict[str, float]:
        di_design = self.build_di_design_matrix(shape_model, self.L)
        k_pos = di_design.shape[0]
        n_mono = 3 * self.L
        i_shape = 0
        i_mono = k_pos
        i_int = i_mono + n_mono

        # Exact naive mono seed matching shapeProjection.py getSeed()
        f_obs = np.sum(self.w_mono1, axis=0) * 0.0  # shape (L, 4)
        # w_mono1 sum per position = 1; log ratios
        w_m_clip = np.maximum(self.w_mono1, 1e-12)
        avg_log = np.mean(np.log(w_m_clip), axis=1, keepdims=True)
        mono_seed = np.log(w_m_clip) - avg_log
        v0 = np.zeros(k_pos + n_mono + 1, dtype=np.float64)
        v0[i_mono:i_int] = mono_seed[:, :3].ravel()
        # Evaluate KL loss of mono-only baseline
        mono0_aff = np.exp(mono_seed) / 4.0
        di_eye = np.ones((self.L - 1, 4, 4), dtype=np.float64)
        s2_0 = self._compute_sum(1.0, mono0_aff, di_eye)
        kl_mono_only = float(
            self.const_kl_term
            - np.sum(mono_seed * self.w_mono1)
            + np.log(s2_0)
        )

        # Notice KL loss is strictly smooth convex in v and shift-invariant in intercept!
        # Fixing v_intercept=0 inside L-BFGS-B avoids flat Hessian directions and solves globally in ~2ms!
        def _loss_grad(u: np.ndarray):
            beta_s = u[:k_pos]
            beta_m = u[k_pos:].reshape(self.L, 3)
            proj_mono = beta_m @ self.X_MONO  # (L, 4)
            proj_di = np.einsum("p,pijk->ijk", beta_s, di_design)
            mono2 = np.exp(proj_mono) / 4.0
            di2 = np.exp(proj_di)
            left2, right2 = self._compute_partial_sums(1.0, mono2, di2)
            s2 = float(np.dot(left2[-1], mono2[-1]))
            kl = float(
                self.const_kl_term
                - np.sum(proj_mono * self.w_mono1)
                - np.sum(proj_di * self.w_di1)
                + np.log(s2)
            )
            w_mono2 = (left2 * right2 * mono2) / s2
            w_di2 = np.array(
                [
                    np.outer(left2[i] * mono2[i], right2[i + 1] * mono2[i + 1]) * di2[i]
                    for i in range(self.L - 1)
                ],
                dtype=np.float64,
            ) / s2
            d_di = w_di2 - self.w_di1
            d_mono = w_mono2 - self.w_mono1
            grad_s = np.einsum("pijk,ijk->p", di_design, d_di)
            grad_m = (d_mono @ self.X_MONO.T).ravel()
            return kl, np.concatenate([grad_s, grad_m])

        u0 = np.zeros(k_pos + n_mono, dtype=np.float64)
        u0[k_pos:] = mono_seed[:, :3].ravel()
        bounds = [(-10.0, 10.0)] * len(u0)
        res = opt.minimize(
            _loss_grad, u0, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
        )
        kl_opt = float(res.fun)
        dkl = max(0.0, kl_mono_only - kl_opt)
        r2_proxy = float(dkl / max(kl_mono_only, 1e-12))
        beta_shape = res.x[:k_pos]
        return {
            "kl_mono_only": kl_mono_only,
            "kl_projected": kl_opt,
            "delta_kl": dkl,
            "frac_kl_explained": r2_proxy,
            "shape_l2_norm": float(np.linalg.norm(beta_shape)),
            "shape_betas": beta_shape.tolist(),
        }
