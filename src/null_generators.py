"""
Self-Contained Matched-Complexity Surrogate Null Generators for Pentamer DNA Tables.

Implements:
  - `ShiftGCExactCRTNull`: Zero-hyperparameter conditional randomization in L^2(p_emp)
    preserving exact `W_fix` (`WS_only`, `WS_G1`, `WS_G2`), cross-channel covariance,
    De Bruijn shift-coboundary / Hamming-order Grams, and RC-even symmetry.
  - `Sparse4merSurrogateNull`: Multi-sparsity 4-mer pocket surrogate family (`K in {4..48}`)
    matched exactly on `W_fix` and 7x7 residual covariance in L^2(p_emp).
  - `NaiveRCPermutationNull`: Synchronous RC-pair pentamer permutation baseline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

CHANNELS_7_EVEN = ("MGW", "ProT", "EP", "Stretch", "Roll_mean", "HelT_mean", "Slide_mean")


def build_pentamer_rc_map() -> Tuple[np.ndarray, np.ndarray]:
    """Return `(canon_512_idx, rc_map_1024)` for 5-mers (A=0, C=1, G=2, T=3)."""
    idx = np.arange(1024, dtype=np.int32)
    rc = np.zeros(1024, dtype=np.int32)
    tmp = idx.copy()
    for _ in range(5):
        rc = (rc << 2) | (3 - (tmp & 3))
        tmp >>= 2
    canon = np.where(idx < rc)[0]
    assert len(canon) == 512
    return canon, rc


CANON_5, RC_5 = build_pentamer_rc_map()


def haar_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exact Haar-uniform orthogonal matrix in O(dim)."""
    if dim <= 1:
        return np.eye(dim, dtype=np.float64)
    g = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(g)
    s = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * s[None, :]


def build_fourier_basis() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct orthonormal Walsh-Hadamard tensor basis U = h^{otimes 5} on {A,C,G,T}^5."""
    h = np.array(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]],
        dtype=np.float64,
    ) / 2.0
    signs = np.rint(np.sum(h * h[::-1], axis=0)).astype(np.int32)
    U = h.copy()
    for _ in range(4):
        U = np.kron(U, h)
    digits = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T
    return U, digits, signs


def _build_exact_algebraic_decomposition() -> Dict[str, object]:
    """Precompute exact RC-even/RC-odd Hamming-order and De Bruijn shift-coboundary subspaces."""
    U, digits, signs = build_fourier_basis()
    pure_ws = np.all((digits == 0) | (digits == 2), axis=1)
    orders = np.sum(digits > 0, axis=1)
    weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
    sup5 = np.sum((digits > 0) * weights[None, :], axis=1)
    adj_supports = {3, 6, 12, 24}

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
    G4_odd = np.column_stack(odd4)
    G4_even = np.column_stack(even4)

    d5 = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T
    p_left = np.ravel_multi_index(d5[:, :4].T, (4,) * 4)
    p_right = np.ravel_multi_index(d5[:, 1:].T, (4,) * 4)

    u_ec, s_ec, _ = np.linalg.svd(G4_odd[p_left, :] - G4_odd[p_right, :], full_matrices=False)
    Q_cob_even = u_ec[:, s_ec > 1e-10]
    u_oc, s_oc, _ = np.linalg.svd(G4_even[p_left, :] - G4_even[p_right, :], full_matrices=False)
    Q_cob_odd = u_oc[:, s_oc > 1e-10]

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
                        (even_adj_r2 if int(sup5[i]) in adj_supports else even_gap_r2).append(v_e)
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
                    (even_adj_r2 if int(sup5[i]) in adj_supports else even_gap_r2).append(v_e)
            odd_by_r[r].append(v_o)

    def _split_coh_tele(W: np.ndarray, Q_cob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_w, s_w, _ = np.linalg.svd(W.T @ Q_cob, full_matrices=True)
        W_rot = W @ u_w
        n_tele = int(np.sum(s_w > 1.0 - 1e-9))
        W_tele = W_rot[:, :n_tele] if n_tele > 0 else np.zeros((1024, 0), dtype=np.float64)
        W_coh = W_rot[:, n_tele:] if n_tele < W_rot.shape[1] else np.zeros((1024, 0), dtype=np.float64)
        return W_coh, W_tele

    even_subblocks = {
        key: _split_coh_tele(np.column_stack(vecs), Q_cob_even)
        for key, vecs in [
            ("r1", even_nonws_by_r[1]),
            ("r2_adj", even_adj_r2),
            ("r2_gap", even_gap_r2),
            ("r3", even_nonws_by_r[3]),
            ("r4", even_nonws_by_r[4]),
            ("r5", even_nonws_by_r[5]),
        ]
    }
    odd_subblocks = {r: _split_coh_tele(np.column_stack(odd_by_r[r]), Q_cob_odd) for r in range(1, 6)}

    return {
        "W_ws": np.column_stack(even_ws),
        "W_r1_nonws": np.column_stack(even_nonws_by_r[1]),
        "W_r2_adj": np.column_stack(even_adj_r2),
        "even_subblocks": even_subblocks,
        "odd_subblocks": odd_subblocks,
    }


_ALGEBRAIC_DECOMP = _build_exact_algebraic_decomposition()


class NaiveRCPermutationNull:
    """Synchronous RC-pair permutation across the 512 canonical pentamers."""

    def __init__(self, F_true: np.ndarray):
        self.F_true = np.asarray(F_true, dtype=np.float64)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        perm = rng.permutation(CANON_5)
        out = np.empty_like(self.F_true)
        out[CANON_5] = self.F_true[perm]
        out[RC_5[CANON_5]] = self.F_true[RC_5[perm]]
        return out


class ShiftGCExactCRTNull:
    """Zero-hyperparameter Fixed-X Conditional Randomization Test null in L^2(p_emp)."""

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
            p_emp = np.full(1024, 1.0 / 1024.0, dtype=np.float64)
        else:
            n_tot = float(n_total_obs) if n_total_obs is not None else (raw_sum if raw_sum > 2.0 else 10000.0)
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
                if Phib.shape[1] >= 1:
                    rot_blocks.append((tag, Phib))

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
        self.rot_list: List[Tuple[str, int, np.ndarray, np.ndarray]] = []
        for tag, Phi_b in rot_blocks:
            C_b = (Phi_b.T * p_emp[None, :]) @ self.F_true
            if np.max(np.abs(C_b)) > 1e-12:
                self.rot_list.append((tag, Phi_b.shape[1], Phi_b, C_b))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.base_proj.copy()
        for _, dim, Phi_b, C_b in self.rot_list:
            q = np.array([[1.0 if rng.random() < 0.5 else -1.0]], dtype=np.float64) if dim == 1 else haar_orthogonal(dim, rng)
            ans += Phi_b @ (q @ C_b)
        return ans

    def audit_invariants(self, F_null: np.ndarray) -> Dict[str, float]:
        p = self.p_emp
        mu_t, mu_n = p @ self.F_true, p @ F_null
        Ft_c, Fn_c = self.F_true - mu_t[None, :], F_null - mu_n[None, :]
        cov_t, cov_n = (Ft_c.T * p[None, :]) @ Ft_c, (Fn_c.T * p[None, :]) @ Fn_c
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


class Sparse4merSurrogateNull:
    """Covariance- & WS_G2-matched sparse 4-mer pocket surrogate table generator."""

    def __init__(self, F_true: np.ndarray, p_raw: np.ndarray, n_total_obs: float, n_pockets_per_ch: int = 12):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.n_pockets = int(n_pockets_per_ch)
        crt = ShiftGCExactCRTNull(self.F_true, p_raw, conditioning="WS_G2", n_total_obs=n_total_obs)
        self.p_emp = crt.p_emp
        self.Phi_fix = crt.Phi_fix
        self.base_proj = crt.base_proj

        R_true = self.F_true - self.base_proj
        Sigma_true = (R_true.T * self.p_emp[None, :]) @ R_true
        evals, evecs = np.linalg.eigh(Sigma_true)
        evals = np.maximum(evals, 1e-10)
        self.L_color = (evecs * np.sqrt(evals)[None, :]) @ evecs.T

        digits5 = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T
        self.l4 = np.ravel_multi_index(digits5[:, :4].T, (4,) * 4)
        self.r4 = np.ravel_multi_index(digits5[:, 1:].T, (4,) * 4)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        R_raw = np.empty_like(self.F_true)
        wP = self.Phi_fix.T * self.p_emp[None, :]
        for ch in range(self.F_true.shape[1]):
            w4 = np.zeros(256, dtype=np.float64)
            pockets = rng.choice(256, size=self.n_pockets, replace=False)
            w4[pockets] = rng.normal(size=self.n_pockets)
            raw_v = w4[self.l4] + w4[self.r4]
            s_even = 0.5 * (raw_v + raw_v[RC_5])
            R_raw[:, ch] = s_even - self.Phi_fix @ (wP @ s_even)

        cov_raw = (R_raw.T * self.p_emp[None, :]) @ R_raw
        ev, ec = np.linalg.eigh(cov_raw)
        inv_sq = ec @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-10))) @ ec.T
        R_matched = R_raw @ inv_sq @ self.L_color
        return self.base_proj + R_matched
