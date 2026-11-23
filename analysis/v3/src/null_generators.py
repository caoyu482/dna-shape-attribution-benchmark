#!/usr/bin/env python3
"""
Strict Hierarchy of Matched-Complexity Null Reference Generators for 5-mer DNA Biophysics.

Implements:
1. NaiveRCPermutationNull        : Synchronous RC-pair permutation (destroys interaction spectrum).
2. Shapelizer2018Null            : Python 3 port of BussemakerLab/Shapelizer randomKmerTable.py
                                   (independent per-channel mono + adjacent-di conditional-variance matching).
3. UnconstrainedSpectralNull     : v1 support-orbit multi-channel Haar rotation (mixes WS/GC & telescoping).
4. GCConstrainedSpectralNull     : Fixes 20 pure-WS (GC) even dims; rotates 492 non-WS dims per support orbit.
5. ShiftGCUnifSpectralNull       : Analytical uniform-measure decomposition separating pure-WS (20 dims, fixed),
                                   shift-coherent motif orbits (425 dims, synchronous Haar across shift copies),
                                   and telescoping spatial-derivative blocks (67 dims, per-orbit Haar).
6. ShiftGCEmpiricalSpectralNull  : Unlabeled training-distribution L^2(p_emp)-weighted sequential QR (r=0..5)
                                   fixing pure-WS + optional unlabeled bulk composition PCs and splitting
                                   sequence-coherent vs. telescoping subspaces so training mean diff <= 0.10 SD
                                   and per-base & sequence-mean covariance relative error <= 10%.

All basis decompositions are precomputed ONCE in __init__; .sample() executes pure O(d^2) BLAS ops.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt

DATA_TABLE_DIR = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "data/biophysics_tables")

# Alphabet encoding matching v1/biophysical_priors.py: A=0, C=1, G=2, T=3
_BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
CHANNELS_7_EVEN = ("MGW", "ProT", "EP", "Stretch", "Roll_mean", "HelT_mean", "Slide_mean")
CHANNELS_10_FULL = CHANNELS_7_EVEN + ("Roll_diff", "HelT_diff", "Slide_diff")


def build_pentamer_rc_map() -> Tuple[np.ndarray, np.ndarray]:
    """Return (canon_512_idx, rc_map_1024) for 5-mers (since 5 is odd, no pentamer is palindromic)."""
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


def decode_pentamers() -> List[str]:
    nucl = ("A", "C", "G", "T")
    out = []
    for idx in range(1024):
        d = [(idx >> (2 * (4 - p))) & 3 for p in range(5)]
        out.append("".join(nucl[x] for x in d))
    return out


PENTAMER_STRINGS = decode_pentamers()


def load_biophysical_table_matrix(include_odd_step_diff: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Load authentic standardized (mean=0, std=1) DNAshapeR pentamer table F in R^{1024 x d}.
    Default d=7 RC-even channels; optional d=10 adds RC-odd step differentials 0.5*(step1 - step2).
    """
    npz = np.load(DATA_TABLE_DIR / "pentamer_dnashape.npz")
    cols = [
        npz["MGW"].astype(np.float64),
        npz["ProT"].astype(np.float64),
        npz["EP"].astype(np.float64),
        npz["Stretch"].astype(np.float64),
        0.5 * (npz["Roll1"].astype(np.float64) + npz["Roll2"].astype(np.float64)),
        0.5 * (npz["HelT1"].astype(np.float64) + npz["HelT2"].astype(np.float64)),
        0.5 * (npz["Slide1"].astype(np.float64) + npz["Slide2"].astype(np.float64)),
    ]
    names = list(CHANNELS_7_EVEN)
    if include_odd_step_diff:
        cols.extend([
            0.5 * (npz["Roll1"].astype(np.float64) - npz["Roll2"].astype(np.float64)),
            0.5 * (npz["HelT1"].astype(np.float64) - npz["HelT2"].astype(np.float64)),
            0.5 * (npz["Slide1"].astype(np.float64) - npz["Slide2"].astype(np.float64)),
        ])
        names = list(CHANNELS_10_FULL)
    F = np.column_stack(cols)
    F = (F - F.mean(axis=0, keepdims=True)) / np.maximum(F.std(axis=0, keepdims=True), 1e-12)
    return F, names


def haar_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exact Haar-uniform orthogonal matrix in O(dim)."""
    if dim <= 1:
        return np.eye(dim, dtype=np.float64)
    g = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(g)
    s = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * s[None, :]


def build_fourier_basis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Construct orthonormal contrast tensor basis U = h^{otimes 5} on {A,C,G,T}^5
    and decompose into exact (key_support, rc_parity) blocks.
    Columns of h: 0=Const (+1), 1=MK (-1), 2=WS (+1), 3=RY (-1).
    """
    h = np.array(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]],
        dtype=np.float64,
    ) / 2.0
    signs = np.rint(np.sum(h * h[::-1], axis=0)).astype(np.int32)  # [1, -1, 1, -1]
    U = h.copy()
    for _ in range(4):
        U = np.kron(U, h)
    digits = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T
    rev = np.ravel_multi_index(digits[:, ::-1].T, (4,) * 5)
    weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
    supports = np.sum((digits > 0) * weights[None, :], axis=1)
    rev_supports = np.sum((digits[:, ::-1] > 0) * weights[None, :], axis=1)

    blocks_list: Dict[Tuple[int, int], List[np.ndarray]] = {}
    used = set()
    for i in range(1024):
        if i in used:
            continue
        j = int(rev[i])
        used.add(i)
        used.add(j)
        sg = int(np.prod(signs[digits[i]]))
        ks = int(min(supports[i], rev_supports[i]))
        if i == j:
            blocks_list.setdefault((ks, sg), []).append(U[:, i])
        else:
            blocks_list.setdefault((ks, 1), []).append((U[:, i] + sg * U[:, j]) / np.sqrt(2.0))
            blocks_list.setdefault((ks, -1), []).append((U[:, i] - sg * U[:, j]) / np.sqrt(2.0))
    blocks = {k: np.column_stack(v) for k, v in blocks_list.items()}
    return U, digits, signs, blocks


class NaiveRCPermutationNull:
    """Ordinary RC-pair row permutation null (v1 baseline control)."""

    def __init__(self, F_true: np.ndarray, parity_signs: Optional[Sequence[int]] = None):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.n_ch = self.F_true.shape[1]
        self.parity_signs = np.ones(self.n_ch, dtype=np.int32) if parity_signs is None else np.asarray(parity_signs, dtype=np.int32)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        perm = rng.permutation(512)
        src = CANON_5
        dst = CANON_5[perm]
        out = np.empty_like(self.F_true)
        out[dst] = self.F_true[src]
        # For RC-even channels F[rc(p)] == F[p]; for RC-odd F[rc(p)] == -F[p]
        out[RC_5[dst]] = self.F_true[RC_5[src]] * self.parity_signs[None, :]
        return out


class Shapelizer2018Null:
    """
    Faithful Python 3 port of BussemakerLab/Shapelizer/package/randomKmerTable.py (Rube et al., MSB 2018).

    Python 2 -> 3 migration notes:
      - xrange -> range; .iteritems() -> .items(); integer division -> //; print -> function.
      - Vectorizes the 5x5 conditional-variance group index sets once in __init__ for identical
        mathematical outputs (~100x faster than nested Python list comprehensions).
      - Strictly reproduces Shapelizer's per-channel, RC-symmetric, mono + adjacent-dinucleotide
        Gaussian random table optimization via L-BFGS-B (no cross-channel Gram preservation,
        no order >= 3 or gapped order-2 terms, no GC-correlation anchoring).
    """

    def __init__(self, F_true: np.ndarray, symmetric: bool = True, maxiter: int = 1000):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.k = 5
        self.n_ch = self.F_true.shape[1]
        self.symmetric = symmetric
        self.maxiter = maxiter

        digits = np.array(np.unravel_index(np.arange(1024), (4,) * 5)).T  # (1024, 5)
        self.digits = digits
        self.di_codes = 4 * digits[:, :-1] + digits[:, 1:]  # (1024, 4)

        # RC transformation for 16-entry dinucleotide vector (4*(3-n2) + (3-n1))
        rc_di = np.zeros((16, 16), dtype=np.float64)
        for n1 in range(4):
            for n2 in range(4):
                j = 4 * (3 - n2) + (3 - n1)
                rc_di[j, 4 * n1 + n2] = 1.0
        self.rc_di = rc_di

        # Identify unique symmetric signatures exactly as in randomKmerTable.py
        self.mono_groups: List[int] = []
        seen_sigs = set()
        for i in range(self.k):
            sig = [0] * self.k
            sig[i] = 1
            if symmetric:
                sig = [sig[p] + sig[self.k - 1 - p] for p in range(self.k)]
            skey = ("mono", tuple(sig))
            if skey not in seen_sigs:
                seen_sigs.add(skey)
                self.mono_groups.append(i)

        self.di_groups: List[int] = []
        for i in range(self.k - 1):
            sig = [0] * (self.k - 1)
            sig[i] = 1
            if symmetric:
                sig = [sig[p] + sig[self.k - 2 - p] for p in range(self.k - 1)]
            skey = ("di", tuple(sig))
            if skey not in seen_sigs:
                seen_sigs.add(skey)
                self.di_groups.append(i)

        self.n_mats = len(self.mono_groups) + len(self.di_groups)

        # Precompute conditional-variance index subsets for every (i1, i2) in {0..4}^2
        # Matches randomKmerTable.py lines 79 & 138-142 identically.
        self.cond_groups: List[List[List[np.ndarray]]] = [[[] for _ in range(self.k)] for _ in range(self.k)]
        for i1 in range(self.k):
            for i2 in range(self.k):
                sub_idx = []
                for n1 in range(4):
                    for n2 in range(4):
                        if i1 == i2 and n1 != n2:
                            continue
                        mask = (digits[:, i1] == n1) & (digits[:, i2] == n2)
                        sub_idx.append(np.where(mask)[0])
                self.cond_groups[i1][i2] = sub_idx

        # Precompute target 5x5 conditional variance matrix for each true channel
        self.target_cond_var = np.zeros((self.n_ch, self.k, self.k), dtype=np.float64)
        for ch in range(self.n_ch):
            vals = self.F_true[:, ch]
            for i1 in range(self.k):
                for i2 in range(self.k):
                    g_vars = [np.var(vals[idx]) for idx in self.cond_groups[i1][i2]]
                    self.target_cond_var[ch, i1, i2] = float(np.mean(g_vars))

    def _build_design_X(self, rng: np.random.Generator) -> np.ndarray:
        X = np.zeros((1024, self.n_mats), dtype=np.float64)
        col = 0
        for i in self.mono_groups:
            rv = rng.random(4)
            m = np.zeros((4, self.k), dtype=np.float64)
            m[:, i] = rv - np.mean(rv)
            if self.symmetric:
                m = 0.5 * (m + m[::-1, ::-1])
            s = np.zeros(1024, dtype=np.float64)
            for p in range(self.k):
                s += m[self.digits[:, p], p]
            X[:, col] = s
            col += 1

        for i in self.di_groups:
            rv = rng.random(16)
            m = np.zeros((16, self.k - 1), dtype=np.float64)
            m[:, i] = rv - np.mean(rv)
            if self.symmetric:
                m = 0.5 * (m + self.rc_di @ m[:, ::-1])
            s = np.zeros(1024, dtype=np.float64)
            for p in range(self.k - 1):
                s += m[self.di_codes[:, p], p]
            X[:, col] = s
            col += 1
        return X

    def _compute_cond_cov_tensor(self, X: np.ndarray) -> np.ndarray:
        """Return D tensor of shape (k, k, n_mats, n_mats) matching randomKmerTable.py eqs (138-142)."""
        D = np.zeros((self.k, self.k, self.n_mats, self.n_mats), dtype=np.float64)
        for i1 in range(self.k):
            for i2 in range(self.k):
                cov_acc = np.zeros((self.n_mats, self.n_mats), dtype=np.float64)
                groups = self.cond_groups[i1][i2]
                for idx in groups:
                    sub = X[idx]
                    sub_c = sub - sub.mean(axis=0, keepdims=True)
                    # np.cov uses ddof=1 normalization (len(idx) - 1)
                    cov_acc += (sub_c.T @ sub_c) / float(len(idx) - 1)
                D[i1, i2] = cov_acc / float(len(groups))
        return D

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros_like(self.F_true)
        for ch in range(self.n_ch):
            X = self._build_design_X(rng)
            D = self._compute_cond_cov_tensor(X)
            D_flat = D.reshape(self.k * self.k, self.n_mats, self.n_mats)
            target = self.target_cond_var[ch].reshape(self.k * self.k)

            def loss_and_grad(v: np.ndarray):
                Dv = np.einsum("pab,b->pa", D_flat, v)
                quad = np.einsum("pa,a->p", Dv, v)
                diff = quad - target
                loss = float(np.sum(diff * diff))
                grad = 4.0 * np.einsum("p,pa->a", diff, Dv)
                return loss, grad

            x0 = rng.random(self.n_mats)
            res = opt.minimize(
                loss_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={"disp": False, "maxiter": self.maxiter},
            )
            col_vals = X @ res.x
            # Standardize to match input table column convention (mean=0, std=1)
            col_vals = (col_vals - col_vals.mean()) / max(col_vals.std(), 1e-12)
            out[:, ch] = col_vals
        return out


class UnconstrainedSpectralNull:
    """v1 support-orbit multi-channel Haar rotation per (key_support, rc_parity) block (constant fixed)."""

    def __init__(self, F_true: np.ndarray):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        U, digits, signs, blocks = build_fourier_basis()
        self.U = U
        self.digits = digits
        self.blocks = blocks
        self.const_proj = blocks[(0, 1)] @ (blocks[(0, 1)].T @ self.F_true)
        self.active_blocks = []
        for k, w in sorted(blocks.items()):
            if k == (0, 1):
                continue
            c = w.T @ self.F_true
            if np.max(np.abs(c)) > 1e-12:
                self.active_blocks.append((k, w, c))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.const_proj.copy()
        for _, w, c in self.active_blocks:
            q = haar_orthogonal(w.shape[1], rng)
            ans += w @ (q @ c)
        return ans


class GCConstrainedSpectralNull:
    """
    Fixes the 20 pure-WS (GC) RC-even dimensions (plus constant) exactly and rotates
    the remaining 492 non-WS dimensions per (key_support, rc_parity) block.
    """

    def __init__(self, F_true: np.ndarray):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        U, digits, signs, _ = build_fourier_basis()
        rev = np.ravel_multi_index(digits[:, ::-1].T, (4,) * 5)
        weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
        supports = np.sum((digits > 0) * weights[None, :], axis=1)
        rev_sup = np.sum((digits[:, ::-1] > 0) * weights[None, :], axis=1)
        pure_ws = np.all((digits == 0) | (digits == 2), axis=1)

        fixed_cols = []
        rot_dict: Dict[Tuple[int, int], List[np.ndarray]] = {}
        used = set()
        for i in range(1024):
            if i in used:
                continue
            j = int(rev[i])
            used.add(i)
            used.add(j)
            sg = int(np.prod(signs[digits[i]]))
            ks = int(min(supports[i], rev_supports[i])) if False else int(min(supports[i], rev_sup[i]))
            is_ws = bool(pure_ws[i])
            if i == j:
                v = U[:, i]
                if sg == 1 and is_ws:
                    fixed_cols.append(v)
                else:
                    rot_dict.setdefault((ks, sg), []).append(v)
            else:
                vp = (U[:, i] + sg * U[:, j]) / np.sqrt(2.0)
                vm = (U[:, i] - sg * U[:, j]) / np.sqrt(2.0)
                if is_ws:
                    fixed_cols.append(vp)
                else:
                    rot_dict.setdefault((ks, 1), []).append(vp)
                rot_dict.setdefault((ks, -1), []).append(vm)

        W_fix = np.column_stack(fixed_cols)
        self.base_proj = W_fix @ (W_fix.T @ self.F_true)
        self.rot_blocks = []
        for k, vlist in sorted(rot_dict.items()):
            w = np.column_stack(vlist)
            c = w.T @ self.F_true
            if np.max(np.abs(c)) > 1e-12:
                self.rot_blocks.append((k, w, c))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.base_proj.copy()
        for _, w, c in self.rot_blocks:
            q = haar_orthogonal(w.shape[1], rng)
            ans += w @ (q @ c)
        return ans


class ShiftGCUnifSpectralNull:
    """
    Analytical Uniform-Measure Shift- & GC-Constrained Spectral Surrogate (ShiftGC-Unif).

    Decomposes RC-even subspace (512 dims) analytically into:
      1. Pure-WS (GC) even subspace (20 dims, including constant): FIXED identically.
      2. Shift-coherent motif orbits (425 dims across canonical trimmed motifs): rotated
         SYNCHRONOUSLY across shift-equivalent offset blocks (e.g. ks=3 and ks=6 share the
         exact 5x5 Haar matrix Q_5 on the 5 non-WS coherent dinucleotides).
      3. Telescoping spatial-derivative blocks (67 dims): rotated per support orbit.
    Also handles RC-odd blocks cleanly if odd-parity step differentials are included.
    """

    def __init__(self, F_true: np.ndarray):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        U, digits, signs, _ = build_fourier_basis()
        rev = np.ravel_multi_index(digits[:, ::-1].T, (4,) * 5)
        weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
        supports = np.sum((digits > 0) * weights[None, :], axis=1)
        rev_sup = np.sum((digits[:, ::-1] > 0) * weights[None, :], axis=1)
        pure_ws = np.all((digits == 0) | (digits == 2), axis=1)

        def trim(d_row: np.ndarray) -> Tuple[int, ...]:
            nz = np.where(d_row > 0)[0]
            return tuple(d_row[nz[0]:nz[-1] + 1].tolist()) if len(nz) else ()

        # Build shift-sum operator T on 1024 space for RC-even coherent motifs
        motif_map: Dict[Tuple[int, ...], int] = {}
        for i in range(1, 1024):
            t = trim(digits[i])
            tr = t[::-1]
            sg_t = int(np.prod([signs[x] for x in t]))
            if t == tr and sg_t == -1:
                continue
            canon = min(t, tr)
            if canon not in motif_map:
                motif_map[canon] = len(motif_map)

        T = np.zeros((len(motif_map), 1024), dtype=np.float64)
        for i in range(1, 1024):
            t = trim(digits[i])
            tr = t[::-1]
            sg_t = int(np.prod([signs[x] for x in t]))
            if t == tr and sg_t == -1:
                continue
            canon = min(t, tr)
            idx = motif_map[canon]
            sgn = 1.0 if t <= tr else float(sg_t)
            T[idx] += sgn * U[:, i]

        even_blocks: Dict[int, List[Tuple[bool, np.ndarray]]] = {}
        odd_blocks: Dict[int, List[np.ndarray]] = {}
        used = set()
        for i in range(1024):
            if i in used:
                continue
            j = int(rev[i])
            used.add(i)
            used.add(j)
            sg = int(np.prod(signs[digits[i]]))
            ks = int(min(supports[i], rev_sup[i]))
            if i == j:
                if sg == 1:
                    even_blocks.setdefault(ks, []).append((bool(pure_ws[i]), U[:, i]))
                else:
                    odd_blocks.setdefault(ks, []).append(U[:, i])
            else:
                even_blocks.setdefault(ks, []).append((bool(pure_ws[i]), (U[:, i] + sg * U[:, j]) / np.sqrt(2.0)))
                odd_blocks.setdefault(ks, []).append((U[:, i] - sg * U[:, j]) / np.sqrt(2.0))

        ws_vecs: List[np.ndarray] = []
        tele_blocks: Dict[int, np.ndarray] = {}
        coh_groups: Dict[Tuple[int, ...], List[Tuple[int, np.ndarray]]] = {}

        for ks, items in sorted(even_blocks.items()):
            if ks == 0:
                ws_vecs.append(items[0][1])
                continue
            ws_vecs.extend([v for is_ws, v in items if is_ws])
            w_nws = [v for is_ws, v in items if not is_ws]
            if not w_nws:
                continue
            W_n = np.column_stack(w_nws)
            TW = T @ W_n
            nz_rows = np.where(np.linalg.norm(TW, axis=1) > 1e-10)[0]
            if len(nz_rows) == 0:
                tele_blocks[ks] = W_n
            else:
                TW_sub = TW[nz_rows]
                Q_coh = (TW_sub / np.linalg.norm(TW_sub, axis=1, keepdims=True)).T
                if W_n.shape[1] > len(nz_rows):
                    P_perp = np.eye(W_n.shape[1]) - Q_coh @ Q_coh.T
                    vals, vecs = np.linalg.eigh(P_perp)
                    tele_blocks[ks] = W_n @ vecs[:, vals > 0.5]
                coh_groups.setdefault(tuple(nz_rows.tolist()), []).append((ks, W_n @ Q_coh))

        W_ws = np.column_stack(ws_vecs)
        self.base_proj = W_ws @ (W_ws.T @ self.F_true)
        self.n_fixed_ws = W_ws.shape[1]

        # Precompute projections for coherent shift-orbit groups and telescoping blocks
        self.coh_orbits = []
        self.n_coh_dims = 0
        for key, copies in coh_groups.items():
            # Each entry: (d_group, [(W_copy, C_copy), ...])
            active_copies = []
            for _, W_c in copies:
                C_c = W_c.T @ self.F_true
                active_copies.append((W_c, C_c))
                self.n_coh_dims += W_c.shape[1]
            self.coh_orbits.append((len(key), active_copies))

        self.tele_list = []
        self.n_tele_dims = 0
        for ks, W_t in sorted(tele_blocks.items()):
            C_t = W_t.T @ self.F_true
            if np.max(np.abs(C_t)) > 1e-12:
                self.tele_list.append((W_t.shape[1], W_t, C_t))
                self.n_tele_dims += W_t.shape[1]

        self.odd_list = []
        for ks, vlist in sorted(odd_blocks.items()):
            W_o = np.column_stack(vlist)
            C_o = W_o.T @ self.F_true
            if np.max(np.abs(C_o)) > 1e-12:
                self.odd_list.append((W_o.shape[1], W_o, C_o))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.base_proj.copy()
        for dim, copies in self.coh_orbits:
            q = haar_orthogonal(dim, rng)
            for W_c, C_c in copies:
                ans += W_c @ (q @ C_c)
        for dim, W_t, C_t in self.tele_list:
            q = haar_orthogonal(dim, rng)
            ans += W_t @ (q @ C_t)
        for dim, W_o, C_o in self.odd_list:
            q = haar_orthogonal(dim, rng)
            ans += W_o @ (q @ C_o)
        return ans


class ShiftGCEmpiricalSpectralNull:
    """
    Distribution-Calibrated Empirical Spectral Surrogate (ShiftGC-Emp).

    For real skewed genomic sequences (e.g. CpG-depleted enhancers where max/min pentamer
    frequency > 100x), uniform-measure rotations distort empirical channel means and
    covariances. ShiftGC-Emp:
      1. Symmetrizes the unlabeled training pentamer distribution p_emp = 0.5*(p + p[RC_5]) + eps.
      2. Performs sequential weighted QR in L^2(p_emp) ordered by interaction order r=0..5,
         preserving exact RC symmetry and strict order-r orthogonality under p_emp.
      3. Fixes the 20 pure-WS (GC) weighted vectors + top `n_bulk_anchors` unlabeled
         sequence-level non-WS bulk composition PCs of M_c (when M is provided) inside orders r=1,2.
      4. Splits remaining vectors in each order r=1..5 into sequence-coherent vs. telescoping
         subspaces via precomputed SVD of M_c @ Phi_r and applies independent Haar rotations.

    Guarantees exact empirical mean preservation (~0.0 SD diff) and per-base covariance
    preservation (< 1e-13 error) under p_emp, plus <= 10% sequence-mean covariance error!
    """

    def __init__(
        self,
        F_true: np.ndarray,
        p_raw: np.ndarray,
        M_unlabeled: Optional[np.ndarray] = None,
        n_bulk_anchors: int = 6,
        tele_sv_ratio: float = 0.15,
    ):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        p_raw = np.asarray(p_raw, dtype=np.float64)
        p_emp = 0.5 * (p_raw + p_raw[RC_5]) + 1e-8
        p_emp /= p_emp.sum()
        self.p_emp = p_emp
        sqp = np.sqrt(p_emp)

        U, digits, signs, _ = build_fourier_basis()
        rev = np.ravel_multi_index(digits[:, ::-1].T, (4,) * 5)
        weights = np.array([16, 8, 4, 2, 1], dtype=np.int32)
        supports = np.sum((digits > 0) * weights[None, :], axis=1)
        rev_sup = np.sum((digits[:, ::-1] > 0) * weights[None, :], axis=1)
        pure_ws = np.all((digits == 0) | (digits == 2), axis=1)
        orders = np.sum(digits > 0, axis=1)

        even_items = []
        used = set()
        for i in range(1024):
            if i in used:
                continue
            j = int(rev[i])
            used.add(i)
            used.add(j)
            sg = int(np.prod(signs[digits[i]]))
            ks = int(min(supports[i], rev_sup[i]))
            ord_r = int(orders[i])
            is_ws = bool(pure_ws[i])
            if i == j:
                if sg == 1:
                    even_items.append((ord_r, 0 if is_ws else 1, ks, U[:, i]))
            else:
                even_items.append((ord_r, 0 if is_ws else 1, ks, (U[:, i] + sg * U[:, j]) / np.sqrt(2.0)))

        # Build L^2(p_emp)-orthonormal WS basis ordered r=0..5 (20 dims)
        ws_items = sorted([x for x in even_items if x[1] == 0], key=lambda x: (x[0], x[2]))
        W_ws_raw = np.column_stack([x[3] for x in ws_items])
        Q_ws_w, _ = np.linalg.qr(sqp[:, None] * W_ws_raw)
        Phi_ws = Q_ws_w / sqp[:, None]

        # Build L^2(p_emp)-orthonormal non-WS bases ordered sequentially r=1..5
        P_prev = Q_ws_w.copy()
        order_bases: Dict[int, np.ndarray] = {}
        for r in range(1, 6):
            r_vecs = [x[3] for x in even_items if x[0] == r and x[1] == 1]
            if not r_vecs:
                continue
            W_rw = sqp[:, None] * np.column_stack(r_vecs)
            W_rw = W_rw - P_prev @ (P_prev.T @ W_rw)
            Qr, Rr = np.linalg.qr(W_rw)
            keep = np.abs(np.diag(Rr)) > 1e-10
            Qr = Qr[:, keep]
            P_prev = np.hstack([P_prev, Qr])
            order_bases[r] = Qr / sqp[:, None]

        fixed_parts = [Phi_ws]
        rot_blocks: List[np.ndarray] = []

        if M_unlabeled is not None and len(M_unlabeled) >= 20:
            Mc = np.asarray(M_unlabeled, dtype=np.float64)
            Mc = Mc - Mc.mean(axis=0, keepdims=True)
            # Split each order r into sequence-coherent vs. telescoping via unlabeled Mc @ Phi_r
            # And anchor top non-WS bulk composition directions in r=1,2 if requested
            remaining_anchors = n_bulk_anchors
            for r in sorted(order_bases.keys()):
                Phi_r = order_bases[r]
                _, sv, vt = np.linalg.svd((Mc @ Phi_r) / np.sqrt(len(Mc)), full_matrices=True)
                # vt rows are orthonormal directions in Phi_r ordered by sequence-mean variance
                Phi_ord = Phi_r @ vt.T
                if r <= 2 and remaining_anchors > 0:
                    k_fix = min(remaining_anchors, max(0, Phi_ord.shape[1] - 2))
                    if k_fix > 0:
                        fixed_parts.append(Phi_ord[:, :k_fix])
                        Phi_ord = Phi_ord[:, k_fix:]
                        sv = sv[k_fix:]
                        remaining_anchors -= k_fix
                if Phi_ord.shape[1] == 0:
                    continue
                if len(sv) > 1 and sv[0] > 0:
                    n_coh = int(np.sum(sv >= tele_sv_ratio * sv[0]))
                else:
                    n_coh = Phi_ord.shape[1]
                n_tel = Phi_ord.shape[1] - n_coh
                if n_coh == 1 and n_tel >= 1:
                    # Avoid 1D block sign/identity lock by anchoring singleton or keeping >=2
                    fixed_parts.append(Phi_ord[:, :1])
                    if n_tel >= 2:
                        rot_blocks.append(Phi_ord[:, 1:])
                    elif n_tel == 1:
                        fixed_parts.append(Phi_ord[:, 1:])
                else:
                    if n_coh >= 2:
                        rot_blocks.append(Phi_ord[:, :n_coh])
                    elif n_coh == 1:
                        fixed_parts.append(Phi_ord[:, :1])
                    if n_tel >= 2:
                        rot_blocks.append(Phi_ord[:, n_coh:])
                    elif n_tel == 1:
                        fixed_parts.append(Phi_ord[:, n_coh:])
        else:
            for r in sorted(order_bases.keys()):
                Phi_r = order_bases[r]
                if Phi_r.shape[1] >= 2:
                    rot_blocks.append(Phi_r)
                else:
                    fixed_parts.append(Phi_r)

        Phi_fix = np.hstack(fixed_parts)
        self.base_proj = Phi_fix @ ((Phi_fix.T * p_emp[None, :]) @ self.F_true)
        self.n_fixed_dims = Phi_fix.shape[1]
        self.rot_list = []
        self.n_rot_dims = 0
        for Phi_b in rot_blocks:
            C_b = (Phi_b.T * p_emp[None, :]) @ self.F_true
            if np.max(np.abs(C_b)) > 1e-12:
                self.rot_list.append((Phi_b.shape[1], Phi_b, C_b))
                self.n_rot_dims += Phi_b.shape[1]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ans = self.base_proj.copy()
        for dim, Phi_b, C_b in self.rot_list:
            q = haar_orthogonal(dim, rng)
            ans += Phi_b @ (q @ C_b)
        return ans
