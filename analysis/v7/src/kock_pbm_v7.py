#!/usr/bin/env python3
"""
Core data curation, 12-bp physical DNA registration, strict-parity sequence/shape/protein
encoders, exact sufficient-statistics two-penalty bilinear Ridge solvers, batched LOGO
multi-sparsity Schur null evaluators, and gene-cluster inference for `biophys_inductive_gfm/v7`.

Reuses verified v3/v4/v5/v6 null generators and v7 FamilyCode exact ports:
  - load_biophysical_table_matrix, ShiftGCExactCRTNull, Shapelizer2018ExactPort, NaiveRCPermutationNull
  - Sparse4merSurrogateNull (K in {4, 8, 12, 24, 48}), fit_predict_two_penalty_null_bank_schur
  - build_multi_sparsity_null_banks
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import json
import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rdata
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh
from scipy.optimize import lsq_linear
from scipy.stats import wilcoxon

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V3_SRC = ROOT / "v3" / "src"
V4_SRC = ROOT / "v4" / "src"
V5_SRC = ROOT / "v5" / "src"
V6_SRC = ROOT / "v6" / "src"
V7_SRC = ROOT / "v7" / "src"
for _p in (V7_SRC, V6_SRC, V5_SRC, V4_SRC, V3_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import familycode_exact as fc  # noqa: E402
import mutant_diff_v6 as md6  # noqa: E402
import null_generators_v4 as ng4  # noqa: E402
import paired_perturbation_v5 as pp5  # noqa: E402
import step1_models as v3_sm  # noqa: E402

# Explicit re-exports required for v7 interoperability & testing
load_biophysical_table_matrix = ng4.load_biophysical_table_matrix
ShiftGCExactCRTNull = ng4.ShiftGCExactCRTNull
Shapelizer2018ExactPort = ng4.Shapelizer2018ExactPort
NaiveRCPermutationNull = ng4.NaiveRCPermutationNull
Sparse4merSurrogateNull = pp5.Sparse4merSurrogateNull
fit_predict_two_penalty_null_bank_schur = pp5.fit_predict_two_penalty_null_bank_schur
build_multi_sparsity_null_banks = md6.build_multi_sparsity_null_banks
revcomp = pp5.revcomp
canonical_rc = pp5.canonical_rc
safe_spearman_pearson = v3_sm.safe_spearman_pearson
fast_batch_spearman_pearson = v3_sm.fast_batch_spearman_pearson

V4_DATA_DIR = ROOT / "v4" / "data"
V7_RAW_DIR = ROOT / "v7" / "data" / "raw"
V7_PROC_DIR = ROOT / "v7" / "data" / "processed"
V7_ART_DIR = ROOT / "v7" / "artifacts"

K_LADDER: Tuple[int, ...] = (4, 8, 12, 24, 48)
K_PWM_DEFAULT: int = 2000
B_NULL_DEFAULT: int = 199

_B2I: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
_I2B: Tuple[str, str, str, str] = ("A", "C", "G", "T")

# Orthonormal 3-column zero-sum contrast per base (A, C, G, T): MK, WS, RY
_MONO_H = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]],
    dtype=np.float64,
) * 0.5

# Precompute canonical dinucleotide map for bulk 10 RC-even counts
_DI_REMAP = np.zeros(16, dtype=np.int32)
_UNIQ_DI = sorted(
    {min(x, ((3 - (x & 3)) << 2) | (3 - (x >> 2))) for x in range(16)}
)
for _i_u, _v_u in enumerate(_UNIQ_DI):
    _DI_REMAP[_v_u] = _i_u

# Amino-acid physicochemical properties: (Kyte-Doolittle, Volume A^3, Charge pH7.4, pI, Grantham polarity)
_AA_PROPS_RAW: Dict[str, Tuple[float, float, float, float, float]] = {
    "A": (1.8, 88.6, 0.0, 6.00, 8.1),
    "R": (-4.5, 173.4, 1.0, 10.76, 10.5),
    "N": (-3.5, 114.1, 0.0, 5.41, 11.6),
    "D": (-3.5, 111.1, -1.0, 2.77, 13.0),
    "C": (2.5, 108.5, 0.0, 5.07, 5.5),
    "Q": (-3.5, 143.8, 0.0, 5.65, 10.5),
    "E": (-3.5, 138.4, -1.0, 3.22, 12.3),
    "G": (-0.4, 60.1, 0.0, 5.97, 9.0),
    "H": (-3.2, 153.2, 0.1, 7.59, 10.4),
    "I": (4.5, 166.7, 0.0, 6.02, 5.2),
    "L": (3.8, 166.7, 0.0, 5.98, 4.9),
    "K": (-3.9, 168.6, 1.0, 9.74, 11.3),
    "M": (1.9, 162.9, 0.0, 5.74, 5.7),
    "F": (2.8, 189.9, 0.0, 5.48, 5.2),
    "P": (-1.6, 112.7, 0.0, 6.30, 8.0),
    "S": (-0.8, 89.0, 0.0, 5.68, 9.2),
    "T": (-0.7, 116.1, 0.0, 5.60, 8.6),
    "W": (-0.9, 227.8, 0.0, 5.89, 5.4),
    "Y": (-1.3, 193.6, 0.0, 5.66, 6.2),
    "V": (4.2, 140.0, 0.0, 5.96, 5.9),
    "-": (0.0, 0.0, 0.0, 7.00, 0.0),
}
_STD_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_MEAN = np.mean([_AA_PROPS_RAW[a] for a in _STD_AA20], axis=0)
_AA_STD = np.maximum(np.std([_AA_PROPS_RAW[a] for a in _STD_AA20], axis=0), 1e-8)
_AA_PROPS_NORM: Dict[str, np.ndarray] = {
    a: (np.array(v, dtype=np.float64) - _AA_MEAN) / _AA_STD
    for a, v in _AA_PROPS_RAW.items()
}
_AA_PROPS_NORM["-"] = np.zeros(5, dtype=np.float64)

_MSA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
_MSA_AA2I = {a: i for i, a in enumerate(_MSA_ALPHABET)}

_SEGMENT_NAMES: Tuple[str, ...] = ("N-arm", "H1", "L12", "H2", "L23", "H3")
_CONTACT_NAMES: Tuple[str, ...] = (
    "minor_groove_N_arm",
    "major_groove_H3_base",
    "backbone_phosphate",
    "core_fold",
)

# Canonical Homeodomain structural segments & DNA contact annotations (positions 2..58)
_MINOR_NARM_POS = {2, 3, 4, 5, 6, 7, 8}
_MAJOR_H3_BASE_POS = {47, 50, 51, 54, 55, 58}
_BACKBONE_PHOS_POS = {25, 28, 31, 43, 44, 46, 48, 53, 57}


def get_hd_segment(pos: int) -> str:
    """Map canonical Pfam Homeodomain position (1..60+) to structural segment."""
    if pos <= 9:
        return "N-arm"
    if pos <= 22:
        return "H1"
    if pos <= 27:
        return "L12"
    if pos <= 38:
        return "H2"
    if pos <= 41:
        return "L23"
    return "H3"


def get_hd_contact_class(pos: int) -> str:
    """Map canonical Pfam Homeodomain position to structural DNA-contact class."""
    if pos in _MINOR_NARM_POS:
        return "minor_groove_N_arm"
    if pos in _MAJOR_H3_BASE_POS:
        return "major_groove_H3_base"
    if pos in _BACKBONE_PHOS_POS:
        return "backbone_phosphate"
    return "core_fold"


def encode_acgt_array(seqs: Sequence[str]) -> np.ndarray:
    """Encode equal-length uppercase ACGT strings to (N, L) int8 array in {0..3}."""
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.int8)
    n = len(seqs)
    L = len(seqs[0])
    arr = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"Expected uniform sequence length {L}, got {len(s)}")
        for j, ch in enumerate(s):
            v = _B2I.get(ch)
            if v is None:
                raise ValueError(f"Invalid DNA base {ch!r} in {s!r}")
            arr[i, j] = v
    return arr


def decode_acgt_array(codes: np.ndarray) -> List[str]:
    """Decode (N, L) integer codes in {0..3} to uppercase ACGT strings."""
    return ["".join(_I2B[int(x)] for x in row) for row in codes]


# ==============================================================================
# 1. WT-Reference-Only Boltzmann Alignment & 12-bp Physical DNA Extraction
# ==============================================================================

_P_GRID_58 = np.concatenate([np.arange(29, dtype=np.int32), np.arange(29, dtype=np.int32)])
_S_GRID_58 = np.concatenate(
    [np.zeros(29, dtype=np.int32), np.ones(29, dtype=np.int32)]
)
_TIE_ORDER_58 = np.lexsort((_P_GRID_58, _S_GRID_58, np.abs(_P_GRID_58 - 14)))
_TIE_RANK_58 = np.empty(58, dtype=np.int32)
_TIE_RANK_58[_TIE_ORDER_58] = np.arange(58, dtype=np.int32)


def precompute_36bp_windows(seq_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute all 58 8-bp placements (29 forward p=0..28, 29 reverse-complement)
    and reverse-complemented 36-bp codes for fast multi-gene PSAM scoring.
    """
    rc_arr = (3 - seq_arr[:, ::-1]).copy()
    fw = np.stack([seq_arr[:, p : p + 8] for p in range(29)], axis=1)
    rw = np.stack([rc_arr[:, 28 - p : 36 - p] for p in range(29)], axis=1)
    all_wins = np.concatenate([fw, rw], axis=1)  # (N, 58, 8)
    return all_wins, rc_arr


def score_36bp_with_psam_tiebreak(
    seq_arr: np.ndarray,
    mat_4x8: np.ndarray,
    precomp_wins: Optional[np.ndarray] = None,
    tie_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Score all 58 8-bp placements of a 4x8 log-affinity PSAM across 36-bp probes.
    Deterministic tie-break for palindromic/exact ties within `tie_tol`:
      prefer smallest |p - 14|, then forward strand (0), then smallest p.
    Returns (best_p, best_strand, f1, total_aff).
    """
    if precomp_wins is None:
        all_wins, _ = precompute_36bp_windows(seq_arr)
    else:
        all_wins = precomp_wins

    mat = np.asarray(mat_4x8, dtype=np.float64)
    if mat.shape != (4, 8):
        raise ValueError(f"Expected (4, 8) log-affinity PSAM, got {mat.shape}")

    pos_idx = np.arange(8, dtype=np.int32)[None, None, :]
    scores = mat[all_wins, pos_idx].sum(axis=2)  # (N, 58)
    smax = scores.max(axis=1, keepdims=True)
    w = np.exp(scores - smax)
    z_sum = w.sum(axis=1)
    f1 = (1.0 / z_sum).astype(np.float64)
    total_aff = (smax[:, 0] + np.log(z_sum)).astype(np.float64)

    is_max = scores >= (smax - tie_tol)
    best_k = np.where(is_max, _TIE_RANK_58[None, :], 999).argmin(axis=1)
    best_p = _P_GRID_58[best_k].astype(np.int32)
    best_strand = _S_GRID_58[best_k].astype(np.int32)
    return best_p, best_strand, f1, total_aff


def extract_12bp_window_codes(
    seq_arr: np.ndarray,
    rc_arr: np.ndarray,
    probe_indices: np.ndarray,
    best_p: np.ndarray,
    best_strand: np.ndarray,
) -> np.ndarray:
    """
    Extract oriented 12-bp physical DNA `[F-2, F-1, N1..N8, F+1, F+2]` codes
    strictly for interior placements `2 <= p <= 26` (zero 'N' padding).
    """
    n = len(probe_indices)
    out = np.empty((n, 12), dtype=np.int8)
    for j, idx in enumerate(probe_indices):
        p = int(best_p[idx])
        if p < 2 or p > 26:
            raise ValueError(f"Non-interior placement p={p} for probe {idx}")
        st = int(best_strand[idx])
        if st == 0:
            out[j] = seq_arr[idx, p - 2 : p + 10]
        else:
            # revcomp(s[p-2 : p+10]) == rc_s[36 - (p+10) : 36 - (p-2)]
            out[j] = rc_arr[idx, 26 - p : 38 - p]
    return out


def extract_12bp_windows_from_36bp(
    seqs36: Sequence[str],
    mat_4x8: np.ndarray,
    f1_thresh: float = 0.70,
    tie_tol: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Convenience helper: align 36-bp probes using WT `mat_4x8` and return interior
    (`2 <= p <= 26`) probes passing `f1 >= f1_thresh` with 12-bp windows.
    """
    seq_arr = encode_acgt_array(seqs36)
    all_wins, rc_arr = precompute_36bp_windows(seq_arr)
    best_p, best_strand, f1, total_aff = score_36bp_with_psam_tiebreak(
        seq_arr, mat_4x8, precomp_wins=all_wins, tie_tol=tie_tol
    )
    mask = (best_p >= 2) & (best_p <= 26) & (f1 >= f1_thresh)
    idx = np.where(mask)[0].astype(np.int32)
    codes12 = extract_12bp_window_codes(seq_arr, rc_arr, idx, best_p, best_strand)
    win12 = np.array(decode_acgt_array(codes12), dtype="<U12")
    return {
        "probe_idx": idx,
        "codes12": codes12,
        "win12": win12,
        "best_p": best_p[idx],
        "best_strand": best_strand[idx],
        "f1": f1[idx],
        "total_aff": total_aff[idx],
    }


def orient_8mers_with_psam(
    kmers8: Sequence[str],
    mat_4x8: np.ndarray,
    tie_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orient 8-mers to `[N1..N8]` via WT ProBound log-affinity PSAM (`mat_4x8`),
    and extract complete internal pentamer indices at `T3` (`pos 0:5`) and `D4` (`pos 1:6`).
    Returns `(oriented_codes_8 (N, 8), t3d4_c5 (N, 2))`.
    """
    codes = encode_acgt_array(kmers8)
    rc_codes = (3 - codes[:, ::-1]).copy()
    mat = np.asarray(mat_4x8, dtype=np.float64)
    pos_idx = np.arange(8, dtype=np.int32)[None, :]
    sc_fwd = mat[codes, pos_idx].sum(axis=1)
    sc_rc = mat[rc_codes, pos_idx].sum(axis=1)
    choose_rc = sc_rc > (sc_fwd + tie_tol)
    oriented = np.where(choose_rc[:, None], rc_codes, codes).astype(np.int8)

    c5_t3 = (
        (
            (
                ((oriented[:, 0].astype(np.int32) << 2) | oriented[:, 1]) << 2
                | oriented[:, 2]
            )
            << 2
            | oriented[:, 3]
        )
        << 2
        | oriented[:, 4]
    )
    c5_d4 = (
        (
            (
                ((oriented[:, 1].astype(np.int32) << 2) | oriented[:, 2]) << 2
                | oriented[:, 3]
            )
            << 2
            | oriented[:, 4]
        )
        << 2
        | oriented[:, 5]
    )
    t3d4_c5 = np.column_stack([c5_t3, c5_d4]).astype(np.int16)
    return oriented, t3d4_c5


# ==============================================================================
# 2. Monotone Cubic P-Spline Specificity Residual
# ==============================================================================

def fit_monotone_pspline_residual(
    z_wt_fit: np.ndarray,
    z_mut_fit: np.ndarray,
    z_wt_eval: np.ndarray,
    z_mut_eval: np.ndarray,
    n_knots: int = 20,
    lam: float = 10.0,
) -> np.ndarray:
    """
    Fit a smooth monotonically non-decreasing cubic I-spline (cumulative B-spline
    basis with non-negative increments and second-difference penalty) predicting
    `z_mut` from `z_wt` across finite interior probes, returning the specificity
    residual `z_mut_eval - spline(z_wt_eval)` (NaN where `z_mut_eval` or `z_wt_eval` is NaN).
    """
    mk_fit = np.isfinite(z_wt_fit) & np.isfinite(z_mut_fit)
    xw = np.asarray(z_wt_fit[mk_fit], dtype=np.float64)
    ym = np.asarray(z_mut_fit[mk_fit], dtype=np.float64)
    if len(xw) < 10 or float(np.std(xw)) < 1e-10:
        return z_mut_eval - z_wt_eval

    x_min, x_max = float(np.min(xw)), float(np.max(xw))
    qs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    knots_raw = np.quantile(xw, qs)
    # Deduplicate knots strictly inside (x_min, x_max)
    knots_int = np.unique(knots_raw[(knots_raw > x_min + 1e-8) & (knots_raw < x_max - 1e-8)])
    if len(knots_int) == 0:
        knots_int = np.array([0.5 * (x_min + x_max)], dtype=np.float64)
    t = np.concatenate([[x_min] * 4, knots_int, [x_max] * 4])

    def _cum_bspline(x_vals: np.ndarray) -> np.ndarray:
        xc = np.clip(x_vals, x_min, x_max)
        B = BSpline.design_matrix(xc, t, 3).toarray()
        return np.cumsum(B[:, ::-1], axis=1)[:, ::-1]

    C_fit = _cum_bspline(xw)
    M = C_fit.shape[1]
    if M >= 3:
        P = np.zeros((M - 2, M), dtype=np.float64)
        idx_r = np.arange(M - 2)
        P[idx_r, idx_r + 1] = -np.sqrt(lam)
        P[idx_r, idx_r + 2] = np.sqrt(lam)
        A = np.vstack([C_fit, P])
        b = np.concatenate([ym, np.zeros(M - 2, dtype=np.float64)])
    else:
        A = C_fit
        b = ym

    lb = np.zeros(M, dtype=np.float64)
    lb[0] = -np.inf
    ub = np.full(M, np.inf, dtype=np.float64)
    sol = lsq_linear(A, b, bounds=(lb, ub), method="trf", tol=1e-10)

    out = np.full(len(z_wt_eval), np.nan, dtype=np.float64)
    mk_ev = np.isfinite(z_wt_eval) & np.isfinite(z_mut_eval)
    if np.any(mk_ev):
        C_ev = _cum_bspline(np.asarray(z_wt_eval[mk_ev], dtype=np.float64))
        out[mk_ev] = z_mut_eval[mk_ev] - (C_ev @ sol.x)
    return out


# ==============================================================================
# 3. Protein Alignment Resolution & Feature Encoders (phi_prot & WT-only)
# ==============================================================================

def resolve_kock_hd_alignments(
    raw_dir: Path = V7_RAW_DIR,
) -> Dict[str, object]:
    """
    Load the 417-HD Pfam alignment (`HD_SELEX_PBM_CV_alignment.rsd` + `add_hmm.sto`),
    resolve duplicate rows (`CRX`, `SIX6`) strictly against `MOESM5_ESM.xls` WT residues
    (guaranteeing 0/92 canonical mismatches), and compute top-8 417-HD MSA SVD basis.
    """
    al_df = rdata.read_rds(raw_dir / "HD_SELEX_PBM_CV_alignment.rsd").reset_index(drop=True)
    sto_lines = [
        ln.strip()
        for ln in (raw_dir / "add_hmm.sto").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and ln != "//"
    ]
    sto_df = pd.DataFrame(
        {
            "name": ["HOXB6", "NKX2-6", "NKX2-4"],
            "alignment": [ln.split()[1] for ln in sto_lines],
        }
    )
    al_all = pd.concat([al_df, sto_df], ignore_index=True)

    # 417-HD MSA top-8 SVD (WT homology space only)
    M_oh = np.zeros((len(al_all), 57 * len(_MSA_ALPHABET)), dtype=np.float64)
    for r_i, seq in enumerate(al_all["alignment"]):
        for p_i, ch in enumerate(seq):
            M_oh[r_i, p_i * len(_MSA_ALPHABET) + _MSA_AA2I.get(ch, 20)] = 1.0
    msa_mean = M_oh.mean(axis=0)
    _, _, vt = np.linalg.svd(M_oh - msa_mean[None, :], full_matrices=False)
    V8 = vt[:8].T.copy()  # (1197, 8)

    # Load MOESM5_ESM.xls variants table (92 mutants)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_df = pd.read_excel(raw_dir / "41467_2024_47396_MOESM5_ESM.xls", sheet_name="variants")

    genes_sorted = sorted(var_df["gene"].astype(str).unique().tolist())
    wt_by_gene: Dict[str, str] = {}
    for g in genes_sorted:
        cands = al_all.loc[al_all["name"] == g, "alignment"].astype(str).tolist()
        g_muts = var_df[var_df["gene"] == g]
        valid = [
            s
            for s in cands
            if all(
                s[int(row["canonical"]) - 2] == str(row["allele"])[len(g) + 1 :][0]
                for _, row in g_muts.iterrows()
            )
        ]
        if len(valid) != 1:
            raise RuntimeError(
                f"Expected unique canonical-matching alignment row for {g}, found {len(valid)}"
            )
        wt_by_gene[g] = valid[0]

    return {
        "var_df": var_df,
        "genes": genes_sorted,
        "wt_by_gene": wt_by_gene,
        "msa_mean": msa_mean,
        "msa_V8": V8,
        "al_417_df": al_all,
    }


def embed_hd_msa_svd8(seq57: str, msa_mean: np.ndarray, msa_V8: np.ndarray) -> np.ndarray:
    """Project a 57-aa Pfam HD alignment sequence onto the top-8 417-HD MSA SVD."""
    v = np.zeros(57 * len(_MSA_ALPHABET), dtype=np.float64)
    for p_i, ch in enumerate(seq57):
        v[p_i * len(_MSA_ALPHABET) + _MSA_AA2I.get(ch, 20)] = 1.0
    return (v - msa_mean) @ msa_V8


def encode_full_hd_protein_features(
    seqs57: Sequence[str],
    msa_mean: np.ndarray,
    msa_V8: np.ndarray,
) -> np.ndarray:
    """
    Encode full 57-aa Homeodomain sequences (`canonical 2..58`, 40 dims) for Regime A
    (`WT-Only Prot x DNA`), ensuring `f(mut, x) - f(wt, x)` is nonzero for every mutant:
      - top-8 417-HD MSA SVD embeddings (8)
      - 6 structural segment mean normalized physicochemical props [kd, vol, chg, pol] (24)
      - N-arm DNA-contact cluster `{2, 3, 5, 6}` mean props (4)
      - H3 base-contact cluster `{47, 50, 51, 54}` mean props (4)
    """
    seg_slices = [
        range(0, 8),    # canonical 2..9 (N-arm)
        range(8, 21),   # canonical 10..22 (H1)
        range(21, 26),  # canonical 23..27 (L12)
        range(26, 37),  # canonical 28..38 (H2)
        range(37, 40),  # canonical 39..41 (L23)
        range(40, 57),  # canonical 42..58 (H3)
    ]
    narm_idx = [p - 2 for p in (2, 3, 5, 6)]
    h3_idx = [p - 2 for p in (47, 50, 51, 54)]
    prop_idx = [0, 1, 2, 4]  # kd, vol, chg, pol

    out = np.empty((len(seqs57), 40), dtype=np.float64)
    for i, seq in enumerate(seqs57):
        svd8 = embed_hd_msa_svd8(seq, msa_mean, msa_V8)
        mat = np.array([_AA_PROPS_NORM[ch][prop_idx] for ch in seq], dtype=np.float64)
        seg_means = np.concatenate([mat[list(sl)].mean(axis=0) for sl in seg_slices])
        c_narm = mat[narm_idx].mean(axis=0)
        c_h3 = mat[h3_idx].mean(axis=0)
        out[i] = np.concatenate([svd8, seg_means, c_narm, c_h3])
    return out


def encode_prot_mutation_features(
    var_df: pd.DataFrame,
    wt_by_gene: Dict[str, str],
    msa_mean: np.ndarray,
    msa_V8: np.ndarray,
    blosum62: Tuple[Dict[str, int], np.ndarray],
) -> Dict[str, object]:
    """
    Build `phi_prot(m)` (shape `(N_mut, 46)`) and structured metadata arrays across
    the 92 Kock mutants:
      - WT physicochemical (5), Mut physicochemical (5), Delta physicochemical (5)
      - BLOSUM62(wt, mut) & BLOSUM62(wt, mut) - BLOSUM62(wt, wt) (2)
      - 6-segment structural one-hot (6)
      - 4-class DNA-contact one-hot (4)
      - canonical position coords: [pos/60, (pos/60)^2, exp(-((pos-5)/4)^2)-exp(-((pos-50)/5)^2)] (3)
      - WT MSA SVD8 (8) + Delta MSA SVD8 (8)
    """
    b62_idx, b62_mat = blosum62
    seg_idx_map = {s: i for i, s in enumerate(_SEGMENT_NAMES)}
    cc_idx_map = {c: i for i, c in enumerate(_CONTACT_NAMES)}

    alleles, genes, mut_seqs, wt_seqs = [], [], [], []
    wt_aas, mut_aas, segments, contact_classes, kock_strata = [], [], [], [], []
    spec_chgs, aff_chgs = [], []
    canon_pos = []
    phi_rows = []

    for _, row in var_df.iterrows():
        g = str(row["gene"])
        allele = str(row["allele"])
        mut_str = allele[len(g) + 1 :]
        wt_aa, mut_aa = mut_str[0], mut_str[-1]
        pos = int(row["canonical"])
        idx0 = pos - 2
        wt_seq = wt_by_gene[g]
        if wt_seq[idx0] != wt_aa:
            raise ValueError(f"Canonical mismatch on {allele}: {wt_seq[idx0]} != {wt_aa}")
        mut_seq = wt_seq[:idx0] + mut_aa + wt_seq[idx0 + 1 :]

        wp = _AA_PROPS_NORM[wt_aa]
        mp = _AA_PROPS_NORM[mut_aa]
        dp = mp - wp

        b_wm = float(b62_mat[b62_idx[wt_aa], b62_idx[mut_aa]]) / 5.0
        b_ww = float(b62_mat[b62_idx[wt_aa], b62_idx[wt_aa]]) / 5.0

        seg_name = get_hd_segment(pos)
        seg_oh = np.zeros(len(_SEGMENT_NAMES), dtype=np.float64)
        seg_oh[seg_idx_map[seg_name]] = 1.0

        cc_name = get_hd_contact_class(pos)
        cc_oh = np.zeros(len(_CONTACT_NAMES), dtype=np.float64)
        cc_oh[cc_idx_map[cc_name]] = 1.0

        pos_coords = np.array(
            [
                pos / 60.0,
                (pos / 60.0) ** 2,
                np.exp(-((pos - 5.0) / 4.0) ** 2) - np.exp(-((pos - 50.0) / 5.0) ** 2),
            ],
            dtype=np.float64,
        )

        w8 = embed_hd_msa_svd8(wt_seq, msa_mean, msa_V8)
        d8 = embed_hd_msa_svd8(mut_seq, msa_mean, msa_V8) - w8

        phi_rows.append(
            np.concatenate([wp, mp, dp, [b_wm, b_wm - b_ww], seg_oh, cc_oh, pos_coords, w8, d8])
        )

        sc = str(row["specificity_change"])
        ac = str(row["affinity_change"])
        k_strat = (
            "specificity_yes"
            if sc == "yes"
            else ("affinity_only" if ac != "no change" else "no_change")
        )

        alleles.append(allele)
        genes.append(g)
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
        wt_aas.append(wt_aa)
        mut_aas.append(mut_aa)
        canon_pos.append(pos)
        segments.append(seg_name)
        contact_classes.append(cc_name)
        spec_chgs.append(sc)
        aff_chgs.append(ac)
        kock_strata.append(k_strat)

    return {
        "alleles": np.array(alleles),
        "genes": np.array(genes),
        "wt_seqs": np.array(wt_seqs),
        "mut_seqs": np.array(mut_seqs),
        "wt_aas": np.array(wt_aas),
        "mut_aas": np.array(mut_aas),
        "canon_pos": np.array(canon_pos, dtype=np.int32),
        "segments": np.array(segments),
        "contact_classes": np.array(contact_classes),
        "specificity_change": np.array(spec_chgs),
        "affinity_change": np.array(aff_chgs),
        "kock_stratum": np.array(kock_strata),
        "phi_prot": np.vstack(phi_rows).astype(np.float64),
    }


def fit_transform_prot_svd(
    phi_tr: np.ndarray,
    phi_te: np.ndarray,
    n_components: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit standardized SVD strictly on training protein rows (zero leakage)."""
    mu = phi_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(phi_tr.std(axis=0, keepdims=True), 1e-8)
    z_tr = (phi_tr - mu) / sd
    z_te = (phi_te - mu) / sd
    _, _, vt = np.linalg.svd(z_tr, full_matrices=False)
    k = min(n_components, vt.shape[0])
    V = vt[:k].T
    return z_tr @ V, z_te @ V


# ==============================================================================
# 4. Strict 12-bp Parity DNA Sequence & Shape Encoders
# ==============================================================================

def encode_seq_g1_12bp(codes12: np.ndarray) -> np.ndarray:
    """Positional mononucleotide orthonormal contrasts on 12-bp `[F-2..F+2]` -> (N, 36)."""
    if codes12.shape[1] != 12:
        raise ValueError(f"Expected 12-bp window, got shape {codes12.shape}")
    return _MONO_H[codes12].reshape(len(codes12), 36).astype(np.float64)


def encode_seq_g2_12bp(codes12: np.ndarray) -> np.ndarray:
    """
    Positional mononucleotide (36) + adjacent dinucleotide contrasts (11 x 9 = 99)
    + bulk GC, GC^2 (2) + 10 RC-canonical dinucleotide freqs (10) -> (N, 147).
    """
    n = len(codes12)
    mf = _MONO_H[codes12]  # (N, 12, 3)
    x_g1 = mf.reshape(n, 36)
    di_adj = np.einsum("npa,npb->npab", mf[:, :-1, :], mf[:, 1:, :]).reshape(n, 99)

    gc = ((codes12 == 1) | (codes12 == 2)).mean(axis=1).astype(np.float64)
    c2 = (codes12[:, :-1].astype(np.int32) << 2) | codes12[:, 1:]
    c2_rc = ((3 - codes12[:, 1:]).astype(np.int32) << 2) | (3 - codes12[:, :-1])
    c2_mapped = _DI_REMAP[np.minimum(c2, c2_rc)]
    di_bulk = np.zeros((n, len(_UNIQ_DI)), dtype=np.float64)
    rows = np.arange(n, dtype=np.int32)[:, None]
    np.add.at(di_bulk, (rows, c2_mapped), 1.0 / 11.0)
    return np.hstack([x_g1, di_adj, gc[:, None], (gc * gc)[:, None], di_bulk]).astype(np.float64)


def encode_seq_pos4mer_n1n4(codes12: np.ndarray) -> np.ndarray:
    """
    Exact 5 positional 4-mers spanning `[F-2..Y6]` (windows `[0:4], [1:5], [2:6], [3:7], [4:8]`),
    matching the exact pentamer span of `Shape_N1N4` -> (N, 1280) float32.
    """
    if codes12.shape[1] != 12:
        raise ValueError(f"Expected 12-bp window, got shape {codes12.shape}")
    n = len(codes12)
    out = np.zeros((n, 5 * 256), dtype=np.float32)
    rows = np.arange(n, dtype=np.int32)
    for k_i in range(5):
        c4 = (
            (
                (
                    ((codes12[:, k_i].astype(np.int32) << 2) | codes12[:, k_i + 1]) << 2
                    | codes12[:, k_i + 2]
                )
                << 2
            )
            | codes12[:, k_i + 3]
        )
        out[rows, k_i * 256 + c4] = 1.0
    return out


def extract_pentamer_indices_12bp(
    codes12: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return pentamer index arrays in `{0..1023}` for `[N1..N8]` on 12-bp `[F-2..F+2]`:
      - `c5_n1n4`: `(N, 4)` at `[N1, N2, T3, D4]` (5-mers `[0:5]..[3:8]`)
      - `c5_t3d4`: `(N, 2)` at `[T3, D4]` (5-mers `[2:7], [3:8]`)
      - `c5_all8`: `(N, 8)` at `[N1..N8]` (5-mers `[0:5]..[7:12]`)
    """
    if codes12.shape[1] != 12:
        raise ValueError(f"Expected 12-bp window, got shape {codes12.shape}")
    cols = []
    for st in range(8):
        c5 = (
            (
                (
                    (
                        ((codes12[:, st].astype(np.int32) << 2) | codes12[:, st + 1])
                        << 2
                        | codes12[:, st + 2]
                    )
                    << 2
                    | codes12[:, st + 3]
                )
                << 2
                | codes12[:, st + 4]
            )
        )
        cols.append(c5)
    c5_all8 = np.column_stack(cols).astype(np.int16)
    return c5_all8[:, 0:4], c5_all8[:, 2:4], c5_all8


def build_shape_from_c5(c5: np.ndarray, F_table: np.ndarray) -> np.ndarray:
    """Lookup (N, K_pos * d_ch) DNA shape features from pentamer indices `c5` (N, K_pos)."""
    return F_table[c5].reshape(c5.shape[0], -1).astype(np.float64)


def build_bilinear_blocks(
    phi_prot_rows: np.ndarray,
    X_dna: np.ndarray,
    phi_prot_int: Optional[np.ndarray] = None,
    include_prot_add: bool = True,
    include_dna_add: bool = True,
) -> np.ndarray:
    """
    Assemble bilinear interaction block `[phi_prot_add, X_dna, phi_prot_int ⊗ X_dna]`.
    """
    parts = []
    if include_prot_add:
        parts.append(np.asarray(phi_prot_rows, dtype=np.float64))
    if include_dna_add:
        parts.append(np.asarray(X_dna, dtype=np.float64))
    p_int = phi_prot_rows if phi_prot_int is None else phi_prot_int
    inter = (
        np.asarray(p_int, dtype=np.float64)[:, :, None]
        * np.asarray(X_dna, dtype=np.float64)[:, None, :]
    ).reshape(len(X_dna), -1)
    parts.append(inter)
    return np.hstack(parts)


# ==============================================================================
# 5. Exact Sufficient-Statistics Two-Penalty Bilinear Ridge Solvers
# ==============================================================================

@dataclass
class GroupSuffStats:
    """Per-group float64 sufficient statistics for exact zero-leakage Ridge & CV."""

    n: np.ndarray          # (G,) int64
    sum_s: np.ndarray      # (G, p_s)
    sum_m: np.ndarray      # (G, p_m)
    sum_y: np.ndarray      # (G,)
    G_ss: np.ndarray       # (G, p_s, p_s)
    G_sm: np.ndarray       # (G, p_s, p_m)
    G_mm: np.ndarray       # (G, p_m, p_m)
    r_s: np.ndarray        # (G, p_s)
    r_m: np.ndarray        # (G, p_m)
    yyy: np.ndarray        # (G,)


def compute_group_suffstats(
    X_s_by_group: Sequence[np.ndarray],
    X_m_by_group: Sequence[np.ndarray],
    y_by_group: Sequence[np.ndarray],
) -> GroupSuffStats:
    """Accumulate exact float64 sufficient statistics per group (e.g. per mutant or gene)."""
    G = len(y_by_group)
    p_s = X_s_by_group[0].shape[1]
    p_m = X_m_by_group[0].shape[1]

    n = np.zeros(G, dtype=np.int64)
    sum_s = np.zeros((G, p_s), dtype=np.float64)
    sum_m = np.zeros((G, p_m), dtype=np.float64)
    sum_y = np.zeros(G, dtype=np.float64)
    G_ss = np.zeros((G, p_s, p_s), dtype=np.float64)
    G_sm = np.zeros((G, p_s, p_m), dtype=np.float64)
    G_mm = np.zeros((G, p_m, p_m), dtype=np.float64)
    r_s = np.zeros((G, p_s), dtype=np.float64)
    r_m = np.zeros((G, p_m), dtype=np.float64)
    yyy = np.zeros(G, dtype=np.float64)

    for g in range(G):
        Xs = np.asarray(X_s_by_group[g], dtype=np.float64)
        Xm = np.asarray(X_m_by_group[g], dtype=np.float64)
        yg = np.asarray(y_by_group[g], dtype=np.float64)
        if len(yg) == 0:
            continue
        n[g] = len(yg)
        sum_s[g] = Xs.sum(axis=0)
        sum_y[g] = float(yg.sum())
        G_ss[g] = Xs.T @ Xs
        r_s[g] = Xs.T @ yg
        yyy[g] = float(yg @ yg)
        if p_m > 0:
            sum_m[g] = Xm.sum(axis=0)
            G_sm[g] = Xs.T @ Xm
            G_mm[g] = Xm.T @ Xm
            r_m[g] = Xm.T @ yg

    return GroupSuffStats(n, sum_s, sum_m, sum_y, G_ss, G_sm, G_mm, r_s, r_m, yyy)


def _aggregate_suffstats(stats: GroupSuffStats, group_indices: Sequence[int]) -> Dict[str, object]:
    idx = np.asarray(group_indices, dtype=np.int64)
    n_tot = int(stats.n[idx].sum())
    sum_s = stats.sum_s[idx].sum(axis=0)
    sum_m = stats.sum_m[idx].sum(axis=0)
    sum_y = float(stats.sum_y[idx].sum())
    G_ss = stats.G_ss[idx].sum(axis=0)
    G_sm = stats.G_sm[idx].sum(axis=0)
    G_mm = stats.G_mm[idx].sum(axis=0)
    r_s = stats.r_s[idx].sum(axis=0)
    r_m = stats.r_m[idx].sum(axis=0)
    yyy = float(stats.yyy[idx].sum())
    return {
        "n": n_tot,
        "sum_s": sum_s,
        "sum_m": sum_m,
        "sum_y": sum_y,
        "G_ss": G_ss,
        "G_sm": G_sm,
        "G_mm": G_mm,
        "r_s": r_s,
        "r_m": r_m,
        "yyy": yyy,
    }


def _standardize_normal_equations(agg: Dict[str, object]) -> Dict[str, object]:
    """Compute exact training standardization & normal equations from sufficient statistics."""
    n = float(agg["n"])
    sum_s = agg["sum_s"]
    sum_m = agg["sum_m"]
    sum_y = float(agg["sum_y"])
    G_ss = agg["G_ss"]
    G_sm = agg["G_sm"]
    G_mm = agg["G_mm"]
    r_s = agg["r_s"]
    r_m = agg["r_m"]

    mu_s = sum_s / n
    var_s = np.maximum(np.diag(G_ss) / n - mu_s * mu_s, 0.0)
    dead_s = var_s <= 1e-12
    sd_s = np.where(dead_s, 1.0, np.sqrt(var_s))
    y_mean = sum_y / n

    Z_ss = (G_ss - n * np.outer(mu_s, mu_s)) / np.outer(sd_s, sd_s)
    Z_ss = 0.5 * (Z_ss + Z_ss.T)
    rhs_s = (r_s - y_mean * sum_s) / sd_s
    if np.any(dead_s):
        Z_ss[dead_s, :] = 0.0
        Z_ss[:, dead_s] = 0.0
        rhs_s[dead_s] = 0.0

    p_m = len(sum_m)
    if p_m > 0:
        mu_m = sum_m / n
        var_m = np.maximum(np.diag(G_mm) / n - mu_m * mu_m, 0.0)
        dead_m = var_m <= 1e-12
        sd_m = np.where(dead_m, 1.0, np.sqrt(var_m))
        Z_sm = (G_sm - n * np.outer(mu_s, mu_m)) / np.outer(sd_s, sd_m)
        Z_mm = (G_mm - n * np.outer(mu_m, mu_m)) / np.outer(sd_m, sd_m)
        Z_mm = 0.5 * (Z_mm + Z_mm.T)
        rhs_m = (r_m - y_mean * sum_m) / sd_m
        if np.any(dead_s):
            Z_sm[dead_s, :] = 0.0
        if np.any(dead_m):
            Z_sm[:, dead_m] = 0.0
            Z_mm[dead_m, :] = 0.0
            Z_mm[:, dead_m] = 0.0
            rhs_m[dead_m] = 0.0
    else:
        mu_m = np.zeros(0, dtype=np.float64)
        sd_m = np.ones(0, dtype=np.float64)
        Z_sm = np.zeros((len(mu_s), 0), dtype=np.float64)
        Z_mm = np.zeros((0, 0), dtype=np.float64)
        rhs_m = np.zeros(0, dtype=np.float64)

    return {
        "mu_s": mu_s,
        "sd_s": sd_s,
        "dead_s": dead_s,
        "mu_m": mu_m,
        "sd_m": sd_m,
        "y_mean": y_mean,
        "Z_ss": Z_ss,
        "Z_sm": Z_sm,
        "Z_mm": Z_mm,
        "rhs_s": rhs_s,
        "rhs_m": rhs_m,
    }


def _weights_to_raw(
    w_s: np.ndarray,
    w_m: np.ndarray,
    std_eq: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, float]:
    ws_raw = w_s / std_eq["sd_s"]
    b_raw = float(std_eq["y_mean"] - np.dot(std_eq["mu_s"], ws_raw))
    if len(w_m) > 0:
        wm_raw = w_m / std_eq["sd_m"]
        b_raw -= float(np.dot(std_eq["mu_m"], wm_raw))
    else:
        wm_raw = np.zeros(0, dtype=np.float64)
    return ws_raw, wm_raw, b_raw


def _sse_on_val_agg(
    ws_raw: np.ndarray,
    wm_raw: np.ndarray,
    b_raw: float,
    val_agg: Dict[str, object],
) -> float:
    """Compute exact validation SSE (up to constant y_val^T y_val) or exact SSE."""
    n_v = float(val_agg["n"])
    # Pred quadratic + linear terms in raw coordinates
    quad = float(ws_raw @ (val_agg["G_ss"] @ ws_raw))
    lin = float(np.dot(ws_raw, val_agg["sum_s"])) * b_raw
    yr = float(np.dot(ws_raw, val_agg["r_s"]))
    if len(wm_raw) > 0:
        quad += 2.0 * float(ws_raw @ (val_agg["G_sm"] @ wm_raw)) + float(wm_raw @ (val_agg["G_mm"] @ wm_raw))
        lin += float(np.dot(wm_raw, val_agg["sum_m"])) * b_raw
        yr += float(np.dot(wm_raw, val_agg["r_m"]))
    return quad + 2.0 * lin + n_v * (b_raw * b_raw) - 2.0 * (yr + b_raw * float(val_agg["sum_y"])) + float(val_agg["yyy"])


def solve_suffstats_two_penalty(
    stats: GroupSuffStats,
    train_groups: Sequence[int],
    lam_s: float,
    lam_m: float,
    X_s_te: np.ndarray,
    X_m_te: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Solve exact two-penalty Ridge on training groups from `GroupSuffStats`
    and predict on `(X_s_te, X_m_te)`. Algebraically identical to direct `scipy.linalg.solve`.
    """
    agg = _aggregate_suffstats(stats, train_groups)
    eq = _standardize_normal_equations(agg)
    p_s = len(eq["mu_s"])
    p_m = len(eq["mu_m"])

    if p_m == 0:
        K = eq["Z_ss"].copy()
        K[np.diag_indices(p_s)] += lam_s
        cf = cho_factor(K, lower=True, check_finite=False)
        w_s = cho_solve(cf, eq["rhs_s"], check_finite=False)
        w_m = np.zeros(0, dtype=np.float64)
    else:
        K = np.empty((p_s + p_m, p_s + p_m), dtype=np.float64)
        K[:p_s, :p_s] = eq["Z_ss"]
        K[:p_s, p_s:] = eq["Z_sm"]
        K[p_s:, :p_s] = eq["Z_sm"].T
        K[p_s:, p_s:] = eq["Z_mm"]
        idx_s = np.arange(p_s)
        idx_m = p_s + np.arange(p_m)
        K[idx_s, idx_s] += lam_s
        K[idx_m, idx_m] += lam_m
        rhs = np.concatenate([eq["rhs_s"], eq["rhs_m"]])
        cf = cho_factor(K, lower=True, check_finite=False)
        w = cho_solve(cf, rhs, check_finite=False)
        w_s, w_m = w[:p_s], w[p_s:]

    ws_raw, wm_raw, b_raw = _weights_to_raw(w_s, w_m, eq)
    pred = b_raw + np.asarray(X_s_te, dtype=np.float64) @ ws_raw
    if p_m > 0 and X_m_te is not None:
        pred = pred + np.asarray(X_m_te, dtype=np.float64) @ wm_raw
    return {
        "pred_te": pred,
        "ws_raw": ws_raw,
        "wm_raw": wm_raw,
        "b_raw": b_raw,
        "w_s": w_s,
        "w_m": w_m,
    }


def _make_balanced_inner_folds(
    train_groups: Sequence[int],
    inner_cv_labels: Sequence[int],
    group_sizes: np.ndarray,
    n_folds: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """Partition `train_groups` cleanly by `inner_cv_labels` (e.g. genes) with balanced sizes."""
    by_lbl: Dict[int, List[int]] = {}
    for g in train_groups:
        by_lbl.setdefault(int(inner_cv_labels[g]), []).append(int(g))
    uniq_lbls = sorted(by_lbl.keys(), key=lambda l: (-int(sum(group_sizes[g] for g in by_lbl[l])), l))
    k = min(n_folds, len(uniq_lbls))
    if k < 2:
        return []
    bins: List[List[int]] = [[] for _ in range(k)]
    bin_sizes = np.zeros(k, dtype=np.int64)
    for l in uniq_lbls:
        b = int(np.argmin(bin_sizes))
        bins[b].extend(by_lbl[l])
        bin_sizes[b] += sum(int(group_sizes[g]) for g in by_lbl[l])
    splits = []
    for f in range(k):
        va_set = set(bins[f])
        tr_g = [int(g) for g in train_groups if int(g) not in va_set]
        va_g = [int(g) for g in train_groups if int(g) in va_set]
        if tr_g and va_g:
            splits.append((tr_g, va_g))
    return splits


def fit_predict_suffstats_two_penalty_ridge_cv(
    stats: GroupSuffStats,
    train_groups: Sequence[int],
    inner_cv_labels: Sequence[int],
    X_s_te: np.ndarray,
    X_m_te: Optional[np.ndarray] = None,
    lams_s: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_m: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    n_inner_folds: int = 5,
) -> Dict[str, object]:
    """
    Tune `(lam_s, lam_m)` strictly inside `train_groups` via gene-grouped inner CV
    using sufficient statistics `S_XX`, then solve on all `train_groups`.
    Uses Schur complement + `eigh(Z_ss)` when `p_m > 0` and `p_s <= 2000` (or fast Cholesky).
    """
    p_s = stats.sum_s.shape[1]
    p_m = stats.sum_m.shape[1]
    ls_arr = np.asarray(lams_s, dtype=np.float64)
    lm_arr = np.asarray(lams_m, dtype=np.float64)
    splits = _make_balanced_inner_folds(train_groups, inner_cv_labels, stats.n, n_folds=n_inner_folds)

    if not splits:
        best_ls = float(ls_arr[len(ls_arr) // 2])
        best_lm = float(lm_arr[len(lm_arr) // 2]) if p_m > 0 else 0.0
    elif p_m == 0:
        cv_sse = np.zeros(len(ls_arr), dtype=np.float64)
        idx_s = np.arange(p_s)
        for tr_g, va_g in splits:
            eq = _standardize_normal_equations(_aggregate_suffstats(stats, tr_g))
            val_agg = _aggregate_suffstats(stats, va_g)
            if p_s > 800 and len(ls_arr) <= 6:
                for i_s, ls in enumerate(ls_arr):
                    K = eq["Z_ss"].copy()
                    K[idx_s, idx_s] += ls
                    cf = cho_factor(K, lower=True, check_finite=False)
                    ws = cho_solve(cf, eq["rhs_s"], check_finite=False)
                    ws_raw, wm_raw, b_raw = _weights_to_raw(ws, np.zeros(0), eq)
                    cv_sse[i_s] += _sse_on_val_agg(ws_raw, wm_raw, b_raw, val_agg)
            else:
                ev, ec = sp_eigh(eq["Z_ss"], check_finite=False, driver="evd")
                ev = np.maximum(ev, 0.0)
                proj = ec.T @ eq["rhs_s"]
                w_all = ec @ (proj[:, None] / (ev[:, None] + ls_arr[None, :]))  # (p_s, n_ls)
                for i_s in range(len(ls_arr)):
                    ws_raw, wm_raw, b_raw = _weights_to_raw(w_all[:, i_s], np.zeros(0), eq)
                    cv_sse[i_s] += _sse_on_val_agg(ws_raw, wm_raw, b_raw, val_agg)
        best_ls = float(ls_arr[int(np.argmin(cv_sse))])
        best_lm = 0.0
    else:
        cv_sse = np.zeros((len(ls_arr), len(lm_arr)), dtype=np.float64)
        idx_m_diag = np.arange(p_m)
        for tr_g, va_g in splits:
            eq = _standardize_normal_equations(_aggregate_suffstats(stats, tr_g))
            val_agg = _aggregate_suffstats(stats, va_g)
            if p_s <= 2200:
                ev_s, Q_s = sp_eigh(eq["Z_ss"], check_finite=False, driver="evd")
                ev_s = np.maximum(ev_s, 0.0)
                qs = Q_s.T @ eq["rhs_s"]
                U = Q_s.T @ eq["Z_sm"]  # (p_s, p_m)
                for i_s, ls in enumerate(ls_arr):
                    inv_d = 1.0 / (ev_s + ls)
                    U_sc = U * inv_d[:, None]
                    S_mm = eq["Z_mm"] - U.T @ U_sc
                    S_mm = 0.5 * (S_mm + S_mm.T)
                    rhs_m_eff = eq["rhs_m"] - U_sc.T @ qs
                    if p_m > 200 and len(lm_arr) <= 8:
                        for i_m, lm in enumerate(lm_arr):
                            S_reg = S_mm.copy()
                            S_reg[idx_m_diag, idx_m_diag] += lm
                            cf_m = cho_factor(S_reg, lower=True, check_finite=False)
                            wm = cho_solve(cf_m, rhs_m_eff, check_finite=False)
                            ws = Q_s @ ((qs - U @ wm) * inv_d)
                            ws_raw, wm_raw, b_raw = _weights_to_raw(ws, wm, eq)
                            cv_sse[i_s, i_m] += _sse_on_val_agg(ws_raw, wm_raw, b_raw, val_agg)
                    else:
                        ev_m, Q_m = sp_eigh(S_mm, check_finite=False, driver="evd")
                        ev_m = np.maximum(ev_m, 0.0)
                        qm = Q_m.T @ rhs_m_eff
                        wm_all = Q_m @ (qm[:, None] / (ev_m[:, None] + lm_arr[None, :]))
                        ws_all = Q_s @ ((qs[:, None] - U @ wm_all) * inv_d[:, None])
                        for i_m in range(len(lm_arr)):
                            ws_raw, wm_raw, b_raw = _weights_to_raw(ws_all[:, i_m], wm_all[:, i_m], eq)
                            cv_sse[i_s, i_m] += _sse_on_val_agg(ws_raw, wm_raw, b_raw, val_agg)
            else:
                # Direct Cholesky fallback if p_s > 2200
                idx_s = np.arange(p_s)
                idx_m = p_s + np.arange(p_m)
                K_base = np.empty((p_s + p_m, p_s + p_m), dtype=np.float64)
                K_base[:p_s, :p_s] = eq["Z_ss"]
                K_base[:p_s, p_s:] = eq["Z_sm"]
                K_base[p_s:, :p_s] = eq["Z_sm"].T
                K_base[p_s:, p_s:] = eq["Z_mm"]
                rhs = np.concatenate([eq["rhs_s"], eq["rhs_m"]])
                for i_s, ls in enumerate(ls_arr):
                    Ks = K_base.copy()
                    Ks[idx_s, idx_s] += ls
                    for i_m, lm in enumerate(lm_arr):
                        Km = Ks.copy()
                        Km[idx_m, idx_m] += lm
                        cf = cho_factor(Km, lower=True, check_finite=False)
                        w = cho_solve(cf, rhs, check_finite=False)
                        ws_raw, wm_raw, b_raw = _weights_to_raw(w[:p_s], w[p_s:], eq)
                        cv_sse[i_s, i_m] += _sse_on_val_agg(ws_raw, wm_raw, b_raw, val_agg)
        bi, bj = np.unravel_index(int(np.argmin(cv_sse)), cv_sse.shape)
        best_ls = float(ls_arr[bi])
        best_lm = float(lm_arr[bj])

    sol = solve_suffstats_two_penalty(stats, train_groups, best_ls, best_lm, X_s_te, X_m_te)
    sol["best_lam_s"] = best_ls
    sol["best_lam_m"] = best_lm
    return sol


# ==============================================================================
# 6. Batched LOGO Multi-Sparsity Schur Null Evaluator (M = 1,593 Tables)
# ==============================================================================

_NULL_WORKER_STATE: Dict[str, object] = {}


def _init_null_bank_worker(state: Dict[str, object]) -> None:
    global _NULL_WORKER_STATE
    _NULL_WORKER_STATE = state


def _eval_logo_null_table_chunk(args: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
    """
    Evaluate a slice `[m0, m1)` of `tables_bank` across ALL 30 LOGO folds simultaneously
    via exact sufficient-statistics batched Schur two-penalty Ridge!
    Returns `(m0, m1, preds_all)` of shape `(m1 - m0, N_total_probes)`.
    """
    m0, m1 = args
    st = _NULL_WORKER_STATE
    tables_chunk = np.asarray(st["tables_bank"][m0:m1], dtype=np.float64)  # (Mc, 1024, 7)
    Mc = tables_chunk.shape[0]

    mut_gene_idx: np.ndarray = st["mut_gene_idx"]
    n_mut = len(mut_gene_idx)
    n_genes = int(st["n_genes"])
    X_s_mut: List[np.ndarray] = st["X_s_mut"]
    c5_mut: List[np.ndarray] = st["c5_mut"]
    psi_mut: np.ndarray = st["psi_mut"]  # (92, K_p) where col 0 == 1.0
    y_mut: List[np.ndarray] = st["y_mut"]
    base_stats: GroupSuffStats = st["base_stats"]  # per-gene stats for X_s alone
    mut_offsets: np.ndarray = st["mut_offsets"]
    n_total_obs = int(mut_offsets[-1])

    K_p = psi_mut.shape[1]
    d_dna = c5_mut[0].shape[1] * 7  # 28 for Shape_N1N4
    p_s = X_s_mut[0].shape[1]
    p_m = K_p * d_dna

    lams_s = np.asarray(st["lams_s"], dtype=np.float64)
    lams_m = np.asarray(st["lams_m"], dtype=np.float64)
    n_inner_folds = int(st["n_inner_folds"])

    # Contract tables_chunk once per mutant & accumulate per-gene sufficient stats for X_m
    T_flat = np.ascontiguousarray(tables_chunk.transpose(1, 0, 2).reshape(1024, Mc * 7), dtype=np.float64)
    n_pos = c5_mut[0].shape[1]

    gene_sum_m = np.zeros((n_genes, Mc, p_m), dtype=np.float64)
    gene_G_sm = np.zeros((n_genes, Mc, p_s, p_m), dtype=np.float64)
    gene_G_mm = np.zeros((n_genes, Mc, p_m, p_m), dtype=np.float64)
    gene_r_m = np.zeros((n_genes, Mc, p_m), dtype=np.float64)

    # Precompute contracted U_mut (Mc, n_i, d_dna) and accumulate Kronecker-factored blocks
    U_te_mut: List[np.ndarray] = []
    for m in range(n_mut):
        g = int(mut_gene_idx[m])
        c5_m = c5_mut[m]
        nm = len(c5_m)
        B_m = np.zeros((nm, n_pos, 1024), dtype=np.float64)
        rows_m = np.arange(nm, dtype=np.int32)[:, None]
        pos_m = np.arange(n_pos, dtype=np.int32)[None, :]
        B_m[rows_m, pos_m, c5_m] = 1.0
        U_m = (
            (B_m.reshape(nm * n_pos, 1024) @ T_flat)
            .reshape(nm, n_pos, Mc, 7)
            .transpose(2, 0, 1, 3)
            .reshape(Mc, nm, d_dna)
        )  # (Mc, nm, d_dna)
        U_te_mut.append(U_m)

        psi = psi_mut[m]  # (K_p,)
        psi_outer = np.outer(psi, psi)  # (K_p, K_p)

        sum_u = U_m.sum(axis=1)  # (Mc, d_dna)
        Xs_T_U = np.einsum("ni,mnj->mij", X_s_mut[m], U_m)  # (Mc, p_s, d_dna)
        U_T_U = np.einsum("mni,mnj->mij", U_m, U_m)  # (Mc, d_dna, d_dna)
        U_T_y = np.einsum("mni,n->mi", U_m, y_mut[m])  # (Mc, d_dna)

        gene_sum_m[g] += (psi[None, :, None] * sum_u[:, None, :]).reshape(Mc, p_m)
        gene_r_m[g] += (psi[None, :, None] * U_T_y[:, None, :]).reshape(Mc, p_m)
        gene_G_sm[g] += (psi[None, None, :, None] * Xs_T_U[:, :, None, :]).reshape(Mc, p_s, p_m)
        gene_G_mm[g] += (
            psi_outer[None, :, None, :, None] * U_T_U[:, None, :, None, :]
        ).transpose(0, 1, 2, 3, 4).reshape(Mc, p_m, p_m)

    tot_n = int(base_stats.n.sum())
    tot_sum_s = base_stats.sum_s.sum(axis=0)
    tot_sum_y = float(base_stats.sum_y.sum())
    tot_G_ss = base_stats.G_ss.sum(axis=0)
    tot_r_s = base_stats.r_s.sum(axis=0)
    tot_yyy = float(base_stats.yyy.sum())

    tot_sum_m = gene_sum_m.sum(axis=0)
    tot_G_sm = gene_G_sm.sum(axis=0)
    tot_G_mm = gene_G_mm.sum(axis=0)
    tot_r_m = gene_r_m.sum(axis=0)

    preds_out = np.empty((Mc, n_total_obs), dtype=np.float64)
    all_genes = list(range(n_genes))
    eye_m = np.eye(p_m, dtype=np.float64)[None, :, :]

    def _batched_std_and_solve_schur(
        n_tr: float,
        s_s: np.ndarray,
        s_m: np.ndarray,
        s_y: float,
        G_ss: np.ndarray,
        G_sm: np.ndarray,
        G_mm: np.ndarray,
        r_s: np.ndarray,
        r_m: np.ndarray,
        ev_s: np.ndarray,
        Q_s: np.ndarray,
        qs: np.ndarray,
        mu_s: np.ndarray,
        sd_s: np.ndarray,
        y_mean: float,
        dead_s: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        mu_m = s_m / n_tr  # (Mc, p_m)
        diag_mm = np.diagonal(G_mm, axis1=-2, axis2=-1)
        var_m = np.maximum(diag_mm / n_tr - mu_m * mu_m, 0.0)
        dead_m = var_m <= 1e-12
        sd_m = np.where(dead_m, 1.0, np.sqrt(var_m))

        Z_sm = (G_sm - n_tr * (mu_s[None, :, None] * mu_m[:, None, :])) / (
            sd_s[None, :, None] * sd_m[:, None, :]
        )
        Z_mm = (G_mm - n_tr * (mu_m[:, :, None] * mu_m[:, None, :])) / (
            sd_m[:, :, None] * sd_m[:, None, :]
        )
        Z_mm = 0.5 * (Z_mm + Z_mm.swapaxes(-1, -2))
        rhs_m = (r_m - y_mean * s_m) / sd_m
        if dead_s is not None and np.any(dead_s):
            Z_sm[:, dead_s, :] = 0.0
        if np.any(dead_m):
            Z_sm = np.where(dead_m[:, None, :], 0.0, Z_sm)
            Z_mm = np.where(dead_m[:, :, None] | dead_m[:, None, :], 0.0, Z_mm)
            rhs_m = np.where(dead_m, 0.0, rhs_m)
        U = np.einsum("ij,mjk->mik", Q_s.T, Z_sm)  # (Mc, p_s, p_m)
        return {
            "mu_m": mu_m,
            "sd_m": sd_m,
            "Z_mm": Z_mm,
            "rhs_m": rhs_m,
            "U": U,
        }

    for g_hold in range(n_genes):
        tr_genes = [g for g in all_genes if g != g_hold]
        splits = _make_balanced_inner_folds(
            tr_genes, all_genes, base_stats.n, n_folds=n_inner_folds
        )
        # Precompute base standardization on outer train genes
        tr_base_agg = {
            "n": tot_n - int(base_stats.n[g_hold]),
            "sum_s": tot_sum_s - base_stats.sum_s[g_hold],
            "sum_m": np.zeros(0),
            "sum_y": tot_sum_y - float(base_stats.sum_y[g_hold]),
            "G_ss": tot_G_ss - base_stats.G_ss[g_hold],
            "G_sm": np.zeros((p_s, 0)),
            "G_mm": np.zeros((0, 0)),
            "r_s": tot_r_s - base_stats.r_s[g_hold],
            "r_m": np.zeros(0),
            "yyy": tot_yyy - float(base_stats.yyy[g_hold]),
        }

        if not splits:
            best_ls = np.full(Mc, float(lams_s[len(lams_s) // 2]), dtype=np.float64)
            best_lm = np.full(Mc, float(lams_m[len(lams_m) // 2]), dtype=np.float64)
        else:
            cv_sse = np.zeros((Mc, len(lams_s), len(lams_m)), dtype=np.float64)
            for tr_in, va_in in splits:
                b_eq = _standardize_normal_equations(_aggregate_suffstats(base_stats, tr_in))
                b_val = _aggregate_suffstats(base_stats, va_in)
                ev_s, Q_s = sp_eigh(b_eq["Z_ss"], check_finite=False, driver="evd")
                ev_s = np.maximum(ev_s, 0.0)
                qs = Q_s.T @ b_eq["rhs_s"]

                idx_tr = np.asarray(tr_in, dtype=np.int64)
                idx_va = np.asarray(va_in, dtype=np.int64)
                n_a = float(base_stats.n[idx_tr].sum())
                m_eq = _batched_std_and_solve_schur(
                    n_a,
                    b_eq["mu_s"] * 0.0 + base_stats.sum_s[idx_tr].sum(axis=0),
                    gene_sum_m[idx_tr].sum(axis=0),
                    float(base_stats.sum_y[idx_tr].sum()),
                    base_stats.G_ss[idx_tr].sum(axis=0),
                    gene_G_sm[idx_tr].sum(axis=0),
                    gene_G_mm[idx_tr].sum(axis=0),
                    base_stats.r_s[idx_tr].sum(axis=0),
                    gene_r_m[idx_tr].sum(axis=0),
                    ev_s,
                    Q_s,
                    qs,
                    b_eq["mu_s"],
                    b_eq["sd_s"],
                    b_eq["y_mean"],
                    b_eq["dead_s"],
                )
                val_sum_m = gene_sum_m[idx_va].sum(axis=0)
                val_G_sm = gene_G_sm[idx_va].sum(axis=0)
                val_G_mm = gene_G_mm[idx_va].sum(axis=0)
                val_r_m = gene_r_m[idx_va].sum(axis=0)
                n_v = float(b_val["n"])

                for i_s, ls in enumerate(lams_s):
                    inv_d = 1.0 / (ev_s + ls)
                    U_sc = m_eq["U"] * inv_d[None, :, None]
                    S_mm = m_eq["Z_mm"] - np.einsum("mki,mkj->mij", m_eq["U"], U_sc)
                    S_mm = 0.5 * (S_mm + S_mm.swapaxes(-1, -2))
                    rhs_m_eff = m_eq["rhs_m"] - np.einsum("mki,k->mi", U_sc, qs)
                    for i_m, lm in enumerate(lams_m):
                        S_reg = S_mm + lm * eye_m
                        wm = np.linalg.solve(S_reg, rhs_m_eff[..., None])[..., 0]
                        ws_q = (qs[None, :] - np.einsum("mik,mk->mi", m_eq["U"], wm)) * inv_d[None, :]
                        ws = ws_q @ Q_s.T
                        ws_raw = ws / b_eq["sd_s"][None, :]
                        wm_raw = wm / m_eq["sd_m"]
                        b_raw = (
                            b_eq["y_mean"]
                            - np.einsum("j,mj->m", b_eq["mu_s"], ws_raw)
                            - np.einsum("mj,mj->m", m_eq["mu_m"], wm_raw)
                        )
                        # Exact validation SSE (omitting constant val_yyy)
                        q_ss = np.einsum("mi,ij,mj->m", ws_raw, b_val["G_ss"], ws_raw)
                        q_sm = 2.0 * np.einsum("mi,mij,mj->m", ws_raw, val_G_sm, wm_raw)
                        q_mm = np.einsum("mi,mij,mj->m", wm_raw, val_G_mm, wm_raw)
                        l_s = 2.0 * b_raw * (ws_raw @ b_val["sum_s"])
                        l_m = 2.0 * b_raw * np.einsum("mj,mj->m", wm_raw, val_sum_m)
                        yr = (ws_raw @ b_val["r_s"]) + np.einsum("mj,mj->m", wm_raw, val_r_m)
                        sse = (
                            q_ss + q_sm + q_mm + l_s + l_m + n_v * (b_raw * b_raw)
                            - 2.0 * (yr + b_raw * float(b_val["sum_y"]))
                        )
                        cv_sse[:, i_s, i_m] += sse

            flat_b = np.argmin(cv_sse.reshape(Mc, -1), axis=1)
            best_ls = lams_s[flat_b // len(lams_m)]
            best_lm = lams_m[flat_b % len(lams_m)]

        # Full outer LOGO fit for held-out gene `g_hold`
        b_eq_f = _standardize_normal_equations(tr_base_agg)
        ev_f, Q_f = sp_eigh(b_eq_f["Z_ss"], check_finite=False, driver="evd")
        ev_f = np.maximum(ev_f, 0.0)
        qf = Q_f.T @ b_eq_f["rhs_s"]
        m_eq_f = _batched_std_and_solve_schur(
            float(tr_base_agg["n"]),
            tr_base_agg["sum_s"],
            tot_sum_m - gene_sum_m[g_hold],
            tr_base_agg["sum_y"],
            tr_base_agg["G_ss"],
            tot_G_sm - gene_G_sm[g_hold],
            tot_G_mm - gene_G_mm[g_hold],
            tr_base_agg["r_s"],
            tot_r_m - gene_r_m[g_hold],
            ev_f,
            Q_f,
            qf,
            b_eq_f["mu_s"],
            b_eq_f["sd_s"],
            b_eq_f["y_mean"],
            b_eq_f["dead_s"],
        )
        inv_df = 1.0 / (ev_f[None, :] + best_ls[:, None])
        Uf_sc = m_eq_f["U"] * inv_df[:, :, None]
        Sf = m_eq_f["Z_mm"] - np.einsum("mki,mkj->mij", m_eq_f["U"], Uf_sc)
        Sf = 0.5 * (Sf + Sf.swapaxes(-1, -2)) + best_lm[:, None, None] * eye_m
        rhs_mf = m_eq_f["rhs_m"] - np.einsum("mki,k->mi", Uf_sc, qf)
        wmf = np.linalg.solve(Sf, rhs_mf[..., None])[..., 0]
        wsf_q = (qf[None, :] - np.einsum("mik,mk->mi", m_eq_f["U"], wmf)) * inv_df
        wsf = wsf_q @ Q_f.T

        ws_raw_f = wsf / b_eq_f["sd_s"][None, :]
        wm_raw_f = wmf / m_eq_f["sd_m"]
        b_raw_f = (
            b_eq_f["y_mean"]
            - np.einsum("j,mj->m", b_eq_f["mu_s"], ws_raw_f)
            - np.einsum("mj,mj->m", m_eq_f["mu_m"], wm_raw_f)
        )

        # Predict on all held-out mutants of `g_hold`
        wm_blocks = wm_raw_f.reshape(Mc, K_p, d_dna)
        for m in np.where(mut_gene_idx == g_hold)[0]:
            s0, s1 = int(mut_offsets[m]), int(mut_offsets[m + 1])
            pred_s = b_raw_f[:, None] + (ws_raw_f @ X_s_mut[m].T)  # (Mc, nm)
            wm_eff = np.einsum("k,mkd->md", psi_mut[m], wm_blocks)  # (Mc, d_dna)
            pred_m = np.einsum("md,mnd->mn", wm_eff, U_te_mut[m])  # (Mc, nm)
            preds_out[:, s0:s1] = pred_s + pred_m

    return m0, m1, preds_out


def evaluate_logo_null_bank_schur(
    tables_bank: np.ndarray,
    mut_gene_idx: np.ndarray,
    X_s_mut: Sequence[np.ndarray],
    c5_mut: Sequence[np.ndarray],
    psi_mut: np.ndarray,
    y_mut: Sequence[np.ndarray],
    lams_s: Sequence[float] = (10.0, 100.0, 1000.0),
    lams_m: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
    n_inner_folds: int = 3,
    n_workers: int = 8,
    chunk_size: int = 100,
) -> np.ndarray:
    """
    Evaluate all `M` tables in `tables_bank` (index 0 = F_true) under exact 30-fold LOGO
    with inner gene-grouped CV using batched Schur complements and per-gene `S_XX`.
    Returns `preds_all` of shape `(M, sum(len(y_mut[m])))`.
    """
    M = tables_bank.shape[0]
    n_genes = int(np.max(mut_gene_idx) + 1)
    n_mut = len(mut_gene_idx)

    # Aggregate X_s and y by gene for base sufficient statistics
    X_s_by_g: List[np.ndarray] = []
    y_by_g: List[np.ndarray] = []
    empty_m: List[np.ndarray] = []
    for g in range(n_genes):
        ms = [m for m in range(n_mut) if int(mut_gene_idx[m]) == g]
        X_s_by_g.append(np.vstack([X_s_mut[m] for m in ms]))
        y_by_g.append(np.concatenate([y_mut[m] for m in ms]))
        empty_m.append(np.zeros((len(y_by_g[-1]), 0), dtype=np.float64))
    base_stats = compute_group_suffstats(X_s_by_g, empty_m, y_by_g)

    mut_offsets = np.zeros(n_mut + 1, dtype=np.int64)
    for m in range(n_mut):
        mut_offsets[m + 1] = mut_offsets[m] + len(y_mut[m])
    n_total_obs = int(mut_offsets[-1])

    state = {
        "tables_bank": np.asarray(tables_bank, dtype=np.float64),
        "mut_gene_idx": np.asarray(mut_gene_idx, dtype=np.int32),
        "n_genes": n_genes,
        "X_s_mut": [np.asarray(x, dtype=np.float64) for x in X_s_mut],
        "c5_mut": [np.asarray(c, dtype=np.int32) for c in c5_mut],
        "psi_mut": np.asarray(psi_mut, dtype=np.float64),
        "y_mut": [np.asarray(y, dtype=np.float64) for y in y_mut],
        "base_stats": base_stats,
        "mut_offsets": mut_offsets,
        "lams_s": lams_s,
        "lams_m": lams_m,
        "n_inner_folds": n_inner_folds,
    }

    chunks = [(s, min(s + chunk_size, M)) for s in range(0, M, chunk_size)]
    preds_all = np.empty((M, n_total_obs), dtype=np.float64)

    if n_workers <= 1 or len(chunks) == 1:
        _init_null_bank_worker(state)
        for ch in chunks:
            m0, m1, block = _eval_logo_null_table_chunk(ch)
            preds_all[m0:m1] = block
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(n_workers, len(chunks)),
            initializer=_init_null_bank_worker,
            initargs=(state,),
        ) as pool:
            for m0, m1, block in pool.map(_eval_logo_null_table_chunk, chunks):
                preds_all[m0:m1] = block

    return preds_all


# ==============================================================================
# 7. Gene-Cluster Inference & Stratified Evaluation
# ==============================================================================

def compute_per_mutant_correlations(
    y_by_mut: Sequence[np.ndarray],
    pred_by_mut: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-mutant Spearman rho and Pearson r across finite probes."""
    n_mut = len(y_by_mut)
    rhos = np.zeros(n_mut, dtype=np.float64)
    pearsons = np.zeros(n_mut, dtype=np.float64)
    for m in range(n_mut):
        sr, pr = safe_spearman_pearson(y_by_mut[m], pred_by_mut[m])
        rhos[m] = sr
        pearsons[m] = pr
    return rhos, pearsons


def gene_cluster_bootstrap_paired(
    per_mut_a: np.ndarray,
    per_mut_b: np.ndarray,
    mut_gene_idx: Sequence[int],
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Paired 5,000x gene-cluster bootstrap (`G = len(unique(mut_gene_idx))` clusters)
    for `per_mut_a - per_mut_b`, reporting BOTH equal-gene-weight (`gene_mean`) and
    mutant-weighted (`mutant_mean`) cluster-bootstrapped CIs & p-values, plus exact
    gene-level paired Wilcoxon signed-rank test.
    """
    a = np.asarray(per_mut_a, dtype=np.float64)
    b = np.asarray(per_mut_b, dtype=np.float64)
    g_arr = np.asarray(mut_gene_idx, dtype=np.int32)
    uniq_g = np.unique(g_arr)
    G = len(uniq_g)

    gene_a = np.array([float(np.mean(a[g_arr == g])) for g in uniq_g], dtype=np.float64)
    gene_b = np.array([float(np.mean(b[g_arr == g])) for g in uniq_g], dtype=np.float64)
    gene_diff = gene_a - gene_b
    gene_counts = np.array([int(np.sum(g_arr == g)) for g in uniq_g], dtype=np.float64)
    gene_sum_diff = np.array([float(np.sum((a - b)[g_arr == g])) for g in uniq_g], dtype=np.float64)
    gene_sum_a = np.array([float(np.sum(a[g_arr == g])) for g in uniq_g], dtype=np.float64)
    gene_sum_b = np.array([float(np.sum(b[g_arr == g])) for g in uniq_g], dtype=np.float64)

    obs_gene_a = float(np.mean(gene_a))
    obs_gene_b = float(np.mean(gene_b))
    obs_gene_diff = float(np.mean(gene_diff))
    obs_mut_a = float(np.mean(a))
    obs_mut_b = float(np.mean(b))
    obs_mut_diff = float(np.mean(a - b))

    rng = np.random.default_rng(seed)
    boot_gene_diffs = np.empty(n_boot, dtype=np.float64)
    boot_mut_diffs = np.empty(n_boot, dtype=np.float64)
    for it in range(n_boot):
        pick = rng.integers(0, G, size=G)
        boot_gene_diffs[it] = float(np.mean(gene_diff[pick]))
        boot_mut_diffs[it] = float(np.sum(gene_sum_diff[pick]) / np.sum(gene_counts[pick]))

    # Exact paired gene-level Wilcoxon signed-rank test
    if G >= 5 and np.any(np.abs(gene_diff) > 1e-12):
        w_res = wilcoxon(gene_a, gene_b, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    else:
        w_stat, w_p = 0.0, 1.0

    p_gt0_gene = float((1.0 + np.sum(boot_gene_diffs <= 0.0)) / float(n_boot + 1))
    p_two_gene = float(min(1.0, 2.0 * min(p_gt0_gene, (1.0 + np.sum(boot_gene_diffs >= 0.0)) / float(n_boot + 1))))
    p_gt0_mut = float((1.0 + np.sum(boot_mut_diffs <= 0.0)) / float(n_boot + 1))
    p_two_mut = float(min(1.0, 2.0 * min(p_gt0_mut, (1.0 + np.sum(boot_mut_diffs >= 0.0)) / float(n_boot + 1))))

    return {
        "n_genes": int(G),
        "n_mutants": int(len(a)),
        "gene_mean_a": obs_gene_a,
        "gene_mean_b": obs_gene_b,
        "gene_mean_diff": obs_gene_diff,
        "gene_ci95": [float(np.quantile(boot_gene_diffs, 0.025)), float(np.quantile(boot_gene_diffs, 0.975))],
        "gene_p_boot_gt0": p_gt0_gene,
        "gene_p_boot_two_sided": p_two_gene,
        "gene_win_frac": float(np.mean(gene_diff > 0.0)),
        "gene_wilcoxon_stat": w_stat,
        "gene_wilcoxon_p": w_p,
        "mutant_mean_a": obs_mut_a,
        "mutant_mean_b": obs_mut_b,
        "mutant_mean_diff": obs_mut_diff,
        "mutant_ci95": [float(np.quantile(boot_mut_diffs, 0.025)), float(np.quantile(boot_mut_diffs, 0.975))],
        "mutant_p_boot_gt0": p_gt0_mut,
        "mutant_p_boot_two_sided": p_two_mut,
        "mutant_win_frac": float(np.mean(a > b)),
    }


def compute_stratified_metrics(
    per_mut_a: np.ndarray,
    per_mut_b: np.ndarray,
    mut_gene_idx: np.ndarray,
    mut_segments: np.ndarray,
    mut_kock_strata: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Compute paired cluster-bootstrap comparisons across pre-registered strata."""
    masks: Dict[str, np.ndarray] = {
        "all_92": np.ones(len(per_mut_a), dtype=bool),
        "seg_N_arm_1_9": mut_segments == "N-arm",
        "seg_H1_H2_linkers": np.isin(mut_segments, ["H1", "L12", "H2", "L23"]),
        "seg_H3_42_58": mut_segments == "H3",
        "kock_specificity_yes": mut_kock_strata == "specificity_yes",
        "kock_affinity_only": mut_kock_strata == "affinity_only",
        "kock_no_change": mut_kock_strata == "no_change",
    }
    out = {}
    for k, mk in masks.items():
        if int(np.sum(mk)) == 0:
            continue
        out[k] = gene_cluster_bootstrap_paired(
            per_mut_a[mk],
            per_mut_b[mk],
            mut_gene_idx[mk],
            n_boot=n_boot,
            seed=seed,
        )
    return out


# ==============================================================================
# 8. WT-vs-WT Chamber Bipartitions & Complete Dataset Builder
# ==============================================================================

def build_wt_vs_wt_bipartitions(
    im_cols: Sequence[str],
    gene_names: Sequence[str],
    min_chambers: int = 4,
) -> List[Dict[str, object]]:
    """
    Enumerate ALL unique balanced disjoint chamber bipartitions `(Half A, Half B)`
    for every REF gene with >= `min_chambers` independent chamber replicates.
    Canonical ordering prevents `(A, B)` / `(B, A)` duplicates on even sizes (`4v4`, `3v3`, `2v2`).
    """
    splits: List[Dict[str, object]] = []
    for g_idx, g in enumerate(gene_names):
        ref = f"{g}-REF"
        reps = sorted(
            [c for c in im_cols if c.startswith(ref + "_rep")],
            key=lambda c: int(c.rsplit("_rep", 1)[1]),
        )
        n_rep = len(reps)
        if n_rep < min_chambers:
            continue
        k_a = n_rep // 2  # 4v4 for 8, 3v3 for 6, 2v3 for 5, 2v2 for 4
        if n_rep % 2 == 0:
            # Fix index 0 in Half A to avoid duplicate symmetric complements -> C(n, n/2)/2
            for rest in itertools.combinations(range(1, n_rep), k_a - 1):
                a_idx = [0] + list(rest)
                b_idx = [i for i in range(n_rep) if i not in a_idx]
                splits.append(
                    {
                        "split_id": f"{g}__{len(a_idx)}v{len(b_idx)}_{len(splits):02d}",
                        "gene": g,
                        "gene_idx": g_idx,
                        "n_chambers": n_rep,
                        "half_a_cols": [reps[i] for i in a_idx],
                        "half_b_cols": [reps[i] for i in b_idx],
                    }
                )
        else:
            # Odd n_rep (e.g. 5 -> 2v3): all C(5, 2) = 10 splits
            for a_idx_tup in itertools.combinations(range(n_rep), k_a):
                a_idx = list(a_idx_tup)
                b_idx = [i for i in range(n_rep) if i not in a_idx]
                splits.append(
                    {
                        "split_id": f"{g}__{len(a_idx)}v{len(b_idx)}_{len(splits):02d}",
                        "gene": g,
                        "gene_idx": g_idx,
                        "n_chambers": n_rep,
                        "half_a_cols": [reps[i] for i in a_idx],
                        "half_b_cols": [reps[i] for i in b_idx],
                    }
                )
    return splits


def build_and_save_v7_dataset(
    raw_dir: Path = V7_RAW_DIR,
    proc_dir: Path = V7_PROC_DIR,
    k_pwm: int = K_PWM_DEFAULT,
) -> Dict[str, object]:
    """
    Build locked `v7_kock_dataset.npz` and `v7_kock_meta.json` cleanly from raw files
    with ZERO mutant leakage in probe alignment/selection and `-Inf -> NaN` masking.
    """
    proc_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pbm = rdata.read_rda(raw_dir / "pbm_8x60k_v1.rda")["pbm_8x60k_v1"]
    dBr = pbm.design[pbm.design["probeID"].astype(str).str.startswith("dBr")].reset_index(drop=True)
    im = rdata.read_rds(raw_dir / "IntensityMatrix.rds")
    seqs36 = im["seq"].astype(str).tolist()
    des36 = [str(s)[:36] for s in dBr["Sequence"]]
    if seqs36 != des36:
        raise RuntimeError("Probe chain mismatch between pbm_8x60k_v1 and IntensityMatrix.rds")

    sample_cols = list(im.columns[2:])
    im_clean = im.iloc[:, 2:].replace([-np.inf, np.inf], np.nan)

    pcc = {
        ln.split()[0]: float(ln.split()[1])
        for ln in (raw_dir / "psamI_pcc.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }

    # Resolve protein alignments & features
    ali_info = resolve_kock_hd_alignments(raw_dir)
    var_df: pd.DataFrame = ali_info["var_df"]
    genes: List[str] = ali_info["genes"]
    wt_by_gene: Dict[str, str] = ali_info["wt_by_gene"]
    msa_mean: np.ndarray = ali_info["msa_mean"]
    msa_V8: np.ndarray = ali_info["msa_V8"]
    blosum62 = fc.load_blosum62(raw_dir / "BLOSUM62.RData")

    prot_data = encode_prot_mutation_features(var_df, wt_by_gene, msa_mean, msa_V8, blosum62)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    mut_gene_idx = np.array([gene_to_idx[str(g)] for g in prot_data["genes"]], dtype=np.int32)

    wt_alignments_aa = np.array([wt_by_gene[g] for g in genes])
    wt_full_prot_feats = encode_full_hd_protein_features(wt_alignments_aa, msa_mean, msa_V8)
    mut_full_prot_feats = encode_full_hd_protein_features(prot_data["mut_seqs"], msa_mean, msa_V8)
    wt_for_mut_full = encode_full_hd_protein_features(prot_data["wt_seqs"], msa_mean, msa_V8)
    min_full_diff = float(np.min(np.linalg.norm(mut_full_prot_feats - wt_for_mut_full, axis=1)))
    if min_full_diff <= 1e-6:
        raise RuntimeError(f"Found mutant with near-zero full_prot difference: {min_full_diff}")

    # Load FamilyCode Step 1 predicted PSAMs aligned to our 92 mutants
    fc_npz = np.load(proc_dir / "familycode_pred_psams.npz", allow_pickle=True)
    fc_name_to_idx = {str(n): i for i, n in enumerate(fc_npz["mutant_names"])}
    fc_reorder = np.array([fc_name_to_idx[str(a)] for a in prot_data["alleles"]], dtype=np.int32)
    fc_pred_psams = fc_npz["pred_psams"][fc_reorder]
    fc_wt_psams = fc_npz["wt_psams"][fc_reorder]

    # Precompute 36-bp probe encodings
    seq_arr = encode_acgt_array(seqs36)
    all_wins, rc_arr = precompute_36bp_windows(seq_arr)
    n_genes = len(genes)
    n_mut = len(prot_data["alleles"])

    wt_best_rep = []
    wt_psam_log_4x8 = np.zeros((n_genes, 4, 8), dtype=np.float64)
    wt_psam_freq_4x8 = np.zeros((n_genes, 4, 8), dtype=np.float64)
    wt_z_all_probes = np.zeros((n_genes, len(seqs36)), dtype=np.float64)
    wt_interior_mask = np.zeros((n_genes, len(seqs36)), dtype=bool)

    probe_idx_f70 = np.zeros((n_genes, k_pwm), dtype=np.int32)
    win12_codes_f70 = np.zeros((n_genes, k_pwm, 12), dtype=np.int8)
    best_p_f70 = np.zeros((n_genes, k_pwm), dtype=np.int8)
    best_strand_f70 = np.zeros((n_genes, k_pwm), dtype=np.int8)
    f1_f70 = np.zeros((n_genes, k_pwm), dtype=np.float32)
    wt_tot_aff_f70 = np.zeros((n_genes, k_pwm), dtype=np.float32)
    z_wt_f70 = np.zeros((n_genes, k_pwm), dtype=np.float64)

    probe_idx_f50 = np.zeros((n_genes, k_pwm), dtype=np.int32)
    win12_codes_f50 = np.zeros((n_genes, k_pwm, 12), dtype=np.int8)
    best_p_f50 = np.zeros((n_genes, k_pwm), dtype=np.int8)
    best_strand_f50 = np.zeros((n_genes, k_pwm), dtype=np.int8)
    f1_f50 = np.zeros((n_genes, k_pwm), dtype=np.float32)
    wt_tot_aff_f50 = np.zeros((n_genes, k_pwm), dtype=np.float32)
    z_wt_f50 = np.zeros((n_genes, k_pwm), dtype=np.float64)

    win12_str_f70: List[List[str]] = []
    win12_str_f50: List[List[str]] = []
    wt_rep_cols_map: Dict[str, List[str]] = {}
    gene_coverage_stats: List[Dict[str, object]] = []

    print("[kock_pbm_v7] Aligning 36-bp probes strictly by WT-reference best-R2 PSAM...", flush=True)
    for g_idx, g in enumerate(genes):
        ref = f"{g}-REF"
        wt_reps = sorted(
            [c for c in sample_cols if c.startswith(ref + "_rep")],
            key=lambda c: int(c.rsplit("_rep", 1)[1]),
        )
        wt_rep_cols_map[g] = wt_reps
        best_rep = max(wt_reps, key=lambda r: pcc[r])
        wt_best_rep.append(best_rep)

        mat_log = np.loadtxt(raw_dir / "pwmsI" / f"{best_rep}.txt", dtype=np.float64)
        wt_psam_log_4x8[g_idx] = mat_log
        wt_psam_freq_4x8[g_idx] = fc.ddG2frequency(mat_log)

        best_p, best_strand, f1, total_aff = score_36bp_with_psam_tiebreak(
            seq_arr, mat_log, precomp_wins=all_wins
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt_z = im_clean[wt_reps].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
        wt_z_all_probes[g_idx] = wt_z
        interior = (best_p >= 2) & (best_p <= 26)
        wt_interior_mask[g_idx] = interior

        cand70 = np.where(interior & (f1 >= 0.70) & np.isfinite(wt_z))[0]
        cand50 = np.where(interior & (f1 >= 0.50) & np.isfinite(wt_z))[0]
        if len(cand70) < k_pwm or len(cand50) < k_pwm:
            raise RuntimeError(
                f"Gene {g} has insufficient probes ({len(cand70)} at f1>=0.70, {len(cand50)} at f1>=0.50)"
            )

        top70 = cand70[np.lexsort((cand70, -total_aff[cand70]))[:k_pwm]].astype(np.int32)
        top50 = cand50[np.lexsort((cand50, -total_aff[cand50]))[:k_pwm]].astype(np.int32)

        probe_idx_f70[g_idx] = top70
        win12_codes_f70[g_idx] = extract_12bp_window_codes(seq_arr, rc_arr, top70, best_p, best_strand)
        best_p_f70[g_idx] = best_p[top70].astype(np.int8)
        best_strand_f70[g_idx] = best_strand[top70].astype(np.int8)
        f1_f70[g_idx] = f1[top70].astype(np.float32)
        wt_tot_aff_f70[g_idx] = total_aff[top70].astype(np.float32)
        z_wt_f70[g_idx] = wt_z[top70]
        win12_str_f70.append(decode_acgt_array(win12_codes_f70[g_idx]))

        probe_idx_f50[g_idx] = top50
        win12_codes_f50[g_idx] = extract_12bp_window_codes(seq_arr, rc_arr, top50, best_p, best_strand)
        best_p_f50[g_idx] = best_p[top50].astype(np.int8)
        best_strand_f50[g_idx] = best_strand[top50].astype(np.int8)
        f1_f50[g_idx] = f1[top50].astype(np.float32)
        wt_tot_aff_f50[g_idx] = total_aff[top50].astype(np.float32)
        z_wt_f50[g_idx] = wt_z[top50]
        win12_str_f50.append(decode_acgt_array(win12_codes_f50[g_idx]))

        gene_coverage_stats.append(
            {
                "gene": g,
                "best_rep": best_rep,
                "n_wt_chambers": len(wt_reps),
                "n_avail_f70": int(len(cand70)),
                "n_avail_f50": int(len(cand50)),
                "k_pwm_selected": int(k_pwm),
            }
        )

    # Compute mutant endpoints (primary dZ, monotone P-spline dZ_spec, f1>=0.50, FamilyCode dE)
    print("[kock_pbm_v7] Computing replicate means, spline specificity residuals, and FamilyCode dE...", flush=True)
    z_mut_f70 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    dz_f70 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    dz_spec_f70 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    z_mut_f50 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    dz_f50 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    dz_spec_f50 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    fc_pred_de_f70 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    fc_pred_de_f50 = np.zeros((n_mut, k_pwm), dtype=np.float64)
    mut_rep_cols_map: Dict[str, List[str]] = {}

    for m in range(n_mut):
        allele = str(prot_data["alleles"][m])
        g_idx = int(mut_gene_idx[m])
        m_reps = sorted(
            [c for c in sample_cols if c.startswith(allele + "_rep")],
            key=lambda c: int(c.rsplit("_rep", 1)[1]),
        )
        mut_rep_cols_map[allele] = m_reps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mz_all = im_clean[m_reps].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)

        wz_all = wt_z_all_probes[g_idx]
        inter = wt_interior_mask[g_idx]

        p70 = probe_idx_f70[g_idx]
        z_mut_f70[m] = mz_all[p70]
        dz_f70[m] = z_mut_f70[m] - z_wt_f70[g_idx]
        dz_spec_f70[m] = fit_monotone_pspline_residual(
            wz_all[inter], mz_all[inter], z_wt_f70[g_idx], z_mut_f70[m]
        )

        p50 = probe_idx_f50[g_idx]
        z_mut_f50[m] = mz_all[p50]
        dz_f50[m] = z_mut_f50[m] - z_wt_f50[g_idx]
        dz_spec_f50[m] = fit_monotone_pspline_residual(
            wz_all[inter], mz_all[inter], z_wt_f50[g_idx], z_mut_f50[m]
        )

        fc_pred_de_f70[m] = fc.score_12bp_with_psam(
            win12_str_f70[g_idx], fc_pred_psams[m], mode="aligned_core"
        ) - fc.score_12bp_with_psam(
            win12_str_f70[g_idx], fc_wt_psams[m], mode="aligned_core"
        )
        fc_pred_de_f50[m] = fc.score_12bp_with_psam(
            win12_str_f50[g_idx], fc_pred_psams[m], mode="aligned_core"
        ) - fc.score_12bp_with_psam(
            win12_str_f50[g_idx], fc_wt_psams[m], mode="aligned_core"
        )

    # WT-vs-WT balanced disjoint chamber bipartitions across ALL 8 REF >= 4 genes (84 splits)
    print("[kock_pbm_v7] Generating all 84 balanced disjoint WT-vs-WT chamber bipartitions...", flush=True)
    bipartitions = build_wt_vs_wt_bipartitions(sample_cols, genes, min_chambers=4)
    n_wt_null = len(bipartitions)
    wt_null_gene_idx = np.zeros(n_wt_null, dtype=np.int32)
    wt_null_gene_names = []
    wt_null_split_ids = []
    wt_null_z_a_f70 = np.zeros((n_wt_null, k_pwm), dtype=np.float64)
    wt_null_z_b_f70 = np.zeros((n_wt_null, k_pwm), dtype=np.float64)
    wt_null_dz_f70 = np.zeros((n_wt_null, k_pwm), dtype=np.float64)
    wt_null_dz_spec_f70 = np.zeros((n_wt_null, k_pwm), dtype=np.float64)

    for s_i, sp in enumerate(bipartitions):
        g_idx = int(sp["gene_idx"])
        wt_null_gene_idx[s_i] = g_idx
        wt_null_gene_names.append(sp["gene"])
        wt_null_split_ids.append(sp["split_id"])
        assert not (set(sp["half_a_cols"]) & set(sp["half_b_cols"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            za_all = im_clean[sp["half_a_cols"]].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
            zb_all = im_clean[sp["half_b_cols"]].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
        p70 = probe_idx_f70[g_idx]
        inter = wt_interior_mask[g_idx]
        wt_null_z_a_f70[s_i] = za_all[p70]
        wt_null_z_b_f70[s_i] = zb_all[p70]
        wt_null_dz_f70[s_i] = za_all[p70] - zb_all[p70]
        wt_null_dz_spec_f70[s_i] = fit_monotone_pspline_residual(
            zb_all[inter], za_all[inter], zb_all[p70], za_all[p70]
        )

    # Kock 8-mer differential endpoint (MOESM7_ESM.txt, T3/D4 complete 5-mer context)
    print("[kock_pbm_v7] Loading MOESM7_ESM.txt 8-mer endpoint (preferring 'This study' for duplicates)...", flush=True)
    m7_path = raw_dir / "optional_esm" / "41467_2024_47396_MOESM7_ESM.txt"
    mut_allele_set = set(prot_data["alleles"].tolist())
    m7 = pd.read_csv(
        m7_path,
        sep="\t",
        usecols=["set", "allele", "seq", "contrastDifference", "contrastResidual", "specificityQ"],
    )
    m7 = m7[m7["allele"].isin(mut_allele_set)].copy()
    # SIX6-H141N and SIX6-T165A appear in both 'This study' (round3) and 'Barrera...'; prioritize 'This study'
    m7["_pri"] = np.where(m7["set"] == "This study", 0, 1)
    m7 = m7.sort_values(["allele", "_pri", "seq"]).drop_duplicates(subset=["allele", "seq"], keep="first")

    kmer8_seqs = np.array(sorted(m7["seq"].unique().tolist()), dtype="<U8")
    if len(kmer8_seqs) != 32896:
        raise RuntimeError(f"Expected 32,896 canonical 8-mers, got {len(kmer8_seqs)}")
    kmer8_to_idx = {s: i for i, s in enumerate(kmer8_seqs)}

    kmer8_contrast_diff = np.empty((n_mut, len(kmer8_seqs)), dtype=np.float32)
    kmer8_contrast_res = np.empty((n_mut, len(kmer8_seqs)), dtype=np.float32)
    kmer8_specificity_q = np.empty((n_mut, len(kmer8_seqs)), dtype=np.float32)
    for m, allele in enumerate(prot_data["alleles"]):
        sub = m7[m7["allele"] == str(allele)].sort_values("seq")
        if len(sub) != len(kmer8_seqs) or not np.array_equal(sub["seq"].to_numpy(dtype="<U8"), kmer8_seqs):
            raise RuntimeError(f"8-mer alignment mismatch for {allele}")
        kmer8_contrast_diff[m] = sub["contrastDifference"].to_numpy(dtype=np.float32)
        kmer8_contrast_res[m] = sub["contrastResidual"].to_numpy(dtype=np.float32)
        kmer8_specificity_q[m] = sub["specificityQ"].to_numpy(dtype=np.float32)

    # Orient all 32,896 8-mers per WT gene and extract complete T3/D4 pentamers (no flanks)
    kmer8_orient_codes = np.zeros((n_genes, len(kmer8_seqs), 8), dtype=np.int8)
    kmer8_t3d4_c5 = np.zeros((n_genes, len(kmer8_seqs), 2), dtype=np.int16)
    for g_idx in range(n_genes):
        oc, t3d4 = orient_8mers_with_psam(kmer8_seqs, wt_psam_log_4x8[g_idx])
        kmer8_orient_codes[g_idx] = oc
        kmer8_t3d4_c5[g_idx] = t3d4

    # Save NPZ & metadata JSON
    npz_path = proc_dir / "v7_kock_dataset.npz"
    np.savez_compressed(
        npz_path,
        gene_names=np.array(genes),
        wt_best_rep=np.array(wt_best_rep),
        wt_alignments_aa=wt_alignments_aa,
        wt_psam_log_4x8=wt_psam_log_4x8,
        wt_psam_freq_4x8=wt_psam_freq_4x8,
        wt_full_prot_feats=wt_full_prot_feats,
        probe_idx_f70=probe_idx_f70,
        win12_f70=np.array(win12_str_f70, dtype="<U12"),
        win12_codes_f70=win12_codes_f70,
        best_p_f70=best_p_f70,
        best_strand_f70=best_strand_f70,
        f1_f70=f1_f70,
        wt_tot_aff_f70=wt_tot_aff_f70,
        z_wt_f70=z_wt_f70,
        probe_idx_f50=probe_idx_f50,
        win12_f50=np.array(win12_str_f50, dtype="<U12"),
        win12_codes_f50=win12_codes_f50,
        best_p_f50=best_p_f50,
        best_strand_f50=best_strand_f50,
        f1_f50=f1_f50,
        wt_tot_aff_f50=wt_tot_aff_f50,
        z_wt_f50=z_wt_f50,
        mut_alleles=prot_data["alleles"],
        mut_genes=prot_data["genes"],
        mut_gene_idx=mut_gene_idx,
        mut_canonical_pos=prot_data["canon_pos"],
        mut_wt_aa=prot_data["wt_aas"],
        mut_mut_aa=prot_data["mut_aas"],
        mut_segments=prot_data["segments"],
        mut_contact_classes=prot_data["contact_classes"],
        mut_specificity_change=prot_data["specificity_change"],
        mut_affinity_change=prot_data["affinity_change"],
        mut_kock_strata=prot_data["kock_stratum"],
        mut_alignments_aa=prot_data["mut_seqs"],
        phi_prot=prot_data["phi_prot"],
        mut_full_prot_feats=mut_full_prot_feats,
        wt_for_mut_full_prot_feats=wt_for_mut_full,
        z_mut_f70=z_mut_f70,
        dz_f70=dz_f70,
        dz_spec_f70=dz_spec_f70,
        z_mut_f50=z_mut_f50,
        dz_f50=dz_f50,
        dz_spec_f50=dz_spec_f50,
        fc_pred_de_f70=fc_pred_de_f70,
        fc_pred_de_f50=fc_pred_de_f50,
        wt_null_split_ids=np.array(wt_null_split_ids),
        wt_null_gene_idx=wt_null_gene_idx,
        wt_null_gene_names=np.array(wt_null_gene_names),
        wt_null_z_a_f70=wt_null_z_a_f70,
        wt_null_z_b_f70=wt_null_z_b_f70,
        wt_null_dz_f70=wt_null_dz_f70,
        wt_null_dz_spec_f70=wt_null_dz_spec_f70,
        kmer8_seqs=kmer8_seqs,
        kmer8_orient_codes=kmer8_orient_codes,
        kmer8_t3d4_c5=kmer8_t3d4_c5,
        kmer8_contrast_diff=kmer8_contrast_diff,
        kmer8_contrast_res=kmer8_contrast_res,
        kmer8_specificity_q=kmer8_specificity_q,
    )

    finite_counts_f70 = [int(np.isfinite(dz_f70[m]).sum()) for m in range(n_mut)]
    finite_counts_f50 = [int(np.isfinite(dz_f50[m]).sum()) for m in range(n_mut)]
    meta = {
        "npz_path": str(npz_path),
        "n_genes": n_genes,
        "n_mutants": n_mut,
        "k_pwm": int(k_pwm),
        "phi_prot_dim": int(prot_data["phi_prot"].shape[1]),
        "full_prot_dim": int(wt_full_prot_feats.shape[1]),
        "min_mut_wt_full_prot_l2": min_full_diff,
        "finite_f70_min_med_max": [
            int(min(finite_counts_f70)),
            int(np.median(finite_counts_f70)),
            int(max(finite_counts_f70)),
        ],
        "finite_f50_min_med_max": [
            int(min(finite_counts_f50)),
            int(np.median(finite_counts_f50)),
            int(max(finite_counts_f50)),
        ],
        "n_wt_null_bipartitions": n_wt_null,
        "wt_null_bipartitions_by_gene": {
            g: int(sum(1 for sp in bipartitions if sp["gene"] == g))
            for g in sorted({sp["gene"] for sp in bipartitions})
        },
        "wt_null_bipartitions": bipartitions,
        "wt_rep_cols_by_gene": wt_rep_cols_map,
        "mut_rep_cols_by_allele": mut_rep_cols_map,
        "gene_coverage_stats": gene_coverage_stats,
        "kmer8_n_unique": int(len(kmer8_seqs)),
        "kmer8_duplicate_policy": "For SIX6-H141N and SIX6-T165A present in both 'This study' and 'Barrera et al., Science (2016)', kept 'This study' matching MOESM5 round3.",
    }
    meta_path = proc_dir / "v7_kock_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {npz_path} & {meta_path}", flush=True)
    return meta


if __name__ == "__main__":
    build_and_save_v7_dataset()
