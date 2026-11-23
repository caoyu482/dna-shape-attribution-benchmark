#!/usr/bin/env python3
"""
Core Data Curation, Sequence Registration, Two-Penalty GroupCV Solvers, and Null Controls
for biophys_inductive_gfm/v5 (Paired Perturbation Preference-Shift Prediction: delta log relKa).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gzip
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

V3_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v3/src")
V4_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v4/src")
for p in (V4_SRC, V3_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import null_generators_v4 as ng4  # noqa: E402
import step1_models as v3_sm  # noqa: E402

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
GSE65073_DIR = ROOT / "v3" / "data" / "raw" / "geo" / "GSE65073" / "extracted"

_TRANTAB = str.maketrans("ACGT", "TGCA")
_B2I = {"A": 0, "C": 1, "G": 2, "T": 3}

# Masked orthonormal base contrasts: rows 0..3 = A,C,G,T; row 4 = 'N' (unobserved flank padding = 0)
_MONO_H_MASKED = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
) * 0.5

# Precompute canonical dinucleotide map for global 10 RC-even counts
_DI_REMAP = np.zeros(16, dtype=np.int32)
_UNIQ_DI = sorted({
    min(x, ((3 - (x & 3)) << 2) | (3 - (x >> 2)))
    for x in range(16)
})
for _i_u, _v_u in enumerate(_UNIQ_DI):
    _DI_REMAP[_v_u] = _i_u


def revcomp(seq: str) -> str:
    return seq.translate(_TRANTAB)[::-1]


def canonical_rc(seq: str) -> str:
    rc = revcomp(seq)
    return seq if seq <= rc else rc


def encode_frame_sequences(seqs: Sequence[str], length: int = 14) -> np.ndarray:
    """Encode strings containing A/C/G/T/N (N -> 4) into int8 array."""
    arr = np.full((len(seqs), length), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            if ch in _B2I:
                arr[i, j] = _B2I[ch]
    return arr


def load_raw_kmer_dict(gz_path: Path) -> Tuple[Dict[str, float], Dict]:
    """Load exact positive relKa rows (relKa > 0) without collapsing or aligning."""
    d: Dict[str, float] = {}
    n_pos = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
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
            if not np.isfinite(ka) or ka <= 0.0:
                continue
            n_pos += 1
            d[km] = ka

    rc_diffs = [abs(d[k] - d[revcomp(k)]) for k in d if revcomp(k) in d]
    meta = {
        "file": gz_path.name,
        "positive_rows": n_pos,
        "unique_directed_strings": len(d),
        "rc_paired_rows": len(rc_diffs),
        "max_rc_diff": float(max(rc_diffs)) if rc_diffs else 0.0,
    }
    return d, meta


def score_exd_hox_10mer(w: str) -> float:
    """
    Sequence-only Exd-Hox alignment score on `A0 T1 G2 A3 T4 [N5 N6] A7 T8 [N9]`.
    Zero weight on `N5, N6, N9` or flanks. Max score = 12.0.
    """
    return (
        (1.0 if w[0] == "A" else 0.0)
        + 2.0 * ((w[1] == "T") + (w[2] == "G") + (w[3] == "A") + (w[4] == "T"))
        + 1.5 * ((w[7] == "A") + (w[8] == "T"))
    )


def align_12mer_to_frame14(seq12: str) -> Dict:
    """
    Deterministic sequence-only registration of a canonical 12-mer into fixed 14-bp frame:
      frame = [-2, -1, A0, T1, G2, A3, T4, N5, N6, A7, T8, N9, N10, N11]
    where missing flanking slots outside the 12-mer are encoded as 'N'.
    """
    best = None
    for st, s in ((0, seq12), (1, revcomp(seq12))):
        for off in (0, 1, 2):
            w = s[off : off + 10]
            sc = score_exd_hox_10mer(w)
            cand = (sc, -abs(off - 1), -st, off, s, w)
            if best is None or cand[:3] > best[:3]:
                best = cand
    sc, _, st_neg, off, s_orient, w10 = best
    # Build 14-char frame string
    pad_left = 2 - off
    pad_right = 14 - pad_left - 12
    frame_str = ("N" * pad_left) + s_orient + ("N" * pad_right)
    assert len(frame_str) == 14 and frame_str[2:12] == w10
    return {
        "align_score": float(sc),
        "strand": int(-st_neg),
        "offset": int(off),
        "oriented_12mer": s_orient,
        "frame14": frame_str,
        "core10": canonical_rc(w10),
        "pocket6": canonical_rc(w10[3:9]),   # A3 T4 N5 N6 A7 T8
        "spacer2": canonical_rc(w10[5:7]),   # N5 N6
        "w10_fwd": w10,
    }


def _extract_dsdna_val(d: Dict[str, float], canon_km: str) -> float:
    v1 = d.get(canon_km)
    v2 = d.get(revcomp(canon_km))
    if v1 is not None and v2 is not None:
        return 0.5 * (v1 + v2)
    return float(v1 if v1 is not None else v2)


def load_paired_12mer_dataset(F_true: np.ndarray) -> Dict:
    """
    Load exact string-intersected GSE65073 12mer tables (`ScrWT`, `ScrR5A`, `ScrR3A`,
    plus secondary `ScrH12AR3A`, `AntpWT`, `AntpHQT` and 14mer marginal check).
    """
    wt12, m_wt = load_raw_kmer_dict(GSE65073_DIR / "GSM1586782_ScrWT_Exd_12mer.txt.gz")
    r5a12, m_r5a = load_raw_kmer_dict(GSE65073_DIR / "GSM1586783_ScrR5A_Exd_12mer.txt.gz")
    r3a12, m_r3a = load_raw_kmer_dict(GSE65073_DIR / "GSM1586784_ScrR3A_Exd_12mer.txt.gz")
    h12ar3a12, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586786_ScrH-12AR3A_Exd_12mer.txt.gz")
    awt12, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586789_AntpWT_Exd_12mer.txt.gz")
    ahqt12, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586793_AntpHQT_Exd_12mer.txt.gz")

    s_wt_r5a = set(wt12) & set(r5a12)
    s_wt_r3a = set(wt12) & set(r3a12)
    s_3way_dir = s_wt_r5a & s_wt_r3a

    ds_wt_r5a = sorted({canonical_rc(k) for k in s_wt_r5a})
    ds_wt_r3a = sorted({canonical_rc(k) for k in s_wt_r3a})
    ds_3way = sorted({canonical_rc(k) for k in s_3way_dir})

    records = []
    for k in ds_3way:
        info = align_12mer_to_frame14(k)
        v_wt = _extract_dsdna_val(wt12, k)
        v_r5a = _extract_dsdna_val(r5a12, k)
        v_r3a = _extract_dsdna_val(r3a12, k)
        info.update({
            "canon12": k,
            "relka_wt": v_wt,
            "relka_r5a": v_r5a,
            "relka_r3a": v_r3a,
            "delta_r5a": float(np.log(v_r5a) - np.log(v_wt)),
            "delta_r3a": float(np.log(v_r3a) - np.log(v_wt)),
            "relka_h12ar3a": _extract_dsdna_val(h12ar3a12, k) if (k in h12ar3a12 or revcomp(k) in h12ar3a12) else None,
            "relka_antp_wt": _extract_dsdna_val(awt12, k) if (k in awt12 or revcomp(k) in awt12) else None,
            "relka_antp_hqt": _extract_dsdna_val(ahqt12, k) if (k in ahqt12 or revcomp(k) in ahqt12) else None,
        })
        records.append(info)

    # Selection bias audit: ds_wt_r5a \ ds_3way vs ds_3way
    s3_set = set(ds_3way)
    excl_r5a = [k for k in ds_wt_r5a if k not in s3_set]
    wt_excl = np.array([_extract_dsdna_val(wt12, k) for k in excl_r5a], dtype=np.float64)
    dr5_excl = np.array([np.log(_extract_dsdna_val(r5a12, k)) - np.log(_extract_dsdna_val(wt12, k)) for k in excl_r5a], dtype=np.float64)
    wt_inc = np.array([r["relka_wt"] for r in records], dtype=np.float64)
    dr5_inc = np.array([r["delta_r5a"] for r in records], dtype=np.float64)

    # MGW at P_N5/P_N6 comparison on aligned subset
    al_inc = [r for r in records if r["align_score"] >= 8.0]
    al_excl = [align_12mer_to_frame14(k) for k in excl_r5a]
    al_excl = [r for r in al_excl if r["align_score"] >= 8.0]

    def _mean_mgw_n5n6(recs: List[Dict]) -> Tuple[float, float]:
        if not recs:
            return 0.0, 0.0
        fc = encode_frame_sequences([r["frame14"] for r in recs], 14)
        c5 = _frame_c5_n5n6(fc)
        mgw = 0.5 * (F_true[c5[:, 0], 0] + F_true[c5[:, 1], 0])
        return float(np.mean(mgw)), float(np.std(mgw))

    mgw_inc_m, mgw_inc_s = _mean_mgw_n5n6(al_inc)
    mgw_exc_m, mgw_exc_s = _mean_mgw_n5n6(al_excl)

    # Consistency ceilings: (i) 12mer vs 14mer marginal, (ii) cross-offset off=0 vs off=1
    wt14, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586782_ScrWT_Exd_14mer.txt.gz")
    r5a14, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586783_ScrR5A_Exd_14mer.txt.gz")
    r3a14, _ = load_raw_kmer_dict(GSE65073_DIR / "GSM1586784_ScrR3A_Exd_14mer.txt.gz")
    s14_3 = sorted({canonical_rc(k) for k in (set(wt14) & set(r5a14) & set(r3a14))})
    sub12_r5, sub12_r3 = {}, {}
    for k14 in s14_3:
        vw = _extract_dsdna_val(wt14, k14)
        v5 = _extract_dsdna_val(r5a14, k14)
        v3 = _extract_dsdna_val(r3a14, k14)
        d5 = float(np.log(v5) - np.log(vw))
        d3 = float(np.log(v3) - np.log(vw))
        for st in (0, 1, 2):
            sub = canonical_rc(k14[st : st + 12])
            sub12_r5.setdefault(sub, []).append(d5)
            sub12_r3.setdefault(sub, []).append(d3)

    c14_rows = [r for r in al_inc if r["canon12"] in sub12_r5 and len(sub12_r5[r["canon12"]]) >= 2]
    r5_12 = np.array([r["delta_r5a"] for r in c14_rows])
    r5_14 = np.array([float(np.mean(sub12_r5[r["canon12"]])) for r in c14_rows])
    r3_12 = np.array([r["delta_r3a"] for r in c14_rows])
    r3_14 = np.array([float(np.mean(sub12_r3[r["canon12"]])) for r in c14_rows])

    core_off: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for r in al_inc:
        core_off.setdefault((r["w10_fwd"], r["offset"]), []).append((r["delta_r5a"], r["delta_r3a"]))
    p01_5, p01_3 = [], []
    for w10 in {r["w10_fwd"] for r in al_inc}:
        if (w10, 0) in core_off and (w10, 1) in core_off:
            p01_5.append((np.mean([x[0] for x in core_off[(w10, 0)]]), np.mean([x[0] for x in core_off[(w10, 1)]])))
            p01_3.append((np.mean([x[1] for x in core_off[(w10, 0)]]), np.mean([x[1] for x in core_off[(w10, 1)]])))
    a5, a3 = np.array(p01_5), np.array(p01_3)

    dr5_all = np.array([r["delta_r5a"] for r in al_inc])
    dr3_all = np.array([r["delta_r3a"] for r in al_inc])
    dr5_unfilt = np.array([r["delta_r5a"] for r in records])
    dr3_unfilt = np.array([r["delta_r3a"] for r in records])

    audit = {
        "raw_file_meta": {"ScrWT": m_wt, "ScrR5A": m_r5a, "ScrR3A": m_r3a},
        "directed_intersection_counts": {
            "WT_and_R5A": len(s_wt_r5a),
            "WT_and_R3A": len(s_wt_r3a),
            "3way": len(s_3way_dir),
        },
        "dsdna_counts": {
            "WT_and_R5A": len(ds_wt_r5a),
            "WT_and_R3A": len(ds_wt_r3a),
            "3way_unfiltered": len(records),
            "aligned_sc8": len(al_inc),
            "strict_sc10": sum(r["align_score"] >= 10.0 for r in records),
        },
        "selection_bias_audit_WT_R5A_vs_3way": {
            "excluded_count_dsdna": len(excl_r5a),
            "excluded_relka_wt_median": float(np.median(wt_excl)),
            "included_relka_wt_median": float(np.median(wt_inc)),
            "excluded_delta_r5a_mean_sd": [float(np.mean(dr5_excl)), float(np.std(dr5_excl))],
            "included_delta_r5a_mean_sd": [float(np.mean(dr5_inc)), float(np.std(dr5_inc))],
            "aligned_excl_count": len(al_excl),
            "aligned_excl_mgw_mean_sd": [mgw_exc_m, mgw_exc_s],
            "aligned_incl_mgw_mean_sd": [mgw_inc_m, mgw_inc_s],
        },
        "empirical_consistency_ceilings": {
            "12mer_vs_14mer_marginal_N": len(c14_rows),
            "12mer_vs_14mer_r5a_spearman": float(spearmanr(r5_12, r5_14).statistic),
            "12mer_vs_14mer_r3a_spearman": float(spearmanr(r3_12, r3_14).statistic),
            "cross_offset_0vs1_cores_N": len(a5),
            "cross_offset_0vs1_r5a_spearman": float(spearmanr(a5[:, 0], a5[:, 1]).statistic),
            "cross_offset_0vs1_r3a_spearman": float(spearmanr(a3[:, 0], a3[:, 1]).statistic),
            "inter_mutant_r5a_vs_r3a_aligned_spearman": float(spearmanr(dr5_all, dr3_all).statistic),
            "inter_mutant_r5a_vs_r3a_aligned_pearson": float(pearsonr(dr5_all, dr3_all).statistic),
            "inter_mutant_r5a_vs_r3a_unfiltered_spearman": float(spearmanr(dr5_unfilt, dr3_unfilt).statistic),
        },
    }
    return {"records": records, "audit": audit}


def _frame_c5_n5n6(frame_codes: np.ndarray) -> np.ndarray:
    """Return (N, 2) base-4 pentamer indices for locked slots P_N5=[5:10] and P_N6=[6:11]."""
    assert np.all(frame_codes[:, 5:11] < 4), "Expected A3..T8 positions to be fully observed."
    c_list = []
    for st in (5, 6):
        c5 = (
            (
                (
                    ((frame_codes[:, st].astype(np.int32) << 2) | frame_codes[:, st + 1]) << 2
                ) | frame_codes[:, st + 2]
            ) << 2 | frame_codes[:, st + 3]
        ) << 2 | frame_codes[:, st + 4]
        c_list.append(c5)
    return np.column_stack(c_list)


def make_core10_disjoint_split(records: List[Dict], dev_frac: float = 0.60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic, zero-label 60% Dev / 40% Held-Out split partitioned strictly by
    canonical RC 10-bp core (`core10`). Verifies zero core or RC overlap.
    """
    by_core: Dict[str, List[int]] = {}
    core_sc: Dict[str, float] = {}
    for idx, r in enumerate(records):
        c = r["core10"]
        by_core.setdefault(c, []).append(idx)
        core_sc[c] = max(core_sc.get(c, 0.0), r["align_score"])
    ordered = sorted(by_core.keys(), key=lambda c: (-int(core_sc[c] >= 10.0), -len(by_core[c]), c))
    dev_cores, held_cores = set(), set()
    # Deterministic modulo-5 stratification: {0, 2, 4} -> 60% Dev, {1, 3} -> 40% Held-Out
    for i, c in enumerate(ordered):
        if (i % 5) in (0, 2, 4):
            dev_cores.add(c)
        else:
            held_cores.add(c)
    assert not (dev_cores & held_cores)
    dev_idx = np.array([i for i, r in enumerate(records) if r["core10"] in dev_cores], dtype=np.int32)
    held_idx = np.array([i for i, r in enumerate(records) if r["core10"] in held_cores], dtype=np.int32)
    dev_12 = {r["canon12"] for r in (records[i] for i in dev_idx)}
    held_12 = {r["canon12"] for r in (records[i] for i in held_idx)}
    assert not (dev_12 & held_12)
    assert not ({records[i]["core10"] for i in dev_idx} & {records[i]["core10"] for i in held_idx})
    return dev_idx, held_idx


def make_balanced_group_folds(group_keys: Sequence[str], n_folds: int = 5) -> np.ndarray:
    """
    Greedy size-balanced k-fold assignment across canonical string groups, ensuring zero group
    leakage even when one group contains >15% of sequences (e.g., Pocket6 ATGAT[NN]AT).
    """
    by_g: Dict[str, List[int]] = {}
    for i, g in enumerate(group_keys):
        by_g.setdefault(g, []).append(i)
    ordered = sorted(by_g.keys(), key=lambda g: (-len(by_g[g]), g))
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    fold_ids = np.empty(len(group_keys), dtype=np.int32)
    for g in ordered:
        f_assign = int(np.argmin(fold_sizes))
        fold_sizes[f_assign] += len(by_g[g])
        for i in by_g[g]:
            fold_ids[i] = f_assign
    for f in range(n_folds):
        g_in = {group_keys[i] for i in range(len(group_keys)) if fold_ids[i] == f}
        g_out = {group_keys[i] for i in range(len(group_keys)) if fold_ids[i] != f}
        assert not (g_in & g_out)
    return fold_ids


def build_feature_suite(records: List[Dict]) -> Dict[str, object]:
    """Build exact registered sequence feature hierarchy M1..M8 + B_n5n6."""
    N = len(records)
    frame_codes = encode_frame_sequences([r["frame14"] for r in records], 14)
    offsets = np.array([r["offset"] for r in records], dtype=np.int32)
    obs12 = v3_sm.encode_sequences_acgt([r["canon12"] for r in records])

    # Mononucleotide contrasts on 14-bp frame (42 dims) + offset one-hot (3 dims) + global GC/GC^2 (2 dims)
    mf = _MONO_H_MASKED[frame_codes]  # (N, 14, 3)
    mono_flat = mf.reshape(N, 42)
    off_oh = np.eye(3, dtype=np.float64)[offsets]
    gc = ((obs12 == 1) | (obs12 == 2)).mean(axis=1).astype(np.float64)
    X_m1 = np.hstack([mono_flat, off_oh, gc[:, None], (gc * gc)[:, None]])  # (N, 47)

    # Adjacent dinucleotides on 14-bp frame (13 x 9 = 117; 0 automatically if either base is N!)
    di_adj = np.einsum("npa,npb->npab", mf[:, :-1, :], mf[:, 1:, :]).reshape(N, 13 * 9)
    c2_obs = (obs12[:, :-1].astype(np.int32) << 2) | obs12[:, 1:]
    c2_obs_rc = ((3 - obs12[:, 1:]).astype(np.int32) << 2) | (3 - obs12[:, :-1])
    c2_mapped = _DI_REMAP[np.minimum(c2_obs, c2_obs_rc)]
    di_bulk = np.zeros((N, len(_UNIQ_DI)), dtype=np.float64)
    for i in range(N):
        di_bulk[i] = np.bincount(c2_mapped[i], minlength=len(_UNIQ_DI)) / 11.0
    X_m2 = np.hstack([X_m1, di_adj, di_bulk])  # (N, 174)

    # All pairwise base-base interactions across the 10 always-observed core bases frame[2:12]
    # Note adjacent core pairs (q == p+1) are already in M2; keep all non-adjacent pairs q >= p+2 (36 * 9 = 324)
    mf_core = mf[:, 2:12, :]  # (N, 10, 3)
    pair_blocks = []
    for p in range(10):
        for q in range(p + 2, 10):
            pair_blocks.append(np.einsum("na,nb->nab", mf_core[:, p, :], mf_core[:, q, :]))
    X_m3 = np.hstack([X_m2, np.concatenate(pair_blocks, axis=1).reshape(N, -1)])  # (N, 174 + 324 = 498)

    # M4: Exact positional 4-mers overlapping N5/N6 ([5:9], [6:10], [7:11]) -> (N, 768) float32
    X_m4_mod = np.zeros((N, 3 * 256), dtype=np.float32)
    rows = np.arange(N, dtype=np.int32)
    for k_idx, st in enumerate((5, 6, 7)):
        c4 = (((((frame_codes[:, st].astype(np.int32) << 2) | frame_codes[:, st + 1]) << 2) | frame_codes[:, st + 2]) << 2 | frame_codes[:, st + 3])
        X_m4_mod[rows, k_idx * 256 + c4] = 1.0

    # M5 & M5b: Exact positional 5-mers at locked (P_N5, P_N6) -> (N, 2048) & shared (N, 1024)
    c5_n5n6 = _frame_c5_n5n6(frame_codes)
    B_n5n6 = np.zeros((N, 2, 1024), dtype=np.float32)
    B_n5n6[rows, 0, c5_n5n6[:, 0]] = 1.0
    B_n5n6[rows, 1, c5_n5n6[:, 1]] = 1.0
    X_m5_mod = B_n5n6.reshape(N, 2048)
    X_m5b_mod = (B_n5n6[:, 0, :] + B_n5n6[:, 1, :]).astype(np.float32)

    # M6: Full positional 4-mers (11 slots) + positional 5-mers (10 slots) across 14-bp frame
    # Note: zero indicator whenever any position in slot is 'N' (=4)
    full_blocks = []
    for st in range(11):
        blk = np.zeros((N, 256), dtype=np.float32)
        seg = frame_codes[:, st : st + 4]
        valid = np.all(seg < 4, axis=1)
        if np.any(valid):
            c4 = (((((seg[valid, 0].astype(np.int32) << 2) | seg[valid, 1]) << 2) | seg[valid, 2]) << 2 | seg[valid, 3])
            blk[rows[valid], c4] = 1.0
        full_blocks.append(blk)
    for st in range(10):
        blk = np.zeros((N, 1024), dtype=np.float32)
        seg = frame_codes[:, st : st + 5]
        valid = np.all(seg < 4, axis=1)
        if np.any(valid):
            c5 = ((((((seg[valid, 0].astype(np.int32) << 2) | seg[valid, 1]) << 2) | seg[valid, 2]) << 2 | seg[valid, 3]) << 2 | seg[valid, 4])
            blk[rows[valid], c5] = 1.0
        full_blocks.append(blk)
    X_m6_mod = np.hstack(full_blocks)

    # Compact HGBDT matrix: M2 (174) + categorical integer bases frame[2:12] + flanking ints + spacer pairs + alignment score
    X_hgb_extra = np.column_stack([
        frame_codes.astype(np.float64),
        ((frame_codes[:, 7].astype(np.int32) << 2) | frame_codes[:, 8]).astype(np.float64),  # N5N6 16-state
        ((frame_codes[:, 7].astype(np.int32) << 4) | (frame_codes[:, 8].astype(np.int32) << 2) | frame_codes[:, 11]).astype(np.float64),  # N5N6N9 64-state
        np.array([r["align_score"] for r in records], dtype=np.float64),
    ])
    X_hgb = np.hstack([X_m2, X_hgb_extra])

    return {
        "frame_codes": frame_codes,
        "c5_n5n6": c5_n5n6,
        "B_n5n6": B_n5n6,
        "X_m1": X_m1,
        "X_m2": X_m2,
        "X_m3": X_m3,
        "X_m4_mod": X_m4_mod,
        "X_m5_mod": X_m5_mod,
        "X_m5b_mod": X_m5b_mod,
        "X_m6_mod": X_m6_mod,
        "X_hgb": X_hgb,
    }


class Sparse4merSurrogateNull:
    """
    Covariance- & WS_G2-matched sparse 4-mer step surrogate table generator.
    Generates 7 channels purely from sparse 4-mer hydrogen-bond / base-step pockets,
    projects out WS_G2 in L^2(p_emp), matches exact L^2(p_emp) 7x7 residual covariance
    Sigma_perp(F_true), and restores exact Phi_G2(F_true).
    """

    def __init__(
        self,
        F_true: np.ndarray,
        p_raw: np.ndarray,
        n_total_obs: float,
        n_pockets_per_ch: int = 12,
    ):
        self.F_true = np.asarray(F_true, dtype=np.float64)
        self.n_pockets = int(n_pockets_per_ch)
        crt = ng4.ShiftGCExactCRTNull(self.F_true, p_raw, conditioning="WS_G2", n_total_obs=n_total_obs)
        self.p_emp = crt.p_emp
        self.Phi_fix = crt.Phi_fix
        self.base_proj = crt.base_proj

        R_true = self.F_true - self.base_proj
        Sigma_true = (R_true.T * self.p_emp[None, :]) @ R_true
        evals, evecs = np.linalg.eigh(Sigma_true)
        evals = np.maximum(evals, 1e-10)
        self.L_color = (evecs * np.sqrt(evals)[None, :]) @ evecs.T

        # Precompute 4-mer left/right indices for all 1024 pentamers (and RC symmetry)
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
            s_even = 0.5 * (raw_v + raw_v[ng4.RC_5])
            R_raw[:, ch] = s_even - self.Phi_fix @ (wP @ s_even)

        cov_raw = (R_raw.T * self.p_emp[None, :]) @ R_raw
        ev, ec = np.linalg.eigh(cov_raw)
        inv_sq = ec @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-10))) @ ec.T
        R_matched = R_raw @ inv_sq @ self.L_color
        return self.base_proj + R_matched


# ==============================================================================
# Exact GroupCV Solvers (Single-Penalty, Two-Penalty Dual/Primal, Poly3, HGBDT, Batched Null Schur)
# ==============================================================================

def _safe_group_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    k = min(n_folds, len(uniq))
    if k < 2 or len(y) < 12:
        return []
    return list(GroupKFold(n_splits=k).split(X, y, groups))


def fit_predict_ridge_groupcv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
    n_folds: int = 5,
    return_oof: bool = False,
) -> Dict:
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-8)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    al_arr = np.asarray(alphas, dtype=np.float64)

    n_tr, p = Z_tr.shape
    splits = _safe_group_splits(Z_tr, yc, groups_tr, n_folds=n_folds)
    oof_best = np.full(n_tr, y_mean, dtype=np.float64)
    if not splits:
        best_a = 30.0
    else:
        cv_mse = np.zeros(len(al_arr), dtype=np.float64)
        oof_grid = np.empty((len(al_arr), n_tr), dtype=np.float64) if return_oof else None
        for tr_i, va_i in splits:
            Za, Zv = Z_tr[tr_i], Z_tr[va_i]
            ym_f = float(y_tr[tr_i].mean())
            ya, yv = y_tr[tr_i] - ym_f, y_tr[va_i] - ym_f
            if Za.shape[0] >= Za.shape[1]:
                ev, ec = sp_eigh(Za.T @ Za, check_finite=False, driver="evd")
                ev = np.maximum(ev, 0.0)
                proj = ec.T @ (Za.T @ ya)
                preds = (Zv @ ec) @ (proj[:, None] / (ev[:, None] + al_arr[None, :]))
            else:
                ev, ec = sp_eigh(Za @ Za.T, check_finite=False, driver="evd")
                ev = np.maximum(ev, 0.0)
                proj = ec.T @ ya
                preds = ((Zv @ Za.T) @ ec) @ (proj[:, None] / (ev[:, None] + al_arr[None, :]))
            err = preds - yv[:, None]
            cv_mse += np.sum(err * err, axis=0)
            if oof_grid is not None:
                oof_grid[:, va_i] = ym_f + preds.T
        best_idx = int(np.argmin(cv_mse))
        best_a = float(al_arr[best_idx])
        if oof_grid is not None:
            oof_best = oof_grid[best_idx]

    if n_tr >= p:
        K = Z_tr.T @ Z_tr
        K[np.diag_indices(p)] += best_a
        cf = cho_factor(K, lower=True, check_finite=False)
        w = cho_solve(cf, Z_tr.T @ yc, check_finite=False)
        pred_te = y_mean + Z_te @ w
    else:
        G = Z_tr @ Z_tr.T
        G[np.diag_indices(n_tr)] += best_a
        cf = cho_factor(G, lower=True, check_finite=False)
        dual = cho_solve(cf, yc, check_finite=False)
        pred_te = y_mean + (Z_te @ Z_tr.T) @ dual

    out = {"pred_te": pred_te, "best_alpha": best_a}
    if return_oof:
        out["pred_oof"] = oof_best
    return out


def fit_predict_two_penalty_ridge_groupcv(
    X_seq_tr: np.ndarray,
    X_mod_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_seq_te: np.ndarray,
    X_mod_te: np.ndarray,
    lams_seq: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_mod: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_folds: int = 5,
    return_oof: bool = False,
) -> Dict:
    """
    Exact two-penalty GroupCV Ridge solving:
      min ||yc - Z_seq w_s - Z_mod w_m||^2 + lam_s ||w_s||^2 + lam_m ||w_m||^2
    Uses dual Gram K_s + (lam_s / lam_m) K_m trick when p_s + p_m > 1200 and n <= 5000,
    or primal eigendecomposition when p_s + p_m <= 1200.
    """
    ms, ss = X_seq_tr.mean(axis=0, keepdims=True), np.maximum(X_seq_tr.std(axis=0, keepdims=True), 1e-8)
    mm, sm = X_mod_tr.mean(axis=0, keepdims=True), np.maximum(X_mod_tr.std(axis=0, keepdims=True), 1e-8)
    Zs_tr = (np.asarray(X_seq_tr, dtype=np.float64) - ms) / ss
    Zs_te = (np.asarray(X_seq_te, dtype=np.float64) - ms) / ss
    Zm_tr = (np.asarray(X_mod_tr, dtype=np.float64) - mm) / sm
    Zm_te = (np.asarray(X_mod_te, dtype=np.float64) - mm) / sm

    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    n_tr, p_s = Zs_tr.shape
    p_m = Zm_tr.shape[1]
    splits = _safe_group_splits(Zs_tr, yc, groups_tr, n_folds=n_folds)
    oof_best = np.full(n_tr, y_mean, dtype=np.float64)

    if not splits:
        best_ls, best_lm = 30.0, 30.0
    else:
        ls_arr = np.asarray(lams_seq, dtype=np.float64)
        lm_arr = np.asarray(lams_mod, dtype=np.float64)
        cv_mse = np.zeros((len(ls_arr), len(lm_arr)), dtype=np.float64)
        oof_grid = np.empty((len(ls_arr), len(lm_arr), n_tr), dtype=np.float64) if return_oof else None

        if p_s + p_m <= 1300:
            for tr_i, va_i in splits:
                Zsa, Zsv = Zs_tr[tr_i], Zs_tr[va_i]
                Zma, Zmv = Zm_tr[tr_i], Zm_tr[va_i]
                ym_f = float(y_tr[tr_i].mean())
                ya, yv = y_tr[tr_i] - ym_f, y_tr[va_i] - ym_f
                if len(tr_i) >= p_s + p_m:
                    Gss = Zsa.T @ Zsa
                    Gsm = Zsa.T @ Zma
                    Gmm = Zma.T @ Zma
                    rhs_s = Zsa.T @ ya
                    rhs_m = Zma.T @ ya
                    K = np.empty((p_s + p_m, p_s + p_m), dtype=np.float64)
                    rhs = np.empty(p_s + p_m, dtype=np.float64)
                    idx_s = np.arange(p_s)
                    idx_m = p_s + np.arange(p_m)
                    for i_s, ls in enumerate(ls_arr):
                        K[:p_s, :p_s] = Gss
                        K[:p_s, p_s:] = Gsm
                        K[p_s:, :p_s] = Gsm.T
                        K[p_s:, p_s:] = Gmm
                        K[idx_s, idx_s] += ls
                        rhs[:p_s] = rhs_s
                        rhs[p_s:] = rhs_m
                        for i_m, lm in enumerate(lm_arr):
                            Km = K.copy()
                            Km[idx_m, idx_m] += lm
                            cf = cho_factor(Km, lower=True, check_finite=False)
                            w = cho_solve(cf, rhs, check_finite=False)
                            pred_v = Zsv @ w[:p_s] + Zmv @ w[p_s:]
                            diff = pred_v - yv
                            cv_mse[i_s, i_m] += float(np.sum(diff * diff))
                            if oof_grid is not None:
                                oof_grid[i_s, i_m, va_i] = ym_f + pred_v
                else:
                    Ks_aa = Zsa @ Zsa.T
                    Km_aa = Zma @ Zma.T
                    Ks_va = Zsv @ Zsa.T
                    Km_va = Zmv @ Zma.T
                    na = len(tr_i)
                    for i_s, ls in enumerate(ls_arr):
                        for i_m, lm in enumerate(lm_arr):
                            G = (1.0 / ls) * Ks_aa + (1.0 / lm) * Km_aa
                            G[np.diag_indices(na)] += 1.0
                            cf = cho_factor(G, lower=True, check_finite=False)
                            al = cho_solve(cf, ya, check_finite=False)
                            pred_v = (1.0 / ls) * (Ks_va @ al) + (1.0 / lm) * (Km_va @ al)
                            diff = pred_v - yv
                            cv_mse[i_s, i_m] += float(np.sum(diff * diff))
                            if oof_grid is not None:
                                oof_grid[i_s, i_m, va_i] = ym_f + pred_v
        else:
            for tr_i, va_i in splits:
                Zsa, Zsv = Zs_tr[tr_i], Zs_tr[va_i]
                Zma, Zmv = Zm_tr[tr_i], Zm_tr[va_i]
                ym_f = float(y_tr[tr_i].mean())
                ya, yv = y_tr[tr_i] - ym_f, y_tr[va_i] - ym_f
                Ks_aa = Zsa @ Zsa.T
                Km_aa = Zma @ Zma.T
                Ks_va = Zsv @ Zsa.T
                Km_va = Zmv @ Zma.T
                na = len(tr_i)
                for i_s, ls in enumerate(ls_arr):
                    for i_m, lm in enumerate(lm_arr):
                        G = (1.0 / ls) * Ks_aa + (1.0 / lm) * Km_aa
                        G[np.diag_indices(na)] += 1.0
                        cf = cho_factor(G, lower=True, check_finite=False)
                        al = cho_solve(cf, ya, check_finite=False)
                        pred_v = (1.0 / ls) * (Ks_va @ al) + (1.0 / lm) * (Km_va @ al)
                        diff = pred_v - yv
                        cv_mse[i_s, i_m] += float(np.sum(diff * diff))
                        if oof_grid is not None:
                            oof_grid[i_s, i_m, va_i] = ym_f + pred_v

        bi, bj = np.unravel_index(int(np.argmin(cv_mse)), cv_mse.shape)
        best_ls = float(ls_arr[bi])
        best_lm = float(lm_arr[bj])
        if oof_grid is not None:
            oof_best = oof_grid[bi, bj]

    if n_tr >= p_s + p_m and p_s + p_m <= 1300:
        K = np.block([[Zs_tr.T @ Zs_tr, Zs_tr.T @ Zm_tr], [Zm_tr.T @ Zs_tr, Zm_tr.T @ Zm_tr]])
        idx_s = np.arange(p_s)
        idx_m = p_s + np.arange(p_m)
        K[idx_s, idx_s] += best_ls
        K[idx_m, idx_m] += best_lm
        rhs = np.concatenate([Zs_tr.T @ yc, Zm_tr.T @ yc])
        cf = cho_factor(K, lower=True, check_finite=False)
        w = cho_solve(cf, rhs, check_finite=False)
        pred_te = y_mean + Zs_te @ w[:p_s] + Zm_te @ w[p_s:]
    else:
        G = (1.0 / best_ls) * (Zs_tr @ Zs_tr.T) + (1.0 / best_lm) * (Zm_tr @ Zm_tr.T)
        G[np.diag_indices(n_tr)] += 1.0
        cf = cho_factor(G, lower=True, check_finite=False)
        al = cho_solve(cf, yc, check_finite=False)
        pred_te = y_mean + (1.0 / best_ls) * ((Zs_te @ Zs_tr.T) @ al) + (1.0 / best_lm) * ((Zm_te @ Zm_tr.T) @ al)

    out = {"pred_te": pred_te, "best_lam_seq": best_ls, "best_lam_mod": best_lm}
    if return_oof:
        out["pred_oof"] = oof_best
    return out


def fit_predict_poly3_kernel_groupcv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    gammas: Sequence[float] = (0.5, 1.0, 2.0),
    lams: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_folds: int = 3,
    return_oof: bool = False,
) -> Dict:
    """Inhomogeneous degree-3 polynomial kernel Ridge: K(z, z') = (1 + gamma * z.T z' / p)^3."""
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-8)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    p = float(Z_tr.shape[1])
    D_tr = (Z_tr @ Z_tr.T) / p
    D_te = (Z_te @ Z_tr.T) / p

    splits = _safe_group_splits(Z_tr, yc, groups_tr, n_folds=n_folds)
    lam_arr = np.asarray(lams, dtype=np.float64)
    oof_best = np.full(len(y_tr), y_mean, dtype=np.float64)
    if not splits:
        best_g, best_l = 1.0, 100.0
    else:
        best_mse = np.inf
        best_g, best_l = 1.0, 100.0
        for g in gammas:
            K_all = np.power(1.0 + g * D_tr, 3)
            mse_g = np.zeros(len(lam_arr), dtype=np.float64)
            oof_g = np.empty((len(lam_arr), len(y_tr)), dtype=np.float64) if return_oof else None
            for tr_i, va_i in splits:
                ym_f = float(y_tr[tr_i].mean())
                Kaa = K_all[np.ix_(tr_i, tr_i)]
                Kva = K_all[np.ix_(va_i, tr_i)]
                ev, ec = sp_eigh(Kaa, check_finite=False, driver="evd")
                ev = np.maximum(ev, 0.0)
                proj = ec.T @ (y_tr[tr_i] - ym_f)
                al_all = ec @ (proj[:, None] / (ev[:, None] + lam_arr[None, :]))
                preds = Kva @ al_all
                err = preds - (y_tr[va_i] - ym_f)[:, None]
                mse_g += np.sum(err * err, axis=0)
                if oof_g is not None:
                    oof_g[:, va_i] = ym_f + preds.T
            i_b = int(np.argmin(mse_g))
            if mse_g[i_b] < best_mse:
                best_mse = float(mse_g[i_b])
                best_g, best_l = float(g), float(lam_arr[i_b])
                if oof_g is not None:
                    oof_best = oof_g[i_b]

    K_full = np.power(1.0 + best_g * D_tr, 3)
    K_full[np.diag_indices(len(Z_tr))] += best_l
    cf = cho_factor(K_full, lower=True, check_finite=False)
    alpha = cho_solve(cf, yc, check_finite=False)
    pred_te = y_mean + np.power(1.0 + best_g * D_te, 3) @ alpha

    out = {"pred_te": pred_te, "best_gamma": best_g, "best_lambda": best_l}
    if return_oof:
        out["pred_oof"] = oof_best
    return out


def fit_predict_hgbdt_groupcv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    n_folds: int = 3,
    return_oof: bool = False,
) -> Dict:
    """GroupCV-tuned HistGradientBoostingRegressor on registered sequence features."""
    splits = _safe_group_splits(X_tr, y_tr, groups_tr, n_folds=n_folds)
    configs = [
        {"learning_rate": 0.04, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 5.0},
        {"learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 2.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0},
    ]
    oof_best = np.full(len(y_tr), float(y_tr.mean()), dtype=np.float64)
    if not splits:
        best_cfg = configs[0]
    else:
        best_mse = np.inf
        best_cfg = configs[0]
        for cfg in configs:
            mse = 0.0
            oof_c = np.empty(len(y_tr), dtype=np.float64) if return_oof else None
            for tr_i, va_i in splits:
                m = HistGradientBoostingRegressor(max_iter=250, early_stopping=True, random_state=42, **cfg)
                m.fit(X_tr[tr_i], y_tr[tr_i])
                pv = m.predict(X_tr[va_i])
                diff = pv - y_tr[va_i]
                mse += float(np.sum(diff * diff))
                if oof_c is not None:
                    oof_c[va_i] = pv
            if mse < best_mse:
                best_mse = mse
                best_cfg = cfg
                if oof_c is not None:
                    oof_best = oof_c

    m_full = HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=42, **best_cfg)
    m_full.fit(X_tr, y_tr)
    pred_te = m_full.predict(X_te)
    out = {"pred_te": pred_te, "best_config": best_cfg}
    if return_oof:
        out["pred_oof"] = oof_best
    return out


def fit_predict_two_penalty_null_bank_schur(
    X_seq_tr: np.ndarray,
    X_seq_te: np.ndarray,
    B_n5n6_tr: np.ndarray,  # (n_tr, 2, 1024)
    B_n5n6_te: np.ndarray,  # (n_te, 2, 1024)
    tables_bank: np.ndarray,  # (M, 1024, d) where [0] = F_true
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    lams_seq: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_mod: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_folds: int = 5,
) -> np.ndarray:
    """
    Batched EXACT Schur-complement two-penalty GroupCV Ridge across M candidate biophysical
    tables simultaneously (each table independently tunes (lam_seq, lam_mod) inside Dev CV!).
    Because Shape_N5N6 has p_m = 2*7 = 14 features, batched np.linalg.eigh over (M, 14, 14)
    computes exact predictions for ~800 tables in ~1.5s! Returns preds_te (M, n_te).
    """
    M, _, d = tables_bank.shape
    n_tr = X_seq_tr.shape[0]
    n_te = X_seq_te.shape[0]
    p_s = X_seq_tr.shape[1]
    p_m = 2 * d

    ms = X_seq_tr.mean(axis=0, keepdims=True)
    ss = np.maximum(X_seq_tr.std(axis=0, keepdims=True), 1e-8)
    Zs_tr = (X_seq_tr - ms) / ss
    Zs_te = (X_seq_te - ms) / ss

    # Contract B_n5n6 with all M tables in float64 -> (M, N, p_m)
    T_flat = np.ascontiguousarray(tables_bank.transpose(1, 0, 2).reshape(1024, M * d), dtype=np.float64)
    Xm_tr = (B_n5n6_tr.reshape(n_tr * 2, 1024).astype(np.float64) @ T_flat).reshape(n_tr, 2, M, d).transpose(2, 0, 1, 3).reshape(M, n_tr, p_m)
    Xm_te = (B_n5n6_te.reshape(n_te * 2, 1024).astype(np.float64) @ T_flat).reshape(n_te, 2, M, d).transpose(2, 0, 1, 3).reshape(M, n_te, p_m)
    mm = Xm_tr.mean(axis=1, keepdims=True)
    sm = np.maximum(Xm_tr.std(axis=1, keepdims=True), 1e-8)
    Zm_tr = (Xm_tr - mm) / sm
    Zm_te = (Xm_te - mm) / sm

    y_mean = float(y_tr.mean())
    yc = y_tr - y_mean
    splits = _safe_group_splits(Zs_tr, yc, groups_tr, n_folds=n_folds)
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)

    if not splits:
        best_ls = np.full(M, 30.0, dtype=np.float64)
        best_lm = np.full(M, 30.0, dtype=np.float64)
    else:
        cv_mse = np.zeros((M, len(ls_arr), len(lm_arr)), dtype=np.float64)
        for tr_i, va_i in splits:
            Zsa, Zsv = Zs_tr[tr_i], Zs_tr[va_i]
            Zma, Zmv = Zm_tr[:, tr_i, :], Zm_tr[:, va_i, :]
            ya, yv = yc[tr_i], yc[va_i]
            if len(tr_i) >= p_s:
                ev_b, Qb = sp_eigh(Zsa.T @ Zsa, check_finite=False, driver="evd")
                ev_b = np.maximum(ev_b, 0.0)
                qb = Qb.T @ (Zsa.T @ ya)
                Zsv_Qb = Zsv @ Qb  # (n_va, p_s)
                G_sm = (Zsa.T @ Zma.transpose(1, 0, 2).reshape(len(tr_i), M * p_m)).reshape(p_s, M, p_m).transpose(1, 0, 2)
                U = np.einsum("ij,mjk->mik", Qb.T, G_sm)  # (M, p_s, p_m)
                G_mm = np.einsum("mni,mnj->mij", Zma, Zma)
                zm_y = np.einsum("mni,n->mi", Zma, ya)
                for i_s, ls in enumerate(ls_arr):
                    inv_d = 1.0 / (ev_b + ls)
                    U_sc = U * inv_d[None, :, None]
                    S = G_mm - np.einsum("mki,mkj->mij", U, U_sc)
                    S = 0.5 * (S + S.swapaxes(-1, -2))
                    rhs_m = zm_y - np.einsum("mki,k->mi", U_sc, qb)
                    ev_t, Qt = np.linalg.eigh(S)
                    ev_t = np.maximum(ev_t, 0.0)
                    proj_t = np.einsum("mik,mi->mk", Qt, rhs_m)
                    for i_m, lm in enumerate(lm_arr):
                        wm = np.einsum("mik,mk->mi", Qt, proj_t / (ev_t + lm))
                        ws_q = (qb[None, :] - np.einsum("mik,mk->mi", U, wm)) * inv_d[None, :]
                        pred_va = ws_q @ Zsv_Qb.T + np.einsum("mni,mi->mn", Zmv, wm)
                        diff = pred_va - yv[None, :]
                        cv_mse[:, i_s, i_m] += np.sum(diff * diff, axis=1)
            else:
                # Dual/small-N fallback: Woodbury over n_a via batched Cholesky/eigh (M is small or n_a < p_s <= 500)
                na = len(tr_i)
                Ks_aa = Zsa @ Zsa.T
                ev_d, Qd = sp_eigh(Ks_aa, check_finite=False, driver="evd")
                ev_d = np.maximum(ev_d, 0.0)
                qd = Qd.T @ ya
                Zm_q = np.einsum("ij,mjk->mik", Qd.T, Zma)  # (M, na, p_m)
                G_mm = np.einsum("mni,mnj->mij", Zma, Zma)
                Ks_va = Zsv @ Zsa.T
                for i_s, ls in enumerate(ls_arr):
                    # (K_s / ls + I_na)^{-1} in Qd eigenbasis is diag(ls / (ev_d + ls))
                    w_d = ls / (ev_d + ls)
                    rhs_d = Qd @ (w_d * qd)  # (na,)
                    S = np.einsum("mni,n,mnj->mij", Zm_q, w_d, Zm_q)
                    S = 0.5 * (S + S.swapaxes(-1, -2))
                    rhs_m = np.einsum("mni,n->mi", Zma, rhs_d)
                    ev_t, Qt = np.linalg.eigh(S)
                    ev_t = np.maximum(ev_t, 0.0)
                    proj_t = np.einsum("mik,mi->mk", Qt, rhs_m)
                    for i_m, lm in enumerate(lm_arr):
                        wm = np.einsum("mik,mk->mi", Qt, proj_t / (ev_t + lm))
                        res_y = ya[None, :] - np.einsum("mni,mi->mn", Zma, wm)
                        dual_s = (Qd @ (w_d[None, :] * (res_y @ Qd)).T).T / ls  # (M, na)
                        pred_va = dual_s @ Ks_va.T + np.einsum("mni,mi->mn", Zmv, wm)
                        diff = pred_va - yv[None, :]
                        cv_mse[:, i_s, i_m] += np.sum(diff * diff, axis=1)

        flat_best = np.argmin(cv_mse.reshape(M, -1), axis=1)
        bi = flat_best // len(lm_arr)
        bj = flat_best % len(lm_arr)
        best_ls = ls_arr[bi]
        best_lm = lm_arr[bj]

    # Full Dev fit using per-table locked (best_ls[m], best_lm[m])
    if n_tr >= p_s:
        ev_bf, Qbf = sp_eigh(Zs_tr.T @ Zs_tr, check_finite=False, driver="evd")
        ev_bf = np.maximum(ev_bf, 0.0)
        qbf = Qbf.T @ (Zs_tr.T @ yc)
        Zte_Qbf = Zs_te @ Qbf
        G_sm_f = (Zs_tr.T @ Zm_tr.transpose(1, 0, 2).reshape(n_tr, M * p_m)).reshape(p_s, M, p_m).transpose(1, 0, 2)
        Uf = np.einsum("ij,mjk->mik", Qbf.T, G_sm_f)
        G_mm_f = np.einsum("mni,mnj->mij", Zm_tr, Zm_tr)
        zm_yf = np.einsum("mni,n->mi", Zm_tr, yc)
        inv_df = 1.0 / (ev_bf[None, :] + best_ls[:, None])  # (M, p_s)
        Uf_sc = Uf * inv_df[:, :, None]
        Sf = G_mm_f - np.einsum("mki,mkj->mij", Uf, Uf_sc)
        Sf += best_lm[:, None, None] * np.eye(p_m)[None, :, :]
        rhs_mf = zm_yf - np.einsum("mki,k->mi", Uf_sc, qbf)
        wmf = np.linalg.solve(Sf, rhs_mf[..., None])[..., 0]
        wsf_q = (qbf[None, :] - np.einsum("mik,mk->mi", Uf, wmf)) * inv_df
        preds_te = y_mean + wsf_q @ Zte_Qbf.T + np.einsum("mni,mi->mn", Zm_te, wmf)
    else:
        preds_te = np.empty((M, n_te), dtype=np.float64)
        Ks_full = Zs_tr @ Zs_tr.T
        Ks_te = Zs_te @ Zs_tr.T
        for m in range(M):
            G = (1.0 / best_ls[m]) * Ks_full + (1.0 / best_lm[m]) * (Zm_tr[m] @ Zm_tr[m].T)
            G[np.diag_indices(n_tr)] += 1.0
            cf = cho_factor(G, lower=True, check_finite=False)
            al = cho_solve(cf, yc, check_finite=False)
            preds_te[m] = y_mean + (1.0 / best_ls[m]) * (Ks_te @ al) + (1.0 / best_lm[m]) * ((Zm_te[m] @ Zm_tr[m].T) @ al)
    return preds_te


def cluster_bootstrap_spearman_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: Sequence[str],
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Fast cluster-bootstrap over independent held-out sequence groups (e.g., Core10)
    for Spearman(y, pred_a) - Spearman(y, pred_b) + p_boot.
    """
    rho_a, p_a = v3_sm.safe_spearman_pearson(y_true, pred_a)
    rho_b, p_b = v3_sm.safe_spearman_pearson(y_true, pred_b)
    diff_obs = float(rho_a - rho_b)

    by_g: Dict[str, np.ndarray] = {}
    tmp: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        tmp.setdefault(g, []).append(i)
    g_names = sorted(tmp.keys())
    g_arrs = [np.array(tmp[g], dtype=np.int32) for g in g_names]
    G = len(g_arrs)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    # Precompute tied ranks for whole dataset vs sampled cluster multiset
    for b in range(n_boot):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([g_arrs[p] for p in pick])
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        ry = np.argsort(np.argsort(yt)).astype(np.float64)
        ra = np.argsort(np.argsort(pa)).astype(np.float64)
        rb = np.argsort(np.argsort(pb)).astype(np.float64)
        ry -= ry.mean()
        ra -= ra.mean()
        rb -= rb.mean()
        yn = np.linalg.norm(ry)
        if yn < 1e-12:
            diffs[b] = 0.0
            continue
        rho_ab = float(np.dot(ra, ry) / (max(np.linalg.norm(ra), 1e-12) * yn))
        rho_bb = float(np.dot(rb, ry) / (max(np.linalg.norm(rb), 1e-12) * yn))
        diffs[b] = rho_ab - rho_bb

    ci_lo = float(np.quantile(diffs, 0.025))
    ci_hi = float(np.quantile(diffs, 0.975))
    p_boot = float((1.0 + np.sum(diffs <= 0.0)) / float(n_boot + 1))
    return {
        "n_groups": int(G),
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "pearson_a": float(p_a),
        "pearson_b": float(p_b),
        "diff": diff_obs,
        "ci95": [ci_lo, ci_hi],
        "p_boot_gt0": p_boot,
    }
