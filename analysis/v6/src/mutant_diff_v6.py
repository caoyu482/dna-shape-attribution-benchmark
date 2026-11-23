#!/usr/bin/env python3
"""
Core data loaders, multi-sparsity null bank builders, and verification helpers
for biophys_inductive_gfm/v6:
  1. WT-free mutant differential endpoint:
       d(x) = log(relKa_ScrR5A(x)) - log(relKa_ScrR3A(x)) = delta_R5A(x) - delta_R3A(x)
     on both the exact v5 3-way cohort (7,774 aligned / 13,172 total) and the WT-free
     2-way R5A∩R3A cohort (12,036 aligned / 19,195 total).
  2. Within-GSE65073 single-mutation Antp differentials:
       AntpHQT - AntpHQ (+T), AntpHQT - AntpQT (+H), AntpHQT - AntpHT (+Q).
  3. Pre-locked multi-sparsity Sparse4merSurrogateNull bank (K in {4, 8, 12, 24, 48})
     compatible with v5 batched Schur solver (`fit_predict_two_penalty_null_bank_schur`).
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

V3_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v3/src")
V4_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v4/src")
V5_SRC = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v5/src")
for _p in (V5_SRC, V4_SRC, V3_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import null_generators_v4 as ng4  # noqa: E402
import paired_perturbation_v5 as pp5  # noqa: E402

GSE65073_DIR = pp5.GSE65073_DIR


def _build_two_sample_records(
    ds_keys: List[str],
    d_a: Dict[str, float],
    d_b: Dict[str, float],
    wt_dict: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Build canonical dsDNA 12-mer records with d(x) = log(A) - log(B) aligned to frame14."""
    records: List[Dict] = []
    for k in ds_keys:
        al = pp5.align_12mer_to_frame14(k)
        va = pp5._extract_dsdna_val(d_a, k)
        vb = pp5._extract_dsdna_val(d_b, k)
        d_val = float(np.log(va) - np.log(vb))
        rec: Dict = {
            **al,
            "canon12": k,
            "relka_a": float(va),
            "relka_b": float(vb),
            "d_diff": d_val,
        }
        if wt_dict is not None:
            vw = pp5._extract_dsdna_val(wt_dict, k)
            rec["relka_wt"] = float(vw)
            rec["delta_r5a"] = float(np.log(va) - np.log(vw))
            rec["delta_r3a"] = float(np.log(vb) - np.log(vw))
        records.append(rec)
    return records


def _audit_12vs14_and_offset(
    aligned_recs: List[Dict],
    all_recs: List[Dict],
    a14: Dict[str, float],
    b14: Dict[str, float],
    common14_keys: List[str],
    wt_for_check: bool = False,
) -> Dict:
    """Audit 12-mer vs 14-mer marginal consistency (min_cnt>=1,2,3) and cross-offset 0 vs 1."""
    sub12: Dict[str, List[float]] = {}
    for k14 in common14_keys:
        va14 = pp5._extract_dsdna_val(a14, k14)
        vb14 = pp5._extract_dsdna_val(b14, k14)
        d14 = float(np.log(va14) - np.log(vb14))
        for st in (0, 1, 2):
            sub = pp5.canonical_rc(k14[st : st + 12])
            sub12.setdefault(sub, []).append(d14)

    res_12vs14: Dict[str, Dict] = {}
    for tag, recs in [("aligned_sc8", aligned_recs), ("all_dsdna", all_recs)]:
        for mcnt in (1, 2, 3):
            rows = [r for r in recs if r["canon12"] in sub12 and len(sub12[r["canon12"]]) >= mcnt]
            if len(rows) < 10:
                continue
            x12 = np.array([r["d_diff"] for r in rows], dtype=np.float64)
            x14 = np.array([float(np.mean(sub12[r["canon12"]])) for r in rows], dtype=np.float64)
            res_12vs14[f"{tag}_mincnt_{mcnt}"] = {
                "N": len(rows),
                "spearman": float(spearmanr(x12, x14).statistic),
                "pearson": float(pearsonr(x12, x14).statistic),
            }

    core_off: Dict[Tuple[str, int], List[float]] = {}
    for r in aligned_recs:
        core_off.setdefault((r["w10_fwd"], r["offset"]), []).append(r["d_diff"])
    pairs = [
        (float(np.mean(core_off[(w, 0)])), float(np.mean(core_off[(w, 1)])))
        for w in {r["w10_fwd"] for r in aligned_recs}
        if (w, 0) in core_off and (w, 1) in core_off
    ]
    a_off = np.array(pairs, dtype=np.float64)
    off_audit = {
        "n_cores": int(len(a_off)),
        "spearman": float(spearmanr(a_off[:, 0], a_off[:, 1]).statistic) if len(a_off) >= 3 else 0.0,
        "pearson": float(pearsonr(a_off[:, 0], a_off[:, 1]).statistic) if len(a_off) >= 3 else 0.0,
    }

    d_al = np.array([r["d_diff"] for r in aligned_recs], dtype=np.float64)
    out: Dict = {
        "d_aligned_dist": {
            "N": int(len(d_al)),
            "mean": float(np.mean(d_al)),
            "sd": float(np.std(d_al)),
            "median": float(np.median(d_al)),
            "iqr": float(np.percentile(d_al, 75) - np.percentile(d_al, 25)),
        },
        "same_library_12mer_vs_14mer_note": "12-mer and 14-mer tables summarize the same SELEX-seq libraries, not independent biological replicates.",
        "12mer_vs_14mer": res_12vs14,
        "cross_offset_0vs1": off_audit,
    }
    if wt_for_check:
        wt_al = np.log(np.array([r["relka_wt"] for r in aligned_recs], dtype=np.float64))
        dr5_al = np.array([r["delta_r5a"] for r in aligned_recs], dtype=np.float64)
        dr3_al = np.array([r["delta_r3a"] for r in aligned_recs], dtype=np.float64)
        _, held_idx = pp5.make_core10_disjoint_split(aligned_recs, dev_frac=0.60)
        out["wt_cancellation_and_coupling"] = {
            "aligned_N": int(len(aligned_recs)),
            "rho_d_vs_logWT_aligned": float(spearmanr(d_al, wt_al).statistic),
            "rho_d_vs_logWT_held": float(spearmanr(d_al[held_idx], wt_al[held_idx]).statistic),
            "held_N": int(len(held_idx)),
            "rho_deltaR5A_vs_deltaR3A_held": float(spearmanr(dr5_al[held_idx], dr3_al[held_idx]).statistic),
            "rho_neglogWT_vs_deltaR3A_held": float(spearmanr(-wt_al[held_idx], dr3_al[held_idx]).statistic),
            "rho_neglogWT_vs_deltaR5A_held": float(spearmanr(-wt_al[held_idx], dr5_al[held_idx]).statistic),
        }
    return out


def load_r5a_r3a_differential_datasets(F_true: np.ndarray) -> Dict:
    """
    Load exact WT-free differential d(x) = log(relKa_ScrR5A) - log(relKa_ScrR3A)
    for BOTH the v5-matched 3-way cohort (WT∩R5A∩R3A) and the WT-free 2-way cohort (R5A∩R3A).
    """
    wt12, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586782_ScrWT_Exd_12mer.txt.gz")
    r512, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586783_ScrR5A_Exd_12mer.txt.gz")
    r312, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586784_ScrR3A_Exd_12mer.txt.gz")

    wt14, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586782_ScrWT_Exd_14mer.txt.gz")
    r514, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586783_ScrR5A_Exd_14mer.txt.gz")
    r314, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586784_ScrR3A_Exd_14mer.txt.gz")

    s3_dir = set(wt12) & set(r512) & set(r312)
    s2_dir = set(r512) & set(r312)
    ds3 = sorted({pp5.canonical_rc(k) for k in s3_dir})
    ds2 = sorted({pp5.canonical_rc(k) for k in s2_dir})

    recs3_all = _build_two_sample_records(ds3, r512, r312, wt_dict=wt12)
    recs3_al = [r for r in recs3_all if r["align_score"] >= 8.0]

    recs2_all = _build_two_sample_records(ds2, r512, r312, wt_dict=None)
    recs2_al = [r for r in recs2_all if r["align_score"] >= 8.0]

    ds14_3 = sorted({pp5.canonical_rc(k) for k in (set(wt14) & set(r514) & set(r314))})
    ds14_2 = sorted({pp5.canonical_rc(k) for k in (set(r514) & set(r314))})

    audit_3way = _audit_12vs14_and_offset(recs3_al, recs3_all, r514, r314, ds14_3, wt_for_check=True)
    audit_2way = _audit_12vs14_and_offset(recs2_al, recs2_all, r514, r314, ds14_2, wt_for_check=False)

    # Selection bias between WT-excluded (2way \ 3way) and 3way
    s3_set = set(ds3)
    excl_recs = [r for r in recs2_all if r["canon12"] not in s3_set]
    excl_al = [r for r in excl_recs if r["align_score"] >= 8.0]
    fc_inc = pp5.encode_frame_sequences([r["frame14"] for r in recs3_al], 14)
    c5_inc = pp5._frame_c5_n5n6(fc_inc)
    mgw_inc = 0.5 * (F_true[c5_inc[:, 0], 0] + F_true[c5_inc[:, 1], 0])
    fc_exc = pp5.encode_frame_sequences([r["frame14"] for r in excl_al], 14)
    c5_exc = pp5._frame_c5_n5n6(fc_exc)
    mgw_exc = 0.5 * (F_true[c5_exc[:, 0], 0] + F_true[c5_exc[:, 1], 0])

    return {
        "cohort_3way": {"all": recs3_all, "aligned": recs3_al, "directed_count": len(s3_dir)},
        "cohort_2way": {"all": recs2_all, "aligned": recs2_al, "directed_count": len(s2_dir)},
        "audit": {
            "cohort_counts": {
                "3way_directed": len(s3_dir),
                "3way_dsdna": len(recs3_all),
                "3way_aligned_sc8": len(recs3_al),
                "2way_directed": len(s2_dir),
                "2way_dsdna": len(recs2_all),
                "2way_aligned_sc8": len(recs2_al),
                "wt_dropout_only_in_2way_dsdna": len(excl_recs),
                "wt_dropout_only_in_2way_aligned": len(excl_al),
            },
            "2way_minus_3way_bias": {
                "excl_d_mean_sd": [float(np.mean([r["d_diff"] for r in excl_recs])), float(np.std([r["d_diff"] for r in excl_recs]))],
                "incl_d_mean_sd": [float(np.mean([r["d_diff"] for r in recs3_all])), float(np.std([r["d_diff"] for r in recs3_all]))],
                "excl_aligned_mgw_mean_sd": [float(np.mean(mgw_exc)), float(np.std(mgw_exc))],
                "incl_aligned_mgw_mean_sd": [float(np.mean(mgw_inc)), float(np.std(mgw_inc))],
            },
            "3way": audit_3way,
            "2way": audit_2way,
        },
    }


def load_antp_single_mutation_diffs() -> Dict[str, Dict]:
    """
    Load the three within-GSE65073 single-substitution Antp differentials:
      - AntpHQT - AntpHQ (+T)
      - AntpHQT - AntpQT (+H)
      - AntpHQT - AntpHT (+Q)
    """
    hqt12, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586793_AntpHQT_Exd_12mer.txt.gz")
    hqt14, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / "GSM1586793_AntpHQT_Exd_14mer.txt.gz")

    pairs_meta = [
        ("AntpHQT_minus_AntpHQ (+T)", "GSM1586790_AntpHQ_Exd_12mer.txt.gz", "GSM1586790_AntpHQ_Exd_14mer.txt.gz"),
        ("AntpHQT_minus_AntpQT (+H)", "GSM1586792_AntpQT_Exd_12mer.txt.gz", "GSM1586792_AntpQT_Exd_14mer.txt.gz"),
        ("AntpHQT_minus_AntpHT (+Q)", "GSM1586791_AntpHT_Exd_12mer.txt.gz", "GSM1586791_AntpHT_Exd_14mer.txt.gz"),
    ]
    out: Dict[str, Dict] = {}
    for label, fn12, fn14 in pairs_meta:
        b12, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / fn12)
        b14, _ = pp5.load_raw_kmer_dict(GSE65073_DIR / fn14)
        ds12 = sorted({pp5.canonical_rc(k) for k in (set(hqt12) & set(b12))})
        recs_all = _build_two_sample_records(ds12, hqt12, b12, wt_dict=None)
        recs_al = [r for r in recs_all if r["align_score"] >= 8.0]
        ds14 = sorted({pp5.canonical_rc(k) for k in (set(hqt14) & set(b14))})
        audit = _audit_12vs14_and_offset(recs_al, recs_all, hqt14, b14, ds14, wt_for_check=False)
        out[label] = {
            "all": recs_all,
            "aligned": recs_al,
            "n_dsdna": len(recs_all),
            "n_aligned": len(recs_al),
            "audit": audit,
        }
    return out


def build_multi_sparsity_null_banks(
    F_true: np.ndarray,
    c5_tr: np.ndarray,
    shap_std: np.ndarray,
    B: int = 199,
    seed: int = 202606,
    k_list: Sequence[int] = (4, 8, 12, 24, 48),
    include_naive: bool = True,
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """
    Build combined table bank starting with F_true at index 0, followed by:
      - Naive_RC_Perm (optional)
      - Shapelizer_2018_Std
      - ShiftGC_ExactCRT_WS_G2
      - Sparse4mer_K{k} for each k in k_list
    """
    rng = np.random.default_rng(seed)
    p_raw = np.bincount(c5_tr.ravel(), minlength=1024).astype(np.float64)
    n_obs = float(c5_tr.size)

    parts = [F_true[None, :, :]]
    slices: Dict[str, slice] = {}
    cur = 1

    if include_naive:
        nv_gen = ng4.NaiveRCPermutationNull(F_true)
        parts.append(np.stack([nv_gen.sample(rng) for _ in range(B)], axis=0))
        slices["Naive_RC_Perm"] = slice(cur, cur + B)
        cur += B

    parts.append(shap_std[:B])
    slices["Shapelizer_2018_Std"] = slice(cur, cur + B)
    cur += B

    crt_g2 = ng4.ShiftGCExactCRTNull(F_true, p_raw, conditioning="WS_G2", n_total_obs=n_obs)
    parts.append(np.stack([crt_g2.sample(rng) for _ in range(B)], axis=0))
    slices["ShiftGC_ExactCRT_WS_G2"] = slice(cur, cur + B)
    cur += B

    for k in k_list:
        sp_gen = pp5.Sparse4merSurrogateNull(F_true, p_raw, n_total_obs=n_obs, n_pockets_per_ch=int(k))
        parts.append(np.stack([sp_gen.sample(rng) for _ in range(B)], axis=0))
        slices[f"Sparse4mer_K{int(k)}"] = slice(cur, cur + B)
        cur += B

    return np.concatenate(parts, axis=0), slices
