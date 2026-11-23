#!/usr/bin/env python3
"""
Faithful Python 3 / NumPy port of BussemakerLab/FCpackage (`FCfunctions.R`)
and Kock et al. (2024) Homeodomain mutant evaluation logic (`HDMutantsPrediction.Rmd`).

All alignment positions in `trainSVD`, `makeSVDModel.singleMutation`, and
`predict.SVD` / `predict.SVDmodel` follow the 1-based alignment indexing of
`FCfunctions.R` (`1 <= j <= L`, where `j = canonical - 1` for the 57-aa Pfam HD
alignment starting at canonical position 2).
"""
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rdata
from scipy.special import logsumexp
from scipy.stats import f as f_dist

DEFAULT_RAW_DIR = (Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2]))) / "v7/data/raw")

DNA_BASES: Tuple[str, str, str, str] = ("A", "C", "G", "T")
MOTIF_POSES_HD: Tuple[str, ...] = ("N1", "N2", "T3", "D4", "A5", "Y6", "N7", "N8")

TETRA_TRANS_MATRIX: np.ndarray = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)
PSSM_TRANS_MATRIX: np.ndarray = np.linalg.pinv(TETRA_TRANS_MATRIX)  # 0.25 * T^T

_BASE_TO_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def DNA() -> List[str]:
    """Return canonical base ordering ['A', 'C', 'G', 'T']."""
    return list(DNA_BASES)


def normalize_sum1(matrix: np.ndarray, rows: bool = True) -> np.ndarray:
    """Normalize each row (`rows=True`) or each column (`rows=False`) to sum to 1."""
    mat = np.asarray(matrix, dtype=np.float64).copy()
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")
    if rows:
        sums = np.sum(mat, axis=1, keepdims=True)
        return mat / sums
    sums = np.sum(mat, axis=0, keepdims=True)
    return mat / sums


def matrix2tetrahedron(matrix: np.ndarray) -> np.ndarray:
    """
    Transform a (4, N) frequency/PSAM matrix into (N, 3) tetrahedron coordinates.
    Matches `matrix2tetrahedron` in `FCfunctions.R`.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, None]
    if mat.shape[0] != 4:
        raise ValueError(f"Expected 4 rows (ACGT), got shape {mat.shape}")
    return normalize_sum1(mat.T, rows=True) @ TETRA_TRANS_MATRIX


def tetrahedron2matrix(tetraMatrix: np.ndarray) -> np.ndarray:
    """
    Transform (N, 3) tetrahedron coordinates back to a (4, N) PSAM matrix
    (normalized so each column maximum is 1.0).
    Matches `tetrahedron2matrix` in `FCfunctions.R`.
    """
    tm = np.asarray(tetraMatrix, dtype=np.float64)
    if tm.ndim == 1:
        tm = tm[None, :]
    if tm.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (x,y,z), got shape {tm.shape}")
    raw = (tm @ PSSM_TRANS_MATRIX + 0.25).T  # (4, N)
    col_max = np.max(raw, axis=0, keepdims=True)
    return raw / col_max


def frequency2ddG(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a frequency/PSAM matrix (4, N) into column-centered -ddG/RT:
    log(x) - mean(log(x)) per column.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    log_m = np.log(mat)
    if mat.ndim == 1:
        return log_m - np.mean(log_m)
    return log_m - np.mean(log_m, axis=0, keepdims=True)


def ddG2frequency(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a -ddG/RT matrix (4, N) into PSAM relative affinity/frequency:
    exp(x - max(x)) per column (column max = 1.0).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim == 1:
        return np.exp(mat - np.max(mat))
    return np.exp(mat - np.max(mat, axis=0, keepdims=True))


def groupedR2(
    x: Sequence[float],
    y: Sequence[float],
    member: Optional[int] = None,
    mean: bool = True,
    throughZero: bool = True,
) -> Union[float, np.ndarray]:
    """
    Faithful port of `groupedR2` in `FCfunctions.R`.

    Splits `x` and `y` into groups of size `member` (default `len(x)`), fits
    `lm(yy ~ xx + 0)` if `throughZero=True` (else `lm(yy ~ xx)`), and returns
    R^2 with R's rule: if slope <= 0 or undefined/NaN, R^2 is set to 0.0.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.shape[0] != yv.shape[0]:
        raise ValueError("x and y have different length")
    n = xv.shape[0]
    if n == 0:
        return float("nan") if mean else np.array([], dtype=np.float64)
    grp = n if member is None else int(member)
    if grp <= 0 or n % grp != 0:
        raise ValueError(f"Length {n} cannot be divided evenly by member={grp}")

    xm = xv.reshape(n // grp, grp)
    ym = yv.reshape(n // grp, grp)
    r2s = np.zeros(xm.shape[0], dtype=np.float64)
    for i in range(xm.shape[0]):
        xx = xm[i]
        yy = ym[i]
        if throughZero:
            dx = float(np.sum(xx * xx))
            dy = float(np.sum(yy * yy))
            sxy = float(np.sum(xx * yy))
        else:
            xc = xx - float(np.mean(xx))
            yc = yy - float(np.mean(yy))
            dx = float(np.sum(xc * xc))
            dy = float(np.sum(yc * yc))
            sxy = float(np.sum(xc * yc))

        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(sxy)):
            r2s[i] = 0.0
            continue
        if dx <= 0.0 or dy <= 0.0:
            r2s[i] = 0.0
            continue
        beta = sxy / dx
        if not np.isfinite(beta) or beta <= 0.0:
            r2s[i] = 0.0
            continue
        r2s[i] = float(np.clip((sxy * sxy) / (dx * dy), 0.0, 1.0))

    return float(np.mean(r2s)) if mean else r2s


def groupedPCC(
    x: Sequence[float],
    y: Sequence[float],
    member: Optional[int] = None,
    mean: bool = True,
) -> Union[float, np.ndarray]:
    """Faithful port of `groupedPCC` in `FCfunctions.R`."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.shape[0] != yv.shape[0]:
        raise ValueError("x and y have different length")
    n = xv.shape[0]
    if n == 0:
        return float("nan") if mean else np.array([], dtype=np.float64)
    grp = n if member is None else int(member)
    if grp <= 0 or n % grp != 0:
        raise ValueError(f"Length {n} cannot be divided evenly by member={grp}")

    xm = xv.reshape(n // grp, grp)
    ym = yv.reshape(n // grp, grp)
    pccs = np.zeros(xm.shape[0], dtype=np.float64)
    for i in range(xm.shape[0]):
        xc = xm[i] - np.mean(xm[i])
        yc = ym[i] - np.mean(ym[i])
        den = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        pccs[i] = float(np.sum(xc * yc) / den) if den > 0 else float("nan")
    return float(np.mean(pccs)) if mean else pccs


def _extract_alignment_df(
    Alignment: Union[pd.DataFrame, Sequence[str]]
) -> pd.DataFrame:
    """Normalize Alignment input to DataFrame with columns ['name', 'alignment']."""
    if isinstance(Alignment, pd.DataFrame):
        if "alignment" not in Alignment.columns:
            raise ValueError("Alignment DataFrame must contain 'alignment' column")
        out = Alignment.copy().reset_index(drop=True)
        if "name" not in out.columns:
            out["name"] = [f"seq_{i}" for i in range(len(out))]
        return out[["name", "alignment"]]
    if isinstance(Alignment, str):
        return pd.DataFrame({"name": ["Sample"], "alignment": [Alignment]})
    seqs = [str(s) for s in Alignment]
    return pd.DataFrame(
        {"name": [f"seq_{i}" for i in range(len(seqs))], "alignment": seqs}
    )


def gene2pos(
    motifs: Sequence[Dict],
    pos: Union[str, int] = "N1",
    nrow: int = 4,
) -> np.ndarray:
    """
    Extract a (4, n_motifs) matrix at motif position `pos` across a list of motif dicts.
    Each motif dict has `matrix` of shape (4, K) (and optional `pos_names`).
    """
    n_gene = len(motifs)
    out = np.zeros((nrow, n_gene), dtype=np.float64)
    for i, m in enumerate(motifs):
        mat = m["matrix"]
        if hasattr(mat, "sel") and isinstance(pos, str):
            col = np.asarray(mat.sel(dim_1=pos).values, dtype=np.float64)
        else:
            arr = np.asarray(mat, dtype=np.float64)
            if isinstance(pos, (int, np.integer)):
                col_idx = int(pos)
            elif "pos_names" in m:
                col_idx = list(m["pos_names"]).index(pos)
            elif pos in MOTIF_POSES_HD and arr.shape[1] == len(MOTIF_POSES_HD):
                col_idx = MOTIF_POSES_HD.index(pos)
            else:
                raise KeyError(f"Cannot resolve motif position {pos!r}")
            col = arr[:, col_idx]
        out[:, i] = col
    return out


def matrixSVD(posMatrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute tetrahedron-centered SVD of a (4, N) position-specific motif matrix.
    Returns dict with keys 'u' (N, 3), 'd' (3,), 'v' (3, 3), 'tetra_mean' (3,).
    """
    tetra_matrix = matrix2tetrahedron(posMatrix)
    tetra_mean = np.mean(tetra_matrix, axis=0)
    tetra_centered = tetra_matrix - tetra_mean[None, :]
    u, d, vt = np.linalg.svd(tetra_centered, full_matrices=False)
    return {
        "u": u,
        "d": d,
        "v": vt.T,
        "tetra_mean": tetra_mean,
    }


def svdANOVA(
    svd: Dict[str, np.ndarray],
    Alignment: Union[pd.DataFrame, Sequence[str]],
) -> pd.DataFrame:
    """
    Perform one-way ANOVA between each SVD U-component and residue identity
    at every 1-based alignment position `1..L`, returning `-log10(p)` table.
    """
    ali_df = _extract_alignment_df(Alignment)
    seqs = ali_df["alignment"].tolist()
    n_seq = len(seqs)
    aln_len = len(seqs[0])
    n_pc = svd["v"].shape[1]
    aov_mat = np.zeros((n_pc, aln_len), dtype=np.float64)

    for j in range(aln_len):
        aas = [s[j] for s in seqs]
        uniq = set(aas)
        if len(uniq) <= 1:
            continue
        k = len(uniq)
        if n_seq <= k:
            aov_mat[:, j] = np.nan
            continue
        groups_idx: Dict[str, List[int]] = {}
        for idx_s, a in enumerate(aas):
            groups_idx.setdefault(a, []).append(idx_s)
        idx_arrays = [np.asarray(v, dtype=np.int64) for v in groups_idx.values()]

        for i in range(n_pc):
            u1 = svd["u"][:, i]
            grand = float(np.mean(u1))
            ss_between = 0.0
            ss_within = 0.0
            for g_idx in idx_arrays:
                vals = u1[g_idx]
                g_mean = float(np.mean(vals))
                ss_between += len(vals) * (g_mean - grand) ** 2
                diff = vals - g_mean
                ss_within += float(np.sum(diff * diff))
            if ss_within <= 0.0:
                p_val = 0.0 if ss_between > 0.0 else 1.0
            else:
                f_stat = (ss_between / (k - 1)) / (ss_within / (n_seq - k))
                p_val = float(f_dist.sf(f_stat, k - 1, n_seq - k))
            if p_val <= 0.0:
                aov_mat[i, j] = 0.0  # R: -log(0, 10) = Inf -> 0
            else:
                aov_mat[i, j] = float(-np.log10(p_val))

    out = pd.DataFrame(aov_mat.T, columns=[f"PC{i + 1}" for i in range(n_pc)])
    out["name"] = np.arange(1, aln_len + 1, dtype=np.int64)
    return out


def _ols_lm_r_style(X_cols: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit OLS `y ~ 1 + X_cols` matching R `lm()` + `coef[is.na(coef)] <- 0`
    for constant or collinear predictor columns.
    """
    n = y.shape[0]
    k = X_cols.shape[1]
    design = np.column_stack([np.ones(n, dtype=np.float64), X_cols])
    kept = [0]
    Q = design[:, [0]] / np.linalg.norm(design[:, 0])
    for j in range(1, k + 1):
        col = design[:, j]
        col_norm = float(np.linalg.norm(col))
        if col_norm <= 1e-12 or float(np.std(col)) <= 1e-12:
            continue
        proj = Q @ (Q.T @ col)
        resid = col - proj
        resid_norm = float(np.linalg.norm(resid))
        if resid_norm > 1e-10 * col_norm:
            kept.append(j)
            Q = np.column_stack([Q, resid / resid_norm])

    coef = np.zeros(k + 1, dtype=np.float64)
    beta_kept, _, _, _ = np.linalg.lstsq(design[:, kept], y, rcond=None)
    for idx, j in enumerate(kept):
        coef[j] = float(beta_kept[idx])
    return coef


def trainSVD(
    svd: Dict[str, np.ndarray],
    Alignment: Union[pd.DataFrame, Sequence[str]],
    no_keyPos: Optional[Sequence[int]] = None,
    keyPos: Optional[Sequence[Sequence[int]]] = None,
) -> Dict:
    """
    Train SVD-regression model from `matrixSVD` output and matched alignment.
    `keyPos` entries are 1-based alignment indices (`1 <= j <= L`).
    Matches `trainSVD` in `FCfunctions.R`.
    """
    ali_df = _extract_alignment_df(Alignment)
    seqs = ali_df["alignment"].tolist()
    n_seq = len(seqs)

    if no_keyPos is None or len(no_keyPos) == 0:
        nkp = [int(np.floor(val) + 1) for val in svd["d"]]
    else:
        nkp = [int(val) for val in no_keyPos]

    if len(svd["d"]) != len(nkp):
        raise ValueError("no_keyPos has different length from svd$d")

    skip = {i for i, k in enumerate(nkp) if k == 0}

    if keyPos is None or len(keyPos) == 0:
        pval_table = svdANOVA(svd, ali_df)
        max_kp = max(nkp) if nkp else 0
        kp_lists: List[List[int]] = []
        for i in range(len(nkp)):
            col_name = f"PC{i + 1}"
            sorted_df = pval_table.sort_values(col_name, ascending=False, kind="mergesort")
            feats = sorted_df["name"].iloc[: nkp[i]].astype(int).tolist()
            sele = [feats[0]] * max_kp
            sele[: nkp[i]] = feats
            kp_lists.append(sele)
    else:
        kp_lists = [[int(p) for p in sub] for sub in keyPos]
        max_kp = max(len(sub) for sub in kp_lists)
        for sub in kp_lists:
            while len(sub) < max_kp:
                sub.append(sub[0])

    # Build uList[pc][pos_1based] -> dict {aa: mean_u} and aa counts at each pos
    u_list: List[Dict[int, Dict[str, float]]] = []
    aa_counts_by_pos: Dict[int, Dict[str, int]] = {}
    for i in range(len(nkp)):
        row_dict: Dict[int, Dict[str, float]] = {}
        u1 = svd["u"][:, i]
        for j_1based in kp_lists[i]:
            if j_1based in row_dict:
                continue
            j0 = j_1based - 1
            groups: Dict[str, List[float]] = {}
            counts: Dict[str, int] = {}
            for s_idx, seq in enumerate(seqs):
                aa = seq[j0]
                groups.setdefault(aa, []).append(float(u1[s_idx]))
                counts[aa] = counts.get(aa, 0) + 1
            row_dict[j_1based] = {aa: float(np.mean(vals)) for aa, vals in groups.items()}
            aa_counts_by_pos[j_1based] = counts
        u_list.append(row_dict)

    # Synthesize U and fit OLS per PC
    coef_list: List[np.ndarray] = []
    upred = np.zeros((n_seq, len(nkp)), dtype=np.float64)
    for i in range(len(nkp)):
        cols = kp_lists[i]
        syn_mat = np.zeros((n_seq, len(cols)), dtype=np.float64)
        for c_idx, j_1based in enumerate(cols):
            j0 = j_1based - 1
            mapping = u_list[i][j_1based]
            syn_mat[:, c_idx] = [mapping[seq[j0]] for seq in seqs]

        if i in skip:
            coef = np.zeros(1 + len(cols), dtype=np.float64)
        else:
            coef = _ols_lm_r_style(syn_mat, svd["u"][:, i])
        coef_list.append(coef)
        upred[:, i] = coef[0] + syn_mat @ coef[1:]

    re_svd = (upred * svd["d"][None, :]) @ svd["v"].T
    true_mat = (svd["u"] * svd["d"][None, :]) @ svd["v"].T
    train_rmsd = float(np.sqrt(np.mean((re_svd - true_mat) ** 2)))

    return {
        "model": coef_list,
        "svd": svd,
        "keyPos": kp_lists,
        "trainRMSD": train_rmsd,
        "trainPred": re_svd,
        "uList": u_list,
        "alignment": ali_df,
        "aa_counts": aa_counts_by_pos,
    }


def makeSVDModel_singleMutation(
    mono_motifs: Sequence[Dict],
    Alignment: Union[pd.DataFrame, Sequence[str]],
    Positions: Sequence[str],
    mutation: int,
) -> Dict[str, Dict]:
    """
    Train per-position single-mutation SVD regression models across `Positions`.
    `mutation` is 1-based alignment index (`canonical - 1` for 57-aa HD alignment).
    Matches `makeSVDModel.singleMutation` in `FCfunctions.R`.
    """
    mut_1based = int(mutation)
    svd_model: Dict[str, Dict] = {}
    for pos in Positions:
        pos_mat = gene2pos(mono_motifs, pos=pos)
        svd = matrixSVD(pos_mat)
        svd_model[pos] = trainSVD(
            svd,
            Alignment,
            no_keyPos=[1, 1, 1],
            keyPos=[[mut_1based], [mut_1based], [mut_1based]],
        )
    return svd_model


class _MakeSVDModelNamespace:
    """Callable namespace exposing `makeSVDModel.singleMutation`."""

    @staticmethod
    def singleMutation(
        mono_motifs: Sequence[Dict],
        Alignment: Union[pd.DataFrame, Sequence[str]],
        Positions: Sequence[str],
        mutation: int,
    ) -> Dict[str, Dict]:
        return makeSVDModel_singleMutation(mono_motifs, Alignment, Positions, mutation)


makeSVDModel = _MakeSVDModelNamespace()


@lru_cache(maxsize=4)
def load_blosum62(
    path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, int], np.ndarray]:
    """Load and cache BLOSUM62 substitution matrix from `BLOSUM62.RData`."""
    b_path = DEFAULT_RAW_DIR / "BLOSUM62.RData" if path is None else Path(path)
    obj = rdata.read_rda(b_path)["BLOSUM62"]
    aa_names = [str(x) for x in obj.coords["dim_0"].values]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_names)}
    mat = np.asarray(obj.values, dtype=np.float64)
    return aa_to_idx, mat


def predict_SVD(
    svd_model: Dict,
    Alignment: Union[pd.DataFrame, Sequence[str], str],
    zero: float = 0.01,
    useSimilarAA: bool = False,
    blosum62: Optional[Tuple[Dict[str, int], np.ndarray]] = None,
    return_confidence: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
    """
    Predict (4, N_test) PSAM column(s) with a single-position SVD model.
    Matches `predict.SVD` in `FCfunctions.R`, including `useSimilarAA=True`
    BLOSUM62 similarity weighting `1 / (BLOSUM62[aa, aa] - BLOSUM62[aa, train_aa])`.
    """
    if len(svd_model["keyPos"]) != 3:
        raise ValueError("Cannot perform tetrahedron transformation, number of PCs should be 3")

    ali_df = _extract_alignment_df(Alignment)
    seqs = ali_df["alignment"].tolist()

    if useSimilarAA and blosum62 is None:
        blosum62 = load_blosum62()

    svd = svd_model["svd"]
    aa_counts_by_pos = svd_model.get("aa_counts", {})
    n_test = len(seqs)
    pred_coords = np.zeros((n_test, 3), dtype=np.float64)
    conf_list: List[float] = []

    for ali_idx, seq in enumerate(seqs):
        pred_u = np.zeros(3, dtype=np.float64)
        conf_num = 0.0
        conf_den = 0.0

        for i in range(3):
            cols = svd_model["keyPos"][i]
            coef = svd_model["model"][i]
            syn_u_test = np.zeros(len(cols), dtype=np.float64)
            confs = np.zeros(len(cols), dtype=np.float64)

            for k_idx, j_1based in enumerate(cols):
                aa = seq[j_1based - 1]
                uset = svd_model["uList"][i][j_1based]
                counts = aa_counts_by_pos.get(j_1based)
                if counts is None:
                    train_seqs = svd_model["alignment"]["alignment"].tolist()
                    counts = {}
                    for ts in train_seqs:
                        caa = ts[j_1based - 1]
                        counts[caa] = counts.get(caa, 0) + 1
                    aa_counts_by_pos[j_1based] = counts
                max_count = max(counts.values())

                if aa in uset:
                    syn_u_test[k_idx] = uset[aa]
                    confs[k_idx] = counts.get(aa, 0) / max_count
                elif not useSimilarAA or aa == "-":
                    syn_u_test[k_idx] = 0.0
                    confs[k_idx] = 0.0
                else:
                    aa_to_idx, b_mat = blosum62
                    ti = aa_to_idx[aa]
                    diag_val = float(b_mat[ti, ti])
                    train_keys = list(uset.keys())
                    diffs = np.array(
                        [diag_val - float(b_mat[ti, aa_to_idx[tk]]) for tk in train_keys],
                        dtype=np.float64,
                    )
                    if np.any(diffs <= 0.0):
                        # Guard degenerate non-standard characters (B/J/Z) if encountered
                        diffs = np.maximum(diffs, 1e-6)
                    sims = 1.0 / diffs
                    means_vec = np.array([uset[tk] for tk in train_keys], dtype=np.float64)
                    syn_u_test[k_idx] = float(np.sum(means_vec * sims) / np.sum(sims))
                    confs[k_idx] = float(np.max(sims) / max_count)

            pred_u[i] = float(coef[0] + np.sum(coef[1:] * syn_u_test))
            conf_num += float(np.sum(confs * np.abs(coef[1:])))
            conf_den += float(np.sum(coef[1:]))

        pred_coords[ali_idx, :] = (pred_u * svd["d"]) @ svd["v"].T
        conf_list.append(float(conf_num / conf_den) if conf_den != 0.0 else float("nan"))

    pred_mat = tetrahedron2matrix(pred_coords + svd["tetra_mean"][None, :])
    pred_mat[pred_mat < zero] = zero
    return (pred_mat, conf_list) if return_confidence else pred_mat


def predict_SVDmodel(
    model: Dict,
    Alignment: Union[pd.DataFrame, Sequence[str], str],
    zero: float = 0.01,
    useSimilarAA: bool = True,
    blosum62: Optional[Tuple[Dict[str, int], np.ndarray]] = None,
    positions: Optional[Sequence[str]] = None,
    return_confidence: bool = False,
):
    """
    Predict PSAM(s) using either a single-position `SVD` model dict or a
    multi-position `{pos: SVD}` dict returned by `makeSVDModel.singleMutation`.

    For a single sample and a multi-position dict, returns a `(4, P)` PSAM matrix.
    For `N > 1` samples, returns an `(N, 4, P)` array.
    """
    if "keyPos" in model:
        return predict_SVD(
            model,
            Alignment,
            zero=zero,
            useSimilarAA=useSimilarAA,
            blosum62=blosum62,
            return_confidence=return_confidence,
        )

    pos_keys = list(model.keys()) if positions is None else list(positions)
    ali_df = _extract_alignment_df(Alignment)
    n_test = len(ali_df)
    n_pos = len(pos_keys)
    out = np.zeros((n_test, 4, n_pos), dtype=np.float64)
    confs_last: List[float] = []
    for p_idx, pos in enumerate(pos_keys):
        col_pred, confs = predict_SVD(
            model[pos],
            ali_df,
            zero=zero,
            useSimilarAA=useSimilarAA,
            blosum62=blosum62,
            return_confidence=True,
        )
        out[:, :, p_idx] = col_pred.T
        confs_last = confs

    if n_test == 1:
        psam = out[0]
        return (psam, confs_last[0]) if return_confidence else psam
    return (out, confs_last) if return_confidence else out


class _PredictNamespace:
    """Callable namespace exposing `predict.SVD` and `predict.SVDmodel`."""

    def __call__(self, model: Dict, *args, **kwargs):
        return predict_SVDmodel(model, *args, **kwargs)

    @staticmethod
    def SVD(svd_model: Dict, *args, **kwargs):
        return predict_SVD(svd_model, *args, **kwargs)

    @staticmethod
    def SVDmodel(model: Dict, *args, **kwargs):
        return predict_SVDmodel(model, *args, **kwargs)


predict = _PredictNamespace()


def compute_tetra_dddg_matrix(
    target_psam: np.ndarray,
    ref_psam: np.ndarray,
    zero: float = 0.01,
    zero_diff_fallback: bool = False,
) -> np.ndarray:
    """
    Compute tetrahedral differential PSAM between `target_psam` and `ref_psam`,
    matching `HDMutantsPrediction.Rmd` lines 287-302 / 512-528.

    Returns the clamped (4, K) frequency matrix; pass through `frequency2ddG`
    to obtain -dddG/RT.
    """
    diff_tetra = matrix2tetrahedron(target_psam) - matrix2tetrahedron(ref_psam)
    mod = False
    if zero_diff_fallback and float(np.sum(diff_tetra)) == 0.0:
        diff_tetra[0, 0] = 0.01
        mod = True
    diff_mat = tetrahedron2matrix(diff_tetra)
    diff_mat[diff_mat < zero] = zero
    if mod:
        diff_mat[0, :] = 1.0
    return diff_mat


def _encode_seqs_array(seqs: Sequence[Union[str, bytes]]) -> np.ndarray:
    """Encode equal-length uppercase DNA strings into int64 matrix (N, L) in {0..4}."""
    arr = np.empty((len(seqs), len(seqs[0])), dtype=np.int64)
    for i, raw_s in enumerate(seqs):
        s_str = raw_s.decode("ascii") if isinstance(raw_s, (bytes, np.bytes_)) else str(raw_s)
        s_up = s_str.upper()
        arr[i, :] = [_BASE_TO_IDX.get(ch, 4) for ch in s_up]
    return arr


def score_12bp_with_psam(
    seqs_12bp: Sequence[str],
    psam_4x8: np.ndarray,
    mode: str = "aligned_core",
    center: str = "mean",
) -> np.ndarray:
    """
    Evaluate a 4x8 PSAM (rows A,C,G,T; cols N1..N8) on 12-bp aligned windows
    `[F-2, F-1, N1..N8, F+1, F+2]` or 8-bp core k-mers `[N1..N8]`.

    Parameters
    ----------
    seqs_12bp : sequence of str
        DNA sequences of length 12 (core at `[2:10]`) or length 8 (`[0:8]`).
        Longer sequences (>= 8 bp) are also supported when `mode` is
        `'boltzmann_rc'` or `'max_rc'`.
    psam_4x8 : array-like, shape (4, 8)
        Positive PSAM weights (e.g. predicted by FamilyCode or ProBound).
    mode : {'aligned_core', 'boltzmann_rc', 'max_rc'}
        - `'aligned_core'` (default): scores the aligned 8-bp core (`seq[2:10]`
          for 12-mers, `seq[0:8]` for 8-mers) directly.
        - `'boltzmann_rc'`: `logsumexp` over all forward and RC 8-mer frames.
        - `'max_rc'`: maximum score over all forward and RC 8-mer frames.
    center : {'mean', 'max'}
        - `'mean'` (default): column-centered `-ddG/RT` via `frequency2ddG`
          (zero mean over A/C/G/T at each position).
        - `'max'`: `log(psam) - max(log(psam))` per column (`<= 0`).
    """
    psam = np.asarray(psam_4x8, dtype=np.float64)
    if psam.shape != (4, 8):
        raise ValueError(f"Expected psam_4x8 shape (4, 8), got {psam.shape}")
    if np.any(psam <= 0.0) or not np.all(np.isfinite(psam)):
        raise ValueError("psam_4x8 entries must be strictly positive and finite")

    if isinstance(seqs_12bp, (str, bytes, np.bytes_)):
        seq_list = [
            seqs_12bp.decode("ascii")
            if isinstance(seqs_12bp, (bytes, np.bytes_))
            else str(seqs_12bp)
        ]
    else:
        seq_list = [
            s.decode("ascii") if isinstance(s, (bytes, np.bytes_)) else str(s)
            for s in seqs_12bp
        ]

    if len(seq_list) == 0:
        return np.zeros(0, dtype=np.float64)

    if center == "mean":
        ddg = frequency2ddG(psam)
    elif center == "max":
        lm = np.log(psam)
        ddg = lm - np.max(lm, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown center={center!r}; expected 'mean' or 'max'")

    # Add 5th row for 'N' matching FCfunctions.R::Scorekmer (mean across bases)
    ddg_n = np.vstack([ddg, np.mean(ddg, axis=0, keepdims=True)])  # (5, 8)
    pos_idx = np.arange(8, dtype=np.int64)

    seq_lens = {len(s) for s in seq_list}
    if len(seq_lens) != 1:
        raise ValueError(f"All sequences must have uniform length, got {sorted(seq_lens)}")
    L = next(iter(seq_lens))

    if mode == "aligned_core":
        if L == 12:
            cores = [s[2:10] for s in seq_list]
        elif L == 8:
            cores = seq_list
        else:
            raise ValueError(
                f"aligned_core mode expects 12-bp windows or 8-mers, got length {L}"
            )
        enc = _encode_seqs_array(cores)  # (N, 8)
        return np.sum(ddg_n[enc, pos_idx[None, :]], axis=1)

    if mode not in ("boltzmann_rc", "max_rc"):
        raise ValueError(f"Unknown mode={mode!r}")
    if L < 8:
        raise ValueError(f"Sequence length must be >= 8, got {L}")

    enc_full = _encode_seqs_array(seq_list)  # (N, L)
    # RC motif reverses A<->T, C<->G (rows 3..0) and columns (7..0)
    ddg_rc = ddg[::-1, ::-1]
    ddg_rc_n = np.vstack([ddg_rc, np.mean(ddg_rc, axis=0, keepdims=True)])  # (5, 8)

    n_frames = L - 8 + 1
    frame_scores = np.empty((len(seq_list), 2 * n_frames), dtype=np.float64)
    for f_i in range(n_frames):
        sl = enc_full[:, f_i : f_i + 8]
        frame_scores[:, 2 * f_i] = np.sum(ddg_n[sl, pos_idx[None, :]], axis=1)
        frame_scores[:, 2 * f_i + 1] = np.sum(ddg_rc_n[sl, pos_idx[None, :]], axis=1)

    if mode == "max_rc":
        return np.max(frame_scores, axis=1)
    return logsumexp(frame_scores, axis=1)


def load_familycode_hd_training(
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame]:
    """
    Load FamilyCode wild-type HD training alignment/motifs (414 HDs) plus the
    3 `add_hmm.sto` sequences (`HOXB6`, `NKX2-6`, `NKX2-4`) -> 417-row `CV_alignment_full`.
    """
    raw = Path(raw_dir)
    cv_ali = rdata.read_rds(raw / "HD_SELEX_PBM_CV_alignment.rsd").reset_index(drop=True)
    cv_mot_raw = rdata.read_rds(raw / "HD_SELEX_PBM_CV_motif.rsd")

    cv_motifs: List[Dict] = []
    for m in cv_mot_raw:
        nm_raw = m["name"]
        nm = str(nm_raw[0]) if isinstance(nm_raw, (list, tuple, np.ndarray)) else str(nm_raw)
        mat = np.asarray(m["matrix"].values, dtype=np.float64)
        cv_motifs.append({"name": nm, "matrix": mat, "pos_names": list(MOTIF_POSES_HD)})

    sto_lines = [
        line.strip()
        for line in (raw / "add_hmm.sto").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and line.strip() != "//"
    ]
    add_rows = [
        {"name": nm, "alignment": line.split()[1]}
        for nm, line in zip(["HOXB6", "NKX2-6", "NKX2-4"], sto_lines)
    ]
    cv_ali_full = pd.concat([cv_ali, pd.DataFrame(add_rows)], ignore_index=True)
    cv_ali_kock = cv_ali_full.iloc[:414].reset_index(drop=True)
    cv_mot_kock = cv_motifs[:414]
    return cv_ali_kock, cv_mot_kock, cv_ali_full
