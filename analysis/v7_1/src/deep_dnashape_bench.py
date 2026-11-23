"""
Authentic Rohs-Lab Deep DNAshape CPU Benchmark Core (`biophys_inductive_gfm/v7_1`).

Loads the official pre-trained TensorFlow v2 checkpoint weights from the locally
cloned Rohs Lab repository (`JinsenLi/deepDNAshape`, Li, Chiu & Rohs,
*Nat. Commun.* 2024, doi:10.1038/s41467-024-45191-5) directly via a pure-Python
LevelDB SSTable + TensorFlow BundleEntryProto parser, and executes the exact
message-passing graph + BatchNorm + Keras GRUCell (`reset_after=True`) +
reverse-complement averaging + `params.json` rescaling forward pass on CPU
using PyTorch (0 GPU, 0 network calls).

Provides:
  1. Exact verification of Deep DNAshape Layers 1/2/4 vs authentic pentamer
     `DNAshapeR` (`pentamer_dnashape.npz`) across all 1,024 pentamers and
     real 36-bp probes, plus strict RC-equivariance verification.
  2. Benchmark A (bHLH `GSE59845` gcPBM, all 6 primary endpoints, 5-fold
     central-10mer `GroupKFold`): static `DNAshapeR (7ch)` vs context-extended
     `DeepDNAshape_L4` (10-pos, 14-pos, and 14-channel sensitivity) on top of
     `Seq_G2`, `Seq_Pos4mer`, and `HGBDT_Seq`, PLUS strict flank-matched 22-bp
     pure-sequence controls (`Seq_G2_Flank22`, `Seq_Pos4mer_Flank22`,
     `HGBDT_Seq_Flank22`) with core-cluster `B=2,000` paired bootstrap 95% CIs
     and cross-6-endpoint Holm-Bonferroni `p_holm`.
  3. Benchmark B (Homeodomain `GSE233827`, 92 mutants, 30-fold LOGO on `dz_f70`
     and `dz_spec_f70`): `Mut x (Seq_G2 / Pos4mer + DeepDNAshape_L4)` and
     `HGBDT_Mut_x_(Seq + DeepDNAshape_L4)` vs `DNAshapeR_N1N4` and strict
     15-bp flank-matched pure-sequence controls (`[F-5..F+2]`), with `n=30`
     gene-cluster `B=5,000` paired bootstrap 95% CIs and gene Wilcoxon tests.

Offline / zero network imports for ScientistPro CoE Pillar 2 AST compliance.
"""
from __future__ import annotations

import gc
import itertools
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve, eigh as sp_eigh, solve_triangular
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V3_SRC = ROOT / "v3" / "src"
V4_SRC = ROOT / "v4" / "src"
V5_SRC = ROOT / "v5" / "src"
V7_SRC = ROOT / "v7" / "src"
V7_1_SRC = ROOT / "v7_1" / "src"
for _p in (V7_1_SRC, V7_SRC, V5_SRC, V4_SRC, V3_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bhlh_cross_family as bf  # noqa: E402
import hgbdt_surrogate_core as hsc  # noqa: E402
import kock_pbm_v7 as kv7  # noqa: E402
import paired_perturbation_v5 as pp5  # noqa: E402

DEEP_DNASHAPE_ROOT = ROOT / "data" / "raw_external" / "deepDNAshape" / "deepDNAshape"
DEEP_DNASHAPE_MODELS_DIR = DEEP_DNASHAPE_ROOT / "models"
DEEP_DNASHAPE_PARAMS_PATH = DEEP_DNASHAPE_ROOT / "params.json"
DNASHAPER_NPZ_PATH = ROOT / "data" / "biophysics_tables" / "pentamer_dnashape.npz"

# Exact 7 RC-even channel intersection with pentamer DNAshapeR (ordered identically to kv7)
CH7_INTERSECTION: Tuple[str, ...] = ("MGW", "ProT", "EP", "Stretch", "Roll", "HelT", "Slide")
# All 14 structural channels in Deep DNAshape (8 intrabase + 6 interbase)
CH14_STRUCTURAL: Tuple[str, ...] = (
    "MGW", "ProT", "EP", "Stretch", "Roll", "HelT", "Slide",
    "Stagger", "Opening", "Rise",
    "Shear", "Buckle", "Shift", "Tilt",
)
INTRABASE_FEATURES = frozenset({"Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", "MGW", "EP"})
INTERBASE_FEATURES = frozenset({"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"})
X_AXIS_RC_ODD_FEATURES = frozenset({"Shift", "Tilt", "Shear", "Buckle"})

# Official Deep DNAshape base ordering from Model_utils.py: ["T", "G", "C", "A"]
_DD_BASES = ("T", "G", "C", "A")
_DD_BASE_MAP: Dict[str, np.ndarray] = {
    b: np.eye(4, dtype=np.float32)[i] for i, b in enumerate(_DD_BASES)
}
_DD_BASE_MAP["N"] = np.full(4, 0.25, dtype=np.float32)

_DD_DI_MAP: Dict[Tuple[str, str], np.ndarray] = {}
for _idx, (_b1, _b2) in enumerate(itertools.product(_DD_BASES, repeat=2)):
    _v = np.zeros(16, dtype=np.float32)
    _v[_idx] = 1.0
    _DD_DI_MAP[(_b1, _b2)] = _v
_DD_DI_MAP[("N", "N")] = np.full(16, 1.0 / 16.0, dtype=np.float32)
for _b1 in _DD_BASES:
    _DD_DI_MAP[(_b1, "N")] = sum(_DD_DI_MAP[(_b1, _b2)] for _b2 in _DD_BASES) / 4.0
    _DD_DI_MAP[("N", _b1)] = sum(_DD_DI_MAP[(_b2, _b1)] for _b2 in _DD_BASES) / 4.0

_RC_TRANTAB_N = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Orthonormal 3-column zero-sum contrast per base (A=0, C=1, G=2, T=3) + N=4 -> [0, 0, 0]
_MONO_H_N = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
) * 0.5

_ASCII_LUT_N = np.full(256, -1, dtype=np.int8)
for _c, _v in (
    ("A", 0), ("a", 0), ("C", 1), ("c", 1),
    ("G", 2), ("g", 2), ("T", 3), ("t", 3),
    ("N", 4), ("n", 4),
):
    _ASCII_LUT_N[ord(_c)] = _v

# Precompute 10 canonical dinucleotide remap & RC map for N-aware codes (0..3, 4=N)
_UNIQ_DI_10 = sorted({min(x, ((3 - (x & 3)) << 2) | (3 - (x >> 2))) for x in range(16)})
_DI_REMAP_10 = np.zeros(16, dtype=np.int32)
for _iu, _vu in enumerate(_UNIQ_DI_10):
    _DI_REMAP_10[_vu] = _iu

# Precompute exact 4-mer probability distributions over 256 states for any 4-char ACGTN tuple
# (maps 5^4 = 625 possible ACGTN 4-mers -> sparse (indices, weights) for N-aware positional 4-mers)
_RC4_MAP = bf._RC4_MAP
_CANON4_IDS = bf._CANON4_IDS
_REMAP_CANON4 = np.full(256, -1, dtype=np.int32)
_REMAP_CANON4[_CANON4_IDS] = np.arange(len(_CANON4_IDS), dtype=np.int32)


def revcomp_n(seq: str) -> str:
    return seq.translate(_RC_TRANTAB_N)[::-1]


def encode_acgtn(seqs: Sequence[str]) -> np.ndarray:
    """Encode equal-length ACGTN strings to (N, L) int8 in {0,1,2,3,4} (4='N')."""
    n = len(seqs)
    L = len(seqs[0])
    buf = "".join(seqs).encode("ascii")
    arr = _ASCII_LUT_N[np.frombuffer(buf, dtype=np.uint8)].reshape(n, L).copy()
    if np.any(arr < 0):
        raise ValueError("Invalid non-ACGTN character encountered")
    return arr


def avail_ram_gib() -> float:
    """Return available system RAM in GiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


# ==============================================================================
# 1. Pure-Python TF-v2 Checkpoint Parser + Exact PyTorch CPU Deep DNAshape Engine
# ==============================================================================

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    res = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7


def _parse_bundle_entry_proto(val: bytes) -> Tuple[int, Tuple[int, ...], int, int]:
    """Parse TensorFlow BundleEntryProto -> `(dtype, shape, offset, size)`."""
    pos = 0
    n = len(val)
    dtype = 0
    shape: List[int] = []
    offset = 0
    size = 0
    while pos < n:
        tag, pos = _read_varint(val, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            v, pos = _read_varint(val, pos)
            if field == 1:
                dtype = v
            elif field == 4:
                offset = v
            elif field == 5:
                size = v
        elif wire == 2:
            ln, pos = _read_varint(val, pos)
            sub = val[pos : pos + ln]
            pos += ln
            if field == 2:
                sp = 0
                while sp < len(sub):
                    st, sp = _read_varint(sub, sp)
                    sf = st >> 3
                    sw = st & 7
                    if sf == 2 and sw == 2:
                        dl, sp = _read_varint(sub, sp)
                        dsub = sub[sp : sp + dl]
                        sp += dl
                        dp = 0
                        dv = 0
                        while dp < len(dsub):
                            dt, dp = _read_varint(dsub, dp)
                            df = dt >> 3
                            dw = dt & 7
                            if dw == 0:
                                val0, dp = _read_varint(dsub, dp)
                                if df == 1:
                                    dv = val0
                            elif dw == 2:
                                dsl, dp = _read_varint(dsub, dp)
                                dp += dsl
                        shape.append(dv)
                    elif sw == 0:
                        _, sp = _read_varint(sub, sp)
                    elif sw == 2:
                        sl, sp = _read_varint(sub, sp)
                        sp += sl
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
    return dtype, tuple(shape), offset, size


def load_tf_checkpoint_weights(prefix: Path) -> Dict[str, torch.Tensor]:
    """
    Parse an uncompressed LevelDB SSTable TF v2 checkpoint (`<prefix>.index` +
    `<prefix>.data-00000-of-00001`) into exact float32 PyTorch tensors.
    """
    idx_path = Path(str(prefix) + ".index")
    dat_path = Path(str(prefix) + ".data-00000-of-00001")
    raw = idx_path.read_bytes()
    footer = raw[-48:]
    magic = struct.unpack("<Q", footer[-8:])[0]
    if magic != 0xDB4775248B80FB57:
        raise ValueError(f"Unexpected LevelDB SSTable magic {hex(magic)} in {idx_path}")

    pos = 0
    _, pos = _read_varint(footer, pos)
    _, pos = _read_varint(footer, pos)
    idx_off, pos = _read_varint(footer, pos)
    idx_sz, pos = _read_varint(footer, pos)

    def _iter_block(off: int, sz: int):
        comp_type = raw[off + sz]
        if comp_type != 0:
            raise ValueError(f"Unsupported SSTable block compression type {comp_type}")
        blk = raw[off : off + sz]
        nr = struct.unpack("<I", blk[-4:])[0]
        lim = len(blk) - 4 * (nr + 1)
        p = 0
        prev = b""
        while p < lim:
            sh, p = _read_varint(blk, p)
            nsh, p = _read_varint(blk, p)
            vl, p = _read_varint(blk, p)
            k = prev[:sh] + blk[p : p + nsh]
            p += nsh
            v = blk[p : p + vl]
            p += vl
            prev = k
            yield k.decode("utf-8", errors="replace"), v

    data = dat_path.read_bytes()
    tensors: Dict[str, torch.Tensor] = {}
    for _, handle in _iter_block(idx_off, idx_sz):
        hp = 0
        bo, hp = _read_varint(handle, hp)
        bs, hp = _read_varint(handle, hp)
        for k, v in _iter_block(bo, bs):
            if k.endswith("/.ATTRIBUTES/VARIABLE_VALUE") and "optimizer" not in k:
                dt, sh, off, sz = _parse_bundle_entry_proto(v)
                expected_bytes = int(np.prod(sh)) * 4
                if dt != 1 or sz != expected_bytes:
                    raise ValueError(f"Unexpected tensor {k}: dtype={dt}, shape={sh}, sz={sz}")
                arr = np.frombuffer(
                    data, dtype="<f4", count=int(np.prod(sh)), offset=off
                ).reshape(sh).copy()
                tensors[k.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")] = torch.from_numpy(arr)
    return tensors


class AuthenticDeepDNAshapeCPU:
    """
    Exact CPU-only PyTorch inference engine for Rohs Lab `deepDNAshape`
    (`Model.py` + `predictor.py`, `mp_layer=7`, `filter_size=64`, `multiply='add'`,
    `BatchNormalization(eps=1e-3)`, Keras `GRUCell(reset_after=True)`).
    """

    def __init__(
        self,
        models_dir: Path = DEEP_DNASHAPE_MODELS_DIR,
        params_path: Path = DEEP_DNASHAPE_PARAMS_PATH,
    ) -> None:
        torch.set_num_threads(1)
        self.models_dir = Path(models_dir)
        self.params: Dict[str, Dict[str, object]] = json.loads(
            Path(params_path).read_text(encoding="utf-8")
        )
        self._weights: Dict[str, Dict[str, torch.Tensor]] = {}

    def _get_weights(self, feature: str) -> Dict[str, torch.Tensor]:
        if feature not in self._weights:
            self._weights[feature] = load_tf_checkpoint_weights(self.models_dir / feature)
        return self._weights[feature]

    def _rescale(self, feature: str, pred: np.ndarray) -> np.ndarray:
        p = self.params[feature]
        m = str(p["method"])
        if m == "minmax":
            return pred * (float(p["max"]) - float(p["min"])) + float(p["min"])
        if m == "minmax2":
            return (pred + 1.0) * (float(p["max"]) - float(p["min"])) / 2.0 + float(p["min"])
        if m == "sin":
            return np.arcsin(np.clip(pred, -1.0, 1.0)) / np.pi * 180.0
        if m == "standard":
            return pred * float(p["std"]) + float(p["mean"])
        return pred * float(p["percentile_range"]) + float(p["median"])

    @staticmethod
    def _encode_batch_inputs(seqs: Sequence[str], is_interbase: bool) -> torch.Tensor:
        """Vectorized encoding of equal-length strings with official `NN` flanking pads."""
        n = len(seqs)
        L = len(seqs[0])
        codes = encode_acgtn(seqs)  # (n, L) in {0:A, 1:C, 2:G, 3:T, 4:N}
        # Map {0:A, 1:C, 2:G, 3:T, 4:N} -> Deep DNAshape order {0:T, 1:G, 2:C, 3:A, 4:N}
        dd_lut = np.array([3, 2, 1, 0, 4], dtype=np.int32)
        c_dd = dd_lut[codes]
        pad = np.full((n, 2), 4, dtype=np.int32)
        c_pad = np.concatenate([pad, c_dd, pad], axis=1)  # (n, L+4)

        if not is_interbase:
            mono_tbl = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                dtype=np.float32,
            )
            return torch.from_numpy(mono_tbl[c_pad])

        # Interbase dinucleotide encoding (5x5 -> 16)
        di_tbl = np.zeros((5, 5, 16), dtype=np.float32)
        names5 = ("T", "G", "C", "A", "N")
        for i, b1 in enumerate(names5):
            for j, b2 in enumerate(names5):
                di_tbl[i, j] = _DD_DI_MAP[(b1, b2)]
        return torch.from_numpy(di_tbl[c_pad[:, :-1], c_pad[:, 1:]])

    @torch.no_grad()
    def predict_equal_len(
        self,
        feature: str,
        seqs: Sequence[str],
        layers: Sequence[int] = (2, 4),
        batch_size: int = 1024,
    ) -> Dict[int, np.ndarray]:
        """
        Predict exact Deep DNAshape profiles for equal-length sequences across
        requested layers `0..7`. Returns `{layer: array}` where array has shape:
          - `(N, L)` for intrabase features
          - `(N, L - 1)` for interbase features (step `s` = dinuc `(s, s+1)`)
        Matches official `predictor.py::predictBatch` exactly (fwd + RC avg, `NN` pad).
        """
        max_layer = int(max(layers))
        if max_layer < 0 or max_layer > 7:
            raise ValueError(f"Layers must be in 0..7, got {layers}")
        W = self._get_weights(feature)
        is_inter = feature in INTERBASE_FEATURES
        is_odd = feature in X_AXIS_RC_ODD_FEATURES
        n = len(seqs)
        L = len(seqs[0])
        out_len = L - 1 if is_inter else L
        req_layers = sorted(set(int(lv) for lv in layers))
        out_map = {lv: np.empty((n, out_len), dtype=np.float32) for lv in req_layers}

        sc_k = W["selfconv/kernel"][0]
        sc_b = W["selfconv/bias"]

        layer_params = []
        for li in range(max_layer):
            p = f"mp_layers/{li}"
            layer_params.append(
                (
                    W[f"{p}/Weight1"],
                    W[f"{p}/Weight2"],
                    W[f"{p}/Bias1"],
                    W[f"{p}/Weight1_term2"],
                    W[f"{p}/Weight2_term2"],
                    W[f"{p}/BiasAll"],
                    W[f"{p}/bn/gamma"],
                    W[f"{p}/bn/beta"],
                    W[f"{p}/bn/moving_mean"],
                    torch.rsqrt(W[f"{p}/bn/moving_variance"] + 1e-3),
                    W[f"{p}/gru_layer/kernel"],
                    W[f"{p}/gru_layer/recurrent_kernel"],
                    W[f"{p}/gru_layer/bias"][0],
                    W[f"{p}/gru_layer/bias"][1],
                )
            )

        def _forward_dir(seq_chunk: Sequence[str]) -> Dict[int, np.ndarray]:
            X = self._encode_batch_inputs(seq_chunk, is_interbase=is_inter)
            x = torch.matmul(X, sc_k) + sc_b
            res: Dict[int, np.ndarray] = {}
            if 0 in req_layers:
                res[0] = x.mean(dim=-1).numpy()
            for li, lp in enumerate(layer_params):
                (
                    w1, w2, b1, w1t, w2t, ball,
                    gamma, beta, mm, inv_sd,
                    gk, grk, gb0, gb1,
                ) = lp
                prev_x = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
                next_x = torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
                aggre = torch.matmul(next_x, w1) + torch.matmul(prev_x, w2) + b1
                aggre = aggre + torch.matmul(next_x, w1t) + torch.matmul(prev_x, w2t)
                aggre = torch.relu(aggre * x + ball)
                aggre = gamma * (aggre - mm) * inv_sd + beta
                mx = torch.matmul(aggre, gk) + gb0
                mh = torch.matmul(x, grk) + gb1
                xz, xr, xh = mx.chunk(3, dim=-1)
                hz, hr, hh = mh.chunk(3, dim=-1)
                z = torch.sigmoid(xz + hz)
                r = torch.sigmoid(xr + hr)
                cand = torch.tanh(xh + r * hh)
                x = z * x + (1.0 - z) * cand
                lv = li + 1
                if lv in req_layers:
                    res[lv] = x.mean(dim=-1).numpy()
            return res

        for b0 in range(0, n, batch_size):
            b1 = min(b0 + batch_size, n)
            chunk = seqs[b0:b1]
            rc_chunk = [revcomp_n(s) for s in chunk]
            fwd_d = _forward_dir(chunk)
            rev_d = _forward_dir(rc_chunk)
            for lv in req_layers:
                pf = self._rescale(feature, fwd_d[lv])
                pr = self._rescale(feature, rev_d[lv])
                if is_odd:
                    pr = -pr
                pr = pr[:, ::-1]
                avg = 0.5 * (pf + pr)
                out_map[lv][b0:b1] = avg[:, 2:-2].astype(np.float32)
        return out_map

    def predict_base_aligned_stack(
        self,
        features: Sequence[str],
        seqs: Sequence[str],
        layers: Sequence[int] = (2, 4),
        batch_size: int = 1024,
    ) -> Dict[int, np.ndarray]:
        """
        Predict base-aligned `(N, L, len(features))` float32 array for equal-length
        sequences of length `L`.
        For intrabase features, position `i` is `pred[:, i]`.
        For interbase features (`Roll`, `HelT`, `Slide`, `Rise`, `Shift`, `Tilt`),
        interior base position `i` (`1 <= i <= L-2`) averages the two flanking
        base-pair steps `0.5 * (step[i-1] + step[i])` — matching DNAshapeR's
        exact RC-even `0.5 * (Roll1 + Roll2)` convention (boundary `i=0` uses
        `step[0]`, `i=L-1` uses `step[L-2]`).
        """
        n = len(seqs)
        L = len(seqs[0])
        c = len(features)
        req_layers = sorted(set(int(lv) for lv in layers))
        out = {lv: np.empty((n, L, c), dtype=np.float32) for lv in req_layers}

        for fi, feat in enumerate(features):
            pred_map = self.predict_equal_len(feat, seqs, layers=req_layers, batch_size=batch_size)
            is_inter = feat in INTERBASE_FEATURES
            for lv in req_layers:
                arr = pred_map[lv]
                if not is_inter:
                    out[lv][:, :, fi] = arr
                else:
                    # arr has shape (n, L-1): step s connects base s and s+1
                    base_arr = np.empty((n, L), dtype=np.float32)
                    base_arr[:, 0] = arr[:, 0]
                    base_arr[:, -1] = arr[:, -1]
                    base_arr[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:])
                    out[lv][:, :, fi] = base_arr
        return out


def verify_deep_dnashape_vs_dnashaper(
    engine: AuthenticDeepDNAshapeCPU,
    sample_36bp_seqs: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Verify authentic Rohs Lab Deep DNAshape against pentamer DNAshapeR
    (`pentamer_dnashape.npz`) across:
      1. All 1,024 pentamers (Layers 1, 2, 4; Layer 2 = exact 5-bp pentamer RF).
      2. Real 36-bp probes (interior positions `4..31`, Layers 1, 2, 4) if provided.
      3. Strict RC-equivariance check across all 14 structural channels.
    """
    d = np.load(DNASHAPER_NPZ_PATH, allow_pickle=False)
    pents = ["".join(t) for t in itertools.product("ACGT", repeat=5)]

    pent_corr: Dict[str, Dict[str, float]] = {}
    for feat in CH7_INTERSECTION:
        pmap = engine.predict_equal_len(feat, pents, layers=(1, 2, 4), batch_size=1024)
        if feat in INTERBASE_FEATURES:
            ref_avg = 0.5 * (d[feat + "1"].astype(np.float64) + d[feat + "2"].astype(np.float64))
            ref_s1 = d[feat + "1"].astype(np.float64)
            rec: Dict[str, float] = {}
            for lv in (1, 2, 4):
                v_avg = 0.5 * (pmap[lv][:, 1].astype(np.float64) + pmap[lv][:, 2].astype(np.float64))
                v_s1 = pmap[lv][:, 1].astype(np.float64)
                rec[f"L{lv}_pearson_avg"] = float(pearsonr(v_avg, ref_avg)[0])
                rec[f"L{lv}_spearman_avg"] = float(spearmanr(v_avg, ref_avg)[0])
                rec[f"L{lv}_pearson_step1"] = float(pearsonr(v_s1, ref_s1)[0])
                rec[f"L{lv}_mae_avg"] = float(np.mean(np.abs(v_avg - ref_avg)))
        else:
            ref = d[feat].astype(np.float64)
            rec = {}
            for lv in (1, 2, 4):
                v = pmap[lv][:, 2].astype(np.float64)
                rec[f"L{lv}_pearson_avg"] = float(pearsonr(v, ref)[0])
                rec[f"L{lv}_spearman_avg"] = float(spearmanr(v, ref)[0])
                rec[f"L{lv}_mae_avg"] = float(np.mean(np.abs(v - ref)))
        pent_corr[feat] = rec

    # RC-equivariance verification across all 14 structural channels on 16 random 36-mers
    rng = np.random.default_rng(2026073)
    test_seqs = ["".join(rng.choice(list("ACGT"), size=36)) for _ in range(16)]
    rc_seqs = [revcomp_n(s) for s in test_seqs]
    rc_max_err: Dict[str, float] = {}
    for feat in CH14_STRUCTURAL:
        pf = engine.predict_equal_len(feat, test_seqs, layers=(4,), batch_size=32)[4]
        pr = engine.predict_equal_len(feat, rc_seqs, layers=(4,), batch_size=32)[4]
        expected = -pf[:, ::-1] if feat in X_AXIS_RC_ODD_FEATURES else pf[:, ::-1]
        rc_max_err[feat] = float(np.max(np.abs(pr - expected)))

    real_probe_corr: Dict[str, Dict[str, float]] = {}
    if sample_36bp_seqs is not None and len(sample_36bp_seqs) > 0:
        sub = list(sample_36bp_seqs[:500])
        codes = bf.encode_sequences_acgt(sub)  # (M, 36)
        c5, _ = bf.extract_c5_pair(codes)  # (M, 32) centered at pos 2..33
        F_true, _ = kv7.load_biophysical_table_matrix(include_odd_step_diff=False)
        ref_36 = F_true[c5]  # (M, 32, 7) -> interior positions 2..33 (slice [2:30] -> pos 4..31)
        pred_36 = engine.predict_base_aligned_stack(CH7_INTERSECTION, sub, layers=(1, 2, 4), batch_size=500)
        for fi, feat in enumerate(CH7_INTERSECTION):
            ref_v = ref_36[:, 2:30, fi].ravel().astype(np.float64)  # pos 4..31
            r_rec: Dict[str, float] = {}
            for lv in (1, 2, 4):
                pv = pred_36[lv][:, 4:32, fi].ravel().astype(np.float64)
                r_rec[f"L{lv}_pearson"] = float(pearsonr(pv, ref_v)[0])
                r_rec[f"L{lv}_spearman"] = float(spearmanr(pv, ref_v)[0])
                r_rec[f"L{lv}_mae"] = float(np.mean(np.abs(pv - ref_v)))
            real_probe_corr[feat] = r_rec

    l2_rs = [pent_corr[f]["L2_pearson_avg"] for f in CH7_INTERSECTION]
    l2_core6_rs = [pent_corr[f]["L2_pearson_avg"] for f in CH7_INTERSECTION if f != "Stretch"]
    return {
        "source_repo": "https://github.com/JinsenLi/deepDNAshape (Li, Chiu & Rohs, Nat. Commun. 2024, doi:10.1038/s41467-024-45191-5)",
        "local_checkpoint_dir": str(engine.models_dir),
        "inference_backend": "pure_python_tf_v2_sstable_loader_plus_exact_pytorch_cpu",
        "receptive_field_architecture_note": (
            "Deep DNAshape uses selfconv (Layer 0: 1-bp intra / 2-bp inter) + 7 message-passing GRU layers: "
            "Layer 1 = 3-bp intra / 4-bp inter; Layer 2 = exact 5-bp pentamer intra / 6-bp inter (DNAshapeR RF); "
            "Layer 4 (official default) = 9-bp intra (+/-4 bp) / 10-bp inter (union +/-5 bp across flanking steps)."
        ),
        "channel_intersection": {
            "dnashaper_rc_even_7ch": list(CH7_INTERSECTION),
            "deep_dnashape_structural_14ch": list(CH14_STRUCTURAL),
            "deep_dnashape_intrabase_8ch": sorted(INTRABASE_FEATURES),
            "deep_dnashape_interbase_6ch": sorted(INTERBASE_FEATURES),
            "deep_dnashape_rc_odd_4ch": sorted(X_AXIS_RC_ODD_FEATURES),
            "exact_7ch_intersection_fraction": "7/7 (100% of DNAshapeR RC-even channels present in Deep DNAshape)",
        },
        "pentamer_1024_vs_dnashaper": pent_corr,
        "pentamer_L2_mean_pearson_7ch": float(np.mean(l2_rs)),
        "pentamer_L2_mean_pearson_core6ch_excl_stretch": float(np.mean(l2_core6_rs)),
        "real_36bp_probes_vs_dnashaper": real_probe_corr,
        "rc_equivariance_max_abs_err_14ch": rc_max_err,
        "rc_equivariance_max_overall": float(max(rc_max_err.values())),
    }


# ==============================================================================
# 2. N-Aware Sequence Encoders (Flank-Matched Pure-Sequence Controls)
# ==============================================================================

def _add_4mer_window_probs(
    out: np.ndarray,
    col_offset: int,
    c4_codes: np.ndarray,
    weight: float = 1.0,
    rc_flip: bool = False,
    canon_fold: bool = False,
) -> None:
    """
    Accumulate exact 4-mer probabilities for shape `(N, 4)` ACGTN codes into
    `out[:, col_offset : col_offset + W]`. For pure ACGT rows (no 'N'=4),
    sets exact one-hot; for boundary rows containing 'N', distributes exact
    uniform `1 / 4^(#N)` weight over matching ACGT 4-mers (matching Deep
    DNAshape's uniform `0.25` 'N' base encoding).
    """
    n = c4_codes.shape[0]
    has_n = np.any(c4_codes == 4, axis=1)
    clean_idx = np.flatnonzero(~has_n)
    if len(clean_idx) > 0:
        cc = c4_codes[clean_idx].astype(np.int32)
        k4 = (((cc[:, 0] << 2) | cc[:, 1]) << 2 | cc[:, 2]) << 2 | cc[:, 3]
        if rc_flip:
            k4 = _RC4_MAP[k4]
        if canon_fold:
            k4 = _REMAP_CANON4[np.minimum(k4, _RC4_MAP[k4])]
        np.add.at(out, (clean_idx, col_offset + k4), weight)

    n_idx = np.flatnonzero(has_n)
    for r in n_idx:
        row = c4_codes[r]
        choices = [range(4) if int(b) == 4 else (int(b),) for b in row]
        combos = list(itertools.product(*choices))
        w = weight / float(len(combos))
        for b0, b1, b2, b3 in combos:
            k4 = (((b0 << 2) | b1) << 2 | b2) << 2 | b3
            if rc_flip:
                k4 = int(_RC4_MAP[k4])
            if canon_fold:
                k4 = int(_REMAP_CANON4[min(k4, int(_RC4_MAP[k4]))])
            out[r, col_offset + k4] += w


def encode_rc_sym_g2_and_pos4mer_acgtn(
    codes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build strictly RC-invariant (`phi(s) == phi(rc(s))`) `X_g1`, `X_g2`, and
    `X_pos4mer` on equal-length `(N, L)` ACGTN arrays (4 = 'N', uniform 0.25
    over A,C,G,T). Reduces identically to `bf.build_rc_sym_ebox_features` when
    all bases are ACGT and `L=14`.
    """
    n, L = codes.shape
    rc_lut = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    rc_codes = rc_lut[codes[:, ::-1]]

    m_f = _MONO_H_N[codes]       # (N, L, 3)
    m_r = _MONO_H_N[rc_codes]    # (N, L, 3)
    m_sym = 0.5 * (m_f + m_r)
    half_m = (L + 1) // 2
    X_g1 = m_sym[:, :half_m, :].reshape(n, half_m * 3)

    di_f = np.einsum("npa,npb->npab", m_f[:, :-1, :], m_f[:, 1:, :]).reshape(n, L - 1, 9)
    di_r = np.einsum("npa,npb->npab", m_r[:, :-1, :], m_r[:, 1:, :]).reshape(n, L - 1, 9)
    di_sym = 0.5 * (di_f + di_r)
    half_d = (L - 1 + 1) // 2
    X_di = di_sym[:, :half_d, :].reshape(n, half_d * 9)

    # N-aware GC fraction: A/T=0, C/G=1, N=0.5
    gc_lut = np.array([0.0, 1.0, 1.0, 0.0, 0.5], dtype=np.float64)
    gc = gc_lut[codes].mean(axis=1)

    # N-aware bulk 10 RC-canonical dinuc counts
    di_w_tbl = np.zeros((5, 5, 10), dtype=np.float64)
    for b1 in range(5):
        c1 = range(4) if b1 == 4 else (b1,)
        for b2 in range(5):
            c2 = range(4) if b2 == 4 else (b2,)
            combos = [(x, y) for x in c1 for y in c2]
            w = 1.0 / float(len(combos))
            for x, y in combos:
                idx2 = (x << 2) | y
                rc2 = ((3 - y) << 2) | (3 - x)
                di_w_tbl[b1, b2, _DI_REMAP_10[min(idx2, rc2)]] += w
    di_bulk = di_w_tbl[codes[:, :-1], codes[:, 1:]].mean(axis=1)

    X_g2 = np.hstack([X_g1, X_di, gc[:, None], (gc * gc)[:, None], di_bulk])

    # RC-folded positional 4-mers across L - 3 slots
    L4 = L - 3
    half = L4 // 2
    n_cols = half * 256 + (len(_CANON4_IDS) if L4 % 2 == 1 else 0)
    X_pos4 = np.zeros((n, n_cols), dtype=np.float64)
    for p in range(half):
        q = L4 - 1 - p
        _add_4mer_window_probs(X_pos4, p * 256, codes[:, p : p + 4], weight=0.5, rc_flip=False, canon_fold=False)
        _add_4mer_window_probs(X_pos4, p * 256, codes[:, q : q + 4], weight=0.5, rc_flip=True, canon_fold=False)
    if L4 % 2 == 1:
        mid = half
        _add_4mer_window_probs(X_pos4, half * 256, codes[:, mid : mid + 4], weight=1.0, rc_flip=False, canon_fold=True)

    return {"X_g1": X_g1, "X_g2": X_g2, "X_pos4mer": X_pos4}


def encode_oriented_g2_acgtn(codes: np.ndarray) -> np.ndarray:
    """
    Oriented `Seq_G2` on `(N, L)` ACGTN window (for homeodomain oriented sites):
    `L*3` mono + `(L-1)*9` adj-dinuc + `2` GC/GC^2 + `10` bulk RC-canon dinuc.
    Matches `kv7.encode_seq_g2_12bp` identically when `L=12` and all ACGT.
    """
    n, L = codes.shape
    mf = _MONO_H_N[codes]  # (N, L, 3)
    x_g1 = mf.reshape(n, L * 3)
    di_adj = np.einsum("npa,npb->npab", mf[:, :-1, :], mf[:, 1:, :]).reshape(n, (L - 1) * 9)

    gc_lut = np.array([0.0, 1.0, 1.0, 0.0, 0.5], dtype=np.float64)
    gc = gc_lut[codes].mean(axis=1)

    di_w_tbl = np.zeros((5, 5, 10), dtype=np.float64)
    for b1 in range(5):
        c1 = range(4) if b1 == 4 else (b1,)
        for b2 in range(5):
            c2 = range(4) if b2 == 4 else (b2,)
            combos = [(x, y) for x in c1 for y in c2]
            w = 1.0 / float(len(combos))
            for x, y in combos:
                idx2 = (x << 2) | y
                rc2 = ((3 - y) << 2) | (3 - x)
                di_w_tbl[b1, b2, _DI_REMAP_10[min(idx2, rc2)]] += w
    di_bulk = di_w_tbl[codes[:, :-1], codes[:, 1:]].mean(axis=1)
    return np.hstack([x_g1, di_adj, gc[:, None], (gc * gc)[:, None], di_bulk]).astype(np.float64)


def encode_oriented_pos4mer_acgtn(
    codes: np.ndarray,
    n_slots: int,
) -> np.ndarray:
    """Oriented positional 4-mers across first `n_slots` positions (`[0:4]..[n_slots-1:n_slots+3]`)."""
    n = codes.shape[0]
    out = np.zeros((n, n_slots * 256), dtype=np.float32)
    for k_i in range(n_slots):
        _add_4mer_window_probs(out, k_i * 256, codes[:, k_i : k_i + 4], weight=1.0, rc_flip=False, canon_fold=False)
    return out


# ==============================================================================
# 3. Benchmark A — bHLH GSE59845 gcPBM (All 6 Endpoints)
# ==============================================================================

def _rc_symmetrize_shape_profile(
    prof: np.ndarray,
    features: Sequence[str],
) -> np.ndarray:
    """
    Strictly RC-symmetrize a `(..., L_win, C)` shape profile on a palindrome-oriented
    E-box core:
      - RC-even channels (`MGW, ProT, EP, Stretch, Roll, HelT, Slide, Stagger, Opening, Rise`):
          `0.5 * (x[p] + x[L - 1 - p])`
      - RC-odd channels (`Shear, Buckle, Shift, Tilt`, where `f(rc(s))[p] = -f(s)[L - 1 - p]`):
          `0.5 * (x[p] - x[L - 1 - p])` (strictly RC-invariant since `h(rc(s)) == +h(s)`!)
    """
    out = np.empty_like(prof, dtype=np.float64)
    rev = prof[..., ::-1, :]
    for ci, feat in enumerate(features):
        if feat in X_AXIS_RC_ODD_FEATURES:
            out[..., :, ci] = 0.5 * (prof[..., :, ci] - rev[..., :, ci])
        else:
            out[..., :, ci] = 0.5 * (prof[..., :, ci] + rev[..., :, ci])
    return out


def align_gse59845_with_full_36bp_flanks(
    seqs36: Sequence[str],
    y_map: Dict[str, np.ndarray],
    engine: AuthenticDeepDNAshapeCPU,
) -> Dict[str, object]:
    """
    Align all 8,000 paired 36-bp `GSE59845` probes identically to
    `bf.align_ebox_14bp_cores`, while extracting:
      - Exact 14-bp cores, targets `y_agg`, central-10mer `groups` (100% parity with Exp 2)
      - Authentic Deep DNAshape predictions run on FULL 36-bp probes (`L=36`),
        oriented identically to each probe's winning `CANNTG` placement, sliced at:
          * 10 interior core positions `2..11` (7ch, L2 & L4 -> exact 70-col parity with DNAshapeR)
          * 14 full core positions `0..13` (7ch L4 -> 98 cols; 14ch L4 -> 196 cols)
        and strictly RC-symmetrized, then averaged across probes sharing each 14-bp core.
      - Exact flank-matched 22-bp (`core14 +/- 4 bp`) pure-sequence features
        (`Seq_G2_Flank22`, `Seq_Pos4mer_Flank22`, `HGBDT_Seq_Flank22`) extracted from
        the EXACT SAME oriented 36-bp probes (`'N'`-padded only on the 3.3% boundary
        probes where `off < 8` or `off + 14 > 36`), RC-symmetrized and averaged per core.
    """
    cores, y_agg, groups, align_meta = bf.align_ebox_14bp_cores(seqs36, y_map, window_len=14)
    core_to_idx = {c: i for i, c in enumerate(cores)}
    n_cores = len(cores)

    # Run authentic Deep DNAshape on ALL 8,000 full 36-bp probes once
    print("[Bench A] Running authentic Deep DNAshape (14ch, L2 & L4) on 8,000 full 36-bp GSE59845 probes...", flush=True)
    # L2 needed only on CH7; L4 needed on CH14
    pred_ch7_l2 = engine.predict_base_aligned_stack(CH7_INTERSECTION, seqs36, layers=(2,), batch_size=1024)[2]
    pred_ch14_l4 = engine.predict_base_aligned_stack(CH14_STRUCTURAL, seqs36, layers=(4,), batch_size=1024)[4]

    # Pre-identify sign masks for orientation flip when winning alignment is on RC strand
    odd_mask_14 = np.array([f in X_AXIS_RC_ODD_FEATURES for f in CH14_STRUCTURAL], dtype=bool)
    center_ref = 0.5 * (len(seqs36[0]) - 6)

    # Record per-probe alignment and aggregate onto the 7,289 unique 14-bp cores
    sum_l2_7ch_10 = np.zeros((n_cores, 10, 7), dtype=np.float64)
    sum_l4_7ch_10 = np.zeros((n_cores, 10, 7), dtype=np.float64)
    sum_l4_7ch_14 = np.zeros((n_cores, 14, 7), dtype=np.float64)
    sum_l4_14ch_14 = np.zeros((n_cores, 14, 14), dtype=np.float64)
    counts = np.zeros(n_cores, dtype=np.int32)

    flank22_strings: List[str] = []
    probe_core_ids = np.empty(len(seqs36), dtype=np.int32)
    n_boundary_padded_22 = 0

    for idx, seq in enumerate(seqs36):
        best = None
        for st, s_cur in ((0, seq), (1, bf.revcomp_str(seq))):
            L = len(s_cur)
            for off in range(4, L - 6 - 4 + 1):
                hex6 = s_cur[off : off + 6]
                sc = (
                    2.0 * (hex6[0] == "C")
                    + 2.0 * (hex6[1] == "A")
                    + 2.0 * (hex6[4] == "T")
                    + 2.0 * (hex6[5] == "G")
                    + 1.0 * (hex6[2:4] == "CG")
                    + 0.5 * ((hex6[2] in "CT") + (hex6[3] in "AG"))
                )
                cand = (sc, -abs(off - center_ref), st, off, s_cur[off - 4 : off + 10], s_cur)
                if best is None or cand[:2] > best[:2]:
                    best = cand
        _, _, st, off, core14, s_cur = best
        c_canon = bf.canonical_rc_str(core14)
        ci = core_to_idx[c_canon]
        probe_core_ids[idx] = ci
        counts[ci] += 1

        # Oriented 36-bp Deep DNAshape profiles
        if st == 0:
            p_l2_7 = pred_ch7_l2[idx]        # (36, 7)
            p_l4_14 = pred_ch14_l4[idx]      # (36, 14)
        else:
            p_l2_7 = pred_ch7_l2[idx, ::-1, :]
            p_l4_14 = pred_ch14_l4[idx, ::-1, :].copy()
            p_l4_14[:, odd_mask_14] = -p_l4_14[:, odd_mask_14]

        # Slice 14 core positions [off-4 : off+10] and 10 interior positions [off-2 : off+8]
        s_l2_10 = _rc_symmetrize_shape_profile(p_l2_7[off - 2 : off + 8], CH7_INTERSECTION)
        s_l4_7_10 = _rc_symmetrize_shape_profile(p_l4_14[off - 2 : off + 8, :7], CH7_INTERSECTION)
        s_l4_7_14 = _rc_symmetrize_shape_profile(p_l4_14[off - 4 : off + 10, :7], CH7_INTERSECTION)
        s_l4_14_14 = _rc_symmetrize_shape_profile(p_l4_14[off - 4 : off + 10, :], CH14_STRUCTURAL)

        sum_l2_7ch_10[ci] += s_l2_10
        sum_l4_7ch_10[ci] += s_l4_7_10
        sum_l4_7ch_14[ci] += s_l4_7_14
        sum_l4_14ch_14[ci] += s_l4_14_14

        # Extract exact 22-bp flank window [off - 8 : off + 14] on oriented s_cur ('N' pad at boundaries)
        left_start = off - 8
        right_end = off + 14
        pad_l = max(0, -left_start)
        pad_r = max(0, right_end - len(s_cur))
        if pad_l > 0 or pad_r > 0:
            n_boundary_padded_22 += 1
        w22 = ("N" * pad_l) + s_cur[max(0, left_start) : min(len(s_cur), right_end)] + ("N" * pad_r)
        assert len(w22) == 22
        # Orient w22 to match canonical_rc(core14) orientation for consistent HGBDT raw codes
        if core14 != c_canon:
            w22 = revcomp_n(w22)
        flank22_strings.append(w22)

    cnt = counts[:, None, None].astype(np.float64)
    X_l2_7ch_10 = (sum_l2_7ch_10 / cnt).reshape(n_cores, 70)
    X_l4_7ch_10 = (sum_l4_7ch_10 / cnt).reshape(n_cores, 70)
    X_l4_7ch_14 = (sum_l4_7ch_14 / cnt).reshape(n_cores, 98)
    # For 14ch: RC-odd channels have anti-symmetric second half (-first half); keep full 14*14 or non-redundant?
    # Keep all 14*14 = 196 columns (Ridge handles symmetric pairs identically to 10-bin DNAshapeR)
    X_l4_14ch_14 = (sum_l4_14ch_14 / cnt).reshape(n_cores, 196)

    # Encode 22-bp pure-sequence features per probe, then average onto the 7,289 cores
    codes22_probes = encode_acgtn(flank22_strings)  # (8000, 22)
    f22_probes = encode_rc_sym_g2_and_pos4mer_acgtn(codes22_probes)

    X_g1_f22 = np.zeros((n_cores, f22_probes["X_g1"].shape[1]), dtype=np.float64)
    X_g2_f22 = np.zeros((n_cores, f22_probes["X_g2"].shape[1]), dtype=np.float64)
    X_pos4_f22 = np.zeros((n_cores, f22_probes["X_pos4mer"].shape[1]), dtype=np.float64)
    codes22_core_first = np.empty((n_cores, 22), dtype=np.int8)
    seen = np.zeros(n_cores, dtype=bool)
    for idx in range(len(seqs36)):
        ci = int(probe_core_ids[idx])
        X_g1_f22[ci] += f22_probes["X_g1"][idx]
        X_g2_f22[ci] += f22_probes["X_g2"][idx]
        X_pos4_f22[ci] += f22_probes["X_pos4mer"][idx]
        if not seen[ci]:
            codes22_core_first[ci] = codes22_probes[idx]
            seen[ci] = True
    cnt1 = counts[:, None].astype(np.float64)
    X_g1_f22 /= cnt1
    X_g2_f22 /= cnt1
    X_pos4_f22 /= cnt1

    # Standard 14-bp core features & DNAshapeR (exact Exp 2 parity)
    feats14 = bf.build_rc_sym_ebox_features(cores, max_pos_bins_seq=14)
    F_true, _ = kv7.load_biophysical_table_matrix(include_odd_step_diff=False)
    X_dnashaper_7ch = bf._contract_b_rc_single(feats14["B_rc"], F_true)

    # Sanity check: correlation between X_dnashaper_7ch and X_l2_7ch_10 on the aligned cores!
    core_corr_l2 = {}
    core_corr_l4 = {}
    arr_ref = X_dnashaper_7ch.reshape(n_cores, 10, 7)
    arr_l2 = X_l2_7ch_10.reshape(n_cores, 10, 7)
    arr_l4 = X_l4_7ch_10.reshape(n_cores, 10, 7)
    for fi, feat in enumerate(CH7_INTERSECTION):
        core_corr_l2[feat] = float(pearsonr(arr_ref[:, :, fi].ravel(), arr_l2[:, :, fi].ravel())[0])
        core_corr_l4[feat] = float(pearsonr(arr_ref[:, :, fi].ravel(), arr_l4[:, :, fi].ravel())[0])

    return {
        "cores": cores,
        "y_agg": y_agg,
        "groups": groups,
        "align_meta": {
            **align_meta,
            "n_probes_flank22_boundary_N_padded": int(n_boundary_padded_22),
            "frac_probes_flank22_unpadded": float(1.0 - n_boundary_padded_22 / float(len(seqs36))),
            "aligned_core_10pos_pearson_L2_vs_DNAshapeR": core_corr_l2,
            "aligned_core_10pos_pearson_L4_vs_DNAshapeR": core_corr_l4,
        },
        "feats14": feats14,
        "X_dnashaper_7ch": X_dnashaper_7ch,
        "X_l2_7ch_10": X_l2_7ch_10,
        "X_l4_7ch_10": X_l4_7ch_10,
        "X_l4_7ch_14": X_l4_7ch_14,
        "X_l4_14ch_14": X_l4_14ch_14,
        "X_g1_f22": X_g1_f22,
        "X_g2_f22": X_g2_f22,
        "X_pos4_f22": X_pos4_f22,
        "codes22_core": codes22_core_first,
    }


def _eval_ridge_base_multi_target_5fold(
    X_seq: np.ndarray,
    mod_map: Dict[str, Optional[np.ndarray]],
    Y_mat: np.ndarray,
    groups: np.ndarray,
    alphas_pure: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
    lams_seq: Sequence[float] = (0.3, 3.0, 30.0, 300.0, 3000.0),
    lams_mod: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_folds: int = 5,
    n_inner_folds: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Exact multi-target primal-Schur 5-fold GroupKFold Ridge solver matching
    `pp5.fit_predict_ridge_groupcv` and `pp5.fit_predict_two_penalty_ridge_groupcv`
    to `<1e-11`, eigendecomposing `Z_seq^T Z_seq` ONCE per fold and sharing it
    across ALL targets `Y_mat (N, K)` and ALL modality blocks in `mod_map`!
    Returns `{model_name: oof (N, K)}`.
    """
    n, K = Y_mat.shape
    al_arr = np.asarray(alphas_pure, dtype=np.float64)
    ls_arr = np.asarray(lams_seq, dtype=np.float64)
    lm_arr = np.asarray(lams_mod, dtype=np.float64)
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    outer_splits = list(gkf.split(X_seq, Y_mat[:, 0], groups))
    out_oofs = {mn: np.empty((n, K), dtype=np.float64) for mn in mod_map}

    for tr_i, va_i in outer_splits:
        keep_s = X_seq[tr_i].std(axis=0) > 1e-12
        Xs_tr_raw, Xs_va_raw = X_seq[tr_i][:, keep_s], X_seq[va_i][:, keep_s]
        ms = Xs_tr_raw.mean(axis=0, keepdims=True)
        ss = np.maximum(Xs_tr_raw.std(axis=0, keepdims=True), 1e-8)
        Zs_tr = (Xs_tr_raw - ms) / ss
        Zs_va = (Xs_va_raw - ms) / ss
        p_s = Zs_tr.shape[1]

        Y_tr = Y_mat[tr_i]
        ym_tr = Y_tr.mean(axis=0, keepdims=True)
        Yc_tr = Y_tr - ym_tr

        # Standardize every modality strictly on outer tr_i
        mods_std: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for mn, Xm in mod_map.items():
            if Xm is None:
                continue
            km = Xm[tr_i].std(axis=0) > 1e-12
            Xmt, Xmv = Xm[tr_i][:, km], Xm[va_i][:, km]
            mm = Xmt.mean(axis=0, keepdims=True)
            sm = np.maximum(Xmt.std(axis=0, keepdims=True), 1e-8)
            mods_std[mn] = ((Xmt - mm) / sm, (Xmv - mm) / sm)

        in_gkf = GroupKFold(n_splits=min(n_inner_folds, len(np.unique(groups[tr_i]))))
        in_splits = list(in_gkf.split(Zs_tr, Y_tr[:, 0], groups[tr_i]))

        pure_names = [mn for mn, Xm in mod_map.items() if Xm is None]
        mod_names = [mn for mn, Xm in mod_map.items() if Xm is not None]
        cv_pure = np.zeros((K, len(al_arr)), dtype=np.float64)
        cv_mod = {mn: np.zeros((K, len(ls_arr), len(lm_arr)), dtype=np.float64) for mn in mod_names}

        for i_tr, i_va in in_splits:
            Zsa, Zsv = Zs_tr[i_tr], Zs_tr[i_va]
            ym_i = Y_tr[i_tr].mean(axis=0, keepdims=True)
            Ya = Y_tr[i_tr] - ym_i
            Yv = Y_tr[i_va] - ym_i

            # Eigendecompose Zsa^T Zsa ONCE (n_inner_tr >= 3800 > p_s <= 2440)
            ev_s, Q_s = sp_eigh(Zsa.T @ Zsa, check_finite=False, driver="evd")
            ev_s = np.maximum(ev_s, 0.0)
            Qs_T = Q_s.T
            qs = Qs_T @ (Zsa.T @ Ya)       # (p_s, K)
            Zsv_Qs = Zsv @ Q_s             # (nv, p_s)

            if pure_names:
                for ia, al in enumerate(al_arr):
                    ws_q = qs / (ev_s[:, None] + al)
                    diff = (Zsv_Qs @ ws_q) - Yv
                    cv_pure[:, ia] += np.sum(diff * diff, axis=0)

            for mn in mod_names:
                Zmt, _ = mods_std[mn]
                Zma, Zmv = Zmt[i_tr], Zmt[i_va]
                U = Qs_T @ (Zsa.T @ Zma)   # (p_s, p_m)
                U_T = U.T
                Gmm = Zma.T @ Zma
                zm_y = Zma.T @ Ya          # (p_m, K)
                for i_s, ls in enumerate(ls_arr):
                    inv_d = 1.0 / (ev_s + ls)
                    U_sc = U * inv_d[:, None]
                    S_mm = Gmm - U_T @ U_sc
                    S_mm = 0.5 * (S_mm + S_mm.T)
                    rhs_m = zm_y - U_sc.T @ qs
                    ev_m, Q_m = sp_eigh(S_mm, check_finite=False, driver="evd")
                    ev_m = np.maximum(ev_m, 0.0)
                    qm = Q_m.T @ rhs_m
                    for i_m, lm in enumerate(lm_arr):
                        wm = Q_m @ (qm / (ev_m[:, None] + lm))
                        ws_q = (qs - U @ wm) * inv_d[:, None]
                        pred_v = Zsv_Qs @ ws_q + Zmv @ wm
                        diff = pred_v - Yv
                        cv_mod[mn][:, i_s, i_m] += np.sum(diff * diff, axis=0)

        # Outer fold eigendecomposition ONCE on full Zs_tr
        ev_f, Q_f = sp_eigh(Zs_tr.T @ Zs_tr, check_finite=False, driver="evd")
        ev_f = np.maximum(ev_f, 0.0)
        Qf_T = Q_f.T
        qf = Qf_T @ (Zs_tr.T @ Yc_tr)      # (p_s, K)
        Zva_Qf = Zs_va @ Q_f               # (n_va, p_s)

        for mn in pure_names:
            for k in range(K):
                al = float(al_arr[int(np.argmin(cv_pure[k]))])
                out_oofs[mn][va_i, k] = ym_tr[0, k] + Zva_Qf @ (qf[:, k] / (ev_f + al))

        for mn in mod_names:
            Zmt, Zmv = mods_std[mn]
            Uf = Qf_T @ (Zs_tr.T @ Zmt)
            Uf_T = Uf.T
            Gmm_f = Zmt.T @ Zmt
            zm_yf = Zmt.T @ Yc_tr
            for k in range(K):
                bi, bj = np.unravel_index(int(np.argmin(cv_mod[mn][k])), (len(ls_arr), len(lm_arr)))
                ls = float(ls_arr[bi])
                lm = float(lm_arr[bj])
                inv_df = 1.0 / (ev_f + ls)
                Uf_sc = Uf * inv_df[:, None]
                Sf = Gmm_f - Uf_T @ Uf_sc
                Sf[np.diag_indices(Zmt.shape[1])] += lm
                rhs_mf = zm_yf[:, k] - Uf_sc.T @ qf[:, k]
                wm = cho_solve(cho_factor(Sf, lower=True, check_finite=False), rhs_mf, check_finite=False)
                ws_q = (qf[:, k] - Uf @ wm) * inv_df
                out_oofs[mn][va_i, k] = ym_tr[0, k] + Zva_Qf @ ws_q + Zmv @ wm

    return out_oofs


def _fit_predict_hgbdt_bhlh_oof(
    X_base: np.ndarray,
    X_mod: Optional[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """Exact 5-fold GroupKFold OOF HGBDT matching `bf._fit_predict_hgbdt_5fold_oof`."""
    X_in = X_base if X_mod is None else np.hstack([X_base, X_mod.astype(np.float32)])
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    splits = list(gkf.split(X_in, y, groups))
    configs = [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 3.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 3.0},
    ]
    oof = np.empty(len(y), dtype=np.float64)
    for tr_i, va_i in splits:
        in_gkf = GroupKFold(n_splits=min(3, len(np.unique(groups[tr_i]))))
        i_tr, i_va = next(in_gkf.split(X_in[tr_i], y[tr_i], groups[tr_i]))
        X_tr, y_tr = X_in[tr_i], y[tr_i]
        best_cfg, best_mse = configs[0], np.inf
        for cfg in configs:
            m = HistGradientBoostingRegressor(max_iter=200, max_bins=64, early_stopping=True, random_state=42, **cfg)
            m.fit(X_tr[i_tr], y_tr[i_tr])
            d = m.predict(X_tr[i_va]) - y_tr[i_va]
            mse = float(np.sum(d * d))
            if mse < best_mse:
                best_mse = mse
                best_cfg = cfg
        m_f = HistGradientBoostingRegressor(max_iter=250, max_bins=64, early_stopping=True, random_state=42, **best_cfg)
        m_f.fit(X_tr, y_tr)
        oof[va_i] = m_f.predict(X_in[va_i])
    return oof


_BHLH_TASK_CTX: Dict[str, object] = {}


def _bhlh_hgb_worker(task: Tuple[str, str]) -> Tuple[str, str, np.ndarray]:
    ep, mname = task
    ctx = _BHLH_TASK_CTX
    y = ctx["y_agg"][ep]
    groups = ctx["groups"]
    xs_key, xm_key = ctx["hgb_specs"][mname]
    oof = _fit_predict_hgbdt_bhlh_oof(
        ctx["mats"][xs_key],
        None if xm_key is None else ctx["mats"][xm_key],
        y,
        groups,
        n_folds=5,
    )
    return ep, mname, oof


def run_benchmark_a_bhlh(
    aligned: Dict[str, object],
    n_boot: int = 2000,
    n_workers: int = 4,
) -> Dict[str, object]:
    """
    Run full Benchmark A across all 6 primary `GSE59845` endpoints:
      - `Seq_G2`, `Seq_Pos4mer`, `HGBDT_Seq` vs `+DNAshapeR (7ch)` vs
        `+DeepDNAshape_L2_7ch_10pos` vs `+DeepDNAshape_L4_7ch_10pos` vs
        `+DeepDNAshape_L4_7ch_14pos` vs `+DeepDNAshape_L4_14ch_14pos`
      - Fair flank-matched pure-sequence controls (`Seq_G2_Flank22`,
        `Seq_Pos4mer_Flank22`, `HGBDT_Seq_Flank22`) and `+DeepDNAshape_L4_7ch_14pos`
      - Core-cluster `B=2,000` paired bootstrap 95% CIs + cross-6-endpoint Holm `p_holm`.
    """
    import time

    endpoints = ["Max", "Mad1", "cMyc", "cMyc_minus_Max", "Mad1_minus_Max", "cMyc_minus_Mad1"]
    Y_mat = np.column_stack([aligned["y_agg"][ep] for ep in endpoints]).astype(np.float64)
    groups = aligned["groups"]

    f14 = aligned["feats14"]
    X_g1_14 = f14["X_g1"]
    X_g2_14 = f14["X_g2"]
    X_p4_14 = f14["X_pos4mer"]
    gc2_14 = X_g2_14[:, -12:-10]
    X_hgb_14 = np.hstack([f14["codes"].astype(np.float32), X_g1_14.astype(np.float32), gc2_14.astype(np.float32)])

    X_g1_22 = aligned["X_g1_f22"]
    X_g2_22 = aligned["X_g2_f22"]
    X_p4_22 = aligned["X_pos4_f22"]
    gc2_22 = X_g2_22[:, -12:-10]
    X_hgb_22 = np.hstack([aligned["codes22_core"].astype(np.float32), X_g1_22.astype(np.float32), gc2_22.astype(np.float32)])

    mats = {
        "hgb_14": X_hgb_14,
        "hgb_22": X_hgb_22,
        "dnashaper_7ch": aligned["X_dnashaper_7ch"],
        "deep_l2_7ch_10": aligned["X_l2_7ch_10"],
        "deep_l4_7ch_10": aligned["X_l4_7ch_10"],
        "deep_l4_7ch_14": aligned["X_l4_7ch_14"],
        "deep_l4_14ch_14": aligned["X_l4_14ch_14"],
    }

    oofs: Dict[str, Dict[str, np.ndarray]] = {ep: {} for ep in endpoints}
    all_model_names: List[str] = []

    # 1. Exact shared-basis multi-target Schur Ridge across all 4 sequence bases
    ridge_groups = [
        (
            "g2_14",
            X_g2_14,
            {
                "Seq_G2": None,
                "Seq_G2+DNAshapeR_7ch": aligned["X_dnashaper_7ch"],
                "Seq_G2+DeepDNAshape_L2_7ch_10pos": aligned["X_l2_7ch_10"],
                "Seq_G2+DeepDNAshape_L4_7ch_10pos": aligned["X_l4_7ch_10"],
                "Seq_G2+DeepDNAshape_L4_7ch_14pos": aligned["X_l4_7ch_14"],
                "Seq_G2+DeepDNAshape_L4_14ch_14pos": aligned["X_l4_14ch_14"],
            },
        ),
        (
            "g2_22",
            X_g2_22,
            {
                "Seq_G2_Flank22": None,
                "Seq_G2_Flank22+DeepDNAshape_L4_7ch_14pos": aligned["X_l4_7ch_14"],
            },
        ),
        (
            "p4_14",
            X_p4_14,
            {
                "Seq_Pos4mer": None,
                "Seq_Pos4mer+DNAshapeR_7ch": aligned["X_dnashaper_7ch"],
                "Seq_Pos4mer+DeepDNAshape_L2_7ch_10pos": aligned["X_l2_7ch_10"],
                "Seq_Pos4mer+DeepDNAshape_L4_7ch_10pos": aligned["X_l4_7ch_10"],
                "Seq_Pos4mer+DeepDNAshape_L4_7ch_14pos": aligned["X_l4_7ch_14"],
                "Seq_Pos4mer+DeepDNAshape_L4_14ch_14pos": aligned["X_l4_14ch_14"],
            },
        ),
        (
            "p4_22",
            X_p4_22,
            {
                "Seq_Pos4mer_Flank22": None,
                "Seq_Pos4mer_Flank22+DeepDNAshape_L4_7ch_14pos": aligned["X_l4_7ch_14"],
            },
        ),
    ]

    for blk_name, X_s, mmap in ridge_groups:
        t_b = time.time()
        res_blk = _eval_ridge_base_multi_target_5fold(X_s, mmap, Y_mat, groups, n_folds=5, n_inner_folds=3)
        for mn, arr_nk in res_blk.items():
            all_model_names.append(mn)
            for k, ep in enumerate(endpoints):
                oofs[ep][mn] = arr_nk[:, k]
        print(f"    [Bench A Ridge] `{blk_name}` ({len(mmap)} models x 6 eps) done in {time.time() - t_b:.1f}s", flush=True)

    # 2. HGBDT track (7 models x 6 endpoints = 42 jobs in parallel)
    hgb_specs: Dict[str, Tuple[str, Optional[str]]] = {
        "HGBDT_Seq": ("hgb_14", None),
        "HGBDT_Seq+DNAshapeR_7ch": ("hgb_14", "dnashaper_7ch"),
        "HGBDT_Seq+DeepDNAshape_L4_7ch_10pos": ("hgb_14", "deep_l4_7ch_10"),
        "HGBDT_Seq+DeepDNAshape_L4_7ch_14pos": ("hgb_14", "deep_l4_7ch_14"),
        "HGBDT_Seq+DeepDNAshape_L4_14ch_14pos": ("hgb_14", "deep_l4_14ch_14"),
        "HGBDT_Seq_Flank22": ("hgb_22", None),
        "HGBDT_Seq_Flank22+DeepDNAshape_L4_7ch_14pos": ("hgb_22", "deep_l4_7ch_14"),
    }
    all_model_names.extend(hgb_specs.keys())
    hgb_jobs = [(ep, mn) for ep in endpoints for mn in hgb_specs]
    t_h = time.time()
    _BHLH_TASK_CTX.clear()
    _BHLH_TASK_CTX.update(
        {
            "y_agg": aligned["y_agg"],
            "groups": groups,
            "mats": mats,
            "hgb_specs": hgb_specs,
        }
    )
    try:
        eff_w = min(max(1, int(n_workers)), len(hgb_jobs))
        if eff_w <= 1:
            for jb in hgb_jobs:
                ep, mn, oof = _bhlh_hgb_worker(jb)
                oofs[ep][mn] = oof
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=eff_w, mp_context=ctx) as ex:
                for ep, mn, oof in ex.map(_bhlh_hgb_worker, hgb_jobs):
                    oofs[ep][mn] = oof
    finally:
        _BHLH_TASK_CTX.clear()
    print(f"    [Bench A HGBDT] {len(hgb_jobs)} jobs done in {time.time() - t_h:.1f}s", flush=True)

    contrast_defs: List[Tuple[str, str, str]] = [
        # G2
        ("G2_DNAshapeR_vs_G2", "Seq_G2+DNAshapeR_7ch", "Seq_G2"),
        ("G2_DeepL2_10_vs_G2", "Seq_G2+DeepDNAshape_L2_7ch_10pos", "Seq_G2"),
        ("G2_DeepL4_10_vs_G2", "Seq_G2+DeepDNAshape_L4_7ch_10pos", "Seq_G2"),
        ("G2_DeepL4_14_vs_G2", "Seq_G2+DeepDNAshape_L4_7ch_14pos", "Seq_G2"),
        ("G2_DeepL4_14_vs_DNAshapeR", "Seq_G2+DeepDNAshape_L4_7ch_14pos", "Seq_G2+DNAshapeR_7ch"),
        ("G2_DeepL4_10_vs_DNAshapeR", "Seq_G2+DeepDNAshape_L4_7ch_10pos", "Seq_G2+DNAshapeR_7ch"),
        ("G2_DeepL4_14ch_vs_G2", "Seq_G2+DeepDNAshape_L4_14ch_14pos", "Seq_G2"),
        ("G2_Flank22_vs_G2", "Seq_G2_Flank22", "Seq_G2"),
        ("G2_DeepL4_14_vs_G2_Flank22", "Seq_G2+DeepDNAshape_L4_7ch_14pos", "Seq_G2_Flank22"),
        ("G2_Flank22_DeepL4_vs_G2_Flank22", "Seq_G2_Flank22+DeepDNAshape_L4_7ch_14pos", "Seq_G2_Flank22"),
        # Pos4mer
        ("Pos4mer_DNAshapeR_vs_Pos4mer", "Seq_Pos4mer+DNAshapeR_7ch", "Seq_Pos4mer"),
        ("Pos4mer_DeepL2_10_vs_Pos4mer", "Seq_Pos4mer+DeepDNAshape_L2_7ch_10pos", "Seq_Pos4mer"),
        ("Pos4mer_DeepL4_10_vs_Pos4mer", "Seq_Pos4mer+DeepDNAshape_L4_7ch_10pos", "Seq_Pos4mer"),
        ("Pos4mer_DeepL4_14_vs_Pos4mer", "Seq_Pos4mer+DeepDNAshape_L4_7ch_14pos", "Seq_Pos4mer"),
        ("Pos4mer_DeepL4_14_vs_DNAshapeR", "Seq_Pos4mer+DeepDNAshape_L4_7ch_14pos", "Seq_Pos4mer+DNAshapeR_7ch"),
        ("Pos4mer_DeepL4_10_vs_DNAshapeR", "Seq_Pos4mer+DeepDNAshape_L4_7ch_10pos", "Seq_Pos4mer+DNAshapeR_7ch"),
        ("Pos4mer_DeepL4_14ch_vs_Pos4mer", "Seq_Pos4mer+DeepDNAshape_L4_14ch_14pos", "Seq_Pos4mer"),
        ("Pos4mer_Flank22_vs_Pos4mer", "Seq_Pos4mer_Flank22", "Seq_Pos4mer"),
        ("Pos4mer_DeepL4_14_vs_Pos4mer_Flank22", "Seq_Pos4mer+DeepDNAshape_L4_7ch_14pos", "Seq_Pos4mer_Flank22"),
        ("Pos4mer_Flank22_DeepL4_vs_Pos4mer_Flank22", "Seq_Pos4mer_Flank22+DeepDNAshape_L4_7ch_14pos", "Seq_Pos4mer_Flank22"),
        # HGBDT
        ("HGBDT_DNAshapeR_vs_HGBDT", "HGBDT_Seq+DNAshapeR_7ch", "HGBDT_Seq"),
        ("HGBDT_DeepL4_10_vs_HGBDT", "HGBDT_Seq+DeepDNAshape_L4_7ch_10pos", "HGBDT_Seq"),
        ("HGBDT_DeepL4_14_vs_HGBDT", "HGBDT_Seq+DeepDNAshape_L4_7ch_14pos", "HGBDT_Seq"),
        ("HGBDT_DeepL4_14_vs_DNAshapeR", "HGBDT_Seq+DeepDNAshape_L4_7ch_14pos", "HGBDT_Seq+DNAshapeR_7ch"),
        ("HGBDT_DeepL4_14ch_vs_HGBDT", "HGBDT_Seq+DeepDNAshape_L4_14ch_14pos", "HGBDT_Seq"),
        ("HGBDT_Flank22_vs_HGBDT", "HGBDT_Seq_Flank22", "HGBDT_Seq"),
        ("HGBDT_DeepL4_14_vs_HGBDT_Flank22", "HGBDT_Seq+DeepDNAshape_L4_7ch_14pos", "HGBDT_Seq_Flank22"),
        ("HGBDT_Flank22_DeepL4_vs_HGBDT_Flank22", "HGBDT_Seq_Flank22+DeepDNAshape_L4_7ch_14pos", "HGBDT_Seq_Flank22"),
    ]

    groups = aligned["groups"]
    ep_results: Dict[str, Dict[str, object]] = {}
    for ep in endpoints:
        y = aligned["y_agg"][ep]
        models_out: Dict[str, Dict[str, float]] = {}
        for mn in all_model_names:
            sr, pr = bf.safe_spearman_pearson(y, oofs[ep][mn])
            models_out[mn] = {"spearman": float(sr), "pearson": float(pr)}
        contrasts_out: Dict[str, Dict[str, object]] = {}
        for cid, ma, mb in contrast_defs:
            contrasts_out[cid] = bf.cluster_bootstrap_spearman_diff_exact(
                y, oofs[ep][ma], oofs[ep][mb], groups, n_boot=n_boot, seed=42
            )
        ep_results[ep] = {
            "n_cores": int(len(y)),
            "n_groups": int(len(np.unique(groups))),
            "models": models_out,
            "contrasts": contrasts_out,
        }

    # Cross-6-endpoint Holm-Bonferroni adjustment for every contrast (both one-sided gt0 & two-sided)
    cross_ep_summary: Dict[str, Dict[str, object]] = {}
    for cid, _, _ in contrast_defs:
        p_gt0 = [float(ep_results[ep]["contrasts"][cid]["p_boot_gt0"]) for ep in endpoints]
        p_two = [float(ep_results[ep]["contrasts"][cid]["p_boot_two_sided"]) for ep in endpoints]
        h_gt0 = bf.holm_bonferroni(p_gt0)
        h_two = bf.holm_bonferroni(p_two)
        diffs = [float(ep_results[ep]["contrasts"][cid]["diff"]) for ep in endpoints]
        for i, ep in enumerate(endpoints):
            ep_results[ep]["contrasts"][cid]["p_holm_6ep_gt0"] = float(h_gt0[i])
            ep_results[ep]["contrasts"][cid]["p_holm_6ep_two_sided"] = float(h_two[i])
        cross_ep_summary[cid] = {
            "mean_diff_6ep": float(np.mean(diffs)),
            "min_diff_6ep": float(np.min(diffs)),
            "max_diff_6ep": float(np.max(diffs)),
            "n_sig_holm005_gt0": int(sum(1 for p in h_gt0 if p <= 0.05)),
            "n_sig_holm005_two_sided": int(sum(1 for p in h_two if p <= 0.05)),
            "per_endpoint_diff": {ep: diffs[i] for i, ep in enumerate(endpoints)},
            "per_endpoint_p_holm_gt0": {ep: float(h_gt0[i]) for i, ep in enumerate(endpoints)},
            "per_endpoint_p_holm_two_sided": {ep: float(h_two[i]) for i, ep in enumerate(endpoints)},
        }

    return {
        "dataset": "GSE59845_gcPBM_bHLH_Max_Mad1_cMyc",
        "cv_scheme": "5-fold central-10mer GroupKFold (exact v7_1 Exp 2 parity)",
        "n_boot": int(n_boot),
        "alignment_meta": aligned["align_meta"],
        "endpoints": ep_results,
        "cross_6_endpoint_summary": cross_ep_summary,
    }


# ==============================================================================
# 4. Benchmark B — Homeodomain GSE233827 (92 Mutants, 30-Fold LOGO)
# ==============================================================================

def load_gse233827_full36_and_deep_dnashape(
    engine: AuthenticDeepDNAshapeCPU,
    cache_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Load locked `v7_kock_dataset.npz` and raw 36-bp PBM probes (`IntensityMatrix.rds`),
    verify 0 mismatches across all 60,000 `(30 genes x 2,000 probes)` oriented 12-bp
    windows, run authentic Deep DNAshape (L2 & L4, 7ch) on the 15,889 unique 36-bp
    probes, and extract:
      - `dnashaper_n1n4_g`: `(30, 2000, 28)` exact DNAshapeR N1..N4
      - `deep_l2_n1n4_g`: `(30, 2000, 28)` Deep DNAshape Layer 2 at N1..N4
      - `deep_l4_n1n4_g`: `(30, 2000, 28)` Deep DNAshape Layer 4 at N1..N4
      - `codes15_g`: `(30, 2000, 15)` int8 ACGTN exact minimal union `[F-5..F+2]`
        covering BOTH original 12-bp `[F-2..F+2]` AND Deep DNAshape L4's full
        receptive field at `N1..N4` (`[F-5..F+1]`), `'N'`-padded identically to
        Deep DNAshape when `q < 3`.
    """
    ds = dict(np.load(ROOT / "v7" / "data" / "processed" / "v7_kock_dataset.npz", allow_pickle=False))
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        meta_p = cache_path.with_suffix(".json")
        if meta_p.exists():
            print(f"[Bench B] Cache hit: {cache_path.name}", flush=True)
            return {
                "ds": ds,
                "dnashaper_n1n4_g": z["dnashaper_n1n4_g"],
                "deep_l2_n1n4_g": z["deep_l2_n1n4_g"],
                "deep_l4_n1n4_g": z["deep_l4_n1n4_g"],
                "codes15_g": z["codes15_g"],
                "alignment_meta": json.loads(meta_p.read_text(encoding="utf-8")),
            }

    import rdata  # offline local RDS reader (already installed in .venv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = rdata.read_rds(ROOT / "v7" / "data" / "raw" / "IntensityMatrix.rds")
    seqs36_all = im["seq"].astype(str).tolist()

    pidx = ds["probe_idx_f70"]      # (30, 2000)
    bp = ds["best_p_f70"]           # (30, 2000)
    bs = ds["best_strand_f70"]      # (30, 2000)
    win12_ref = ds["win12_f70"]     # (30, 2000)
    n_genes, k_pwm = pidx.shape

    uniq_idx = np.unique(pidx)
    idx_to_u = {int(pi): ui for ui, pi in enumerate(uniq_idx)}
    uniq_seqs36 = [seqs36_all[int(pi)] for pi in uniq_idx]

    print(
        f"[Bench B] Running authentic Deep DNAshape (7ch, L2 & L4) on {len(uniq_seqs36)} unique 36-bp GSE233827 probes...",
        flush=True,
    )
    pred_u = engine.predict_base_aligned_stack(CH7_INTERSECTION, uniq_seqs36, layers=(2, 4), batch_size=1024)
    u_l2 = pred_u[2]  # (U, 36, 7)
    u_l4 = pred_u[4]  # (U, 36, 7)

    F_true, _ = kv7.load_biophysical_table_matrix(include_odd_step_diff=False)
    dnashaper_n1n4_g = np.empty((n_genes, k_pwm, 28), dtype=np.float32)
    deep_l2_n1n4_g = np.empty((n_genes, k_pwm, 28), dtype=np.float32)
    deep_l4_n1n4_g = np.empty((n_genes, k_pwm, 28), dtype=np.float32)
    codes15_g = np.empty((n_genes, k_pwm, 15), dtype=np.int8)

    n_mismatch = 0
    n_flank15_padded = 0
    for g in range(n_genes):
        c12 = ds["win12_codes_f70"][g].astype(np.uint8)
        c5_n1n4, _, _ = kv7.extract_pentamer_indices_12bp(c12)
        dnashaper_n1n4_g[g] = F_true[c5_n1n4].reshape(k_pwm, 28).astype(np.float32)

        f15_strs: List[str] = []
        for j in range(k_pwm):
            pi = int(pidx[g, j])
            ui = idx_to_u[pi]
            p = int(bp[g, j])
            st = int(bs[g, j])
            s36 = seqs36_all[pi]
            if st == 0:
                s_ori = s36
                q = p - 2
                p_l2 = u_l2[ui]
                p_l4 = u_l4[ui]
            else:
                s_ori = revcomp_n(s36)
                q = 26 - p
                p_l2 = u_l2[ui, ::-1, :]
                p_l4 = u_l4[ui, ::-1, :]

            if s_ori[q : q + 12] != str(win12_ref[g, j]):
                n_mismatch += 1

            # N1, N2, T3, D4 sit at q+2, q+3, q+4, q+5 on oriented 36-bp probe
            deep_l2_n1n4_g[g, j] = p_l2[q + 2 : q + 6, :].reshape(28)
            deep_l4_n1n4_g[g, j] = p_l4[q + 2 : q + 6, :].reshape(28)

            # Minimal union 15-bp window [q - 3 : q + 12] = [F-5..F+2]
            ls = q - 3
            pad_l = max(0, -ls)
            if pad_l > 0:
                n_flank15_padded += 1
            w15 = ("N" * pad_l) + s_ori[max(0, ls) : q + 12]
            assert len(w15) == 15
            f15_strs.append(w15)
        codes15_g[g] = encode_acgtn(f15_strs)

    if n_mismatch != 0:
        raise RuntimeError(f"GSE233827 36-bp alignment mismatch count = {n_mismatch}")

    # Verify Layer 2 & Layer 4 vs DNAshapeR across all 60,000 aligned N1..N4 instances
    corr_l2 = {}
    corr_l4 = {}
    r_ref = dnashaper_n1n4_g.reshape(-1, 4, 7)
    r_l2 = deep_l2_n1n4_g.reshape(-1, 4, 7)
    r_l4 = deep_l4_n1n4_g.reshape(-1, 4, 7)
    for fi, feat in enumerate(CH7_INTERSECTION):
        corr_l2[feat] = float(pearsonr(r_ref[:, :, fi].ravel(), r_l2[:, :, fi].ravel())[0])
        corr_l4[feat] = float(pearsonr(r_ref[:, :, fi].ravel(), r_l4[:, :, fi].ravel())[0])

    align_meta = {
        "n_genes": int(n_genes),
        "k_pwm": int(k_pwm),
        "n_unique_36bp_probes": int(len(uniq_seqs36)),
        "win12_exact_match_mismatches": int(n_mismatch),
        "flank15_window": "[F-5, F-4, F-3, F-2, F-1, N1..N8, F+1, F+2] (exact minimal union of 12-bp + DeepDNAshape_L4 N1..N4 RF)",
        "n_instances_flank15_left_N_padded": int(n_flank15_padded),
        "frac_instances_flank15_unpadded": float(1.0 - n_flank15_padded / float(n_genes * k_pwm)),
        "n1n4_60k_pearson_L2_vs_DNAshapeR": corr_l2,
        "n1n4_60k_pearson_L4_vs_DNAshapeR": corr_l4,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            dnashaper_n1n4_g=dnashaper_n1n4_g,
            deep_l2_n1n4_g=deep_l2_n1n4_g,
            deep_l4_n1n4_g=deep_l4_n1n4_g,
            codes15_g=codes15_g,
        )
        cache_path.with_suffix(".json").write_text(json.dumps(align_meta, indent=2), encoding="utf-8")

    return {
        "ds": ds,
        "dnashaper_n1n4_g": dnashaper_n1n4_g,
        "deep_l2_n1n4_g": deep_l2_n1n4_g,
        "deep_l4_n1n4_g": deep_l4_n1n4_g,
        "codes15_g": codes15_g,
        "alignment_meta": align_meta,
    }


def _gene_cluster_bootstrap_paired(
    rhos_a: np.ndarray,
    rhos_b: np.ndarray,
    mut_gene_idx: np.ndarray,
    n_genes: int = 30,
    n_boot: int = 5000,
    seed: int = 42,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Exact gene-cluster (`n=30` genes) paired bootstrap + gene-mean Wilcoxon signed-rank
    for `rhos_a - rhos_b` (both mutant-weighted and gene-balanced averages).
    """
    if mask is not None:
        ra = rhos_a[mask]
        rb = rhos_b[mask]
        mg = mut_gene_idx[mask]
    else:
        ra = rhos_a
        rb = rhos_b
        mg = mut_gene_idx

    uniq_g = np.unique(mg)
    G = len(uniq_g)
    g_ra = np.array([float(np.mean(ra[mg == g])) for g in uniq_g], dtype=np.float64)
    g_rb = np.array([float(np.mean(rb[mg == g])) for g in uniq_g], dtype=np.float64)
    g_n = np.array([int(np.sum(mg == g)) for g in uniq_g], dtype=np.float64)
    g_diff = g_ra - g_rb

    mut_mean_a = float(np.mean(ra))
    mut_mean_b = float(np.mean(rb))
    mut_diff_obs = mut_mean_a - mut_mean_b
    gene_mean_a = float(np.mean(g_ra))
    gene_mean_b = float(np.mean(g_rb))
    gene_diff_obs = gene_mean_a - gene_mean_b

    rng = np.random.default_rng(seed)
    boot_gene = np.empty(n_boot, dtype=np.float64)
    boot_mut = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, G, size=G)
        boot_gene[b] = float(np.mean(g_diff[pick]))
        w = g_n[pick]
        boot_mut[b] = float(np.sum(g_diff[pick] * w) / max(np.sum(w), 1.0))

    nz = g_diff[np.abs(g_diff) > 1e-12]
    if len(nz) >= 3:
        w_stat, w_p = wilcoxon(nz, alternative="two-sided")
        w_p_gt0 = wilcoxon(nz, alternative="greater").pvalue
    else:
        w_stat, w_p, w_p_gt0 = 0.0, 1.0, 1.0

    return {
        "n_mutants": int(len(ra)),
        "n_genes": int(G),
        "mut_mean_a": mut_mean_a,
        "mut_mean_b": mut_mean_b,
        "mut_diff": float(mut_diff_obs),
        "mut_ci95": [float(np.quantile(boot_mut, 0.025)), float(np.quantile(boot_mut, 0.975))],
        "gene_mean_a": gene_mean_a,
        "gene_mean_b": gene_mean_b,
        "gene_diff": float(gene_diff_obs),
        "gene_ci95": [float(np.quantile(boot_gene, 0.025)), float(np.quantile(boot_gene, 0.975))],
        "gene_wilcoxon_stat": float(w_stat),
        "gene_wilcoxon_p_two_sided": float(w_p),
        "gene_wilcoxon_p_gt0": float(w_p_gt0),
        "n_genes_positive": int(np.sum(g_diff > 0)),
    }


def _kron_add_mat(
    target: np.ndarray,
    sl_r: slice,
    sl_c: slice,
    A_rc: np.ndarray,
    G_rc: np.ndarray,
) -> None:
    """In-place exact Kronecker accumulation `target[sl_r, sl_c] += kron(A_rc, G_rc)` via broadcasting."""
    pr, pc = A_rc.shape
    qr, qc = G_rc.shape
    target[sl_r, sl_c] += (A_rc[:, None, :, None] * G_rc[None, :, None, :]).reshape(pr * qr, pc * qc)


def _eval_bilinear_ridge_logo_multi_target(
    phi_prot: np.ndarray,
    mut_gene_idx: np.ndarray,
    y_mats: Sequence[np.ndarray],
    dna_g2_g: np.ndarray,
    dna_pos4_g: Optional[np.ndarray],
    dna_mod_g: Optional[np.ndarray],
    mode: str,
    n_workers: int = 4,
) -> List[np.ndarray]:
    """
    Exact zero-leakage 30-fold LOGO bilinear Ridge evaluator for `GSE233827`
    solving ALL `K` targets (`y_mats`, e.g. `[dz_f70, dz_spec_f70]`) sharing
    identical finite probe masks simultaneously using:
      - Gene-grouped upper-triangle Kronecker Gram assembly
      - Single-pass `solve_triangular(L, Z_sm, lower=True)` Schur complement
      - Zero outer-fold leakage (`Vp16`, `Vp8`, `Vd64`, active cols, scaling & inner CV
        fit strictly on the 29 training genes).
    Returns list of `K` per-mutant Spearman arrays, each shape `(92,)`.
    """
    K_tgt = len(y_mats)
    n_mut = len(mut_gene_idx)
    n_genes, k_pwm, d_g2 = dna_g2_g.shape
    L_win = (d_g2 - 3) // 12
    d_g1 = 3 * L_win
    has_mod = dna_mod_g is not None
    d_mod = 0 if not has_mod else dna_mod_g.shape[2]
    D0 = 1 + d_g2 + d_mod

    fin_masks = [np.isfinite(y_mats[0][m]) for m in range(n_mut)]
    for k in range(1, K_tgt):
        for m in range(n_mut):
            if not np.array_equal(fin_masks[m], np.isfinite(y_mats[k][m])):
                raise ValueError("All targets in multi-target LOGO solver must share finite masks")

    # Y_fin[m] has shape (n_m, K_tgt)
    Y_fin = [
        np.column_stack([y_mats[k][m, fin_masks[m]].astype(np.float64) for k in range(K_tgt)])
        for m in range(n_mut)
    ]
    miss_idx = [np.flatnonzero(~fin_masks[m]) for m in range(n_mut)]
    g_muts = [[m for m in range(n_mut) if int(mut_gene_idx[m]) == g] for g in range(n_genes)]
    g_sizes = [sum(int(np.sum(fin_masks[m])) for m in g_muts[g]) for g in range(n_genes)]

    ones_col = np.ones((k_pwm, 1), dtype=np.float64)
    D_base_g = np.empty((n_genes, k_pwm, D0), dtype=np.float64)
    G_base_g = np.empty((n_genes, D0, D0), dtype=np.float64)
    for g in range(n_genes):
        if has_mod:
            Db = np.hstack([ones_col, dna_g2_g[g].astype(np.float64), dna_mod_g[g].astype(np.float64)])
        else:
            Db = np.hstack([ones_col, dna_g2_g[g].astype(np.float64)])
        D_base_g[g] = Db
        G_base_g[g] = Db.T @ Db

    G_base_miss_m: Dict[int, np.ndarray] = {}
    r_base_m = np.empty((n_mut, D0, K_tgt), dtype=np.float64)
    n_m_arr = np.empty(n_mut, dtype=np.int32)
    sum_y_m = np.empty((n_mut, K_tgt), dtype=np.float64)
    sum_yy_m = np.empty((n_mut, K_tgt), dtype=np.float64)
    for m in range(n_mut):
        g = int(mut_gene_idx[m])
        mk = fin_masks[m]
        Yv = Y_fin[m]  # (n_m, K_tgt)
        n_m_arr[m] = len(Yv)
        sum_y_m[m] = Yv.sum(axis=0)
        sum_yy_m[m] = np.sum(Yv * Yv, axis=0)
        mi = miss_idx[m]
        if len(mi) == 0:
            r_base_m[m] = D_base_g[g].T @ Yv
        else:
            Dm_miss = D_base_g[g][mi]
            G_base_miss_m[m] = Dm_miss.T @ Dm_miss
            r_base_m[m] = D_base_g[g][mk].T @ Yv

    G44_g = G4base_g = sum4_g = r4_m = None
    if mode == "p4":
        assert dna_pos4_g is not None
        p4_raw = dna_pos4_g.shape[2]
        G44_g = np.empty((n_genes, p4_raw, p4_raw), dtype=np.float64)
        G4base_g = np.empty((n_genes, p4_raw, D0), dtype=np.float64)
        sum4_g = np.empty((n_genes, p4_raw), dtype=np.float64)
        for g in range(n_genes):
            X4 = dna_pos4_g[g].astype(np.float64)
            G44_g[g] = X4.T @ X4
            G4base_g[g] = X4.T @ D_base_g[g]
            sum4_g[g] = X4.sum(axis=0)
        r4_m = np.empty((n_mut, p4_raw, K_tgt), dtype=np.float64)
        for m in range(n_mut):
            g = int(mut_gene_idx[m])
            X4m = dna_pos4_g[g][fin_masks[m]].astype(np.float64)
            r4_m[m] = X4m.T @ Y_fin[m]

    _LOGO_RIDGE_CTX.clear()
    _LOGO_RIDGE_CTX.update(
        {
            "K_tgt": K_tgt,
            "phi_prot": phi_prot.astype(np.float64),
            "mut_gene_idx": mut_gene_idx,
            "fin_masks": fin_masks,
            "miss_idx": miss_idx,
            "Y_fin": Y_fin,
            "dna_g2_g": dna_g2_g.astype(np.float64),
            "dna_pos4_g": dna_pos4_g,
            "dna_mod_g": None if dna_mod_g is None else dna_mod_g.astype(np.float64),
            "D_base_g": D_base_g,
            "G_base_g": G_base_g,
            "G_base_miss_m": G_base_miss_m,
            "r_base_m": r_base_m,
            "n_m_arr": n_m_arr,
            "sum_y_m": sum_y_m,
            "sum_yy_m": sum_yy_m,
            "G44_g": G44_g,
            "G4base_g": G4base_g,
            "sum4_g": sum4_g,
            "r4_m": r4_m,
            "mode": mode,
            "d_g1": d_g1,
            "d_g2": d_g2,
            "d_mod": d_mod,
            "D0": D0,
            "k_pwm": k_pwm,
            "n_genes": n_genes,
            "n_mut": n_mut,
            "g_muts": g_muts,
            "g_sizes": g_sizes,
        }
    )
    rhos_out = [np.zeros(n_mut, dtype=np.float64) for _ in range(K_tgt)]
    try:
        max_w = 3 if (mode == "p4" and dna_pos4_g is not None and dna_pos4_g.shape[2] > 2000) else int(n_workers)
        eff_w = min(max(1, max_w), n_genes)
        if eff_w <= 1:
            for gh in range(n_genes):
                _, te_m, te_rk = _logo_ridge_fold_worker(gh)
                for m, rv_k in zip(te_m, te_rk):
                    for k in range(K_tgt):
                        rhos_out[k][m] = rv_k[k]
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=eff_w, mp_context=ctx) as ex:
                for _, te_m, te_rk in ex.map(_logo_ridge_fold_worker, range(n_genes)):
                    for m, rv_k in zip(te_m, te_rk):
                        for k in range(K_tgt):
                            rhos_out[k][m] = rv_k[k]
    finally:
        _LOGO_RIDGE_CTX.clear()
    return rhos_out


_LOGO_RIDGE_CTX: Dict[str, object] = {}


def _build_unit_xs_xm(
    m: int,
    c: Dict[str, object],
    p16: Optional[np.ndarray],
    p8: Optional[np.ndarray],
    act4: Optional[np.ndarray],
    mu4: Optional[np.ndarray],
    V_d64: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    g = int(c["mut_gene_idx"][m])
    mk = c["fin_masks"][m]
    g2 = c["dna_g2_g"][g][mk]
    d_g1 = int(c["d_g1"])
    g1 = g2[:, :d_g1]
    nm = len(g1)
    pr = np.broadcast_to(c["phi_prot"][m], (nm, c["phi_prot"].shape[1]))
    xb1 = kv7.build_bilinear_blocks(pr, g1, include_prot_add=True, include_dna_add=False)

    if c["mode"] == "g2":
        p16r = np.broadcast_to(p16[m], (nm, p16.shape[1]))
        xdi = kv7.build_bilinear_blocks(p16r, g2[:, d_g1:], include_prot_add=False, include_dna_add=False)
        xs = np.hstack([xb1, g2, xdi])
    else:
        p8r = np.broadcast_to(p8[m], (nm, p8.shape[1]))
        x_base = np.hstack([xb1, g2])
        x4a = c["dna_pos4_g"][g][mk][:, act4].astype(np.float64)
        x4_pc = (x4a - mu4[None, :]) @ V_d64
        x4_int = (p8r[:, :, None] * x4_pc[:, None, :]).reshape(nm, -1)
        xs = np.hstack([x_base, x4a, x4_int])

    if c["dna_mod_g"] is None:
        xm = None
    else:
        mod = c["dna_mod_g"][g][mk]
        xm = kv7.build_bilinear_blocks(pr, mod, include_prot_add=False, include_dna_add=True)
    return xs, xm


def _logo_ridge_fold_worker(g_hold: int) -> Tuple[int, List[int], List[List[float]]]:
    c = _LOGO_RIDGE_CTX
    K_tgt = int(c["K_tgt"])
    n_genes = int(c["n_genes"])
    k_pwm = int(c["k_pwm"])
    phi_prot: np.ndarray = c["phi_prot"]
    d_p = phi_prot.shape[1]
    d_g1 = int(c["d_g1"])
    d_g2 = int(c["d_g2"])
    d_rest = d_g2 - d_g1
    d_mod = int(c["d_mod"])
    D0 = int(c["D0"])
    g_muts: List[List[int]] = c["g_muts"]
    g_sizes: List[int] = c["g_sizes"]
    mode: str = str(c["mode"])
    has_mod = d_mod > 0

    tr_g = [g for g in range(n_genes) if g != g_hold]
    tr_m_all = [m for g in tr_g for m in g_muts[g]]
    te_m = g_muts[g_hold]

    p_tr = phi_prot[tr_m_all]
    mu_p = p_tr.mean(axis=0)
    sd_p = np.maximum(p_tr.std(axis=0), 1e-8)
    pz_tr = (p_tr - mu_p[None, :]) / sd_p[None, :]
    _, _, Vt_p = np.linalg.svd(pz_tr, full_matrices=False)
    Vp16 = Vt_p[:16].T
    Vp8 = Vt_p[:8].T
    pz_all = (phi_prot - mu_p[None, :]) / sd_p[None, :]
    p16 = pz_all @ Vp16
    p8 = pz_all @ Vp8

    act4 = mu4 = V_d64 = None
    p4_a = 0
    k_pc = 0
    if mode == "p4":
        G44_g = c["G44_g"]
        sum4_g = c["sum4_g"]
        n_tr_rows = float(len(tr_g) * k_pwm)
        sum4_tot = np.sum([sum4_g[g] for g in tr_g], axis=0)
        mu4_all = sum4_tot / n_tr_rows
        diag_tot = np.sum([np.diag(G44_g[g]) for g in tr_g], axis=0)
        col_var = (diag_tot / n_tr_rows) - (mu4_all * mu4_all)
        act4 = col_var > 1e-10
        p4_a = int(np.sum(act4))
        mu4 = mu4_all[act4]
        G44_tr_sum = np.zeros((p4_a, p4_a), dtype=np.float64)
        for g in tr_g:
            G44_tr_sum += G44_g[g][np.ix_(act4, act4)]
        cov = (G44_tr_sum / n_tr_rows) - np.outer(mu4, mu4)
        cov = 0.5 * (cov + cov.T)
        _, ec = sp_eigh(cov, check_finite=False, driver="evd")
        k_pc = min(64, ec.shape[1])
        V_d64 = ec[:, -k_pc:][:, ::-1].copy()
        del G44_tr_sum, cov, ec

    splits = kv7._make_balanced_inner_folds(tr_g, list(range(n_genes)), g_sizes, n_folds=3)
    gene_to_inner_fold: Dict[int, int] = {}
    for fi_k, (_, in_va_g) in enumerate(splits):
        for g in in_va_g:
            gene_to_inner_fold[int(g)] = fi_k

    s_one = slice(0, 1)
    s_g1 = slice(1, 1 + d_g1)
    s_g2 = slice(1, 1 + d_g2)
    s_gr = slice(1 + d_g1, 1 + d_g2)
    s_mod = slice(1 + d_g2, D0)
    one_a = np.ones(1, dtype=np.float64)

    sl_b0 = slice(0, d_p)
    sl_b1 = slice(d_p, d_p + d_p * d_g1)
    sl_b2 = slice(sl_b1.stop, sl_b1.stop + d_g2)
    if mode == "g2":
        sl_b3 = slice(sl_b2.stop, sl_b2.stop + p16.shape[1] * d_rest)
        p_s = sl_b3.stop
    else:
        sl_b3 = slice(sl_b2.stop, sl_b2.stop + p4_a)
        sl_b4 = slice(sl_b3.stop, sl_b3.stop + p8.shape[1] * k_pc)
        p_s = sl_b4.stop

    if has_mod:
        sl_m0 = slice(0, d_mod)
        sl_m1 = slice(d_mod, d_mod + d_p * d_mod)
        p_m = sl_m1.stop
    else:
        p_m = 0

    n_in = len(splits)
    fold_stats: List[Dict[str, object]] = [
        {
            "n": 0,
            "sum_y": np.zeros(K_tgt, dtype=np.float64),
            "sum_yy": np.zeros(K_tgt, dtype=np.float64),
            "sum_s": np.zeros(p_s, dtype=np.float64),
            "s_ss": np.zeros((p_s, p_s), dtype=np.float64),
            "s_sy": np.zeros((p_s, K_tgt), dtype=np.float64),
            "sum_m": np.zeros(p_m, dtype=np.float64),
            "s_sm": np.zeros((p_s, p_m), dtype=np.float64),
            "s_mm": np.zeros((p_m, p_m), dtype=np.float64),
            "s_my": np.zeros((p_m, K_tgt), dtype=np.float64),
        }
        for _ in range(n_in)
    ]

    G_base_g = c["G_base_g"]
    G_base_miss_m = c["G_base_miss_m"]
    r_base_m = c["r_base_m"]
    n_m_arr = c["n_m_arr"]
    sum_y_m = c["sum_y_m"]
    sum_yy_m = c["sum_yy_m"]
    miss_idx = c["miss_idx"]

    for g in tr_g:
        st = fold_stats[gene_to_inner_fold[int(g)]]
        muts_g = g_muts[g]
        nm_g = len(muts_g)
        Gg = G_base_g[g]

        # Gene-level protein matrices (nm_g, p) -> ONE outer sum per gene!
        P_g = phi_prot[muts_g]                       # (nm_g, d_p)
        One_g = np.ones((nm_g, 1), dtype=np.float64) # (nm_g, 1)
        if mode == "g2":
            P16_g = p16[muts_g]                      # (nm_g, 16)
            base_g_blocks = [
                (sl_b0, P_g, s_one),
                (sl_b1, P_g, s_g1),
                (sl_b2, One_g, s_g2),
                (sl_b3, P16_g, s_gr),
            ]
        else:
            P8_g = p8[muts_g]                        # (nm_g, 8)
            base_g_blocks = [
                (sl_b0, P_g, s_one),
                (sl_b1, P_g, s_g1),
                (sl_b2, One_g, s_g2),
            ]

        # Upper-triangle gene-grouped accumulation on Gg (29 calls instead of 89!)
        for sl_u, A_u, d_u in base_g_blocks:
            st["sum_s"][sl_u] += np.outer(A_u.sum(axis=0), Gg[0, d_u]).ravel()
        for i_u, (sl_u, A_u, d_u) in enumerate(base_g_blocks):
            for i_v in range(i_u, len(base_g_blocks)):
                sl_v, A_v, d_v = base_g_blocks[i_v]
                _kron_add_mat(st["s_ss"], sl_u, sl_v, A_u.T @ A_v, Gg[d_u, d_v])

        if has_mod:
            m_g_blocks = [(sl_m0, One_g, s_mod), (sl_m1, P_g, s_mod)]
            for sl_u, A_u, d_u in m_g_blocks:
                st["sum_m"][sl_u] += np.outer(A_u.sum(axis=0), Gg[0, d_u]).ravel()
            for i_u, (sl_u, A_u, d_u) in enumerate(m_g_blocks):
                for i_v in range(i_u, len(m_g_blocks)):
                    sl_v, A_v, d_v = m_g_blocks[i_v]
                    _kron_add_mat(st["s_mm"], sl_u, sl_v, A_u.T @ A_v, Gg[d_u, d_v])
            for sl_u, A_u, d_u in base_g_blocks:
                for sl_v, A_v, d_v in m_g_blocks:
                    _kron_add_mat(st["s_sm"], sl_u, sl_v, A_u.T @ A_v, Gg[d_u, d_v])

        if mode == "p4":
            G44_a_g = c["G44_g"][g][np.ix_(act4, act4)]
            G4b_a_g = c["G4base_g"][g][act4]
            s4_g = c["sum4_g"][g][act4]
            sb_g = Gg[0]
            mu_V = mu4 @ V_d64
            s4_V = s4_g @ V_d64
            G44_V = G44_a_g @ V_d64
            G4pc_g = G44_V - np.outer(s4_g, mu_V)
            Gpc_b_g = (V_d64.T @ G4b_a_g) - np.outer(mu_V, sb_g)
            Gpc_pc_g = (
                (V_d64.T @ G44_V)
                - np.outer(mu_V, s4_V)
                - np.outer(s4_V, mu_V)
                + float(k_pwm) * np.outer(mu_V, mu_V)
            )
            spc_g = s4_V - float(k_pwm) * mu_V

            st["sum_s"][sl_b3] += float(nm_g) * s4_g
            st["sum_s"][sl_b4] += np.outer(P8_g.sum(axis=0), spc_g).ravel()
            st["s_ss"][sl_b3, sl_b3] += float(nm_g) * G44_a_g
            _kron_add_mat(st["s_ss"], sl_b3, sl_b4, One_g.T @ P8_g, G4pc_g)
            _kron_add_mat(st["s_ss"], sl_b4, sl_b4, P8_g.T @ P8_g, Gpc_pc_g)
            for sl_u, A_u, d_u in base_g_blocks:
                _kron_add_mat(st["s_ss"], sl_u, sl_b3, A_u.T @ One_g, G4b_a_g[:, d_u].T)
                _kron_add_mat(st["s_ss"], sl_u, sl_b4, A_u.T @ P8_g, Gpc_b_g[:, d_u].T)
            if has_mod:
                for sl_v, A_v, d_v in m_g_blocks:
                    _kron_add_mat(st["s_sm"], sl_b3, sl_v, One_g.T @ A_v, G4b_a_g[:, d_v])
                    _kron_add_mat(st["s_sm"], sl_b4, sl_v, P8_g.T @ A_v, Gpc_b_g[:, d_v])

        # Per-mutant RHS (s_sy, s_my) + exact subtraction for rare missing probes (18/184,000)
        for m in muts_g:
            pr = phi_prot[m]
            rb = r_base_m[m]  # (D0, K_tgt)
            st["n"] += int(n_m_arr[m])
            st["sum_y"] += sum_y_m[m]
            st["sum_yy"] += sum_yy_m[m]

            b_m = [(sl_b0, pr, s_one), (sl_b1, pr, s_g1), (sl_b2, one_a, s_g2)]
            if mode == "g2":
                b_m.append((sl_b3, p16[m], s_gr))
            for sl_u, a_u, d_u in b_m:
                st["s_sy"][sl_u] += (a_u[:, None, None] * rb[d_u][None, :, :]).reshape(-1, K_tgt)
            if has_mod:
                m_m = [(sl_m0, one_a, s_mod), (sl_m1, pr, s_mod)]
                for sl_u, a_u, d_u in m_m:
                    st["s_my"][sl_u] += (a_u[:, None, None] * rb[d_u][None, :, :]).reshape(-1, K_tgt)
            if mode == "p4":
                p8m = p8[m]
                r4m = c["r4_m"][m][act4]  # (p4_a, K_tgt)
                rpcm = (V_d64.T @ r4m) - np.outer(mu_V, sum_y_m[m])
                st["s_sy"][sl_b3] += r4m
                st["s_sy"][sl_b4] += (p8m[:, None, None] * rpcm[None, :, :]).reshape(-1, K_tgt)

            mi = miss_idx[m]
            if len(mi) > 0:
                G_miss = G_base_miss_m[m]
                for sl_u, a_u, d_u in b_m:
                    st["sum_s"][sl_u] -= np.outer(a_u, G_miss[0, d_u]).ravel()
                for i_u, (sl_u, a_u, d_u) in enumerate(b_m):
                    for i_v in range(i_u, len(b_m)):
                        sl_v, a_v, d_v = b_m[i_v]
                        _kron_add_mat(st["s_ss"], sl_u, sl_v, -np.outer(a_u, a_v), G_miss[d_u, d_v])
                if has_mod:
                    for sl_u, a_u, d_u in m_m:
                        st["sum_m"][sl_u] -= np.outer(a_u, G_miss[0, d_u]).ravel()
                    for i_u, (sl_u, a_u, d_u) in enumerate(m_m):
                        for i_v in range(i_u, len(m_m)):
                            sl_v, a_v, d_v = m_m[i_v]
                            _kron_add_mat(st["s_mm"], sl_u, sl_v, -np.outer(a_u, a_v), G_miss[d_u, d_v])
                    for sl_u, a_u, d_u in b_m:
                        for sl_v, a_v, d_v in m_m:
                            _kron_add_mat(st["s_sm"], sl_u, sl_v, -np.outer(a_u, a_v), G_miss[d_u, d_v])
                if mode == "p4":
                    X4_miss = c["dna_pos4_g"][g][mi][:, act4].astype(np.float64)
                    Db_miss = c["D_base_g"][g][mi]
                    PC_miss = (X4_miss - mu4[None, :]) @ V_d64
                    st["sum_s"][sl_b3] -= X4_miss.sum(axis=0)
                    st["sum_s"][sl_b4] -= np.outer(p8[m], PC_miss.sum(axis=0)).ravel()
                    st["s_ss"][sl_b3, sl_b3] -= (X4_miss.T @ X4_miss)
                    _kron_add_mat(st["s_ss"], sl_b3, sl_b4, -np.outer(one_a, p8[m]), X4_miss.T @ PC_miss)
                    _kron_add_mat(st["s_ss"], sl_b4, sl_b4, -np.outer(p8[m], p8[m]), PC_miss.T @ PC_miss)
                    G4b_miss = X4_miss.T @ Db_miss
                    Gpcb_miss = PC_miss.T @ Db_miss
                    for sl_u, a_u, d_u in b_m:
                        _kron_add_mat(st["s_ss"], sl_u, sl_b3, -np.outer(a_u, one_a), G4b_miss[:, d_u].T)
                        _kron_add_mat(st["s_ss"], sl_u, sl_b4, -np.outer(a_u, p8[m]), Gpcb_miss[:, d_u].T)
                    if has_mod:
                        for sl_v, a_v, d_v in m_m:
                            _kron_add_mat(st["s_sm"], sl_b3, sl_v, -np.outer(one_a, a_v), G4b_miss[:, d_v])
                            _kron_add_mat(st["s_sm"], sl_b4, sl_v, -np.outer(p8[m], a_v), Gpcb_miss[:, d_v])

    # Symmetrize upper-triangular s_ss & s_mm in each fold bucket once
    for st in fold_stats:
        s_ss = st["s_ss"]
        iu = np.triu_indices(p_s, k=1)
        s_ss[iu[1], iu[0]] = s_ss[iu]
        if has_mod:
            s_mm = st["s_mm"]
            im = np.triu_indices(p_m, k=1)
            s_mm[im[1], im[0]] = s_mm[im]

    tot_stats: Dict[str, object] = {
        "n": sum(fs["n"] for fs in fold_stats),
        "sum_y": sum(fs["sum_y"] for fs in fold_stats),
        "sum_yy": sum(fs["sum_yy"] for fs in fold_stats),
        "sum_s": sum(fs["sum_s"] for fs in fold_stats),
        "s_ss": sum(fs["s_ss"] for fs in fold_stats),
        "s_sy": sum(fs["s_sy"] for fs in fold_stats),
        "sum_m": sum(fs["sum_m"] for fs in fold_stats),
        "s_sm": sum(fs["s_sm"] for fs in fold_stats),
        "s_mm": sum(fs["s_mm"] for fs in fold_stats),
        "s_my": sum(fs["s_my"] for fs in fold_stats),
    }

    def _tr_minus_fold(k: int) -> Dict[str, object]:
        va = fold_stats[k]
        return {
            "n": tot_stats["n"] - va["n"],
            "sum_y": tot_stats["sum_y"] - va["sum_y"],
            "sum_yy": tot_stats["sum_yy"] - va["sum_yy"],
            "sum_s": tot_stats["sum_s"] - va["sum_s"],
            "s_ss": tot_stats["s_ss"] - va["s_ss"],
            "s_sy": tot_stats["s_sy"] - va["s_sy"],
            "sum_m": tot_stats["sum_m"] - va["sum_m"],
            "s_sm": tot_stats["s_sm"] - va["s_sm"],
            "s_mm": tot_stats["s_mm"] - va["s_mm"],
            "s_my": tot_stats["s_my"] - va["s_my"],
        }

    def _std_eq(agg: Dict[str, object]) -> Dict[str, object]:
        n_t = float(agg["n"])
        mu_s = agg["sum_s"] / n_t
        y_mean = agg["sum_y"] / n_t  # (K_tgt,)
        C_ss = agg["s_ss"] - n_t * np.outer(mu_s, mu_s)
        var_s = np.maximum(np.diag(C_ss) / n_t, 0.0)
        sd_s = np.sqrt(var_s)
        dead_s = sd_s < 1e-10
        inv_s = np.where(dead_s, 0.0, 1.0 / np.where(dead_s, 1.0, sd_s))
        Z_ss = C_ss * np.outer(inv_s, inv_s)
        Z_ss = 0.5 * (Z_ss + Z_ss.T)
        rhs_s = (agg["s_sy"] - np.outer(mu_s, agg["sum_y"])) * inv_s[:, None]  # (p_s, K_tgt)

        if p_m == 0:
            return {
                "mu_s": mu_s, "inv_s": inv_s, "y_mean": y_mean,
                "Z_ss": Z_ss, "rhs_s": rhs_s,
                "mu_m": np.zeros(0), "inv_m": np.zeros(0),
                "Z_sm": np.zeros((p_s, 0)), "Z_mm": np.zeros((0, 0)), "rhs_m": np.zeros((0, K_tgt)),
            }
        mu_m = agg["sum_m"] / n_t
        C_mm = agg["s_mm"] - n_t * np.outer(mu_m, mu_m)
        var_m = np.maximum(np.diag(C_mm) / n_t, 0.0)
        sd_m = np.sqrt(var_m)
        dead_m = sd_m < 1e-10
        inv_m = np.where(dead_m, 0.0, 1.0 / np.where(dead_m, 1.0, sd_m))
        Z_mm = C_mm * np.outer(inv_m, inv_m)
        Z_mm = 0.5 * (Z_mm + Z_mm.T)
        C_sm = agg["s_sm"] - n_t * np.outer(mu_s, mu_m)
        Z_sm = C_sm * np.outer(inv_s, inv_m)
        rhs_m = (agg["s_my"] - np.outer(mu_m, agg["sum_y"])) * inv_m[:, None]  # (p_m, K_tgt)
        return {
            "mu_s": mu_s, "inv_s": inv_s, "y_mean": y_mean,
            "Z_ss": Z_ss, "rhs_s": rhs_s,
            "mu_m": mu_m, "inv_m": inv_m,
            "Z_sm": Z_sm, "Z_mm": Z_mm, "rhs_m": rhs_m,
        }

    def _val_sse_vec(Ws_r: np.ndarray, Wm_r: np.ndarray, b_r: np.ndarray, va: Dict[str, object]) -> np.ndarray:
        """Exact ||X_s W_s + X_m W_m + b - Y||^2 across all K_tgt columns simultaneously."""
        n_v = float(va["n"])
        ss_w = va["s_ss"] @ Ws_r  # (p_s, K_tgt)
        val = (
            va["sum_yy"]
            + n_v * (b_r * b_r)
            - 2.0 * b_r * va["sum_y"]
            + np.sum(Ws_r * ss_w, axis=0)
            + 2.0 * b_r * (va["sum_s"] @ Ws_r)
            - 2.0 * np.sum(va["s_sy"] * Ws_r, axis=0)
        )
        if p_m > 0:
            mm_w = va["s_mm"] @ Wm_r
            sm_w = va["s_sm"] @ Wm_r
            val = val + (
                np.sum(Wm_r * mm_w, axis=0)
                + 2.0 * np.sum(Ws_r * sm_w, axis=0)
                + 2.0 * b_r * (va["sum_m"] @ Wm_r)
                - 2.0 * np.sum(va["s_my"] * Wm_r, axis=0)
            )
        return val

    if mode == "p4" and has_mod:
        ls_grid = (30.0, 300.0, 3000.0)
        lm_grid = (3.0, 30.0, 300.0, 3000.0)
    else:
        ls_grid = (3.0, 30.0, 300.0, 3000.0)
        lm_grid = (3.0, 30.0, 300.0, 3000.0) if has_mod else (0.0,)

    if not has_mod:
        cv_sse = np.zeros((K_tgt, len(ls_grid)), dtype=np.float64)
        for fi_k in range(n_in):
            eq = _std_eq(_tr_minus_fold(fi_k))
            va = fold_stats[fi_k]
            for i_s, ls in enumerate(ls_grid):
                K_ss = eq["Z_ss"].copy()
                K_ss[np.diag_indices(p_s)] += ls
                Ws = cho_solve(cho_factor(K_ss, lower=True, check_finite=False), eq["rhs_s"], check_finite=False)
                Ws_r = Ws * eq["inv_s"][:, None]
                b_r = eq["y_mean"] - (eq["mu_s"] @ Ws_r)
                cv_sse[:, i_s] += _val_sse_vec(Ws_r, np.zeros((0, K_tgt)), b_r, va)
            del eq
        best_ls_k = [float(ls_grid[int(np.argmin(cv_sse[k]))]) for k in range(K_tgt)]
        best_lm_k = [0.0 for _ in range(K_tgt)]
    else:
        cv_sse = np.zeros((K_tgt, len(ls_grid), len(lm_grid)), dtype=np.float64)
        for fi_k in range(n_in):
            eq = _std_eq(_tr_minus_fold(fi_k))
            va = fold_stats[fi_k]
            for i_s, ls in enumerate(ls_grid):
                K_ss = eq["Z_ss"].copy()
                K_ss[np.diag_indices(p_s)] += ls
                L_s, _ = cho_factor(K_ss, lower=True, check_finite=False)
                # ONE triangular solve L_s^{-1} [Z_sm, rhs_s] (2x faster than cho_solve!)
                Y_all = solve_triangular(
                    L_s, np.hstack([eq["Z_sm"], eq["rhs_s"]]), lower=True, check_finite=False
                )
                Y_sm = Y_all[:, :p_m]
                y0 = Y_all[:, p_m:]  # (p_s, K_tgt)
                S_mm = eq["Z_mm"] - (Y_sm.T @ Y_sm)
                S_mm = 0.5 * (S_mm + S_mm.T)
                rhs_m_eff = eq["rhs_m"] - (Y_sm.T @ y0)
                for i_m, lm in enumerate(lm_grid):
                    S_reg = S_mm.copy()
                    S_reg[np.diag_indices(p_m)] += lm
                    Wm = cho_solve(cho_factor(S_reg, lower=True, check_finite=False), rhs_m_eff, check_finite=False)
                    # Back-substitute L_s^T Ws = y0 - Y_sm @ Wm (only K_tgt=2 cols!)
                    Ws = solve_triangular(L_s.T, y0 - (Y_sm @ Wm), lower=False, check_finite=False)
                    Ws_r = Ws * eq["inv_s"][:, None]
                    Wm_r = Wm * eq["inv_m"][:, None]
                    b_r = eq["y_mean"] - (eq["mu_s"] @ Ws_r) - (eq["mu_m"] @ Wm_r)
                    cv_sse[:, i_s, i_m] += _val_sse_vec(Ws_r, Wm_r, b_r, va)
            del eq
        best_ls_k = []
        best_lm_k = []
        for k in range(K_tgt):
            bi, bj = np.unravel_index(int(np.argmin(cv_sse[k])), (len(ls_grid), len(lm_grid)))
            best_ls_k.append(float(ls_grid[bi]))
            best_lm_k.append(float(lm_grid[bj]))

    # Final outer fit on tot_stats (group by unique best_ls to share Cholesky!)
    eq_out = _std_eq(tot_stats)
    Ws_out = np.empty((p_s, K_tgt), dtype=np.float64)
    Wm_out = np.empty((p_m, K_tgt), dtype=np.float64)
    b_out = np.empty(K_tgt, dtype=np.float64)
    by_ls: Dict[float, List[int]] = defaultdict(list)
    for k in range(K_tgt):
        by_ls[best_ls_k[k]].append(k)
    for ls, ks in by_ls.items():
        K_ss = eq_out["Z_ss"].copy()
        K_ss[np.diag_indices(p_s)] += ls
        L_s, _ = cho_factor(K_ss, lower=True, check_finite=False)
        if not has_mod:
            Ws = cho_solve((L_s, True), eq_out["rhs_s"][:, ks], check_finite=False)
            Ws_r = Ws * eq_out["inv_s"][:, None]
            Ws_out[:, ks] = Ws_r
            b_out[ks] = eq_out["y_mean"][ks] - (eq_out["mu_s"] @ Ws_r)
        else:
            Y_all = solve_triangular(
                L_s, np.hstack([eq_out["Z_sm"], eq_out["rhs_s"][:, ks]]), lower=True, check_finite=False
            )
            Y_sm = Y_all[:, :p_m]
            y0 = Y_all[:, p_m:]
            S_mm = eq_out["Z_mm"] - (Y_sm.T @ Y_sm)
            S_mm = 0.5 * (S_mm + S_mm.T)
            rhs_m_eff = eq_out["rhs_m"][:, ks] - (Y_sm.T @ y0)
            for j_loc, k in enumerate(ks):
                S_reg = S_mm.copy()
                S_reg[np.diag_indices(p_m)] += best_lm_k[k]
                wm = cho_solve(cho_factor(S_reg, lower=True, check_finite=False), rhs_m_eff[:, j_loc], check_finite=False)
                ws = solve_triangular(L_s.T, y0[:, j_loc] - (Y_sm @ wm), lower=False, check_finite=False)
                ws_r = ws * eq_out["inv_s"]
                wm_r = wm * eq_out["inv_m"]
                Ws_out[:, k] = ws_r
                Wm_out[:, k] = wm_r
                b_out[k] = float(eq_out["y_mean"][k] - np.dot(eq_out["mu_s"], ws_r) - np.dot(eq_out["mu_m"], wm_r))

    te_rhos_k: List[List[float]] = []
    for m in te_m:
        xs, xm = _build_unit_xs_xm(m, c, p16, p8, act4, mu4, V_d64)
        pred = b_out[None, :] + (xs @ Ws_out)
        if has_mod and xm is not None:
            pred = pred + (xm @ Wm_out)
        Ym = c["Y_fin"][m]
        te_rhos_k.append([float(kv7.safe_spearman_pearson(Ym[:, k], pred[:, k])[0]) for k in range(K_tgt)])

    del fold_stats, tot_stats, eq_out
    gc.collect()
    return g_hold, te_m, te_rhos_k


_LOGO_HGB_CTX: Dict[str, object] = {}


def _logo_hgb_fold_worker(g_hold: int) -> Tuple[int, List[int], List[float]]:
    c = _LOGO_HGB_CTX
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    mut_gene_idx: np.ndarray = c["mut_gene_idx"]
    phi_prot: np.ndarray = c["phi_prot"]
    dna_g: np.ndarray = c["dna_g"]
    fin_masks: Sequence[np.ndarray] = c["fin_masks"]
    y_fin: Sequence[np.ndarray] = c["y_fin"]
    cfg: Dict[str, object] = c["cfg"]
    n_mut = len(mut_gene_idx)

    tr_m = [m for m in range(n_mut) if int(mut_gene_idx[m]) != g_hold]
    te_m = [m for m in range(n_mut) if int(mut_gene_idx[m]) == g_hold]

    n_tr = sum(len(y_fin[m]) for m in tr_m)
    d_p = phi_prot.shape[1]
    d_x = dna_g.shape[2]
    X_tr = np.empty((n_tr, d_p + d_x), dtype=np.float32)
    y_tr = np.empty(n_tr, dtype=np.float32)
    cur = 0
    for m in tr_m:
        nm = len(y_fin[m])
        g = int(mut_gene_idx[m])
        X_tr[cur : cur + nm, :d_p] = phi_prot[m]
        X_tr[cur : cur + nm, d_p:] = dna_g[g][fin_masks[m]]
        y_tr[cur : cur + nm] = y_fin[m]
        cur += nm

    model = HistGradientBoostingRegressor(**cfg)
    model.fit(X_tr, y_tr)
    del X_tr, y_tr

    te_rhos: List[float] = []
    for m in te_m:
        nm = len(y_fin[m])
        X_te = np.empty((nm, d_p + d_x), dtype=np.float32)
        X_te[:, :d_p] = phi_prot[m]
        X_te[:, d_p:] = dna_g[g_hold][fin_masks[m]]
        pred = model.predict(X_te)
        te_rhos.append(float(kv7.safe_spearman_pearson(y_fin[m].astype(np.float64), pred)[0]))
    gc.collect()
    return g_hold, te_m, te_rhos


def _eval_hgbdt_logo_full_k2000(
    phi_prot: np.ndarray,
    mut_gene_idx: np.ndarray,
    y_mat: np.ndarray,
    dna_g: np.ndarray,
    n_workers: int = 2,
) -> np.ndarray:
    """
    Exact full-K=2,000 30-fold LOGO HGBDT (`CFG_HGB_100`), matching `v7` & `v7_1` Exp 1b
    with memory-lean pre-allocated arrays (<= 2.5 GiB peak across `n_workers=2`).
    """
    n_mut = len(mut_gene_idx)
    n_genes = dna_g.shape[0]
    fin_masks = [np.isfinite(y_mat[m]) for m in range(n_mut)]
    y_fin = [y_mat[m, fin_masks[m]].astype(np.float32) for m in range(n_mut)]

    _LOGO_HGB_CTX.clear()
    _LOGO_HGB_CTX.update(
        {
            "phi_prot": phi_prot.astype(np.float32),
            "mut_gene_idx": mut_gene_idx,
            "dna_g": dna_g.astype(np.float32),
            "fin_masks": fin_masks,
            "y_fin": y_fin,
            "cfg": hsc.CFG_HGB_100,
        }
    )
    rhos = np.zeros(n_mut, dtype=np.float64)
    try:
        eff_w = min(max(1, int(n_workers)), n_genes)
        if eff_w <= 1:
            for gh in range(n_genes):
                _, te_m, te_r = _logo_hgb_fold_worker(gh)
                for m, rv in zip(te_m, te_r):
                    rhos[m] = rv
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=eff_w, mp_context=ctx) as ex:
                for _, te_m, te_r in ex.map(_logo_hgb_fold_worker, range(n_genes)):
                    for m, rv in zip(te_m, te_r):
                        rhos[m] = rv
    finally:
        _LOGO_HGB_CTX.clear()
    return rhos


def run_benchmark_b_gse233827(
    hd_bundle: Dict[str, object],
    n_boot: int = 5000,
    n_workers_ridge: int = 4,
    n_workers_hgb: int = 2,
    part_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Run full Benchmark B on Homeodomain `GSE233827` (92 mutants, 30-fold LOGO)
    across BOTH `dz_f70` and `dz_spec_f70`:
      - Bilinear Ridge `Mut x Seq_G2` ladder (12-bp, +DNAshapeR, +DeepL2, +DeepL4, Flank15, Flank15+DeepL4)
      - Bilinear Ridge `Mut x Seq_Pos4mer` ladder (12-bp, +DNAshapeR, +DeepL4, Flank15, Flank15+DeepL4)
      - Full-K=2,000 `HGBDT` ladder (12-bp, +DNAshapeR, +DeepL4, Flank15, Flank15+DeepL4)
    Supports per-model checkpointing into `part_dir` for resume safety.
    """
    ds = hd_bundle["ds"]
    phi_prot = ds["phi_prot"].astype(np.float64)
    mut_gene_idx = ds["mut_gene_idx"].astype(np.int32)
    mut_segments = ds["mut_segments"]
    mut_kock_strata = ds["mut_kock_strata"]
    n_genes = len(ds["gene_names"])
    k_pwm = ds["win12_codes_f70"].shape[1]

    # Precompute DNA feature tensors across all 30 genes
    codes12_g = ds["win12_codes_f70"].astype(np.uint8)
    codes15_g = hd_bundle["codes15_g"]  # (30, 2000, 15) int8 ACGTN

    g2_12_g = np.stack([kv7.encode_seq_g2_12bp(codes12_g[g]) for g in range(n_genes)], axis=0)
    pos4_12_g = np.stack([kv7.encode_seq_pos4mer_n1n4(codes12_g[g]) for g in range(n_genes)], axis=0)

    g2_15_g = np.stack([encode_oriented_g2_acgtn(codes15_g[g]) for g in range(n_genes)], axis=0)
    # 11 positional 4-mers on 15-bp [F-5..F+2]: spans [0:4]..[10:14] = [F-5..F+1] (exact L4 interbase RF at N1..N4!)
    pos4_15_g = np.stack([encode_oriented_pos4mer_acgtn(codes15_g[g], n_slots=11) for g in range(n_genes)], axis=0)

    # HGBDT sequence DNA blocks: [codes, g1, gc2] matching v7/v7_1 exact feature columns
    hgb_seq12_g = np.stack(
        [
            np.hstack([codes12_g[g].astype(np.float32), g2_12_g[g, :, :36].astype(np.float32), g2_12_g[g, :, -12:-10].astype(np.float32)])
            for g in range(n_genes)
        ],
        axis=0,
    )
    hgb_seq15_g = np.stack(
        [
            np.hstack([codes15_g[g].astype(np.float32), g2_15_g[g, :, :45].astype(np.float32), g2_15_g[g, :, -12:-10].astype(np.float32)])
            for g in range(n_genes)
        ],
        axis=0,
    )

    sh_dnashaper = hd_bundle["dnashaper_n1n4_g"]
    sh_deep_l2 = hd_bundle["deep_l2_n1n4_g"]
    sh_deep_l4 = hd_bundle["deep_l4_n1n4_g"]

    strata_masks: Dict[str, Optional[np.ndarray]] = {
        "all_92": None,
        "seg_N_arm_1_9": (mut_segments == "N-arm"),
        "kock_specificity_yes": (mut_kock_strata == "specificity_yes"),
    }

    contrast_defs: List[Tuple[str, str, str]] = [
        # G2
        ("G2_DNAshapeR_vs_G2", "Mut_x_(Seq_G2+DNAshapeR_N1N4)", "Mut_x_Seq_G2"),
        ("G2_DeepL2_vs_G2", "Mut_x_(Seq_G2+DeepDNAshape_L2_N1N4)", "Mut_x_Seq_G2"),
        ("G2_DeepL4_vs_G2", "Mut_x_(Seq_G2+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_G2"),
        ("G2_DeepL4_vs_DNAshapeR", "Mut_x_(Seq_G2+DeepDNAshape_L4_N1N4)", "Mut_x_(Seq_G2+DNAshapeR_N1N4)"),
        ("G2_Flank15_vs_G2", "Mut_x_Seq_G2_Flank15", "Mut_x_Seq_G2"),
        ("G2_DeepL4_vs_G2_Flank15", "Mut_x_(Seq_G2+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_G2_Flank15"),
        ("G2_Flank15_DeepL4_vs_G2_Flank15", "Mut_x_(Seq_G2_Flank15+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_G2_Flank15"),
        # Pos4mer
        ("Pos4mer_DNAshapeR_vs_Pos4mer", "Mut_x_(Seq_Pos4mer_N1N4+DNAshapeR_N1N4)", "Mut_x_Seq_Pos4mer_N1N4"),
        ("Pos4mer_DeepL4_vs_Pos4mer", "Mut_x_(Seq_Pos4mer_N1N4+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_Pos4mer_N1N4"),
        ("Pos4mer_DeepL4_vs_DNAshapeR", "Mut_x_(Seq_Pos4mer_N1N4+DeepDNAshape_L4_N1N4)", "Mut_x_(Seq_Pos4mer_N1N4+DNAshapeR_N1N4)"),
        ("Pos4mer_Flank15_vs_Pos4mer", "Mut_x_Seq_Pos4mer_Flank15", "Mut_x_Seq_Pos4mer_N1N4"),
        ("Pos4mer_DeepL4_vs_Pos4mer_Flank15", "Mut_x_(Seq_Pos4mer_N1N4+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_Pos4mer_Flank15"),
        ("Pos4mer_Flank15_DeepL4_vs_Pos4mer_Flank15", "Mut_x_(Seq_Pos4mer_Flank15+DeepDNAshape_L4_N1N4)", "Mut_x_Seq_Pos4mer_Flank15"),
        # HGBDT
        ("HGBDT_DNAshapeR_vs_HGBDT", "HGBDT_Mut_x_(Seq+DNAshapeR_N1N4)", "HGBDT_Mut_x_Seq"),
        ("HGBDT_DeepL4_vs_HGBDT", "HGBDT_Mut_x_(Seq+DeepDNAshape_L4_N1N4)", "HGBDT_Mut_x_Seq"),
        ("HGBDT_DeepL4_vs_DNAshapeR", "HGBDT_Mut_x_(Seq+DeepDNAshape_L4_N1N4)", "HGBDT_Mut_x_(Seq+DNAshapeR_N1N4)"),
        ("HGBDT_Flank15_vs_HGBDT", "HGBDT_Mut_x_Seq_Flank15", "HGBDT_Mut_x_Seq"),
        ("HGBDT_DeepL4_vs_HGBDT_Flank15", "HGBDT_Mut_x_(Seq+DeepDNAshape_L4_N1N4)", "HGBDT_Mut_x_Seq_Flank15"),
        ("HGBDT_Flank15_DeepL4_vs_HGBDT_Flank15", "HGBDT_Mut_x_(Seq_Flank15+DeepDNAshape_L4_N1N4)", "HGBDT_Mut_x_Seq_Flank15"),
    ]

    def _load_or_run(key: str, fn):
        if part_dir is not None:
            part_dir.mkdir(parents=True, exist_ok=True)
            p = part_dir / f"{key}.npy"
            if p.exists():
                print(f"    [Bench B cache hit] {key}", flush=True)
                return np.load(p)
        arr = fn()
        if part_dir is not None:
            np.save(part_dir / f"{key}.npy", arr)
        return arr

    targets_order = ("dz_f70", "dz_spec_f70")
    y_mats_both = [ds[t] for t in targets_order]
    rhos_by_tgt: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in targets_order}

    # 1. Simultaneous 2-target (dz_f70 + dz_spec_f70) Ridge evaluation (2x faster!)
    ridge_specs = [
        ("Mut_x_Seq_G2", g2_12_g, None, None, "g2"),
        ("Mut_x_(Seq_G2+DNAshapeR_N1N4)", g2_12_g, None, sh_dnashaper, "g2"),
        ("Mut_x_(Seq_G2+DeepDNAshape_L2_N1N4)", g2_12_g, None, sh_deep_l2, "g2"),
        ("Mut_x_(Seq_G2+DeepDNAshape_L4_N1N4)", g2_12_g, None, sh_deep_l4, "g2"),
        ("Mut_x_Seq_G2_Flank15", g2_15_g, None, None, "g2"),
        ("Mut_x_(Seq_G2_Flank15+DeepDNAshape_L4_N1N4)", g2_15_g, None, sh_deep_l4, "g2"),
        ("Mut_x_Seq_Pos4mer_N1N4", g2_12_g, pos4_12_g, None, "p4"),
        ("Mut_x_(Seq_Pos4mer_N1N4+DNAshapeR_N1N4)", g2_12_g, pos4_12_g, sh_dnashaper, "p4"),
        ("Mut_x_(Seq_Pos4mer_N1N4+DeepDNAshape_L4_N1N4)", g2_12_g, pos4_12_g, sh_deep_l4, "p4"),
        ("Mut_x_Seq_Pos4mer_Flank15", g2_15_g, pos4_15_g, None, "p4"),
        ("Mut_x_(Seq_Pos4mer_Flank15+DeepDNAshape_L4_N1N4)", g2_15_g, pos4_15_g, sh_deep_l4, "p4"),
    ]
    print("[Bench B] Running 11 simultaneous 2-target (dz_f70 + dz_spec_f70) LOGO Ridge specs...", flush=True)
    import time as _time
    for mn, dg2, dp4, dmod, md in ridge_specs:
        safe_m = mn.replace("(", "").replace(")", "").replace("+", "_plus_")
        paths = [
            (part_dir / f"{t}__{safe_m}.npy") if part_dir is not None else None
            for t in targets_order
        ]
        if all(p is not None and p.exists() for p in paths):
            print(f"    [Bench B cache hit] {mn} (both targets)", flush=True)
            for t, p in zip(targets_order, paths):
                rhos_by_tgt[t][mn] = np.load(p)
        else:
            t_m = _time.time()
            res_both = _eval_bilinear_ridge_logo_multi_target(
                phi_prot, mut_gene_idx, y_mats_both, dg2, dp4, dmod, mode=md, n_workers=n_workers_ridge
            )
            dt_m = _time.time() - t_m
            if part_dir is not None:
                part_dir.mkdir(parents=True, exist_ok=True)
            for t, p, arr in zip(targets_order, paths, res_both):
                rhos_by_tgt[t][mn] = arr
                if p is not None:
                    np.save(p, arr)
                g_means = [float(np.mean(arr[mut_gene_idx == g])) for g in range(n_genes)]
                print(f"    {t:11s} | {mn:48s} mut={np.mean(arr):.4f}  gene={np.mean(g_means):.4f} ({dt_m:.1f}s)", flush=True)

    # 2. Full-K=2,000 HGBDT models with strict RAM guard
    hgb_specs = [
        ("HGBDT_Mut_x_Seq", hgb_seq12_g),
        ("HGBDT_Mut_x_(Seq+DNAshapeR_N1N4)", np.concatenate([hgb_seq12_g, sh_dnashaper], axis=2)),
        ("HGBDT_Mut_x_(Seq+DeepDNAshape_L4_N1N4)", np.concatenate([hgb_seq12_g, sh_deep_l4], axis=2)),
        ("HGBDT_Mut_x_Seq_Flank15", hgb_seq15_g),
        ("HGBDT_Mut_x_(Seq_Flank15+DeepDNAshape_L4_N1N4)", np.concatenate([hgb_seq15_g, sh_deep_l4], axis=2)),
    ]
    for tgt in targets_order:
        y_mat = ds[tgt]
        print(f"[Bench B] Evaluating HGBDT models on `{tgt}`...", flush=True)
        for mn, d_in in hgb_specs:
            safe_k = f"{tgt}__{mn.replace('(', '').replace(')', '').replace('+', '_plus_')}"
            cached_p = (part_dir / f"{safe_k}.npy") if part_dir is not None else None
            if cached_p is None or not cached_p.exists():
                ram_gb = avail_ram_gib()
                eff_hgb_w = n_workers_hgb if ram_gb >= 7.0 else 1
                print(
                    f"    [RAM check] MemAvailable={ram_gb:.2f} GiB -> running `{mn}` with n_workers={eff_hgb_w}",
                    flush=True,
                )
            else:
                eff_hgb_w = n_workers_hgb
            t_h = _time.time()
            rhos_by_tgt[tgt][mn] = _load_or_run(
                safe_k,
                lambda _din=d_in, _nw=eff_hgb_w, _ym=y_mat: _eval_hgbdt_logo_full_k2000(
                    phi_prot, mut_gene_idx, _ym, _din, n_workers=_nw
                ),
            )
            arr = rhos_by_tgt[tgt][mn]
            g_means = [float(np.mean(arr[mut_gene_idx == g])) for g in range(n_genes)]
            print(f"    {tgt:11s} | {mn:48s} mut={np.mean(arr):.4f}  gene={np.mean(g_means):.4f} ({_time.time() - t_h:.1f}s)", flush=True)

    target_results: Dict[str, Dict[str, object]] = {}
    for tgt in targets_order:
        rhos_map = rhos_by_tgt[tgt]
        models_summary: Dict[str, Dict[str, float]] = {}
        for mn, r_arr in rhos_map.items():
            g_arr = np.array([float(np.mean(r_arr[mut_gene_idx == g])) for g in range(n_genes)])
            models_summary[mn] = {
                "mut_mean_spearman": float(np.mean(r_arr)),
                "mut_median_spearman": float(np.median(r_arr)),
                "gene_mean_spearman": float(np.mean(g_arr)),
                "gene_median_spearman": float(np.median(g_arr)),
                "n_arm_mut_mean": float(np.mean(r_arr[mut_segments == "N-arm"])),
                "spec_yes_mut_mean": float(np.mean(r_arr[mut_kock_strata == "specificity_yes"])),
            }

        contrasts_by_stratum: Dict[str, Dict[str, Dict[str, object]]] = {}
        for s_name, s_mask in strata_masks.items():
            c_out: Dict[str, Dict[str, object]] = {}
            for cid, ma, mb in contrast_defs:
                c_out[cid] = _gene_cluster_bootstrap_paired(
                    rhos_map[ma], rhos_map[mb], mut_gene_idx,
                    n_genes=n_genes, n_boot=n_boot, seed=42, mask=s_mask,
                )
            contrasts_by_stratum[s_name] = c_out

        target_results[tgt] = {
            "n_mutants": int(len(mut_gene_idx)),
            "n_genes": int(n_genes),
            "models": models_summary,
            "contrasts": contrasts_by_stratum,
            "per_mutant_spearman": {mn: [float(x) for x in arr] for mn, arr in rhos_map.items()},
        }

    return {
        "dataset": "GSE233827_Kock2024_Homeodomain_92mutants",
        "cv_scheme": "30-fold Leave-One-Gene-Out (LOGO), f1>=0.70, interior 2<=p<=26, K=2,000 probes/gene",
        "n_boot": int(n_boot),
        "alignment_meta": hd_bundle["alignment_meta"],
        "targets": target_results,
    }


def check_finite_recursive(obj: object, path: str = "root") -> None:
    """Assert 0 NaNs / 0 Infs recursively across JSON-serializable output."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_finite_recursive(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_finite_recursive(v, f"{path}[{i}]")
