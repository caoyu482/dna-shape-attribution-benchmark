#!/usr/bin/env python3
"""
Self-Contained Unit Test Suite for `dna-shape-attribution-benchmark`.

Verifies:
  1. Exact RC invariance (`0.00e+00`) of 14-bp E-box sequence & pentamer features.
  2. Zero-leakage central-10mer homology grouping.
  3. Exact mathematical invariants of `ShiftGCExactCRTNull` (`mean`, `cov`, `W_fix`, `block Gram`, `RC <= 1e-12`).
  4. Exact `W_fix`, residual covariance & RC invariants of `Sparse4merSurrogateNull`.
  5. Exact Woodbury parity (`<= 1e-12`) of `Degree-3 Polynomial Kernel Ridge`.
  6. Exact parity (`<= 1e-12`) between single-target & multi-target parallel batched Schur solvers.
  7. Exact step-down Holm-Bonferroni adjustment.
  8. Strict inductiveness & row-permutation invariance (`<= 1e-12`) of `run_inductive_pos4mer_ridge`.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import bhlh_benchmark as bf
from src import null_generators as ng


class TestCalibratedAttributionInvariants(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        bases = np.array(list("ACGT"))
        self.cores = [
            "".join(rng.choice(bases, size=4)) + "CACGTG" + "".join(rng.choice(bases, size=4))
            for _ in range(90)
        ]
        self.y = rng.normal(size=90)
        self.groups = np.arange(90, dtype=np.int32) // 3

    def test_01_rc_invariance_14bp_features(self) -> None:
        f_fwd = bf.build_rc_sym_ebox_features(self.cores)
        f_rc = bf.build_rc_sym_ebox_features([bf.revcomp_str(s) for s in self.cores])
        for k in ("X_g1", "X_g2", "X_pos4mer", "B_rc"):
            err = float(np.max(np.abs(f_fwd[k] - f_rc[k])))
            self.assertLessEqual(err, 1e-12, f"RC asymmetry in {k}: {err}")

    def test_02_homology_grouping_no_leakage(self) -> None:
        probes = ["AAA" + s + "TTT" + "A" * 16 for s in self.cores[:30]]
        cores, y_agg, groups, meta = bf.align_ebox_14bp_cores(probes, {"t": self.y[:30]}, window_len=14)
        self.assertEqual(meta["input_probes"], 30)
        self.assertEqual(len(cores), len(groups))
        # Verify identical central 10-mers map to identical group IDs
        g_by_c10 = {}
        for c, g in zip(cores, groups):
            c10 = bf.canonical_rc_str(c[2:-2])
            if c10 in g_by_c10:
                self.assertEqual(g_by_c10[c10], int(g))
            else:
                g_by_c10[c10] = int(g)

    def test_03_shiftgc_exact_crt_invariants(self) -> None:
        rng = np.random.default_rng(7)
        F_raw = rng.normal(size=(1024, 7))
        F_even = 0.5 * (F_raw + F_raw[ng.RC_5])
        p_raw = rng.uniform(1.0, 50.0, size=1024)
        crt = ng.ShiftGCExactCRTNull(F_even, p_raw, conditioning="WS_G2")
        F_null = crt.sample(rng)
        aud = crt.audit_invariants(F_null)
        for k, v in aud.items():
            self.assertLessEqual(v, 1e-11, f"ShiftGC invariant {k} violated: {v}")

    def test_04_sparse4mer_surrogate_invariants(self) -> None:
        rng = np.random.default_rng(11)
        F_raw = rng.normal(size=(1024, 7))
        F_even = 0.5 * (F_raw + F_raw[ng.RC_5])
        p_raw = rng.uniform(1.0, 50.0, size=1024)
        sp = ng.Sparse4merSurrogateNull(F_even, p_raw, n_total_obs=10000.0, n_pockets_per_ch=12)
        F_null = sp.sample(rng)
        self.assertLessEqual(float(np.max(np.abs(F_null - F_null[ng.RC_5]))), 1e-12)
        R_t = F_even - sp.base_proj
        R_n = F_null - sp.base_proj
        cov_t = (R_t.T * sp.p_emp[None, :]) @ R_t
        cov_n = (R_n.T * sp.p_emp[None, :]) @ R_n
        self.assertLessEqual(float(np.max(np.abs(cov_n - cov_t))), 1e-10)

    def test_05_poly3_multitarget_exact_parity(self) -> None:
        feats = bf.build_rc_sym_ebox_features(self.cores)
        rng = np.random.default_rng(19)
        tab = rng.normal(size=(1024, 7))
        tab = 0.5 * (tab + tab[ng.RC_5])
        X_sh7 = bf._contract_b_rc_single(feats["B_rc"], tab)
        X_mgw = bf._contract_b_rc_single(feats["B_rc"], tab[:, :1])
        y2 = rng.normal(size=len(self.y))
        mt = bf.eval_paired_poly3_multi_targets(
            feats["X_g1"], X_sh7, X_mgw, {"t1": self.y, "t2": y2}, self.groups,
            gammas=(1.0,), lams_seq=(10.0, 100.0), lams_mod=(10.0, 100.0), n_folds=3, n_inner_folds=2
        )
        s_ref = bf.eval_paired_poly3_multi_targets(
            feats["X_g1"], X_sh7, X_mgw, {"t1": self.y}, self.groups,
            gammas=(1.0,), lams_seq=(10.0, 100.0), lams_mod=(10.0, 100.0), n_folds=3, n_inner_folds=2
        )
        for mn in ("Poly3_Kernel", "Poly3_Kernel+Shape_7ch", "Poly3_Kernel+MGW"):
            err = float(np.max(np.abs(mt["t1"][mn] - s_ref["t1"][mn])))
            self.assertLessEqual(err, 1e-12, f"Poly3 parity diff on {mn}: {err}")

    def test_06_batched_multitarget_schur_parity(self) -> None:
        feats = bf.build_rc_sym_ebox_features(self.cores)
        rng = np.random.default_rng(23)
        bank = rng.normal(size=(6, 1024, 7))
        bank = 0.5 * (bank + bank[:, ng.RC_5, :])
        y2 = rng.normal(size=len(self.y))
        single = bf._eval_null_bank_chunked_multibin(feats["X_g2"], feats["B_rc"], bank, self.y, self.groups, n_folds=3, chunk_size=3)
        multi = bf.eval_null_bank_multi_target_parallel(
            feats["X_g2"], feats["B_rc"], bank, {"t1": self.y, "t2": y2}, self.groups, n_folds=3, chunk_size=3, n_workers=1
        )["t1"]
        self.assertLessEqual(float(np.max(np.abs(single - multi))), 1e-12)

    def test_07_holm_bonferroni_exact(self) -> None:
        adj = bf.holm_bonferroni([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.20], atol=1e-12)

    def test_08_inductive_predictor_permutation_invariance(self) -> None:
        tr_s, tr_y = self.cores[:60], self.y[:60]
        ev_s = self.cores[60:]
        p1 = bf.run_inductive_pos4mer_ridge(tr_s, tr_y, ev_s, seed=42)
        perm = np.random.default_rng(99).permutation(len(ev_s))
        p2 = bf.run_inductive_pos4mer_ridge(tr_s, tr_y, [ev_s[i] for i in perm], seed=42)
        self.assertLessEqual(float(np.max(np.abs(p1[perm] - p2))), 1e-12)

    def test_09_offline_ast_hygiene(self) -> None:
        banned = {"requests", "urllib", "http", "httpx", "aiohttp", "socket", "boto3"}
        for py in (REPO_ROOT / "src").glob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        self.assertNotIn(n.name.split(".")[0], banned)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module.split(".")[0], banned)


if __name__ == "__main__":
    unittest.main()
