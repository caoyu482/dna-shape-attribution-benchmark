#!/usr/bin/env python3
"""
Experiment 3: Authentic Rohs-Lab Deep DNAshape Comparative Control (`biophys_inductive_gfm/v7_1`).

Compares static pentamer `DNAshapeR` vs context-extended `Deep DNAshape` (Layer 4,
plus Layer 2 pentamer-RF control) on top of strong pure-sequence models (`Seq_G2`,
`Seq_Pos4mer`, `HGBDT_Seq`) AND fair flank-matched pure-sequence controls (`Flank22`
on bHLH `GSE59845`; `Flank15` on Homeodomain `GSE233827`), strictly CPU-only (0 GPU).

Outputs:
  - `<BIOPHYS_ROOT>/v7_1/artifacts/ckpt_v7_1_exp3_deep_dnashape.json`
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V3_COH = ROOT / "v3" / "data" / "processed" / "cohorts"
V7_1 = ROOT / "v7_1"
V7_1_SRC = V7_1 / "src"
V7_1_ART = V7_1 / "artifacts"
PARTS_DIR = V7_1_ART / ".deep_dnashape_parts"
OUT_JSON = V7_1_ART / "ckpt_v7_1_exp3_deep_dnashape.json"

if str(V7_1_SRC) not in sys.path:
    sys.path.insert(0, str(V7_1_SRC))

import deep_dnashape_bench as ddb  # noqa: E402


def _run_self_checks() -> Dict[str, object]:
    """Quick deterministic parity & RC-invariance self-checks before main run."""
    import ast

    # 1. Verify 0 network imports in v7_1/src/deep_dnashape_bench.py (CoE Pillar 2)
    src_path = V7_1_SRC / "deep_dnashape_bench.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    forbidden = {"urllib", "requests", "http", "socket", "ftplib", "ssl", "aiohttp", "httpx"}
    found_net = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in forbidden:
                    found_net.append(al.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                found_net.append(node.module)
    if found_net:
        raise RuntimeError(f"Forbidden network imports in deep_dnashape_bench.py: {found_net}")

    # 2. Verify ACGTN encoders match v7 / v7_1 reference encoders to <1e-12 on clean ACGT
    rng = np.random.default_rng(12345)
    seqs14 = ["".join(rng.choice(list("ACGT"), size=14)) for _ in range(32)]
    f_ref = ddb.bf.build_rc_sym_ebox_features(seqs14)
    f_new = ddb.encode_rc_sym_g2_and_pos4mer_acgtn(ddb.encode_acgtn(seqs14))
    err_g2_14 = float(np.max(np.abs(f_ref["X_g2"] - f_new["X_g2"])))
    err_p4_14 = float(np.max(np.abs(f_ref["X_pos4mer"] - f_new["X_pos4mer"])))
    assert err_g2_14 < 1e-12, f"G2 14bp parity error {err_g2_14}"
    assert err_p4_14 < 1e-6, f"Pos4mer 14bp parity error {err_p4_14}"

    # Verify strict RC invariance on 22-bp sequences with 'N' boundary padding
    seqs22 = ["NN" + "".join(rng.choice(list("ACGT"), size=18)) + "NN" for _ in range(16)]
    rc22 = [ddb.revcomp_n(s) for s in seqs22]
    f22_a = ddb.encode_rc_sym_g2_and_pos4mer_acgtn(ddb.encode_acgtn(seqs22))
    f22_b = ddb.encode_rc_sym_g2_and_pos4mer_acgtn(ddb.encode_acgtn(rc22))
    rc_err_g2_22 = float(np.max(np.abs(f22_a["X_g2"] - f22_b["X_g2"])))
    rc_err_p4_22 = float(np.max(np.abs(f22_a["X_pos4mer"] - f22_b["X_pos4mer"])))
    assert rc_err_g2_22 < 1e-12, f"G2 22bp RC error {rc_err_g2_22}"
    assert rc_err_p4_22 < 1e-12, f"Pos4mer 22bp RC error {rc_err_p4_22}"

    # Verify oriented 12-bp G2 & Pos4mer match kv7 exactly
    seqs12 = ["".join(rng.choice(list("ACGT"), size=12)) for _ in range(32)]
    c12 = ddb.encode_acgtn(seqs12)
    err_hd_g2 = float(np.max(np.abs(ddb.kv7.encode_seq_g2_12bp(c12) - ddb.encode_oriented_g2_acgtn(c12))))
    err_hd_p4 = float(np.max(np.abs(ddb.kv7.encode_seq_pos4mer_n1n4(c12) - ddb.encode_oriented_pos4mer_acgtn(c12, n_slots=5))))
    assert err_hd_g2 < 1e-12, f"HD G2 12bp parity error {err_hd_g2}"
    assert err_hd_p4 < 1e-12, f"HD Pos4mer 12bp parity error {err_hd_p4}"

    return {
        "offline_zero_network_imports": True,
        "bhlh_14bp_g2_parity_err": err_g2_14,
        "bhlh_14bp_pos4mer_parity_err": err_p4_14,
        "bhlh_22bp_N_rc_invariance_g2_err": rc_err_g2_22,
        "bhlh_22bp_N_rc_invariance_pos4mer_err": rc_err_p4_22,
        "hd_12bp_g2_parity_err": err_hd_g2,
        "hd_12bp_pos4mer_parity_err": err_hd_p4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v7_1 Exp 3 Authentic Deep DNAshape Control")
    parser.add_argument("--stage", choices=("all", "verify", "bench_a", "bench_b"), default="all")
    parser.add_argument("--workers-a", type=int, default=4)
    parser.add_argument("--workers-b-ridge", type=int, default=4)
    parser.add_argument("--workers-b-hgb", type=int, default=2)
    args = parser.parse_args()

    t0 = time.time()
    V7_1_ART.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Exp 3] Running offline & encoder parity self-checks...", flush=True)
    self_checks = _run_self_checks()
    print(f"  Self-checks PASSED: {self_checks}", flush=True)

    engine = ddb.AuthenticDeepDNAshapeCPU()

    # Load GSE59845 paired 36-bp probes (used both for verification on real probes and Benchmark A)
    df_max = pd.read_parquet(V3_COH / "disc_Max_gcPBM.parquet")
    df_mad = pd.read_parquet(V3_COH / "disc_Mad1_gcPBM.parquet")
    df_myc = pd.read_parquet(V3_COH / "disc_cMyc_gcPBM.parquet")
    assert df_max["seq"].tolist() == df_mad["seq"].tolist() == df_myc["seq"].tolist()
    seqs36_bhlh = df_max["seq"].tolist()

    # Step 1: Deep DNAshape verification vs pentamer DNAshapeR
    ver_path = PARTS_DIR / "verification.json"
    if ver_path.exists() and args.stage != "verify":
        verification = json.loads(ver_path.read_text(encoding="utf-8"))
        print("[Exp 3] Cache hit: verification.json", flush=True)
    else:
        print("[Exp 3] Verifying authentic Deep DNAshape (L1/L2/L4) vs DNAshapeR & RC equivariance...", flush=True)
        verification = ddb.verify_deep_dnashape_vs_dnashaper(engine, sample_36bp_seqs=seqs36_bhlh)
        ver_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
        print(
            f"  Pentamer L2 mean r (7ch)={verification['pentamer_L2_mean_pearson_7ch']:.4f}, "
            f"(core 6ch)={verification['pentamer_L2_mean_pearson_core6ch_excl_stretch']:.4f}, "
            f"RC max err={verification['rc_equivariance_max_overall']:.2e}",
            flush=True,
        )
    if args.stage == "verify":
        return

    # Step 2: Benchmark A — bHLH GSE59845 gcPBM (all 6 primary endpoints)
    bench_a_path = PARTS_DIR / "benchmark_a_bhlh.json"
    if args.stage in ("all", "bench_a"):
        if bench_a_path.exists() and args.stage == "all":
            bench_a = json.loads(bench_a_path.read_text(encoding="utf-8"))
            print("[Exp 3] Cache hit: benchmark_a_bhlh.json", flush=True)
        else:
            ta = time.time()
            y_max = df_max["y"].to_numpy(dtype=np.float64)
            y_mad = df_mad["y"].to_numpy(dtype=np.float64)
            y_myc = df_myc["y"].to_numpy(dtype=np.float64)
            y_map = {
                "Max": y_max,
                "Mad1": y_mad,
                "cMyc": y_myc,
                "cMyc_minus_Max": y_myc - y_max,
                "Mad1_minus_Max": y_mad - y_max,
                "cMyc_minus_Mad1": y_myc - y_mad,
            }
            aligned = ddb.align_gse59845_with_full_36bp_flanks(seqs36_bhlh, y_map, engine)
            print(
                f"  [Bench A] Aligned {aligned['align_meta']['input_probes']} probes -> "
                f"{aligned['align_meta']['unique_aligned_ebox_cores']} cores; "
                f"running 6 endpoints x 23 models (n_workers={args.workers_a})...",
                flush=True,
            )
            bench_a = ddb.run_benchmark_a_bhlh(aligned, n_boot=2000, n_workers=args.workers_a)
            bench_a["runtime_sec"] = round(float(time.time() - ta), 2)
            bench_a_path.write_text(json.dumps(bench_a, indent=2), encoding="utf-8")
            print(f"  [Bench A] Completed in {bench_a['runtime_sec']:.1f}s", flush=True)
            del aligned
            gc.collect()
        if args.stage == "bench_a":
            return
    else:
        bench_a = json.loads(bench_a_path.read_text(encoding="utf-8")) if bench_a_path.exists() else {}

    # Step 3: Benchmark B — Homeodomain GSE233827 (92 mutants, 30-fold LOGO on dz_f70 & dz_spec_f70)
    bench_b_path = PARTS_DIR / "benchmark_b_gse233827.json"
    if args.stage in ("all", "bench_b"):
        tb = time.time()
        hd_bundle = ddb.load_gse233827_full36_and_deep_dnashape(
            engine, cache_path=PARTS_DIR / "bench_b_deep_cache.npz"
        )
        bench_b = ddb.run_benchmark_b_gse233827(
            hd_bundle,
            n_boot=5000,
            n_workers_ridge=args.workers_b_ridge,
            n_workers_hgb=args.workers_b_hgb,
            part_dir=PARTS_DIR / "bench_b_models",
        )
        bench_b["runtime_sec"] = round(float(time.time() - tb), 2)
        bench_b_path.write_text(json.dumps(bench_b, indent=2), encoding="utf-8")
        print(f"  [Bench B] Completed in {bench_b['runtime_sec']:.1f}s", flush=True)
        del hd_bundle
        gc.collect()
        if args.stage == "bench_b":
            return
    else:
        bench_b = json.loads(bench_b_path.read_text(encoding="utf-8")) if bench_b_path.exists() else {}

    # Combine & verify 0 NaNs / Infs
    # Also compute exact parity check against v7_1 Exp 2 and v7 Step 3
    parity_checks: Dict[str, object] = {}
    exp2_path = V7_1_ART / "ckpt_v7_1_exp2_bhlh_cross_family.json"
    if exp2_path.exists() and "endpoints" in bench_a:
        e2 = json.loads(exp2_path.read_text(encoding="utf-8"))
        bhlh_diffs = {}
        for ep in ("Max", "Mad1", "cMyc", "cMyc_minus_Max", "Mad1_minus_Max", "cMyc_minus_Mad1"):
            if ep in e2.get("primary_gcpbm_endpoints", {}):
                m_e2 = e2["primary_gcpbm_endpoints"][ep]["models"]
                m_a = bench_a["endpoints"][ep]["models"]
                bhlh_diffs[ep] = {
                    "Seq_G2_abs_diff": abs(float(m_a["Seq_G2"]["spearman"]) - float(m_e2["Seq_G2"]["spearman"])),
                    "Seq_Pos4mer_abs_diff": abs(float(m_a["Seq_Pos4mer"]["spearman"]) - float(m_e2["Seq_Pos4mer"]["spearman"])),
                    "Seq_Pos4mer_7ch_abs_diff": abs(float(m_a["Seq_Pos4mer+DNAshapeR_7ch"]["spearman"]) - float(m_e2["Seq_Pos4mer+Shape_7ch"]["spearman"])),
                    "HGBDT_Seq_abs_diff": abs(float(m_a["HGBDT_Seq"]["spearman"]) - float(m_e2["HGBDT_Seq"]["spearman"])),
                }
        parity_checks["bhlh_vs_v7_1_exp2"] = bhlh_diffs

    v7_full_path = ROOT / "v7" / "artifacts" / "v7_gse233827_full_results.json"
    if v7_full_path.exists() and "targets" in bench_b:
        v7f = json.loads(v7_full_path.read_text(encoding="utf-8"))
        s3 = v7f.get("step3_mut_x_dna_benchmark", {})
        hd_diffs = {}
        for tgt, v7_sec in [
            ("dz_f70", s3.get("primary_endpoint_dz_f70", {}).get("regime_b_primary_logo", {})),
            ("dz_spec_f70", s3.get("secondary_endpoint_dz_spec_f70", {}).get("regime_b_primary_logo", {})),
        ]:
            m_v7 = v7_sec.get("models", {})
            m_b = bench_b["targets"][tgt]["models"]
            d_t = {}
            for k_b, k_v7 in [
                ("Mut_x_Seq_G2", "Mut_x_Seq_G2"),
                ("Mut_x_(Seq_G2+DNAshapeR_N1N4)", "Mut_x_(Seq_G2 + Shape_N1N4)"),
                ("Mut_x_Seq_Pos4mer_N1N4", "Mut_x_Seq_Pos4mer_N1N4"),
                ("Mut_x_(Seq_Pos4mer_N1N4+DNAshapeR_N1N4)", "Mut_x_(Seq_Pos4mer_N1N4 + Shape_N1N4)"),
                ("HGBDT_Mut_x_Seq", "HGBDT_Mut_x_Seq"),
                ("HGBDT_Mut_x_(Seq+DNAshapeR_N1N4)", "HGBDT_Mut_x_(Seq + Shape_N1N4)"),
            ]:
                if k_v7 in m_v7 and k_b in m_b:
                    v7_mut = float(m_v7[k_v7]["mut_mean_spearman"])
                    v7_gene = float(m_v7[k_v7]["gene_mean_spearman"])
                    d_t[k_b] = {
                        "exp3_mut_mean": float(m_b[k_b]["mut_mean_spearman"]),
                        "v7_mut_mean": v7_mut,
                        "abs_diff_mut": abs(float(m_b[k_b]["mut_mean_spearman"]) - v7_mut),
                        "exp3_gene_mean": float(m_b[k_b]["gene_mean_spearman"]),
                        "v7_gene_mean": v7_gene,
                        "abs_diff_gene": abs(float(m_b[k_b]["gene_mean_spearman"]) - v7_gene),
                    }
            hd_diffs[tgt] = d_t
        parity_checks["gse233827_vs_v7_step3"] = hd_diffs

    final_obj = {
        "experiment": "v7_1_exp3_authentic_deep_dnashape_control",
        "cpu_only": True,
        "self_checks": self_checks,
        "deep_dnashape_verification": verification,
        "benchmark_a_bhlh_gse59845": bench_a,
        "benchmark_b_homeodomain_gse233827": bench_b,
        "reference_parity_checks": parity_checks,
        "total_runtime_sec": round(float(time.time() - t0), 2),
    }

    ddb.check_finite_recursive(final_obj)
    OUT_JSON.write_text(json.dumps(final_obj, indent=2), encoding="utf-8")
    print(f"[Exp 3] ALL DONE cleanly in {final_obj['total_runtime_sec']:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
