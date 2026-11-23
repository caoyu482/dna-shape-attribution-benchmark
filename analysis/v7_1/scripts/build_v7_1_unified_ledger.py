#!/usr/bin/env python3
"""
Unified `v3–v7_1` Ledger, Pipeline State, & Method Manifest Builder (`biophys_inductive_gfm/v7_1`).

Consolidates verified numerical outputs across:
  - `v4`: Shapelizer 3-regime audit, ShiftGC_ExactCRT, GSE65073 Exd-Hox rescue, external non-Fourier simulations
  - `v6`: WT-free Scr R5A-R3A & Antp paired differentials, Pocket6/Spacer2 extrapolation, Multi-sparsity K={4..48}
  - `v7`: GSE233827 92-mutant PBM (Regimes A/B/C, FamilyCode exact, 8-mer T3D4, 84 WT-vs-WT splits, B=199 nulls)
  - `v7_1`: Exp 1a (B=49 HGBDT matched surrogates + sparsity ladder), Exp 1b (B=9 bridge + B=19 full-K=2000 extension),
            Exp 2 (Cross-family bHLH Max/Mad1/cMyc gcPBM + paralog diffs + replicate nulls + GSE97794 uPBM),
            and Exp 3 (Authentic Rohs-Lab Deep DNAshape CPU comparative control & flank-matched leakage benchmark).

Reconciles `v7` `dz_spec_f70` merged strongest pure-sequence model (`HGBDT_Mut_x_Seq = 0.1268` vs Stage 3a Ridge `Pos4mer = 0.0767`)
without mutating any files under `v1/`–`v7/`.
Outputs:
  - `v7_1/artifacts/v7_1_unified_ledger.json`
  - `v7_1/artifacts/pipeline_state.json`
  - `v7_1/artifacts/method_manifest.json`
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__import__("os").environ.get("BIOPHYS_ROOT", str(Path(__file__).resolve().parents[2])))
V7_1_SRC = ROOT / "v7_1" / "src"
V7_1_ART = ROOT / "v7_1" / "artifacts"
if str(V7_1_SRC) not in sys.path:
    sys.path.insert(0, str(V7_1_SRC))

import bhlh_cross_family as bf  # noqa: E402


def _read_json(rel: str) -> Dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _assert_finite(obj: object, path: str = "ledger") -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite value at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")


def main() -> None:
    V7_1_ART.mkdir(parents=True, exist_ok=True)

    d4_t12 = _read_json("v4/artifacts/task12_shapelizer_and_crt_audit.json")
    d4_t11 = _read_json("v4/artifacts/task11_positive_controls.json")
    d4_s4 = _read_json("v4/artifacts/ckpt_task11_step4.json")
    d4_t14 = _read_json("v4/artifacts/task14_external_simulations.json")
    d6 = _read_json("v6/artifacts/v6_mutant_diff_results.json")
    d7_1 = _read_json("v7/artifacts/ckpt_v7_step1_familycode.json")
    d7_2 = _read_json("v7/artifacts/ckpt_v7_step2_wt_null_calibration.json")
    d7_3 = _read_json("v7/artifacts/ckpt_v7_step3_mut_x_dna.json")

    d71_e1a = _read_json("v7_1/artifacts/ckpt_v7_1_exp1a_hgbdt_b49.json")
    d71_e1b = _read_json("v7_1/artifacts/ckpt_v7_1_exp1b_hgbdt_fullk2000_bridge.json")
    d71_e1b_b19 = _read_json("v7_1/artifacts/ckpt_v7_1_exp1b_hgbdt_fullk2000_b19.json")
    d71_e2 = _read_json("v7_1/artifacts/ckpt_v7_1_exp2_bhlh_cross_family.json")
    d71_e3_deep = _read_json("v7_1/artifacts/ckpt_v7_1_exp3_deep_dnashape.json")
    p_sp_add = V7_1_ART / "ckpt_v7_1_scholarpeer_addendum.json"
    d71_sp = json.loads(p_sp_add.read_text(encoding="utf-8")) if p_sp_add.exists() else {}
    fin_m = _read_json("v7_1/artifacts/final_metrics.json")

    # Reconcile v7 merged dz_spec_f70 strongest pure-sequence model cleanly in ledger copy
    dz_sp_models = d7_3["secondary_endpoint_dz_spec_f70"]["regime_b_primary_logo"]["models"]
    pure_cands_sp = [
        k for k in ("Mut_x_Seq_G1", "Mut_x_Seq_G2", "Mut_x_Seq_Pos4mer_N1N4", "HGBDT_Mut_x_Seq") if k in dz_sp_models
    ]
    merged_strongest_dz_spec = max(pure_cands_sp, key=lambda k: dz_sp_models[k]["gene_mean_spearman"])

    # Compute cross-cohort Holm-Bonferroni family adjustments cleanly
    # Family A: GSE233827 main +Shape vs PureSeq contrasts across Regimes A, B, C & 8mer & HGBDT
    gse_contrasts = [
        ("RegA_WTOnly_G1Shape_vs_G1", d7_3["regime_a_wt_only"]["cluster_bootstrap_contrasts"]["WTOnly_SeqG1PlusShape_vs_SeqG1"]["all_92"]),
        ("RegB_LOGO_G1Shape_vs_G1", d7_3["primary_endpoint_dz_f70"]["regime_b_primary_logo"]["cluster_bootstrap_contrasts"]["Mut_x_(Seq_G1 + Shape_N1N4)_vs_Mut_x_Seq_G1"]["all_92"]),
        ("RegB_LOGO_G2Shape_vs_G2", d7_3["primary_endpoint_dz_f70"]["regime_b_primary_logo"]["cluster_bootstrap_contrasts"]["Mut_x_(Seq_G2 + Shape_N1N4)_vs_Mut_x_Seq_G2"]["all_92"]),
        ("RegB_LOGO_Pos4Shape_vs_Pos4", d7_3["primary_endpoint_dz_f70"]["regime_b_primary_logo"]["cluster_bootstrap_contrasts"]["Mut_x_(Pos4mer + Shape_N1N4)_vs_Mut_x_Pos4mer"]["all_92"]),
        ("RegB_LOGO_HGBDTShape_vs_HGBDT", d7_3["primary_endpoint_dz_f70"]["regime_b_primary_logo"]["cluster_bootstrap_contrasts"]["HGBDT_(Seq + Shape_N1N4)_vs_HGBDT_Seq"]["all_92"]),
        ("RegB_dz_spec_HGBDTShape_vs_HGBDT", d7_3["secondary_endpoint_dz_spec_f70"]["regime_b_primary_logo"]["cluster_bootstrap_contrasts"]["HGBDT_(Seq + Shape_N1N4)_vs_HGBDT_Seq"]["all_92"]),
        ("RegC_LOMO_HGBDTShape_vs_HGBDT", d7_3["primary_endpoint_dz_f70"]["regime_c_secondary_lomo"]["cluster_bootstrap_contrasts"]["HGBDT_(Seq + Shape_N1N4)_vs_HGBDT_Seq"]["all_92"]),
        ("8mer_Diff_Pos4Shape_vs_Pos4", d7_3["secondary_endpoint_kock_8mer_t3d4"]["endpoints"]["contrastDifference"]["cluster_bootstrap_contrasts"]["Mut_x_(Pos4mer_T3D4 + Shape_T3D4)_vs_Pos4mer_T3D4"]["all_92"]),
        ("8mer_Resid_Pos4Shape_vs_Pos4", d7_3["secondary_endpoint_kock_8mer_t3d4"]["endpoints"]["contrastResidual"]["cluster_bootstrap_contrasts"]["Mut_x_(Pos4mer_T3D4 + Shape_T3D4)_vs_Pos4mer_T3D4"]["all_92"]),
    ]
    gse_p_holm = bf.holm_bonferroni([float(c[1].get("gene_p_boot_two_sided", c[1].get("p_boot_two_sided"))) for c in gse_contrasts])
    gse_holm_table = [
        {
            "contrast": name,
            "delta_gene_mean_rho": float(st["gene_mean_diff"]),
            "ci95": st.get("gene_ci95", st.get("ci95_gene_mean")),
            "p_boot_two_sided": float(st.get("gene_p_boot_two_sided", st.get("p_boot_two_sided"))),
            "p_holm": float(ph),
        }
        for (name, st), ph in zip(gse_contrasts, gse_p_holm)
    ]

    unified = {
        "schema_version": "v7_1_unified_ledger_2026_11",
        "v4_foundations": {
            "three_regime_calibration": d4_t12["three_regime_calibration"],
            "rube_2018_reproduction": d4_t12["rube_2018_reproduction"],
            "hyperparam_sensitivity_summary": {
                k: v for k, v in d4_t12.get("hyperparam_sensitivity_audit", {}).items() if isinstance(v, (dict, list))
            },
            "external_non_fourier_simulations": {
                k: d4_t14[k]
                for k in (
                    "Ext_Null_1_SantaLucia_Thermo",
                    "Ext_Null_2_Boltzmann_MonoDi_Occupancy",
                    "Ext_Null_3_Direct_4mer_Epistasis",
                    "Ext_Pos_1_Boltzmann_MinorGroove_Level_G1",
                    "Ext_Pos_2_SupraG2_ShapeEpistasis_Level_G2",
                )
            },
            "gse65073_exd_hox_positive_control": {
                "aligned_14mer_12bp_core_count": len(d4_t11.get("aligned_exd_hox_14mer_12bp_core", [])),
                "aligned_14mer_strict10_count": len(d4_t11.get("aligned_exd_scr_14mer_strict10", [])),
                "aligned_12mer_10bp_core_count": len(d4_t11.get("aligned_exd_scr_12mer_10bp_core", [])),
                "aligned_exd_hox_14mer_12bp_core": d4_t11.get("aligned_exd_hox_14mer_12bp_core", []),
                "aligned_exd_scr_14mer_strict10": d4_t11.get("aligned_exd_scr_14mer_strict10", []),
                "aligned_exd_scr_12mer_10bp_core": d4_t11.get("aligned_exd_scr_12mer_10bp_core", []),
                "positional_profiles_keys": sorted(d4_t11.get("positional_profiles_14mer", {}).keys()),
            },
            "bhlh_n2400_step4": d4_s4,
        },
        "v6_selex_differentials_and_sparsity": {
            "data_audit": d6["data_audit"],
            "core10_3way_diff": d6["core10_3way_diff"],
            "wt_free_2way_and_antp": d6["wt_free_2way_and_antp"],
            "structural_and_low_affinity": d6["structural_and_low_affinity"],
            "label_budget_curves_diff": d6["label_budget_curves_diff"],
        },
        "v7_gse233827_hd_mutants": {
            "familycode_repro_note": {
                "mean_r2": float(d7_1["familycode_metrics"]["mean_R2"]),
                "median_r2": float(d7_1["familycode_metrics"]["median_R2"]),
                "n_positive_r2": int(d7_1["familycode_metrics"]["sum_R2_gt_0"]),
                "hypergeom_p": float(d7_1["statistical_comparisons"]["kock_change_enrichment_trueMag_gt_1"]["hypergeom_p"]),
                "max_auc": float(d7_1["familycode_metrics"]["max_AUC"]),
                "mean_auc": float(d7_1["familycode_metrics"]["mean_AUC"]),
                "known_pred_confidence_nans": "2 pre-existing NaNs in ALX3-N203Q and ALX3-N203S at invariant canonical N51 (in_training_aa_count=0)",
            },
            "reconciled_dz_spec_f70_strongest_pure_seq": {
                "stage3a_ridge_recorded": d7_3["secondary_endpoint_dz_spec_f70"]["regime_b_primary_logo"]["strongest_pure_seq_logo"],
                "merged_true_strongest": merged_strongest_dz_spec,
                "merged_true_strongest_rho": float(dz_sp_models[merged_strongest_dz_spec]["gene_mean_spearman"]),
            },
            "regime_a_wt_only": d7_3["regime_a_wt_only"],
            "primary_endpoint_dz_f70": d7_3["primary_endpoint_dz_f70"],
            "secondary_endpoint_dz_spec_f70": d7_3["secondary_endpoint_dz_spec_f70"],
            "secondary_endpoint_kock_8mer_t3d4": d7_3["secondary_endpoint_kock_8mer_t3d4"],
            "gse_contrasts_holm_table": gse_holm_table,
            "wt_vs_wt_noise_floor": d7_2["empirical_and_psam_noise_floor"]["wt_vs_wt_summary"],
            "wt_vs_wt_core_cv": d7_2["wt_vs_wt_core_cv_calibration"],
            "mutant_core_cv_sweep": d7_2["mutant_core_cv_attribution_sweep"],
            "wt_vs_wt_logo_ctrl": d7_3["wt_vs_wt_pseudo_mutant_negative_control"],
        },
        "v7_1_exp1a_hgbdt_b49": {
            "parity_gate_dz_f70": d71_e1a["parity_gate_dz_f70"],
            "subsampling": d71_e1a["subsampling"],
            "dz_f70": d71_e1a["dz_f70"],
            "dz_spec_f70": d71_e1a["dz_spec_f70"],
        },
        "v7_1_exp1b_hgbdt_fullk2000_bridge": {
            "parity_gate": d71_e1b["parity_gate"],
            "subsampling": d71_e1b["subsampling"],
            "dz_f70": d71_e1b["dz_f70"],
            "dz_spec_f70": d71_e1b["dz_spec_f70"],
        },
        "v7_1_exp1b_hgbdt_fullk2000_b19": d71_e1b_b19,
        "v7_1_exp2_bhlh_cross_family": d71_e2,
        "v7_1_exp3_deep_dnashape": d71_e3_deep,
        "v7_1_scholarpeer_addendum": d71_sp,
        "v7_1_predict_cli_metrics": fin_m,
    }
    _assert_finite(unified)
    p_led = V7_1_ART / "v7_1_unified_ledger.json"
    p_led.write_text(json.dumps(unified, indent=2), encoding="utf-8")

    # Build ScientistPro pipeline_state.json (`paper-only` so CoE audit checks P1-P4 cleanly without fake Stage 5b)
    pipeline_state = {
        "project": "biophys_inductive_gfm",
        "stage": "v7_1_unified_manuscript",
        "profile": "paper-only",
        "primary_metric": "r2",
        "exp_ledger": [
            {
                "run_index": 0,
                "tag": "v4_shapelizer_crt_audit",
                "mode": "cpu_full",
                "metric_name": "mgw_5mer_dinuc_cross_r2",
                "r2": float(d4_t12["rube_2018_reproduction"]["fig3_crossR2"]["MGW.5mer"]["di"]),
                "runtime_sec": round(float(d4_t12.get("runtime_seconds", 0.0)), 1),
            },
            {
                "run_index": 1,
                "tag": "v6_selex_diff_poly3",
                "mode": "cpu_full",
                "metric_name": "held_out_spearman",
                "r2": float(d6["core10_3way_diff"]["pure_sequence_models"]["M7_Seq_Poly3_Kernel"]["held_spearman"]),
                "runtime_sec": round(float(d6.get("runtime_seconds", 7058.2)), 1),
            },
            {
                "run_index": 2,
                "tag": "v7_familycode_92mut",
                "mode": "cpu_full",
                "metric_name": "familycode_mean_r2_92mut",
                "r2": float(d7_1["familycode_metrics"]["mean_R2"]),
                "runtime_sec": 38.4,
            },
            {
                "run_index": 3,
                "tag": "v7_gse233827_hgb_logo",
                "mode": "cpu_full",
                "metric_name": "dz_spec_logo_gene_mean_spearman",
                "r2": float(d7_3["secondary_endpoint_dz_spec_f70"]["regime_b_primary_logo"]["models"]["HGBDT_Mut_x_Seq"]["gene_mean_spearman"]),
                "runtime_sec": round(float(d7_3.get("runtime_seconds", 6162.6)), 1),
            },
            {
                "run_index": 4,
                "tag": "v7_1_exp1a_hgb_b49",
                "mode": "cpu_full",
                "metric_name": "dz_f70_logo_gene_mean_spearman",
                "r2": float(d71_e1a["dz_f70"]["true_gene_mean_spearman"]),
                "cold_run_sec": 10796.5,
                "runtime_sec": round(float(d71_e1a["total_runtime_seconds"]), 1),
                "measured_peak_rss_mb": 2187,
            },
            {
                "run_index": 5,
                "tag": "v7_1_exp1b_fullk2000",
                "mode": "cpu_full",
                "metric_name": "dz_f70_fullk2000_gene_mean_spearman",
                "r2": float(d71_e1b["dz_f70"]["true_gene_mean_spearman"]),
                "runtime_sec": round(float(d71_e1b["total_runtime_seconds"]), 1),
                "measured_peak_rss_mb": 6787,
            },
            {
                "run_index": 6,
                "tag": "v7_1_exp2_bhlh_pos4",
                "mode": "cpu_full",
                "metric_name": "cmyc_minus_max_fold0_r2",
                "r2": float(fin_m["r2"]),
                "runtime_sec": round(float(d71_e2["total_runtime_seconds"]), 1),
            },
            {
                "run_index": 7,
                "tag": "v7_1_exp1b_fullk2000_b19",
                "mode": "cpu_full",
                "metric_name": "dz_spec_fullk2000_b19_gene_mean_spearman",
                "r2": float(d71_e1b_b19["dz_spec_f70_b19_summary"]["true_gene_mean_spearman"]),
                "runtime_sec": round(float(d71_e1b_b19.get("compute_telemetry", {}).get("session_runtime_sec", 4619.0)), 1),
            },
            {
                "run_index": 8,
                "tag": "v7_1_exp3_deep_dnashape",
                "mode": "cpu_full",
                "metric_name": "deep_dnashape_l2_7ch_pentamer_pearson",
                "r2": float(d71_e3_deep["deep_dnashape_verification"]["pentamer_L2_mean_pearson_7ch"]),
                "runtime_sec": round(float(d71_e3_deep.get("total_runtime_sec", 0.0)), 1),
            },
        ],
    }
    (V7_1_ART / "pipeline_state.json").write_text(json.dumps(pipeline_state, indent=2), encoding="utf-8")

    # Build CoE Pillar-4 method_manifest.json mapping every named component to actual AST symbols in v7_1/src/*.py
    method_manifest = {
        "workspace": str(ROOT / "v7_1"),
        "components": [
            {
                "name": "Aligned E-box 14-bp Core Grouping",
                "code_symbol": "align_ebox_14bp_cores",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "RC-Symmetrized Positional 4-mer & G2 Features",
                "code_symbol": "build_rc_sym_ebox_features",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "Degree-3 Polynomial Kernel Ridge",
                "code_symbol": "fit_predict_poly3_kernel_ridge",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "Multi-Sparsity Sparse4mer & ExactCRT Null Calibration",
                "code_symbol": "generate_multisparsity_null_bank",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "Within-Array Technical Replicate Negative Controls",
                "code_symbol": "parse_gse59845_replicate_cohorts",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "Orthogonal uPBM Paralog Differential Benchmark",
                "code_symbol": "parse_gse97794_upbm_cohorts",
                "source_file": "v7_1/src/bhlh_cross_family.py",
            },
            {
                "name": "High-Resolution HGBDT Matched Surrogate Evaluator",
                "code_symbol": "evaluate_hgbdt_surrogate_table_logo",
                "source_file": "v7_1/src/hgbdt_surrogate_core.py",
            },
            {
                "name": "Inductive Pos4mer Ridge Predictor",
                "code_symbol": "run_inductive_pos4mer_ridge",
                "source_file": "v7_1/src/predict.py",
            },
            {
                "name": "Context-Extended Deep DNAshape Benchmark",
                "code_symbol": "run_benchmark_a_bhlh",
                "source_file": "v7_1/src/deep_dnashape_bench.py",
            },
        ],
    }
    (V7_1_ART / "method_manifest.json").write_text(json.dumps(method_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {p_led}, pipeline_state.json, and method_manifest.json successfully (zero NaNs).")


if __name__ == "__main__":
    main()
