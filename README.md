# Calibrated Biophysical Attribution and Cross-Family Benchmarking of DNA Shape Features in Transcription Factor Binding

Code, verified results, manuscript source, and publication figures for:

> **Calibrated Biophysical Attribution and Cross-Family Benchmarking of Pentamer DNA Shape Features in Transcription Factor Binding and Variant Specificity Shifts** (2026).

## Summary

Explicit structural DNA shape features—from static pentamer lookup tables (*DNAshapeR*) to context-extended neural predictors (*Deep DNAshape*)—are routinely added to sequence models of transcription factor (TF)–DNA binding, with performance gains interpreted as evidence of biophysical shape readout. Because sequence-indexed shape descriptors are deterministic functions of local or flanking nucleotide sequence, naive `+Shape` comparisons confound genuine structural inductive bias with higher-order $k$-mer basis expansion, outer-flank receptive-field leakage, and continuous split-proxy compression in tree learners.

This repository implements:

- **`ShiftGC_ExactCRT`** (`src/null_generators.py`): a zero-hyperparameter conditional randomization test preserving empirical mononucleotide–dinucleotide–GC subspaces, cross-channel covariance, De Bruijn shift-coboundary and Hamming-order Gram spectra, and exact reverse-complement symmetry;
- **`Sparse4mer`** (`src/null_generators.py`): a multi-sparsity ($K \in \{4, 8, 12, 24, 48\}$) 4-mer pocket surrogate family matched on low-order sequence content and residual covariance;
- **Reverse-complement-symmetric sequence, kernel, and tree benchmarks** (`src/bhlh_benchmark.py`, `src/predict.py`): `Seq_G1`, `Seq_G2`, `Seq_Pos4mer`, `Degree-3 Polynomial Kernel Ridge`, and `HistGradientBoostingRegressor` (`HGBDT`), paired with empirical WT-vs-WT replicate negative controls and receptive-field flank-matched controls (`Flank15`, `Flank22`).

### Key Findings

1. **Null calibration**: *Shapelizer* exhibits $4\times$ Type-I error inflation ($20.0\%$ at $\alpha = 0.05$) under correlated 7-channel genomic dinucleotide nulls and $34.6\%$ under 4-mer epistasis, whereas `ShiftGC_ExactCRT` stays calibrated in simulations ($5.2\%$) and on real multi-chamber homeodomain ($4.8\%$) and within-array bHLH ($0.0\%$) WT-vs-WT replicates.
2. **Pure-sequence saturation**: Upgrading linear/kernel baselines to positional 4-mers (`Seq_Pos4mer`) or polynomial kernels saturates apparent `+Shape` gains across homeodomain SELEX-seq differentials, 92 human homeodomain mutants (`GSE233827`, `0/9` Holm-significant contrasts), and all 11 bHLH gcPBM/uPBM endpoints (`0/11` significant under `ShiftGC_ExactCRT`).
3. **Nonlinear tree proxy decomposition (`B = 49` & full-$K=2{,}000$ `B = 19`)**: Random complexity-matched lookup tables reproduce $100\%$ of apparent `HGBDT + Shape` lifts across all 11/11 bHLH gcPBM/uPBM endpoints (`0/11` significant at $B = 49$) and $64\%\text{--}74\%$ of the full-dataset ($K = 2{,}000$) homeodomain $\Delta z_{\mathrm{spec}}$ lift under formal $\alpha = 0.05$ power ($B = 19$, $p_{\min} = 0.050$: `ShiftGC` $p = 0.050$, $p_{\mathrm{Holm}} = 0.100$; `Sparse4mer_K12` $p = 0.150$ with $2/19$ random draws exceeding real DNAshape; N-arm $p = 0.350\text{--}0.500$; `0/6` Holm-significant strata; `specificity_yes` weakens to $p = 0.100$ upon excluding `HOXC10-K271E`).
4. **Context-extended *Deep DNAshape* & receptive-field flank leakage**: Authentic CPU benchmarking of pre-trained *Deep DNAshape* (Layer 4, 9–10 bp receptive field; L2 vs `DNAshapeR` $r = 0.963/0.977$, exact RC equivariance) shows no gain over static `DNAshapeR` on 14-bp bHLH `Seq_G2` (`-0.0012`) or `HGBDT` (`-0.0034`) and zero linear gain on `GSE233827`. Matching pure-sequence flanks to Deep DNAshape's receptive field (`Pos4mer_Flank15` on homeodomain: `+0.0276/+0.0283` vs `+DeepL4` `-0.0008/+0.0006`; `Pos4mer_Flank22` on bHLH single TFs) exposes outer-flank leakage while bounding modest flank-matched residuals (`+0.0050` on bHLH `Pos4mer_Flank22`; nonlinear `HGBDT_Flank15/22`).

## Repository Contents

```text
paper/
  manuscript.md                              Full manuscript (Markdown)
  main.tex                                   LaTeX manuscript source
  references.bib                             59 verified BibTeX references
  figures/fig1..fig7.{png,pdf,svg}           Publication Figures 1-7 (600-DPI PNG + vector PDF/SVG)
protocol/
  evaluation_protocol.md                     Evaluation, flank-matching, and statistical protocol
src/
  null_generators.py                         ShiftGC_ExactCRT, Sparse4mer, and naive surrogate generators
  bhlh_benchmark.py                          RC-symmetric sequence, polynomial kernel, ridge, and tree modules
  predict.py                                 Strictly inductive CLI positional 4-mer ridge predictor
scripts/
  generate_figures.py                        Regenerates Figures 1-7 directly from results/
  verify_results.py                          Audits all result JSONs, parity invariants, and Figures 1-7
results/
  unified_benchmark_results.json             Consolidated numerical ledger across all benchmarks
  homeodomain_familycode_repro.json          Exact FamilyCode reproduction on GSE233827
  homeodomain_hgbdt_b49.json                 Homeodomain HGBDT matched surrogates (B=49 + sparsity ladder)
  homeodomain_hgbdt_fullk2000_bridge.json    Full-K=2,000 HGBDT surrogate bridge (B=9)
  homeodomain_hgbdt_fullk2000_b19.json       Full-K=2,000 HGBDT formal alpha=0.05 extension (B=19)
  bhlh_cross_family_results.json             bHLH GSE59845 gcPBM, replicate nulls, and GSE97794 uPBM
  hoxc10_and_upbm_surrogates.json            HOXC10 leave-one-gene-out and uPBM HGBDT surrogates
  deep_dnashape_comparison.json              Authentic Deep DNAshape comparative & flank-leakage benchmark
  verified_references.json                   DOI/PMID-verified reference metadata (59 entries)
tests/
  test_core_invariants.py                    9-test self-contained unit test suite
```

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run all 9 core unit tests (RC symmetry, ExactCRT, Sparse4mer, Woodbury/Schur, Holm, AST hygiene)
python -m unittest discover -s tests -v

# 2. Verify numerical integrity of all result JSONs and Figures 1-7
python scripts/verify_results.py

# 3. Regenerate all 7 publication figures (PNG, PDF, SVG) from results/
python scripts/generate_figures.py
```

Predicting binding for new sequences with the strictly inductive positional 4-mer ridge CLI:

```bash
python src/predict.py --train train.csv --eval eval.csv --out preds.csv --seed 42
```

## Data Availability

All benchmark datasets are publicly available from NCBI GEO: Exd–Hox SELEX-seq (`GSE65073`), 92-mutant human homeodomain universal PBMs (`GSE233827`), bHLH genomic-context PBMs (`GSE59845`), and bHLH universal PBMs (`GSE97794`), along with official pre-trained *Deep DNAshape* checkpoints (`JinsenLi/deepDNAshape`). Verified numerical outputs used to generate all tables and figures are archived in `results/`.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Yu Cao, Faculty of Dentistry, National University of Singapore (`caoyu482@u.nus.edu`)
