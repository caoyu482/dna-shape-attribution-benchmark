# DNA shape features add little beyond sequence complexity in predicting transcription factor binding specificity

Code, results and figures for the preprint:

> Cao Y. DNA shape features add little beyond sequence complexity in predicting transcription factor binding specificity. *bioRxiv* (2026).

The bioRxiv link will be added here once the preprint is posted.

## Summary

DNA shape features from pentamer lookup tables (such as DNAshapeR) are often added to sequence models of transcription factor (TF) binding, and the resulting gains are interpreted as evidence of shape readout. Because each table entry is a fixed function of a 5-bp sequence, these gains may instead reflect higher-order sequence information that weak baselines miss.

This repository implements:

- an **exact conditional randomization test (CRT)** that generates surrogate shape tables matched to the real tables in low-order sequence content, cross-channel covariance, variance at each sequence order and reverse-complement symmetry;
- **sparse 4-mer surrogate tables** as a complementary control;
- a ladder of reverse-complement-symmetric sequence models (mononucleotide, dinucleotide, positional 4-mer, degree-3 polynomial kernel and gradient-boosted trees) with and without shape features;
- **flank-matched sequence controls** for context-dependent shape predictors such as Deep DNAshape, whose sequence windows cover the predictor's full context.

Main findings:

1. The widely used Shapelizer control shows fourfold type I error inflation under realistic multichannel nulls, whereas the exact CRT remains calibrated in simulations and on experimental replicates.
2. Positional 4-mer and polynomial kernel models match or exceed the gains from adding shape to dinucleotide models, across homeodomain SELEX-seq differentials, 92 human homeodomain mutants and two bHLH PBM platforms.
3. In gradient-boosted trees, random surrogate tables of matched complexity reproduce all apparent shape gains on bHLH endpoints and 49%–74% of the gains on homeodomain mutants; no mutation class retains a significant residual after correction.
4. Deep DNAshape does not outperform the pentamer tables. Its apparent advantage in linear models comes largely from flanking sequence outside the window given to sequence models; small residual gains remain, mainly in tree models.

## Repository contents

```text
paper/
  figures/                     Figures 1-6 and Supplementary Figure S1 (PNG and PDF)
protocol/
  evaluation_protocol.md       Evaluation and statistical analysis protocol
src/
  null_generators.py           Exact CRT, sparse 4-mer and naive surrogate generators
  bhlh_benchmark.py            Sequence features, kernel, ridge and tree models
  predict.py                   Command-line positional 4-mer ridge predictor
scripts/
  generate_figures.py          Plotting functions (values read from results/)
  relabel_figures.py           Produces the publication figures in paper/figures/
  verify_results.py            Checks result files and figures
results/                       Numerical results (JSON) and reference metadata
analysis/                      Pipeline behind the tree-surrogate and Deep DNAshape results (see analysis/README.md)
tests/
  test_core_invariants.py      Unit tests for surrogate and model invariants
```

Result files of note: `homeodomain_hgbdt_fullk2000_b19.json` (gradient-boosted trees with all training probes and 19 surrogates per family) and `deep_dnashape_comparison.json` (Deep DNAshape versus DNAshapeR and flank-matched sequence models).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Unit tests
python -m unittest discover -s tests -v

# Check result files and figures
python scripts/verify_results.py

# Regenerate Figures 1-6 and S1 from results/
python scripts/relabel_figures.py paper/figures
```

Predicting binding for new sequences with the positional 4-mer ridge model:

```bash
python src/predict.py --train train.csv --eval eval.csv --out preds.csv --seed 42
```

## Data

All datasets are publicly available from NCBI GEO: Exd-Hox SELEX-seq (GSE65073), homeodomain mutant PBMs (GSE233827), bHLH genomic-context PBMs (GSE59845) and bHLH universal PBMs (GSE97794). Pretrained Deep DNAshape models are available from https://github.com/JinsenLi/deepDNAshape. Processed results used in the paper are provided in `results/`.

## Citation

If you use this code, please cite the preprint above.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Yu Cao, Faculty of Dentistry, National University of Singapore (caoyu482@u.nus.edu)
