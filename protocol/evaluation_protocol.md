# Evaluation and Statistical Analysis Protocol

## 1. Core Design Principles

1. **Matched Pairwise Comparisons & Receptive-Field Flank Matching**: Every explicit DNA shape evaluation compares `Model(Sequence)` against `Model(Sequence + Shape)` using identical physical flanking sequences matched to the **full receptive field** of the shape predictor (e.g., 12-bp `[F-2..F+2]` / 14-bp `NNNNCANNTGNNNN` for static pentamer `DNAshapeR`; 15-bp `[F-5..F+2]` on homeodomain and 22-bp on bHLH for context-extended *Deep DNAshape* Layer 4), identical protein/mutant encodings, identical training samples, and matched hyperparameter tuning budgets.
2. **Exact Physical Flanking Context (Zero Core Padding)**: For all pentamer-indexed DNAshapeR features (`MGW`, `ProT`, `EP`, `Stretch`, `Roll`, `HelT`, `Slide`), binding sites are restricted to interior placements with complete physical flanking bases (zero `N`-padding across the core). For context-extended *Deep DNAshape* (`Flank15` / `Flank22`), boundary `'N'` padding occurs exclusively outside the 12-bp / 14-bp core and is matched identically across pure-sequence and shape models.
3. **Homology-Grouped & Unseen-Gene Evaluation**:
   - **Homeodomain 92-Mutant Panel (`GSE233827`)**: Primary 30-fold Leave-One-Gene-Out (LOGO) cross-validation summarizing per-mutant Spearman correlations aggregated first by gene ($\bar{\rho}_g$) with gene-clustered paired bootstrap 95% CIs ($B = 5{,}000$) and two-sided gene Wilcoxon tests; secondary 92-fold Leave-One-Mutant-Out (LOMO).
   - **bHLH `gcPBM` (`GSE59845`, `N = 7,289`) & `uPBM` (`GSE97794`, `N = 3,620`)**: Sequence-only zero-leakage `CANNTG` alignment followed by 5-fold `GroupKFold` grouped on RC-canonical central 10-mers (`1,238` gcPBM / `2,043` uPBM groups) and core-cluster paired bootstrap 95% CIs ($B = 2{,}000$).
4. **Complexity-Matched Surrogate Null Controls**:
   - `ShiftGC_ExactCRT_WS_G2`: Zero-hyperparameter conditional randomization preserving exact `W_fix` (`40`d mononucleotide + dinucleotide + GC subspace), $L^2(p_{\mathrm{emp}})$ channel covariance, De Bruijn shift-coboundary and Hamming-order ($r = 3, 4, 5$) Grams, and RC-even symmetry. Required when conditioning on `Seq_Pos4mer`.
   - `Sparse4mer_K` ($K \in \{4, 8, 12, 24, 48\}$): Multi-sparsity 4-mer pocket surrogate family matched on `W_fix` and $7 \times 7$ residual covariance. Calibrated on `Seq_G2` and nonlinear tree learners (`HGBDT`).
   - Exact finite-sample Monte Carlo $p$-values: $p_{\mathrm{mc}} = (1 + \sum_{b=1}^{B} \mathbb{I}[\rho_b \ge \rho_{\mathrm{true}}]) / (B + 1)$ (`B = 49` in `Exp 1a` and bHLH `HGBDT`; formally powered `B = 19`, $p_{\min} = 0.050$ on full-$K=2{,}000$ `Exp 1b` $\Delta z_{\mathrm{spec}}$).
5. **Empirical Experimental Replicate Negative Controls**:
   - **Independent Multi-Chamber Replicates (`GSE233827`)**: 84 balanced WT-vs-WT bipartitions across 8 genes with $\ge 4$ reference chambers.
   - **Within-Array Technical Replicates (`GSE59845`)**: 9 pairwise replicate differences (`r1-r2`, `r1-r3`, `r2-r3` within identical `o1`/`o2` probe orientations) across `Max`, `Mad1`, and `cMyc` (rank $\le 6$ due to linear dependence; reported descriptively alongside multi-chamber benchmarks).
6. **Multiple-Testing Control & Sensitivity Disclosures**:
   - Step-down Holm–Bonferroni adjustment ($p_{\mathrm{Holm}}$) applied across endpoints or strata within each null family.
   - Full disclosure of supervision regimes (*FamilyCode* WT-only vs. mutant-supervised LOGO/LOMO), leave-one-gene-out single-allele outlier leverage (`HOXC10-K271E`), and honest residual boundaries for flank-matched *Deep DNAshape* (`Pos4mer_Flank22` on bHLH; nonlinear `HGBDT_Flank15/22`).
