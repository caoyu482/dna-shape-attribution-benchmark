"""Regenerate the publication figures (Figure1-6 and FigureS1) from results/.

Usage: python scripts/relabel_figures.py [output_dir]   (default: paper/figures)
"""
import re,importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('g','scripts/generate_figures.py'); g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
import matplotlib.pyplot as plt
OUT=Path(sys.argv[1] if len(sys.argv)>1 else "paper/figures"); OUT.mkdir(parents=True,exist_ok=True)
STEM={'fig1_calibration_and_simulations':'Figure1','fig2_selex_and_label_efficiency':'Figure2','fig3_gse233827_hd_mutants_and_familycode':'Figure3',
'fig4_hgbdt_surrogate_decomposition':'Figure4','fig5_bhlh_cross_family_gcpbm_upbm':'Figure5','fig6_unified_attribution_framework':'Figure6','fig7_deep_dnashape_comparison':'FigureS1'}
EX={
# fig1
'Empirical Rejection Rate (FPR at α = 0.05)':'Type I error (α = 0.05)','Rejection Rate (FPR / Power at α = 0.05)':'Rejection rate (α = 0.05)',
'Empirical Null FPR at α = 0.05 (on Pos4mer)':'False-positive rate (α = 0.05)',
'Regime I\n1-ch Aligned\nG1 Null':'1 channel,\naligned','Regime II\n1-ch GC 50bp\nNull':'1 channel,\nGC-stratified','Regime III\n7-ch Enhancer\nG2 Null':'7 channels,\ngenomic',
'SantaLucia\nThermo\n(Null)':'Thermo-\ndynamic','Boltzmann\nMono+Di\n(Null)':'Boltzmann','Direct 4-mer\nEpistasis\n(Null)':'4-mer\nepistasis',
'Minor-Groove\nLevel G1\n(Positive)':'Minor\ngroove','Supra-G2\nShape Epist.\n(Positive)':'Shape\nepistasis',
'Null Regimes (FPR)':'No shape effect','Positive (Power)':'Shape effect',
'Naive\nRC Perm':'Naive','Naive RC Perm':'Naive permutation','Shapelizer\n(2018)':'Shapelizer','Shapelizer (2018)':'Shapelizer',
'ShiftGC\nExactCRT':'Exact\nCRT','ShiftGC_ExactCRT':'Exact CRT','Sp4\nK=12':'Sparse\nK = 12','Sp4\nK=4':'Sparse\nK = 4','Sp4\nK=48':'Sparse\nK = 48',
'Nominal α = 0.05':'α = 0.05','HD Multi-Chamber Δz (84 splits)':'Homeodomain, Δz','HD Multi-Chamber Δz_spec (84 splits)':'Homeodomain, Δz$_{spec}$',
'bHLH Within-Array 7ch (9 cohorts)':'bHLH, 7 channels','bHLH Within-Array MGW (9 cohorts)':'bHLH, MGW only',
# fig2
'+DNAshape (7ch)':'+ Shape','Pure Sequence':'Sequence only','Best Pure + Shape':'Best sequence + shape','Best Pure Seq (Poly3/HGB)':'Best sequence model',
'Full Pos\n4/5-mer':'Positional\n4/5-mer','G2\nAllPairs':'All-pair\ndinuc.','G2 AllPairs (Pure)':'All-pair dinucleotide','HGBDT':'GBT',
'Held-Out Core-Grouped Spearman ρ':'Held-out Spearman ρ','Held-Out Spearman ρ (Mean ± 0.5 SD)':'Held-out Spearman ρ','Out-of-Distribution Spearman ρ':'Spearman ρ',
'Poly3\nKernel':'Poly-\nnomial','Pos4mer\n(N5-N6)':'Positional\n4-mer','Pos4mer (N5-N6)':'Positional 4-mer','Seq_G1':'Mono-\nnuc.','Seq_G2':'Dinucleotide',
'Seq_G2 + Shape':'Dinucleotide + shape','ShiftGC Surrogate Med.':'Exact CRT surrogates (median)','Training Cores per SELEX Library (Log Scale)':'Training cores',
'Unseen\nPocket6\n5-Fold':'Unseen\npocket','Unseen\nSpacer2\n5-Fold':'Unseen\nspacer','WT-Free\nScr 2-Way\n(N=4,799)':'Two-way\ndifferential',
# fig3
'30-Fold LOGO Gene-Mean Spearman ρ̄_g':'Gene-averaged Spearman ρ','8-mer Diff (T3D4): Pos4+Sh':'8-mer difference, 4-mer','8-mer Resid (T3D4): Pos4+Sh':'8-mer residual, 4-mer',
'All 92 Mutants':'All 92 mutants','FamilyCode\n(WT-Only)':'Recognition\ncode','FamilyCode Native ΔΔΔG/RT  R²':'R²','HGBDT\n(K=2k)':'GBT',
'Homeodomain 8-bp Core Position':'Core position','Mut×G1':'Mono-\nnuc.','Mut×G2':'Dinuc.','Mut×Pos4':'Positional\n4-mer','N1–N4 Minor-Groove / N-arm':'Positions N1–N4',
'Paired Δρ̄_g (+Shape − Pure Seq, 95% CI)':'Difference in ρ (shape − sequence)','Reg.A WT-Only: G1+Sh vs G1':'WT-only, mononucleotide',
'Reg.B (Δz_spec): HGB+Sh vs HGB':'GBT, Δz$_{spec}$','Reg.B LOGO (Δz): G1+Sh vs G1':'Mononucleotide, Δz','Reg.B LOGO (Δz): G2+Sh vs G2':'Dinucleotide, Δz',
'Reg.B LOGO (Δz): HGB+Sh vs HGB':'GBT, Δz','Reg.B LOGO (Δz): Pos4+Sh vs Pos4':'Positional 4-mer, Δz','Reg.C LOMO (Δz): HGB+Sh vs HGB':'GBT, Δz (LOMO)',
'|ΔΔΔG/RT| > 1 Subset':'Large-effect subset','Δz_f70 + Shape (N1–N4)':'Δz, + shape','Δz_f70 Pure Seq':'Δz, sequence','Δz_spec + Shape (N1–N4)':'Δz$_{spec}$, + shape','Δz_spec Pure Seq':'Δz$_{spec}$, sequence',
# fig4
'Affinity\nOnly':'Affinity\nonly','Exp 1a\nΔz_f70\n(B=49)':'Δz\n500','Exp 1a\nΔz_spec\n(B=49)':'Δz$_{spec}$\n500','Exp 1b\nΔz_f70\n(Full, B=9)':'Δz\n2,000','Exp 1b\nΔz_spec\n(Full, B=9)':'Δz$_{spec}$\n2,000',
'F_true (< Null Median)':'Real shape (below surrogate median)','F_true DNAshape (0.0860)':'Real shape','HGBDT Pure Sequence':'Sequence only','HGBDT Seq_Only (0.0859)':'Sequence only',
'N-arm\n(1–9)':'N-terminal\narm','No\nChange':'No\nchange','Null 95th Pct.':'Surrogate 95th percentile','Null Median':'Surrogate median','Random Surrogate Proxy Lift':'Gain from surrogates',
'Residual Above Surrogate Med.':'Residual gain','Seq_Only (Exp 1a)':'Sequence only','ShiftGC\n(B=49)':'Exact\nCRT','ShiftGC Med. (+q95)':'Exact CRT surrogates',
'Sp4 K=12\n(B=49)':'K = 12','Sp4 K=24\n(B=19)':'K = 24','Sp4 K=4\n(B=19)':'K = 4','Sp4 K=48\n(B=19)':'K = 48','Sp4 K=8\n(B=19)':'K = 8','Spec.\nYes':'Specificity\naltering',
'Stratum Mutant-Mean Spearman (Δz_spec)':'Spearman ρ (Δz$_{spec}$)','True DNAshape (Exp 1a)':'Real shape','p=0.56\n(null > true)':'P = 0.56',
# fig5
'5-Fold Core-Grouped Spearman ρ (N = 7,289)':'Spearman ρ','5-Fold GroupKFold Spearman ρ (HGBDT)':'Spearman ρ','HGBDT + True Shape':'GBT + real shape','HGBDT Seq_Only':'GBT, sequence only',
'On Seq_G2':'Dinucleotide baseline','On Seq_Pos4mer (0/11 sig.)':'Positional 4-mer baseline','Seq_G2 + Shape (7ch)':'Dinucleotide + shape','Seq_Pos4mer (Pure)':'Positional 4-mer',
'Seq_Pos4mer + Shape (7ch)':'Positional 4-mer + shape','ShiftGC Med. (Range)':'Exact CRT surrogates','ShiftGC_ExactCRT Monte Carlo p-value (B = 199)':'Exact CRT P value',
'Sparse4mer_K12 Med.':'Sparse 4-mer surrogates','α = 0.05 threshold':'P = 0.05','cMyc':'c-Myc','cMyc (gc)':'c-Myc (gc)','cMyc−\nMad1':'c-Myc−\nMad1','cMyc−\nMax':'c-Myc−\nMax',
'cMyc−Mad1 (gc)':'c-Myc−Mad1 (gc)','cMyc−Max (gc)':'c-Myc−Max (gc)','Max (uPBM)':'Max (u)','MadMax (u)':'Mad–Max (u)','MycMax (u)':'Myc–Max (u)',
# fig6
'HD PBM (92 Mut): Δz_f70 LOGO':'Homeodomain mutants, Δz','HD PBM (92 Mut): Δz_spec LOGO':'Homeodomain mutants, Δz$_{spec}$','HD SELEX: Scr R5A–R3A (3-Way)':'Homeodomain SELEX',
'HGBDT Surrogate\nResidual (True−Null)':'GBT residual\nafter surrogates','Mean Spearman Gain Δρ (+Shape)':'Gain in Spearman ρ from shape','N/A\n(Linear/K)':'NA',
'Nonlinear HGBDT\nNaive +Shape':'GBT\n+ shape','Strong Seq (Pos4/Poly3)\nNaive +Shape':'Strong sequence\n+ shape','Weak Seq (G2)\nNaive +Shape':'Dinucleotide\n+ shape',
'bHLH gcPBM: Max / Mad1 / cMyc (Mean)':'bHLH gcPBM, single TFs','bHLH gcPBM: Paralog Diffs (Mean)':'bHLH gcPBM, paralog differences','bHLH uPBM: GSE97794 (5-Ep. Mean)':'bHLH uPBM',
'+0.0054\n(1a:+0.010)':'+0.0054',
'Step 1: Exact Flanks & Homology-Grouped CV':'1. True flanks and grouped cross-validation',
'Step 2: Strong Pure-Sequence Saturation Ladder':'2. Ladder of strong sequence baselines',
'Step 3: Matched-Complexity Surrogate Controls':'3. Complexity-matched surrogate tables',
'Step 4: Empirical Replicate Nulls & Holm Control':'4. Replicate controls and multiple-testing correction',
'Extract true physical flanks (zero padding); enforce central-core\nGroupKFold or Leave-One-Gene-Out (LOGO) & RC symmetry.':'Use true flanking bases; group folds by homology or gene;\nenforce reverse-complement symmetry.',
'Benchmark against Seq_G2, Pos4mer, Poly3, and HGBDT_Seq with\nidentical flanks & tuning budgets; report paired clustered 95% CIs.':'Compare against dinucleotide, positional 4-mer, polynomial\nkernel and GBT models with identical inputs and tuning.',
'Test F_true against ShiftGC_ExactCRT (exact 5-mer variance, valid on\nPos4mer) & Sparse4mer K∈{4..48} (on G2 & HGBDT) to strip proxy lifts.':'Test the real table against exact CRT and sparse 4-mer\nsurrogates matched in sequence complexity.',
'Audit FPR@0.05 on real WT-vs-WT / technical replicate diffs and\nreport Holm–Bonferroni p_Holm alongside stratum localization.':'Check false-positive rates on WT-versus-WT replicates;\napply Holm correction across endpoints.',
}

# --- v2 additions (B = 19 extension, Deep DNAshape) ---
EX.update({
'Exp 1b\nΔz_f70\n(2k, B=9)':'Δz\n2,000','Exp 1b\nΔz_spec\nCRT B=19':'Δz$_{spec}$\n2,000\nexact CRT','Exp 1b\nΔz_spec\nK12 B=19':'Δz$_{spec}$\n2,000\nK = 12',
'p=0.56\n(null>true)':'P = 0.56','p=0.05\np_H=0.10\n(64% proxy)':'P = 0.05\n64%','p=0.15\n2/19>true\n(74% proxy)':'P = 0.15\n74%',
'Gene-Balanced Stratum Spearman ρ̄_g (Δz_spec)':'Gene-averaged Spearman ρ (Δz$_{spec}$)',
'CRT:0.35 | K12:0.50\n(p_H6=0.80)':'P = 0.35 / 0.50','CRT:0.05 | K12:0.05\n(p_H6=0.30)':'P = 0.05 / 0.05','CRT:0.50 | K12:0.50\n(p_H6=0.80)':'P = 0.50 / 0.50','CRT:0.25 | K12:0.15\n(p_H6=0.80)':'P = 0.25 / 0.15',
'Seq_Only (Full K=2k)':'Sequence only','ShiftGC B=19 Med. (+q95)':'Exact CRT surrogates','True DNAshape (Full K=2k)':'Real shape',
'+0.0058\n(K12:+0.004)':'+0.0058',
'Step 1: Receptive-Field Flank Matching & Grouped CV':'1. Matched flanks and grouped cross-validation',
'Match physical flanks to the full receptive field of any shape model\n(e.g., 15–22 bp for Deep DNAshape); use GroupKFold/LOGO & RC symmetry.':'Give sequence models the same flanking bases the shape model sees;\ngroup folds by homology or gene; enforce reverse-complement symmetry.',
'Step 2: Flank-Matched Pure-Sequence Saturation Ladder':'2. Ladder of strong sequence baselines',
'Benchmark Seq_G2, Pos4mer, Poly3, and HGBDT_Seq on identical matched\nflanks & tuning budgets to expose flank leakage; report clustered 95% CIs.':'Compare against dinucleotide, positional 4-mer, polynomial\nkernel and GBT models with identical inputs and tuning.',
# Figure S1 (Deep DNAshape)
'Pentamer r vs DNAshapeR':'Correlation with DNAshapeR (r)','L2 7ch mean r=0.963':'Mean, 7 channels','L2 core-6ch r=0.977':'Mean, 6 channels',
'Deep L2 (5-bp RF)':'Deep DNAshape, 5-bp context','Deep L4 (9–10 bp RF)':'Deep DNAshape, 9–10-bp context',
'6-Ep Δρ (DeepL4−ShapeR)':'Δρ, Deep DNAshape − DNAshapeR','Seq_G2\n(14-bp)':'Dinucleotide','HGBDT\n(14-bp)':'GBT','Pos4mer\n(14-bp*)':'Positional\n4-mer',
'-0.0012\n(0/6 Holm)':'−0.0012','-0.0034\n(0/6 Holm)':'−0.0034','+0.0047\n(5/6 Holm*)':'+0.0047',
'bHLH 5-Fold ρ':'Spearman ρ','Pos4mer (14-bp)':'4-mer, 14 bp','Pos4mer + DeepL4':'4-mer + Deep DNAshape','Pos4mer Flank22 (Pure)':'4-mer, 22 bp','Flank22 + DeepL4':'4-mer, 22 bp + Deep DNAshape',
'LOGO Gene-Mean ρ̄_g':'Gene-averaged Spearman ρ','HD Δz_f70\n(30-Fold LOGO)':'Δz','HD Δz_spec\n(30-Fold LOGO)':'Δz$_{spec}$',
'Pos4mer (12-bp)':'4-mer, 12 bp','Pos4mer Flank15 (Pure)':'4-mer, 15 bp','Flank15 + DeepL4':'4-mer, 15 bp + Deep DNAshape',
'HGBDT\n(12-bp)':'GBT\n12 bp','+DNAshapeR\n(7ch)':'+ DNAshapeR','+DeepL4\n(7ch)':'+ Deep\nDNAshape','Flank15\n(Pure Seq)':'GBT\n15 bp','Flank15\n+DeepL4':'15 bp + Deep\nDNAshape',
'Δz_f70 (30-Fold LOGO)':'Δz','Δz_spec (30-Fold LOGO)':'Δz$_{spec}$',
'12-bp DeepL4 vs DNAshapeR:\n  Δz: +0.0034 (Wp=0.54)\n  Δz_spec: -0.0024 (Wp=0.58)\nFlank15 +DeepL4 vs Flank15:\n  Δz: +0.0090 (Wp=0.21)\n  Δz_spec: +0.0249 (Wp=0.0028)':'',
})
TICK={'Seq_G2':'Dinuc.','Pos4mer\n(N5-N6)':'Pos.\n4-mer','Full Pos\n4/5-mer':'Pos.\n4/5-mer','G2\nAllPairs':'All-pair\ndinuc.'}
def fix(t):
    if t in EX: return EX[t]
    m=re.fullmatch(r'p=([\d.]+) \| p_H=([\d.]+)',t)
    if m: return f'P = {m.group(1)} (adj. {m.group(2)})'
    m=re.fullmatch(r'p=([\d.]+)\n\((\d+)% proxy\)',t)
    if m: return f'P = {m.group(1)}\n{m.group(2)}%'
    m=re.fullmatch(r'1a: p=([\d.]+)\n1b: p=([\d.]+)',t)
    if m: return f'P = {m.group(1)} / {m.group(2)}'
    m=re.fullmatch(r'p=([\d.]+)',t)
    if m: return f'P = {m.group(1)}'
    return t
LEFT=set()
def save(fig,stem):
    for ax in fig.axes:
        ax.set_title('')
        ax.set_xlabel(fix(ax.get_xlabel())); ax.set_ylabel(fix(ax.get_ylabel()))
        for get,setf in ((ax.get_xticklabels,ax.set_xticklabels),(ax.get_yticklabels,ax.set_yticklabels)):
            labs=get()
            if any(l.get_text() and not re.fullmatch(r'[\d.−\-+ ]+', l.get_text()) for l in labs):
                ticks=ax.get_xticks() if get==ax.get_xticklabels else ax.get_yticks()
                new=[TICK.get(l.get_text(), fix(l.get_text())) for l in labs]
                if get==ax.get_xticklabels: ax.set_xticks(ticks); ax.set_xticklabels(new)
                else: ax.set_yticks(ticks); ax.set_yticklabels(new)
        for t in ax.texts: t.set_text(fix(t.get_text()))
        lg=ax.get_legend()
        if lg:
            for t in lg.get_texts(): t.set_text(fix(t.get_text()))
        for s in [ax.get_xlabel(),ax.get_ylabel()]+[t.get_text() for t in ax.texts]+([t.get_text() for t in lg.get_texts()] if lg else []):
            if re.search(r'_|Seq|HGB|ShiftGC|Sp4|Exp 1|FamilyCode|GSE',s): LEFT.add((stem,s))
    for t in fig.texts: t.set_text(fix(t.get_text()))
    name=STEM[stem]
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else 300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print('saved',name)
g._save_fig=save
if __name__=="__main__":
    g.main()
    print("LEFT",LEFT)
