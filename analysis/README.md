# Analysis pipeline for the gradient-boosted tree surrogate and Deep DNAshape analyses

This folder contains the code, as it was run, behind the following result files in `../results/`:

| Result file | Script |
| --- | --- |
| `homeodomain_hgbdt_b49.json` | `v7_1/scripts/run_v7_1_exp1a_hgbdt_b49.py` (500 training probes per mutant, 49 surrogates) |
| `homeodomain_hgbdt_fullk2000_bridge.json` | `v7_1/scripts/run_v7_1_exp1b_hgbdt_fullk2000_bridge.py` (all 2,000 probes, 9 surrogates) |
| `homeodomain_hgbdt_fullk2000_b19.json` | `v7_1/scripts/run_v7_1_exp1b_b19_extension.py` (all 2,000 probes, 19 surrogates for Δz_spec) |
| `deep_dnashape_comparison.json` | `v7_1/scripts/run_v7_1_exp3_deep_dnashape.py` with `v7_1/src/deep_dnashape_bench.py` |
| `unified_benchmark_results.json` | `v7_1/scripts/build_v7_1_unified_ledger.py` (collects the checkpoints above) |

The `v3/`–`v7_1/` subfolders keep the module layout of the original project, because later modules import earlier ones (`v7_1/src` → `v7/src` → `v6/src` → `v5/src` → `v4/src` → `v3/src`). The core methods used in the paper are also available in a cleaner form in `../src/`.

## Setup

```bash
pip install -r ../requirements.txt
pip install torch rdata   # Deep DNAshape inference (CPU) and reading the homeodomain PBM .rds file
```

Paths are resolved relative to a project root. By default this is this `analysis/` folder; set `BIOPHYS_ROOT` to use another location:

```bash
export BIOPHYS_ROOT=/path/to/project_root
```

## Input data

Download the public datasets and place them under the project root:

| Data | Expected location |
| --- | --- |
| Homeodomain mutant PBMs (GEO GSE233827), including `IntensityMatrix.rds` | `v7/data/raw/` |
| bHLH genomic-context PBMs (GSE59845) and universal PBMs (GSE97794) | `v3/data/raw/geo/<accession>/` |
| Exd–Hox SELEX-seq (GSE65073) | `v3/data/raw/geo/GSE65073/extracted/` |
| DNAshapeR pentamer table | `data/biophysics_tables/pentamer_dnashape.npz` |
| Pretrained Deep DNAshape models (https://github.com/JinsenLi/deepDNAshape) | `data/raw_external/deepDNAshape/deepDNAshape/` |

The processed homeodomain dataset (`v7/data/processed/v7_kock_dataset.npz`) is built from the raw GSE233827 files by `python v7/src/kock_pbm_v7.py`. The Deep DNAshape analysis also reads the processed bHLH genomic-context cohorts (`v3/data/processed/cohorts/disc_{Max,Mad1,cMyc}_gcPBM.parquet`), which were produced by an earlier processing step that is not included here; the same alignment is implemented in `../src/bhlh_benchmark.py`. Outputs are written to `v7_1/artifacts/`.

## Run

```bash
python v7/src/kock_pbm_v7.py      # build the processed homeodomain dataset
cd v7_1/scripts
python run_v7_1_exp1a_hgbdt_b49.py
python run_v7_1_exp1b_hgbdt_fullk2000_bridge.py
python run_v7_1_exp1b_b19_extension.py
python run_v7_1_exp3_deep_dnashape.py
python build_v7_1_unified_ledger.py
```

The tree analyses are CPU-bound and run for many hours with all 2,000 probes per mutant; intermediate results are checkpointed and reused when a script is restarted.
