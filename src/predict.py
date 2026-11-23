#!/usr/bin/env python3
"""
Strictly Inductive Pos4mer + Ridge CLI Predictor on Aligned 14-bp DNA Cores.

Zero eval-row leakage, zero network calls, deterministic, and invariant to evaluation
row permutation.

Usage:
  python src/predict.py --train train.csv --eval eval.csv --out preds.csv --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bhlh_benchmark import run_inductive_pos4mer_ridge


def main() -> None:
    ap = argparse.ArgumentParser(description="Strictly inductive Pos4mer Ridge predictor")
    ap.add_argument("--train", required=True, help="Path to training CSV (columns: id, seq, label)")
    ap.add_argument("--eval", required=True, help="Path to evaluation CSV (columns: id, seq[, label])")
    ap.add_argument("--out", required=True, help="Output predictions CSV path (columns: id, pred)")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = ap.parse_args()

    with Path(args.train).open("r", encoding="utf-8", newline="") as fh:
        tr_rows = list(csv.DictReader(fh))
    with Path(args.eval).open("r", encoding="utf-8", newline="") as fh:
        ev_rows = list(csv.DictReader(fh))

    train_seqs = [r["seq"].strip().upper() for r in tr_rows]
    train_y = np.array([float(r["label"]) for r in tr_rows], dtype=np.float64)
    eval_seqs = [r["seq"].strip().upper() for r in ev_rows]

    preds = run_inductive_pos4mer_ridge(train_seqs, train_y, eval_seqs, seed=args.seed)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "pred"])
        w.writeheader()
        for r, pr in zip(ev_rows, preds):
            w.writerow({"id": r["id"], "pred": f"{float(pr):.8f}"})


if __name__ == "__main__":
    main()
