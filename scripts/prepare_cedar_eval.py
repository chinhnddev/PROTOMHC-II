"""Prepare clean positive-only evaluation set from CEDAR export.

Usage:
python scripts/prepare_cedar_eval.py \
    --input data/processed/epitope_table_export_1765214297.csv \
    --train data/processed/mhc2_cancer_train_final.csv \
    --output data/processed/cedar_eval_pos.csv \
    --min-len 13 --max-len 25
"""
import argparse
import csv
import re
from pathlib import Path

STANDARD_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def load_epitopes(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []

    # Older export style has a first row full of "Epitopes"; newer exports have field names.
    start = 1 if all(cell == "Epitopes" for cell in rows[0]) else 0
    headers = rows[start]

    # Find the epitope column by exact name or suffix (e.g., "Epitopes - Epitope").
    epi_idx = None
    for i, h in enumerate(headers):
        h_norm = h.lower().strip()
        if h == "Epitope" or h_norm.endswith("epitope"):
            epi_idx = i
            break
    if epi_idx is None:
        raise ValueError(f"Epitope column not found in headers: {headers}")

    peptides = []
    for r in rows[start + 1 :]:
        if len(r) <= epi_idx:
            continue
        seq = r[epi_idx].strip()
        if seq:
            peptides.append(seq)
    return peptides


def split_standard(peptides):
    standard, removed = [], []
    for p in peptides:
        (standard if STANDARD_AA_RE.match(p) else removed).append(p)
    return standard, removed


def filter_lengths(peptides, min_len=None, max_len=None):
    kept, dropped = [], []
    for p in peptides:
        l = len(p)
        if (min_len and l < min_len) or (max_len and l > max_len):
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def load_train_sequences(path: Path):
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row.get("sequence", "").strip() for row in reader if row.get("sequence", "").strip()}


def dedup_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Clean CEDAR positives for evaluation")
    parser.add_argument("--input", type=Path, default=Path("data/processed/epitope_table_export_1765214297.csv"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/mhc2_cancer_train_final.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cedar_eval_pos.csv"))
    parser.add_argument("--min-len", type=int, default=None, help="Minimum peptide length to keep")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum peptide length to keep")
    args = parser.parse_args()

    peptides = load_epitopes(args.input)
    standard, modified = split_standard(peptides)
    length_filtered, length_dropped = filter_lengths(standard, args.min_len, args.max_len)

    train_peps = load_train_sequences(args.train)
    no_overlap = [p for p in length_filtered if p not in train_peps]
    deduped = dedup_preserve_order(no_overlap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        for seq in deduped:
            writer.writerow([seq, 1])

    print(f"Input peptides: {len(peptides)}")
    print(f"Dropped modified/non-standard: {len(modified)}")
    print(f"Dropped by length: {len(length_dropped)}")
    print(f"Overlap with train removed: {len(length_filtered) - len(no_overlap)}")
    print(f"Unique kept: {len(deduped)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
