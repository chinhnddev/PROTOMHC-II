"""Collect key artifacts for paper submission into a dated folder."""
import os
import shutil
from datetime import datetime
from pathlib import Path


def main():
    date = datetime.now().strftime("%Y%m%d")
    paper_dir = Path(f"Paper_Export_{date}")
    paper_dir.mkdir(exist_ok=True)

    # List of files to collect; add more as needed
    artifacts = [
        "results/tables/Table_1_Antigenicity_Results.xlsx",
        "results/tables/Table_1.md",
        "results/predictions/exp02_prototype_test_preds.pkl",
        "results/predictions/esm2_transformer_test_preds.pkl",
        "results/predictions/protbert_test_preds.pkl",
        "results/predictions/cnn_bilstm_test_preds.pkl",
        "results/predictions/ensemble_test_preds.pkl",
        "results/figures/prototype_motifs_heatmap.pdf",
        "results/figures/pr_curve.pdf",
        "notebooks/04_Visualize_Prototype_Motifs.ipynb",
    ]

    for path_str in artifacts:
        src = Path(path_str)
        if src.exists():
            shutil.copy(src, paper_dir / src.name)
        else:
            print(f"Warning: missing {src}, skipping.")

    print(f"Exported artifacts to: {paper_dir}")


if __name__ == "__main__":
    main()
