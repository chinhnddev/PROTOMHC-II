"""Collect key artifacts for paper submission into a dated folder."""
import os
import shutil
from datetime import datetime


date = datetime.now().strftime("%Y%m%d")
paper_dir = f"Paper_Export_{date}"
os.makedirs(paper_dir, exist_ok=True)

artifacts = [
    "results/tables/Table_1_Antigenicity_Results.xlsx",
    "results/figures/prototype_motifs_heatmap.pdf",
    "notebooks/04_Visualize_Prototype_Motifs.ipynb",
    "results/figures/pr_curve.pdf",
]

for path in artifacts:
    if os.path.exists(path):
        shutil.copy(path, os.path.join(paper_dir, os.path.basename(path)))
    else:
        print(f"Warning: missing {path}, skipping.")

print(f"Exported artifacts to: {paper_dir}")
