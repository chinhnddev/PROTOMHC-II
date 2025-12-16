# src/data/dataset.py
import re
import warnings
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
NON_STANDARD_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


class MHCII15merDataset(Dataset):
    def __init__(self, df):
        self.peptides = df["peptide"].tolist()
        self.mhc_seqs = df["mhc_sequence"].tolist()
        self.labels = df["label"].values.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.peptides[idx],
            self.mhc_seqs[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class MHCIIMILDataset(Dataset):
    """Core-aware MIL dataset that creates 9-mer windows for each 15-mer peptide."""

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        split: Optional[str] = None,
        skip_invalid_length: bool = False,
        invalid_aa_policy: str = "replace",  # "replace" or "error"
        invalid_aa_replacement: str = "X",
    ):
        """
        Args:
            data: DataFrame or path to parquet/csv with columns peptide, label, optional allele and split.
            split: If provided, filter rows to this split value.
            skip_invalid_length: If True, drop peptides with length != 15; otherwise raise.
            invalid_aa_policy: "replace" will swap non-standard AAs with invalid_aa_replacement, "error" raises.
            invalid_aa_replacement: Replacement character when invalid_aa_policy="replace".
        """
        self.skip_invalid_length = skip_invalid_length
        self.invalid_aa_policy = invalid_aa_policy
        self.invalid_aa_replacement = invalid_aa_replacement

        df = self._load_dataframe(data)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]

        self.peptides: List[str] = []
        self.labels: List[int] = []
        self.alleles: List[Optional[str]] = []
        self.invalid_length_count = 0
        self.replaced_aa_count = 0

        for _, row in df.iterrows():
            peptide = str(row["peptide"])
            label = int(row["label"])
            allele = row["allele"] if "allele" in df.columns else None

            if len(peptide) != 15:
                if self.skip_invalid_length:
                    self.invalid_length_count += 1
                    continue
                raise ValueError(f"Expected 15-mer, got length {len(peptide)} for peptide: {peptide}")

            peptide, replaced = self._sanitize_peptide(peptide)
            if replaced:
                self.replaced_aa_count += 1

            self.peptides.append(peptide)
            self.labels.append(label)
            if allele is None or (isinstance(allele, float) and pd.isna(allele)):
                self.alleles.append(None)
            else:
                self.alleles.append(str(allele))

        if self.invalid_length_count:
            warnings.warn(
                f"Skipped {self.invalid_length_count} peptides with invalid length (skip_invalid_length=True)."
            )

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _load_dataframe(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        path = str(data)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        raise ValueError("Unsupported data input. Provide a DataFrame or path to parquet/csv.")

    def _sanitize_peptide(self, peptide: str):
        if not NON_STANDARD_RE.search(peptide):
            return peptide, False

        if self.invalid_aa_policy == "replace":
            cleaned = "".join(
                aa if aa in STANDARD_AA else self.invalid_aa_replacement for aa in peptide
            )
            return cleaned, cleaned != peptide

        raise ValueError(f"Peptide contains non-standard amino acids: {peptide}")

    @staticmethod
    def _windows_from_peptide(peptide: str) -> List[str]:
        return [peptide[i : i + 9] for i in range(7)]

    def __getitem__(self, idx: int):
        peptide = self.peptides[idx]
        sample = {
            "windows": self._windows_from_peptide(peptide),
            "label": int(self.labels[idx]),
            "peptide": peptide,
            "allele": self.alleles[idx],
        }
        return sample


def mhcii_mil_collate_fn(batch):
    windows = [item["windows"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    peptides = [item["peptide"] for item in batch]
    alleles = [item["allele"] for item in batch]
    return {"windows": windows, "labels": labels, "peptide": peptides, "allele": alleles}


if __name__ == "__main__":
    # Quick demo / sanity check
    demo_df = pd.DataFrame(
        {
            "peptide": ["ACDEFGHIKLMNPQR", "RSTVWYACDEFGHIK"],  # both length 15
            "label": [1, 0],
            "allele": ["DRB1*04:01", "DRB1*15:01"],
        }
    )

    dataset = MHCIIMILDataset(demo_df, skip_invalid_length=False, invalid_aa_policy="replace")
    first = dataset[0]
    print("Single item windows count:", len(first["windows"]))
    print("First item windows:", first["windows"])

    loader = DataLoader(dataset, batch_size=2, collate_fn=mhcii_mil_collate_fn)
    batch = next(iter(loader))
    print("Batch windows shape:", len(batch["windows"]), "x", len(batch["windows"][0]))
    print("Labels tensor shape:", batch["labels"].shape)
