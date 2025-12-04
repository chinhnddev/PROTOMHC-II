"""Lightning DataModule for 15-mer antigenicity dataset."""
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class PeptideOnlyDataset(Dataset):
    def __init__(self, df):
        self.peptides = df["peptide"].tolist()
        self.labels = df["label"].astype(float).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]


def collate_fn(batch):
    peptides, labels = zip(*batch)
    return list(peptides), torch.tensor(labels, dtype=torch.float32)


class AntigenicityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        data_path: str = "15mer_antigenicity_dataset.parquet",
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        df = pd.read_parquet(self.data_path)
        required_cols = {"peptide", "label", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        self.train_df = df[df["split"] == "train"]
        self.val_df = df[df["split"] == "val"]
        self.test_df = df[df["split"] == "test"]

    def train_dataloader(self):
        return DataLoader(
            PeptideOnlyDataset(self.train_df),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            PeptideOnlyDataset(self.val_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            PeptideOnlyDataset(self.test_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# Backwards compatibility alias
MHCIIDataModule = AntigenicityDataModule
