# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class MHCII15merDataset(Dataset):
    def __init__(self, df):
        self.peptides = df['peptide'].tolist()
        self.mhc_seqs = df['mhc_sequence'].tolist()
        self.labels = df['label'].values.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.peptides[idx],
            self.mhc_seqs[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )