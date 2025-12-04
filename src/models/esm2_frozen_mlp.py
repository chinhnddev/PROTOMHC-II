"""Frozen ESM-2 pooled embedding with MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl


class ESM2FrozenMLP(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(1280, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33].mean(dim=1)
        return self.mlp(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
