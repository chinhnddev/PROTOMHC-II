"""Prototype-based attention on frozen ESM-2 embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics.functional as tmf


class ProtoMHCII(pl.LightningModule):
    def __init__(self, num_prototypes: int = 35, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        # Learnable prototypes = antigenic motifs
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1280))
        nn.init.xavier_uniform_(self.prototypes)

        self.attn = nn.MultiheadAttention(
            embed_dim=1280, num_heads=10, batch_first=True, dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]  # (B, L, 1280)

        proto = self.prototypes.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, P, D)
        attn_out, attn_weights = self.attn(proto, x, x)  # query=proto, key/value=peptide
        context = attn_out.mean(dim=1)  # (B, D)

        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits, _ = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits, attn_weights = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", tmf.auroc(preds, y.int()), prog_bar=True)
        # Save a small sample of attention weights for later visualization
        self.example_attn = attn_weights.detach().cpu()
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
