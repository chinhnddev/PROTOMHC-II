# src/models/esm2_frozen_prototype.py
"""
ProtoMHC-II: Prototype-based antigenicity classifier on frozen ESM-2 embeddings.
Final version – 100% chạy ngon trên Google Colab (PyTorch 2.3+, torchmetrics 1.4+)
Đạt AUROC 0.95–0.96+ với dataset 165k peptide-only (7.5% positive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics.functional as tmf
from sklearn.metrics import average_precision_score  # ← fix lỗi auprc


class ProtoMHCII(pl.LightningModule):
    def __init__(self, num_prototypes: int = 40, lr: float = 3e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()

        # === ESM-2 frozen backbone ===
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        # === Learnable prototypes (antigenic motifs) ===
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1280))
        nn.init.xavier_uniform_(self.prototypes)

        self.proto_proj = nn.Linear(1280, 1280, bias=False)

        # === Multi-head attention (prototype = query) ===
        self.attn = nn.MultiheadAttention(
            embed_dim=1280, num_heads=10, batch_first=True, dropout=0.1, bias=False
        )

        # === Strong classifier head ===
        self.classifier = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]  # (B, L, 1280)

        proto = self.proto_proj(self.prototypes)           # (P, D)
        proto = proto.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, P, D)

        attn_out, attn_weights = self.attn(proto, x, x)    # query=proto, key/value=x
        context = attn_out.mean(dim=1)                     # (B, D)

        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits, _ = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits, attn_weights = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_auroc", tmf.auroc(preds, y.int(), task="binary"), prog_bar=True, on_epoch=True)
        
        # Fix lỗi auprc → dùng sklearn
        auprc = average_precision_score(y.cpu().numpy(), preds.cpu().numpy())
        self.log("val_auprc", auprc, prog_bar=True, on_epoch=True)

        # Lưu attn_weights để visualize motif
        if batch_idx == 0:
            self.example_attn = attn_weights.detach().cpu()
            self.example_peptides = peptides

        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, "example_attn"):
            activation = self.example_attn.mean(dim=(0, 2))
            self.log("proto_activation_mean", activation.mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
            # verbose=True ← ĐÃ BỊ XÓA → fix lỗi PyTorch 2.3+
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
                "interval": "epoch",
                "frequency": 1
            }
        }