# src/models/esm2_frozen_prototype.py
"""ProtoMHC-II: Prototype-based antigenicity classifier on frozen ESM-2 embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics.functional as tmf


class ProtoMHCII(pl.LightningModule):
    def __init__(self, num_prototypes: int = 40, lr: float = 3e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # --- ESM-2 frozen backbone ---
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        # --- Learnable prototypes (antigenic motifs) ---
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1280))
        nn.init.xavier_uniform_(self.prototypes)

        # Optional: projector để prototypes học tốt hơn
        self.proto_proj = nn.Linear(1280, 1280)

        # --- Multi-head attention (prototype = query) ---
        self.attn = nn.MultiheadAttention(
            embed_dim=1280, num_heads=10, batch_first=True, dropout=0.1
        )

        # --- Classifier head ---
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

        # --- Focal loss cho imbalance cực mạnh (7.5% positive) ---
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]  # (B, L, 1280)

        # Project prototypes
        proto = self.proto_proj(self.prototypes)  # (P, D)
        proto = proto.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, P, D)

        # Attention: prototype là query → học motif nào được activate
        attn_out, attn_weights = self.attn(proto, x, x)
        context = attn_out.mean(dim=1)  # (B, D)

        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits, _ = self(peptides)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits, attn_weights = self(peptides)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_auroc", tmf.auroc(preds, y.int(), task="binary"), prog_bar=True, on_epoch=True)
        self.log("val_auprc", tmf.auprc(preds, y.int(), task="binary"), prog_bar=True, on_epoch=True)

        # Lưu attn_weights của batch đầu tiên để visualize sau
        if batch_idx == 0:
            self.example_attn = attn_weights.detach().cpu()
            self.example_peptides = peptides

        return loss

    def on_validation_epoch_end(self):
        # Log số prototype được activate (để kiểm tra có bị dead prototype không)
        if hasattr(self, "example_attn"):
            activation = self.example_attn.mean(dim=(0, 2))  # mean over batch & seq_len
            self.log("proto_activation_mean", activation.mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc", "interval": "epoch"}
        }
