# src/models/cnn_bilstm_scratch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision


class CNNBiLSTMScratch(pl.LightningModule):
    def __init__(self, vocab_size=25, embed_dim=128, lr=1e-3, pos_weight=12.33):
        super().__init__()
        self.save_hyperparameters()

        # Mapping chuẩn 20+4 aa (X=21, U=22, O=23, B=24, Z=25)
        self.aa_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWXYUOBZ")}
        
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)  # +1 vì có 0
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True, dropout=0.3)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight))

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

    def seq_to_idx(self, s):
        return torch.tensor([
            self.aa_to_idx.get(c.upper(), 0) for c in s
        ], dtype=torch.long)

    def forward(self, peptides):
        x = torch.stack([self.seq_to_idx(s) for s in peptides]).to(self.device)
        x = self.embedding(x)           # (B, L, D)
        x = x.transpose(1, 2)           # (B, D, L)
        x = self.conv(x).transpose(1, 2) # (B, L, D)
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)                # (B, 512)
        logits = self.classifier(x).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        preds = torch.sigmoid(logits)
        self.val_auroc.update(preds, y.int())
        self.val_auprc.update(preds, y.int())
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc", auprc, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"}
        }
