# src/models/cnn_bilstm_scratch.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.functional.classification import precision_recall_curve
import torch.nn.functional as F


class CNNBiLSTMScratch(pl.LightningModule):
    def __init__(self, vocab_size=25, embed_dim=128, lr=1e-3, pos_weight=12.33, dropout: float = 0.4):
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
        lstm_dropout = 0.0  # num_layers=1 so dropout not applied
        self.bilstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight))

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_preds = []
        self.val_targets = []

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
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc", auprc, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

        # Compute optimal-threshold metrics from full-epoch precision-recall curve
        all_preds = torch.cat(self.val_preds) if self.val_preds else torch.tensor([])
        all_targets = torch.cat(self.val_targets) if self.val_targets else torch.tensor([])
        if all_preds.numel() > 0 and all_targets.numel() > 0:
            all_targets_int = all_targets.int()
            precision, recall, thresholds = precision_recall_curve(all_preds, all_targets_int, task="binary")
            if thresholds.numel() > 0:
                f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
                best_idx = torch.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                best_precision = precision[best_idx + 1]
                best_recall = recall[best_idx + 1]

            self.log("val_f1_opt", best_f1, prog_bar=True)
            self.log("val_precision_opt", best_precision)
            self.log("val_recall_opt", best_recall)
            self.log("val_threshold_opt", best_threshold)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auprc"}
        }
