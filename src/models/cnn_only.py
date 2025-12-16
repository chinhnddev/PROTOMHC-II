# src/models/cnn_only.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.functional.classification import precision_recall_curve


class CNNOnlyScratch(pl.LightningModule):
    def __init__(self, vocab_size=25, embed_dim=128, lr=1e-3, pos_weight=12.33, pooling: str = "mean"):
        super().__init__()
        self.save_hyperparameters()

        # Mapping chuỗi 20+4 aa (X=21, U=22, O=23, B=24, Z=25)
        self.aa_to_idx = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWXYUOBZ")}

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  # +1 với idx 0
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Project to match baseline classifier input dim (512)
        self.proj = nn.Linear(256, 512)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.pooling = pooling

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_preds = []
        self.val_targets = []

    def seq_to_idx(self, s):
        return torch.tensor([self.aa_to_idx.get(c.upper(), 0) for c in s], dtype=torch.long)

    def forward(self, peptides):
        x = torch.stack([self.seq_to_idx(s) for s in peptides]).to(self.device)
        x = self.embedding(x)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)  # (B, 256, L)

        if self.pooling == "max":
            pooled, _ = torch.max(x, dim=2)  # (B, 256)
        else:
            pooled = x.mean(dim=2)  # (B, 256)

        pooled = self.proj(pooled)  # (B, 512)
        logits = self.classifier(pooled).squeeze(-1)
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
        # Base metrics
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc", auprc, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

        # Aggregate predictions/targets
        all_preds = torch.cat(self.val_preds) if self.val_preds else torch.tensor([])
        all_targets = torch.cat(self.val_targets) if self.val_targets else torch.tensor([])

        if all_preds.numel() > 0 and all_targets.numel() > 0:
            precision, recall, thresholds = precision_recall_curve(all_preds, all_targets, task="binary")
            if thresholds.numel() > 0:
                f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
                best_idx = torch.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                best_precision = precision[best_idx + 1]
                best_recall = recall[best_idx + 1]

                preds_bin = (all_preds >= best_threshold).int()
                acc = (preds_bin == all_targets.int()).float().mean()

                self.log("val_threshold_opt", best_threshold)
                self.log("val_f1", best_f1, prog_bar=True)
                self.log("val_precision", best_precision)
                self.log("val_recall", best_recall)
                self.log("val_acc", acc)

            # Top-K screening metrics
            sorted_indices = torch.argsort(all_preds, descending=True)
            sorted_targets = all_targets[sorted_indices]
            for k in (20, 50):
                if sorted_targets.numel() >= k:
                    topk = sorted_targets[:k]
                    precision_k = topk.float().mean()
                    recall_k = topk.sum() / (all_targets.sum() + 1e-8)
                    self.log(f"val_precision@{k}", precision_k)
                    self.log(f"val_recall@{k}", recall_k)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auprc"},
        }
