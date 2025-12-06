# src/models/protbert_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision
from sklearn.metrics import average_precision_score


class ProtBERTTransformer(pl.LightningModule):
    def __init__(self, lr: float = 3e-4, pos_weight: float = 12.33):
        super().__init__()
        self.save_hyperparameters()

        # ProtBERT frozen
        self.model, self.alphabet = esm.pretrained.protbert_bfd()
        self.batch_converter = self.alphabet.get_batch_converter()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()  # keep frozen backbone deterministic

        # Transformer head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 4 layer tốt hơn 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1)
        )

        # pos_weight – BẮT BUỘC để tránh model đoán toàn 0
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

    def forward(self, peptides):
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        padding_idx = self.alphabet.padding_idx
        cls_idx = getattr(self.alphabet, "cls_idx", None)
        eos_idx = getattr(self.alphabet, "eos_idx", None)
        key_padding_mask = tokens.eq(padding_idx)
        valid_tokens = ~key_padding_mask
        if cls_idx is not None:
            valid_tokens &= tokens.ne(cls_idx)
        if eos_idx is not None:
            valid_tokens &= tokens.ne(eos_idx)

        with torch.no_grad():
            result = self.model(tokens, repr_layers=[30])
        x = result["representations"][30]

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        token_lengths = valid_tokens.sum(dim=1, keepdim=True).clamp(min=1)
        x = (x * valid_tokens.unsqueeze(-1)).sum(dim=1) / token_lengths
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
