# src/models/esm2_frozen_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics.functional as tmf
from sklearn.metrics import average_precision_score


class ESM2FrozenTransformer(pl.LightningModule):
    def __init__(self, num_layers: int = 6, lr: float = 3e-4, pos_weight: float = 12.33):
        super().__init__()
        self.save_hyperparameters()

        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=20, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 1)
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]

        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        # SỬA TẠI ĐÂY
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss = loss * self.pos_weight.to(y.device)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss = loss * self.pos_weight.to(y.device)
        
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_auroc", tmf.auroc(preds, y.int(), task="binary"), prog_bar=True, on_epoch=True)
        auprc = average_precision_score(y.cpu().numpy(), preds.cpu().numpy())
        self.log("val_auprc", auprc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"}
        }