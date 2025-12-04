"""ESM-2 frozen encoder with Transformer head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics.functional as tmf


class ESM2FrozenTransformer(pl.LightningModule):
    def __init__(self, num_layers: int = 4, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for p in self.esm.parameters():
            p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=20, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]  # (B, L, 1280)

        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", tmf.auroc(preds, y.int()), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
