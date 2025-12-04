"""ProtBERT frozen encoder with Transformer head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl


class ProtBERTTransformer(pl.LightningModule):
    def __init__(self, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.alphabet = esm.pretrained.protbert_bfd()
        for p in self.model.parameters():
            p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, peptides):
        batch_converter = self.alphabet.get_batch_converter()
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            result = self.model(tokens, repr_layers=[30])
        x = result["representations"][30]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
