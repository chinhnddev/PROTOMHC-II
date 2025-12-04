"""Linear probe on frozen ESM-3 8B embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, EsmModel


class ESM3LinearProbe(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = EsmModel.from_pretrained("facebook/esm3_8B")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm3_8B")
        for p in self.model.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(2560, 1)  # ESM-3 8B hidden size

    def forward(self, peptides):
        inputs = self.tokenizer(
            peptides,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
