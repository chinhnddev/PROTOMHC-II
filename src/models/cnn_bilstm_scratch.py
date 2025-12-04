"""CNN + BiLSTM from scratch on character indices."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNNBiLSTMScratch(pl.LightningModule):
    def __init__(self, vocab_size=25, embed_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, peptides):
        # Simple char-to-index (A=1, C=2, ..., padding=0)
        def seq_to_idx(s):
            return torch.tensor(
                [max(0, ord(c.upper()) - 64) if c.isalpha() else 0 for c in s],
                dtype=torch.long,
            )

        x = torch.stack([seq_to_idx(s) for s in peptides]).to(self.device)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
