# src/models/esm2_frozen_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision  # ← SỬA IMPORT ĐÚNG
from sklearn.metrics import average_precision_score  # backup nếu cần


class ESM2FrozenTransformer(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 6,
        lr: float = 3e-4,
        pos_weight: float = 12.33,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load & freeze ESM-2
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        for p in self.esm.parameters():
            p.requires_grad = False

        plm_dim = 1280

        # 2) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=plm_dim,
            nhead=20,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(plm_dim),
            nn.Linear(plm_dim, 1),
        )

        # 4) pos_weight – ĐÚNG CÁCH (không dùng trong BCE trực tiếp!)
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

        # 5) Metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

    def forward(self, peptides):
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = self.batch_converter(data)
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

        # SỬA TẠI ĐÂY – KHÔNG DÙNG pos_weight TRONG BCE!
        bce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none')
        loss = (bce * self.pos_weight.to(y.device) * y.float() + bce * (1 - y.float())).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)

        bce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none')
        loss = (bce * self.pos_weight.to(y.device) * y.float() + bce * (1 - y.float())).mean()

        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y.int())
        self.val_auprc.update(probs, y.int())

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"}
        }