# src/models/esm2_frozen_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
import torchmetrics as tm


class ESM2FrozenTransformer(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 6,
        lr: float = 3e-4,
        pos_weight: float = 12.33,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load ESM-2 và freeze
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        for p in self.esm.parameters():
            p.requires_grad = False

        plm_dim = 1280

        # 2) Transformer encoder head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=plm_dim,
            nhead=20,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 3) Classifier (có thể để đơn giản như bạn muốn)
        self.classifier = nn.Sequential(
            nn.LayerNorm(plm_dim),
            nn.Linear(plm_dim, 1),
        )

        # 4) pos_weight cho BCEWithLogitsLoss
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

        # 5) Metrics epoch-level (nếu muốn dùng torchmetrics cho đẹp)
        self.val_auroc = tm.AUROC(task="binary")
        self.val_auprc = tm.AveragePrecision(task="binary")

    def forward(self, peptides):
        # peptides: list[str]
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        x = results["representations"][33]  # (B, L, D)

        x = self.transformer(x)      # (B, L, D)
        x = x.mean(dim=1)            # mean pooling
        logits = self.classifier(x).squeeze(-1)  # (B,)
        return logits

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        y = y.float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=self.pos_weight,
        )

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits = self(peptides)
        y = y.float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)

        # Cập nhật metrics
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
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
            },
        }
