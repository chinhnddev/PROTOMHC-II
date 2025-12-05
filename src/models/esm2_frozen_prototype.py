# src/models/esm2_frozen_prototype.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision  # ← SỬA IMPORT NÀY (đúng cho torchmetrics 1.4+)

class ProtoMHCII(pl.LightningModule):
    def __init__(
        self,
        num_prototypes: int = 70,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 12.3,
        esm_model_name: str = "esm2_t33_650M_UR50D",
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load & freeze ESM-2
        self.esm, self.alphabet = getattr(esm.pretrained, esm_model_name)()
        self.batch_converter = self.alphabet.get_batch_converter()
        for p in self.esm.parameters():
            p.requires_grad = False
        plm_dim = self.esm.embed_dim  # 1280

        # 2) Prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, plm_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.proto_proj = nn.Linear(plm_dim, plm_dim, bias=False)

        # 3) Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=plm_dim, num_heads=10, batch_first=True, dropout=0.1
        )

        # 4) Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(plm_dim),
            nn.Linear(plm_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # 5) Pos weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

        # 6) Epoch-level metrics
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

        # Để visualize
        self.example_attn = None
        self.example_peptides = None

    def encode_peptides_with_esm(self, peptides):
        data = [(i, seq) for i, seq in enumerate(peptides)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33])
        return results["representations"][33]  # (B, L, D)

    def forward(self, peptides):
        x = self.encode_peptides_with_esm(peptides)
        B, L, D = x.size()

        proto = self.proto_proj(self.prototypes).unsqueeze(0).expand(B, -1, -1)

        attn_out, attn_weights = self.attn(proto, x, x)
        context = attn_out.mean(dim=1)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights

    def training_step(self, batch, batch_idx):
        peptides, y = batch
        logits, _ = self(peptides)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peptides, y = batch
        logits, attn_weights = self(peptides)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.val_auroc.update(probs, y.int())
        self.val_auprc.update(probs, y.int())

        if batch_idx == 0:
            self.example_attn = attn_weights.detach().cpu()
            self.example_peptides = list(peptides)

        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc", auprc, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

        # Log prototype activation (rất quan trọng cho interpretability)
        if self.example_attn is not None:
            activation_per_proto = self.example_attn.mean(dim=(0, 2))  # (P,) ← SỬA CHỖ NÀY
            self.log("proto_activation_mean", activation_per_proto.mean(), prog_bar=True)
            self.log("proto_activation_std", activation_per_proto.std(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
                "interval": "epoch",
                "frequency": 1,
            },
        }