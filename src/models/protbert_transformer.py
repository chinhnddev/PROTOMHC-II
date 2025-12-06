import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision


class ProtBERTTransformer(pl.LightningModule):
    def __init__(self, lr: float = 5e-5, pos_weight: float = 6.0):
        super().__init__()
        self.save_hyperparameters()

        # ProtBERT backbone (Rostlab) – frozen
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Transformer head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1)
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight))

        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

    def forward(self, peptides):
        # ProtBERT expects space-separated, upper-case amino acids
        spaced = [" ".join(list(seq)) for seq in peptides]
        encoded = self.tokenizer(
            spaced,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (B, L, 1024)

        # Mask out padding and special tokens ([CLS]=0, [SEP]=last)
        key_padding_mask = attention_mask.eq(0)
        valid_tokens = attention_mask.bool()
        if valid_tokens.shape[1] >= 1:
            valid_tokens[:, 0] = False
        if valid_tokens.shape[1] >= 2:
            # last non-pad position is SEP
            last_pos = attention_mask.sum(dim=1) - 1
            valid_tokens[torch.arange(valid_tokens.size(0)), last_pos] = False

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"},
        }
