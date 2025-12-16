import torch
import torch.nn as nn
import torch.nn.functional as F


class MILAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        conv_channels: int = 64,
        dropout: float = 0.1,
        pooling_mode: str = "attention",
    ):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.conv3 = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)

        encoder_out = conv_channels * 2
        self.instance_head = nn.Linear(encoder_out, 1)

        if pooling_mode == "attention":
            self.attn_W = nn.Linear(encoder_out, encoder_out)
            self.attn_v = nn.Linear(encoder_out, 1, bias=False)
            self.bag_head = nn.Sequential(
                nn.Linear(encoder_out, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def encode_instances(self, windows: torch.Tensor) -> torch.Tensor:
        """
        windows: LongTensor [B, 7, 9]
        Returns: h [B, 7, encoder_out]
        """
        bsz, inst, seq = windows.shape
        x = self.embedding(windows.view(bsz * inst, seq))  # [B*7, 9, emb]
        x = x.transpose(1, 2)  # [B*7, emb, 9]

        h3 = F.relu(self.conv3(x))
        h5 = F.relu(self.conv5(x))
        h3 = torch.max(h3, dim=2).values
        h5 = torch.max(h5, dim=2).values
        h = torch.cat([h3, h5], dim=1)  # [B*7, 2C]
        h = self.dropout(h)
        h = h.view(bsz, inst, -1)
        return h

    def forward(self, windows: torch.Tensor):
        """
        windows: [B, 7, 9]
        Returns:
            bag_logit: [B]
            instance_logits: [B, 7]
            attn: [B, 7] or None
        """
        h = self.encode_instances(windows)
        instance_logits = self.instance_head(h).squeeze(-1)

        if self.pooling_mode == "max":
            bag_logit, _ = instance_logits.max(dim=1)
            return bag_logit, instance_logits, None

        attn_scores = self.attn_v(torch.tanh(self.attn_W(h))).squeeze(-1)  # [B,7]
        attn_weights = torch.softmax(attn_scores, dim=1)
        bag_repr = torch.sum(attn_weights.unsqueeze(-1) * h, dim=1)  # [B,D]
        bag_logit = self.bag_head(bag_repr).squeeze(-1)
        return bag_logit, instance_logits, attn_weights
