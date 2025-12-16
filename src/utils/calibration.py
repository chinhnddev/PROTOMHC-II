# src/utils/calibration.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for binary logits."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        # log_T guarantees T > 0 via softplus
        self.log_T = nn.Parameter(torch.tensor(float(init_temp)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward(logits)

    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000, lr: float = 1e-2) -> float:
        """Fit temperature on validation logits/labels by minimizing BCE-with-logits."""
        self.train()
        optimizer = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        labels = labels.float()

        def _closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.temperature.item()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature
