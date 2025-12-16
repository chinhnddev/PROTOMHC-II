import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    precision_recall_curve,
)

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.datamodule import AntigenicityDataModule, collate_fn, PeptideOnlyDataset
from src.utils.calibration import TemperatureScaler, apply_temperature


@dataclass
class MetricResult:
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    acc: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    brier: float
    ece: float
    threshold: float


def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(False)


def load_model(ckpt_path: str, map_location: str = "cpu") -> pl.LightningModule:
    """Load Lightning model from checkpoint; requires class to be importable."""
    return pl.LightningModule.load_from_checkpoint(ckpt_path, map_location=map_location)


def get_logits_labels(model: pl.LightningModule, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    logits_list, labels_list = [], []
    with torch.no_grad():
        for peptides, y in dataloader:
            logits = model(peptides)
            logits_list.append(logits.detach().cpu())
            labels_list.append(torch.tensor(y).detach().cpu())
    if not logits_list:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(logits_list), torch.cat(labels_list)


def select_best_threshold(probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float]:
    targets_int = targets.int()
    precision, recall, thresholds = precision_recall_curve(probs, targets_int, task="binary")
    if thresholds.numel() == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
    best_idx = torch.argmax(f1_scores)
    best_threshold = thresholds[best_idx].item()
    best_f1 = f1_scores[best_idx].item()
    best_precision = precision[best_idx + 1].item()
    best_recall = recall[best_idx + 1].item()
    return best_threshold, best_f1, best_precision, best_recall


def precision_recall_at_k(probs: torch.Tensor, targets: torch.Tensor, k: int) -> Tuple[float, float]:
    if probs.numel() < k:
        return float("nan"), float("nan")
    idx = torch.argsort(probs, descending=True)
    topk_targets = targets[idx][:k]
    precision_k = topk_targets.float().mean().item()
    recall_k = (topk_targets.sum() / (targets.sum() + 1e-8)).item()
    return precision_k, recall_k


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((probs - targets.float()) ** 2).item()


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Simple ECE with equal-width bins."""
    targets = targets.int()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = probs.numel()
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = targets[mask].float().mean()
        ece += (mask.sum().item() / total) * torch.abs(bin_conf - bin_acc).item()
    return ece


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, threshold: float, k_list: Sequence[int]) -> MetricResult:
    targets_int = targets.int()
    auroc = binary_auroc(probs, targets_int).item()
    auprc = binary_average_precision(probs, targets_int).item()

    preds_bin = (probs >= threshold).int()
    tp = (preds_bin * targets_int).sum().item()
    fp = (preds_bin * (1 - targets_int)).sum().item()
    fn = ((1 - preds_bin) * targets_int).sum().item()
    tn = ((1 - preds_bin) * (1 - targets_int)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    precision_at_k = {}
    recall_at_k = {}
    for k in k_list:
        pk, rk = precision_recall_at_k(probs, targets_int, k)
        precision_at_k[k] = pk
        recall_at_k[k] = rk

    brier = brier_score(probs, targets_int)
    ece = expected_calibration_error(probs, targets_int)

    return MetricResult(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        precision=precision,
        recall=recall,
        acc=acc,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        brier=brier,
        ece=ece,
        threshold=threshold,
    )


def build_dataloaders(data_path: str, batch_size: int, num_workers: int = 4):
    dm = AntigenicityDataModule(batch_size=batch_size, data_path=data_path, num_workers=num_workers)
    dm.setup()
    return dm, dm.val_dataloader(), dm.test_dataloader()


def to_markdown(before: MetricResult, after: MetricResult, k_list: Sequence[int]) -> str:
    headers = ["Stage", "AUROC", "AUPRC", "F1", "P@20", "P@50", "R@20", "R@50", "Brier", "ECE", "Threshold"]
    rows = [headers]

    def fmt_row(name: str, m: MetricResult):
        return [
            name,
            f"{m.auroc:.4f}",
            f"{m.auprc:.4f}",
            f"{m.f1:.4f}",
            f"{m.precision_at_k.get(20, float('nan')):.4f}",
            f"{m.precision_at_k.get(50, float('nan')):.4f}",
            f"{m.recall_at_k.get(20, float('nan')):.4f}",
            f"{m.recall_at_k.get(50, float('nan')):.4f}",
            f"{m.brier:.4f}",
            f"{m.ece:.4f}",
            f"{m.threshold:.4f}",
        ]

    rows.append(fmt_row("before", before))
    rows.append(fmt_row("after", after))

    # Markdown table (tab-separated for simplicity)
    md_lines = ["\t".join(headers)]
    for r in rows[1:]:
        md_lines.append("\t".join(r))
    return "\n".join(md_lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature scaling calibration and evaluation.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Lightning checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset with split column.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--k_list", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    print(f"Loading model from {args.ckpt_path}")
    model = load_model(args.ckpt_path, map_location=device)

    dm, val_loader, test_loader = build_dataloaders(args.data_path, args.batch_size, num_workers=args.num_workers)

    # Collect logits and labels
    val_logits, val_labels = get_logits_labels(model, val_loader, device)
    test_logits, test_labels = get_logits_labels(model, test_loader, device)

    if val_logits.numel() == 0 or test_logits.numel() == 0:
        raise ValueError("Empty logits/labels collected; check data_path and splits.")

    # Before calibration
    val_probs_before = torch.sigmoid(val_logits)
    test_probs_before = torch.sigmoid(test_logits)
    best_thr, _, _, _ = select_best_threshold(val_probs_before, val_labels)
    before_metrics = compute_metrics(test_probs_before, test_labels, threshold=best_thr, k_list=args.k_list)

    # Fit temperature on val
    scaler = TemperatureScaler()
    scaler.to(device)
    T = scaler.fit_temperature(val_logits.to(device), val_labels.to(device))
    print(f"Fitted temperature: {T:.4f}")

    # After calibration
    val_probs_after = torch.sigmoid(apply_temperature(val_logits, T))
    test_probs_after = torch.sigmoid(apply_temperature(test_logits, T))
    best_thr_after, _, _, _ = select_best_threshold(val_probs_after, val_labels)
    after_metrics = compute_metrics(test_probs_after, test_labels, threshold=best_thr_after, k_list=args.k_list)

    # Save reports
    report = {
        "temperature": T,
        "best_threshold_before": best_thr,
        "best_threshold_after": best_thr_after,
        "before_metrics": asdict(before_metrics),
        "after_metrics": asdict(after_metrics),
    }

    json_path = output_dir / "calibration_metrics.json"
    md_path = output_dir / "calibration_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_text = to_markdown(before_metrics, after_metrics, args.k_list)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text + "\n")

    print("=== Calibration Report ===")
    print(md_text)
    print(f"\nSaved JSON to: {json_path}")
    print(f"Saved Markdown to: {md_path}")


if __name__ == "__main__":
    main()
