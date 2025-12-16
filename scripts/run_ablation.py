import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    precision_recall_curve,
)

from src.data.datamodule import AntigenicityDataModule
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention
from src.models.cnn_bilstm_scratch import CNNBiLSTMScratch
from src.models.cnn_only import CNNOnlyScratch


@dataclass
class EvalResult:
    model: str
    pooling: str
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    acc: float
    precision_at_20: float
    precision_at_50: float
    recall_at_20: float
    recall_at_50: float
    threshold: float


def get_preds_and_targets(model: pl.LightningModule, dataloader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for peptides, y in dataloader:
            logits = model(peptides)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(y.cpu())
    if not all_preds:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(all_preds), torch.cat(all_targets)


def select_best_threshold(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(preds, targets)
    if thresholds.numel() == 0:
        return 0.5, 0.0, 0.0, 0.0, 0.0

    f1_scores = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
    best_idx = torch.argmax(f1_scores)
    best_threshold = thresholds[best_idx].item()
    best_f1 = f1_scores[best_idx].item()
    best_precision = precision[best_idx + 1].item()
    best_recall = recall[best_idx + 1].item()
    return best_threshold, best_f1, best_precision, best_recall, best_idx.item()


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, float]:
    if preds.numel() == 0:
        return {k: float("nan") for k in ["auroc", "auprc", "f1", "precision", "recall", "acc"]}

    auroc = binary_auroc(preds, targets.int()).item()
    auprc = binary_average_precision(preds, targets.int()).item()

    preds_bin = (preds >= threshold).int()
    tp = (preds_bin * targets.int()).sum().item()
    fp = (preds_bin * (1 - targets.int())).sum().item()
    fn = ((1 - preds_bin) * targets.int()).sum().item()
    tn = ((1 - preds_bin) * (1 - targets.int())).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "acc": acc,
    }


def precision_recall_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> Tuple[float, float]:
    if preds.numel() < k:
        return float("nan"), float("nan")
    sorted_indices = torch.argsort(preds, descending=True)
    topk_targets = targets[sorted_indices][:k]
    precision_k = topk_targets.float().mean().item()
    recall_k = (topk_targets.sum() / (targets.sum() + 1e-8)).item()
    return precision_k, recall_k


def train_single_model(
    model_name: str,
    pooling: str,
    model_fn: Callable[[], pl.LightningModule],
    datamodule: AntigenicityDataModule,
    args,
) -> EvalResult:
    pl.seed_everything(args.seed, workers=True)

    model = model_fn()
    device = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks = [
        EarlyStopping(monitor="val_auprc", mode="max", patience=args.early_stop_patience, verbose=True)
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)

    # Collect predictions
    val_preds, val_targets = get_preds_and_targets(model, datamodule.val_dataloader(), device)
    best_threshold, best_f1, best_precision, best_recall, _ = select_best_threshold(val_preds, val_targets)

    test_preds, test_targets = get_preds_and_targets(model, datamodule.test_dataloader(), device)
    metric_dict = compute_metrics(test_preds, test_targets, threshold=best_threshold)

    p20, r20 = precision_recall_at_k(test_preds, test_targets, 20)
    p50, r50 = precision_recall_at_k(test_preds, test_targets, 50)

    return EvalResult(
        model=model_name,
        pooling=pooling,
        auroc=metric_dict["auroc"],
        auprc=metric_dict["auprc"],
        f1=metric_dict["f1"],
        precision=metric_dict["precision"],
        recall=metric_dict["recall"],
        acc=metric_dict["acc"],
        precision_at_20=p20,
        precision_at_50=p50,
        recall_at_20=r20,
        recall_at_50=r50,
        threshold=best_threshold,
    )


def write_reports(results: List[EvalResult], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "ablation_results.md")
    json_path = os.path.join(output_dir, "ablation_results.json")

    headers = [
        "Model",
        "Pooling",
        "AUROC",
        "AUPRC",
        "F1",
        "P@20",
        "P@50",
    ]
    lines = ["\t".join(headers)]
    for r in results:
        lines.append(
            "\t".join(
                [
                    r.model,
                    r.pooling,
                    f"{r.auroc:.4f}",
                    f"{r.auprc:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.precision_at_20:.4f}" if not torch.isnan(torch.tensor(r.precision_at_20)) else "nan",
                    f"{r.precision_at_50:.4f}" if not torch.isnan(torch.tensor(r.precision_at_50)) else "nan",
                ]
            )
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("Model\tPooling\tAUROC\tAUPRC\tF1\tP@20\tP@50\n")
        for line in lines[1:]:
            f.write(f"{line}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print("=== Ablation Results ===")
    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved JSON to: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation across CNN variants.")
    parser.add_argument("--data_path", type=str, default="15mer_antigenicity_dataset.parquet")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--cnn_pooling", type=str, choices=["mean", "max"], default="mean")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="reports")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    datamodule = AntigenicityDataModule(
        batch_size=args.batch_size,
        data_path=args.data_path,
    )
    datamodule.setup()

    models: List[Tuple[str, str, Callable[[], pl.LightningModule]]] = [
        (
            "CNN-only",
            args.cnn_pooling,
            lambda: CNNOnlyScratch(lr=args.lr, pos_weight=args.pos_weight, pooling=args.cnn_pooling),
        ),
        (
            "CNN+BiLSTM",
            "mean",
            lambda: CNNBiLSTMScratch(lr=args.lr, pos_weight=args.pos_weight),
        ),
        (
            "CNN+BiLSTM",
            "attention",
            lambda: CNNBiLSTMAttention(lr=args.lr, pos_weight=args.pos_weight),
        ),
    ]

    results: List[EvalResult] = []
    for model_name, pooling, model_fn in models:
        print(f"\n=== Training {model_name} ({pooling}) ===")
        res = train_single_model(model_name, pooling, model_fn, datamodule, args)
        results.append(res)

    write_reports(results, args.output_dir)


if __name__ == "__main__":
    main()
