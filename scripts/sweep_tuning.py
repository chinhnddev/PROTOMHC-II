import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    precision_recall_curve,
)

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.datamodule import AntigenicityDataModule
from src.models.cnn_only import CNNOnlyScratch
from src.models.cnn_bilstm_scratch import CNNBiLSTMScratch
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention


@dataclass
class ValMetrics:
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]


def parse_list(arg: str, cast=float) -> List:
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)


def get_preds_targets(model: pl.LightningModule, dataloader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    preds_list, targets_list = [], []
    with torch.no_grad():
        for peptides, y in dataloader:
            logits = model(peptides)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu())
            targets_list.append(y.cpu())
    if not preds_list:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(preds_list), torch.cat(targets_list)


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


def compute_val_metrics(probs: torch.Tensor, targets: torch.Tensor, threshold: float, k_list: Sequence[int]) -> ValMetrics:
    targets_int = targets.int()
    auroc = binary_auroc(probs, targets_int).item()
    auprc = binary_average_precision(probs, targets_int).item()

    preds_bin = (probs >= threshold).int()
    tp = (preds_bin * targets_int).sum().item()
    fp = (preds_bin * (1 - targets_int)).sum().item()
    fn = ((1 - preds_bin) * targets_int).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    precision_at_k = {}
    recall_at_k = {}
    for k in k_list:
        pk, rk = precision_recall_at_k(probs, targets_int, k)
        precision_at_k[k] = pk
        recall_at_k[k] = rk

    return ValMetrics(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        precision=precision,
        recall=recall,
        threshold=threshold,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
    )


def build_model_factory(model_name: str, lr: float, pos_weight: float, dropout: Optional[float]) -> Tuple[Callable[[], pl.LightningModule], str]:
    drop = dropout if dropout is not None else 0.4
    if model_name == "cnn_only":
        return (
            lambda: CNNOnlyScratch(lr=lr, pos_weight=pos_weight, pooling="mean", dropout=drop),
            "mean",
        )
    if model_name == "cnn_bilstm_mean":
        return (
            lambda: CNNBiLSTMScratch(lr=lr, pos_weight=pos_weight, dropout=drop),
            "mean",
        )
    if model_name == "cnn_bilstm_attention":
        return (
            lambda: CNNBiLSTMAttention(lr=lr, pos_weight=pos_weight, dropout=drop),
            "attention",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Small sweep over pos_weight / lr / dropout.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn_only", "cnn_bilstm_mean", "cnn_bilstm_attention"], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="reports/sweep")
    parser.add_argument("--pos_weight_list", type=str, required=True, help="Comma-separated floats, e.g., '1,3,6,12'")
    parser.add_argument("--lr_list", type=str, required=True, help="Comma-separated floats, e.g., '1e-3,5e-4,2e-4'")
    parser.add_argument("--dropout_list", type=str, default="", help="Comma-separated floats; empty to use model default")
    parser.add_argument("--k_list", type=str, default="20,50", help="Comma-separated ints for Precision@K/Recall@K")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = output_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    pos_weights = parse_list(args.pos_weight_list, float)
    lrs = parse_list(args.lr_list, float)
    dropouts = parse_list(args.dropout_list, float) if args.dropout_list else [None]
    k_list = [int(k) for k in parse_list(args.k_list, float)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for pos_weight in pos_weights:
        for lr in lrs:
            for dropout in dropouts:
                set_seed(args.seed)

                model_fn, pooling = build_model_factory(args.model, lr=lr, pos_weight=pos_weight, dropout=dropout)
                model = model_fn()

                run_name = f"{args.model}_pw{pos_weight}_lr{lr}_do{dropout if dropout is not None else 'def'}"
                ckpt_dir = ckpt_root / run_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_cb = ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="best",
                    monitor="val_auprc",
                    mode="max",
                    save_last=False,
                )
                early_stop_cb = EarlyStopping(monitor="val_auprc", mode="max", patience=10, verbose=False)

                trainer = pl.Trainer(
                    max_epochs=args.max_epochs,
                    accelerator="gpu" if device.type == "cuda" else "cpu",
                    devices=1,
                    callbacks=[checkpoint_cb, early_stop_cb],
                    log_every_n_steps=50,
                    enable_checkpointing=True,
                    logger=False,
                    num_sanity_val_steps=0,
                )

                dm = AntigenicityDataModule(
                    batch_size=args.batch_size,
                    data_path=args.data_path,
                    num_workers=args.num_workers,
                )
                dm.setup()

                trainer.fit(model, datamodule=dm)

                best_ckpt = checkpoint_cb.best_model_path
                if not best_ckpt:
                    print(f"[WARN] No checkpoint saved for {run_name}; skipping.")
                    continue

                # Reload best model for evaluation
                best_model = type(model).load_from_checkpoint(best_ckpt, map_location=device)

                val_loader = dm.val_dataloader()
                val_probs, val_targets = get_preds_targets(best_model, val_loader, device)
                if val_probs.numel() == 0:
                    print(f"[WARN] No val predictions for {run_name}; skipping.")
                    continue
                best_thr, best_f1, best_prec, best_rec = select_best_threshold(val_probs, val_targets)
                val_metrics = compute_val_metrics(val_probs, val_targets, threshold=best_thr, k_list=k_list)

                results.append(
                    {
                        "model": args.model,
                        "pooling": pooling,
                        "pos_weight": pos_weight,
                        "lr": lr,
                        "dropout": dropout if dropout is not None else "default",
                        "best_val_auprc": val_metrics.auprc,
                        "best_val_auroc": val_metrics.auroc,
                        "val_f1": val_metrics.f1,
                        "val_p@20": val_metrics.precision_at_k.get(20, float("nan")),
                        "val_p@50": val_metrics.precision_at_k.get(50, float("nan")),
                        "val_r@20": val_metrics.recall_at_k.get(20, float("nan")),
                        "val_r@50": val_metrics.recall_at_k.get(50, float("nan")),
                        "best_threshold": val_metrics.threshold,
                        "ckpt_path": best_ckpt,
                    }
                )

    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)
    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)

    # Sort top 10
    df_sorted = df.sort_values(by=["best_val_auprc", "val_p@20"], ascending=[False, False]).head(10)
    md_lines = ["model\tpooling\tpos_weight\tlr\tdropout\tbest_val_auprc\tbest_val_auroc\tval_f1\tval_p@20\tval_p@50\tval_r@20\tval_r@50\tbest_threshold\tckpt_path"]
    for _, row in df_sorted.iterrows():
        md_lines.append(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["pooling"]),
                    str(row["pos_weight"]),
                    str(row["lr"]),
                    str(row["dropout"]),
                    f"{row['best_val_auprc']:.4f}",
                    f"{row['best_val_auroc']:.4f}",
                    f"{row['val_f1']:.4f}",
                    f"{row['val_p@20']:.4f}",
                    f"{row['val_p@50']:.4f}",
                    f"{row['val_r@20']:.4f}",
                    f"{row['val_r@50']:.4f}",
                    f"{row['best_threshold']:.4f}",
                    str(row["ckpt_path"]),
                ]
            )
        )

    md_path = output_dir / "sweep_top10.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Saved sweep CSV to {csv_path}")
    print(f"Saved top-10 markdown to {md_path}")
    print("Top 5 runs:\n", df_sorted.head(5))


if __name__ == "__main__":
    main()
