"""Run inference for the ProtoMHC-II model on the test split and save predictions."""
import argparse
import os

import joblib
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.data.datamodule import AntigenicityDataModule
from src.models.esm2_frozen_prototype import ProtoMHCII


def parse_args():
    parser = argparse.ArgumentParser(description="Predict test split with ProtoMHC-II checkpoint")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/ProtoMHC-II_SOTA/best-v1.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/predictions/exp02_prototype_test_preds.pkl",
        help="Where to save (y_true, y_pred)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")

    state = torch.load(args.ckpt, map_location="cpu")
    model = ProtoMHCII(**state["hyper_parameters"])
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device).eval()

    dm = AntigenicityDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    dm.setup()
    loader = dm.test_dataloader()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preds, labels = [], []
    with torch.no_grad():
        for peptides, y in tqdm(loader, desc="Predicting", unit="batch"):
            logits, _ = model(peptides)
            preds.append(torch.sigmoid(logits).cpu())
            labels.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    joblib.dump((y_true, y_pred), args.out)

    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    print(f"Saved predictions to: {args.out}")
    print(f"Test AUROC={auroc:.4f} AUPRC={auprc:.4f}")


if __name__ == "__main__":
    main()
