"""Run inference for CNN+BiLSTM scratch model on the test split and save predictions."""
import argparse
import os
import joblib
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.datamodule import AntigenicityDataModule
from src.models.cnn_bilstm_scratch import CNNBiLSTMScratch


def parse_args():
    parser = argparse.ArgumentParser(description="Predict test split with CNN+BiLSTM scratch model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/CNN_BiLSTM_Scratch/best.ckpt",
        help="Path to checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/predictions/cnn_bilstm_test_preds.pkl",
        help="Output pickle file (y_true, y_pred)",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")

    model = CNNBiLSTMScratch.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    dm = AntigenicityDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()
    loader = dm.test_dataloader()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for peptides, labels in tqdm(loader, desc="Predicting test set", unit="batch"):
            logits = model(peptides)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    joblib.dump((y_true, y_pred), args.out)
    print(f"Predictions saved to: {args.out}")
    print(f"Test AUROC = {roc_auc_score(y_true, y_pred):.4f}")
    print(f"Test AUPRC = {average_precision_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
