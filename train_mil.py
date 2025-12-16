import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import MHCIIMILDataset, mhcii_mil_collate_fn
from src.metrics import aggregate_metrics, classification_metrics, ranking_metrics
from src.model_mil import MILAttentionModel
from src.tokenizer import VOCAB_SIZE, batch_encode


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(labels):
    pos = labels.sum()
    neg = len(labels) - pos
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        windows = batch_encode(batch["windows"], device=device)  # [B,7,9]
        labels = batch["labels"].float().to(device)
        bag_logit, _, _ = model(windows)
        loss = criterion(bag_logit, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    probs, labels, attn_list, peptides, windows_list = [], [], [], [], []
    for batch in loader:
        windows_tensor = batch_encode(batch["windows"], device=device)
        bag_logit, _, attn = model(windows_tensor)
        prob = torch.sigmoid(bag_logit).cpu().numpy()
        probs.extend(prob.tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())
        peptides.extend(batch["peptide"])
        windows_list.extend(batch["windows"])
        if attn is not None:
            attn_list.extend(attn.cpu().numpy().tolist())
        else:
            attn_list.extend([None] * len(batch["peptide"]))
    return np.array(labels), np.array(probs), attn_list, peptides, windows_list


def log_top_examples(peptides, windows, probs, attn_list, k=5, mode="attention"):
    idxs = np.argsort(-probs)[:k]
    print("Top-5 positive-like predictions:")
    for i in idxs:
        print(f"- peptide: {peptides[i]}")
        print(f"  prob: {probs[i]:.4f}")
        print(f"  windows: {windows[i]}")
        if mode == "attention" and attn_list[i] is not None:
            print(f"  attention: {attn_list[i]}")
        else:
            print("  attention: N/A (max pooling)")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Train MIL model for MHC-II 15-mer immunogenicity.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--pooling_mode", type=str, default="attention", choices=["attention", "max"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/mil"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MHCIIMILDataset(args.data_path, split="train", skip_invalid_length=True)
    val_ds = MHCIIMILDataset(args.data_path, split="val", skip_invalid_length=True)
    test_ds = MHCIIMILDataset(args.data_path, split="test", skip_invalid_length=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mhcii_mil_collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mhcii_mil_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mhcii_mil_collate_fn)

    pos_weight = compute_pos_weight(np.array(train_ds.labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    model = MILAttentionModel(vocab_size=VOCAB_SIZE, pooling_mode=args.pooling_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_auprc = -1
    patience = 8
    epochs_no_improve = 0
    best_ckpt = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        y_val, p_val, _, _, _ = run_eval(model, val_loader, device)
        val_metrics = aggregate_metrics(y_val, p_val)
        val_auprc = val_metrics["auprc"]

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_auprc={val_auprc:.4f}, val_f1={val_metrics['f1']:.4f}"
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best checkpoint from {best_ckpt}")

    # Choose threshold from val, then report on test
    y_val, p_val, attn_val, pep_val, win_val = run_eval(model, val_loader, device)
    val_metrics = aggregate_metrics(y_val, p_val)
    thr = val_metrics["best_threshold"]
    print("Validation metrics:", val_metrics)

    y_test, p_test, attn_test, pep_test, win_test = run_eval(model, test_loader, device)
    test_metrics = aggregate_metrics(y_test, p_test)
    cls_at_thr = classification_metrics(y_test, p_test, thr)
    rank_at_k = ranking_metrics(y_test, p_test)
    print(f"Test metrics (threshold from val={thr:.4f}):", {**test_metrics, **cls_at_thr, **rank_at_k})

    log_top_examples(pep_test, win_test, p_test, attn_test, k=5, mode=args.pooling_mode)


if __name__ == "__main__":
    main()
