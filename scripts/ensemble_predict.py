"""Generate ensemble predictions on the test set and save AUROC/AUPRC."""
import os
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.datamodule import AntigenicityDataModule
from src.models.esm2_frozen_transformer import ESM2FrozenTransformer
from src.models.esm2_frozen_prototype import ProtoMHCII
from src.models.esm2_frozen_mlp import ESM2FrozenMLP
from src.models.protbert_transformer import ProtBERTTransformer
from src.models.cnn_bilstm_scratch import CNNBiLSTMScratch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_TEMPLATE = "checkpoints/{folder}/best.ckpt"
PRED_OUT = "results/predictions/ensemble_test_preds.pkl"
os.makedirs("results/predictions", exist_ok=True)


def load_model(ckpt_path, cls):
    model = cls()
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    model_entries = [
        ("exp01_esm2_transformer", ESM2FrozenTransformer),
        ("exp02_prototype", ProtoMHCII),
        ("exp03_mlp", ESM2FrozenMLP),
        ("exp04_protbert", ProtBERTTransformer),
        ("exp05_cnn_bilstm", CNNBiLSTMScratch),
        # ESM-3 8B omitted (too heavy for most environments)
    ]

    models = []
    for folder, cls in model_entries:
        ckpt_path = CKPT_TEMPLATE.format(folder=folder)
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint {ckpt_path}, skipping.")
            continue
        models.append(load_model(ckpt_path, cls))

    if not models:
        raise RuntimeError("No models loaded for ensemble.")

    dm = AntigenicityDataModule()
    dm.setup()
    loader = dm.test_dataloader()

    all_preds, y_true = [], []
    with torch.no_grad():
        for peptides, labels in loader:
            batch_preds = []
            for model in models:
                logits = model(peptides)
                batch_preds.append(torch.sigmoid(logits))
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            all_preds.append(ensemble_pred.cpu())
            y_true.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(y_true).numpy()

    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    print(f"Ensemble AUROC={auroc:.4f} AUPRC={auprc:.4f}")

    joblib.dump((y_true, y_pred), PRED_OUT)
    print(f"Saved predictions to {PRED_OUT}")


if __name__ == "__main__":
    main()
