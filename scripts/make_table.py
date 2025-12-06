"""Aggregate per-model predictions into a results table (AUROC/AUPRC)."""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


ckpt_dir = "checkpoints"
pred_dir = "results/predictions"
os.makedirs("results/tables", exist_ok=True)

models = [
    ("esm2_transformer", "ESM-2 + Transformer"),          # file: esm2_transformer_test_preds.pkl
    ("exp02_prototype", "ProtoMHC-II (Ours)"),            # file: exp02_prototype_test_preds.pkl
    ("exp03_mlp", "ESM-2 + MLP"),                          # file may be missing; skip if absent
    ("protbert", "ProtBERT + Transformer"),               # file: protbert_test_preds.pkl
    ("cnn_bilstm", "CNN+BiLSTM (scratch)"),               # file: cnn_bilstm_test_preds.pkl
]

results = []

for folder, name in models:
    pred_path = os.path.join(pred_dir, f"{folder}_test_preds.pkl")
    if not os.path.exists(pred_path):
        print(f"Missing predictions for {name}, skipping.")
        continue

    y_true, y_pred = joblib.load(pred_path)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    results.append(
        {
            "Model": name,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
        }
    )

# Add ensemble if available
ensemble_path = os.path.join(pred_dir, "ensemble_test_preds.pkl")
if os.path.exists(ensemble_path):
    y_true, y_pred = joblib.load(ensemble_path)
    results.append(
        {
            "Model": "Ensemble (Ours)",
            "AUROC": round(roc_auc_score(y_true, y_pred), 4),
            "AUPRC": round(average_precision_score(y_true, y_pred), 4),
        }
    )

df = pd.DataFrame(results)
df.to_excel("results/tables/Table_1_Antigenicity_Results.xlsx", index=False)
df.to_markdown("results/tables/Table_1.md", index=False)
print("Table 1 generated:")
print(df)
