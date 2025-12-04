"""Preprocess raw MHC-II data into a standardized 15-mer parquet with splits."""
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = "data/processed/mhc2_cancer_train_final.csv"
OUT_PATH = "15mer_antigenicity_dataset.parquet"


def main():
    df = pd.read_csv(RAW_PATH)
    print("Original columns:", df.columns.tolist())
    print("Shape:", df.shape)
    if "label" in df.columns:
        print("Label distribution:\n", df["label"].value_counts())

    # Normalize column names to expected schema
    df = df.rename(
        columns={
            "seq": "peptide",
            "core": "peptide",
            "hla": "mhc_sequence",
            "allele_seq": "mhc_sequence",
            "binding": "label",
            "target": "label",
        }
    ).filter(["peptide", "mhc_sequence", "label"])

    # Create splits if missing
    if "split" not in df.columns:
        train_val, test = train_test_split(
            df, test_size=0.15, stratify=df["label"], random_state=42
        )
        train, val = train_test_split(
            train_val, test_size=0.15, stratify=train_val["label"], random_state=42
        )

        train = train.copy()
        train["split"] = "train"
        val = val.copy()
        val["split"] = "val"
        test = test.copy()
        test["split"] = "test"
        df = pd.concat([train, val, test])

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
