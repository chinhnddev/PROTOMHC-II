"""Prepare leak-free peptide dataset with split by unique peptide."""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/processed/mhc2_cancer_train_final.csv"
OUT_PATH = "data/processed/15mer_antigenicity_dataset_FINAL.parquet"


def main():
    df_raw = pd.read_csv(RAW_PATH)
    print(f"Raw: {len(df_raw)} samples, {df_raw['sequence'].nunique()} unique sequences")

    # Normalize columns to expected schema
    df = df_raw.rename(
        columns={
            "sequence": "peptide",
            "allele_seq": "mhc_sequence",
            "hla": "mhc_sequence",
            "binding": "label",
            "target": "label",
        }
    ).filter(["peptide", "mhc_sequence", "label"])

    # Remove peptides with conflicting labels
    label_per_pep = df.groupby("peptide")["label"].nunique()
    conflicts = label_per_pep[label_per_pep > 1].index
    print(f"Conflicting peptides: {len(conflicts)}")
    df = df[~df["peptide"].isin(conflicts)]

    # Deduplicate peptides (keep first occurrence)
    df_unique = df.drop_duplicates(subset="peptide", keep="first")
    print(f"After dedup: {len(df_unique)} samples, {df_unique['peptide'].nunique()} peptides")

    peptides = df_unique["peptide"]
    labels = df_unique["label"]

    # Split by unique peptide to avoid leakage
    pep_train_val, pep_test, y_tv, y_test = train_test_split(
        peptides, labels, test_size=0.15, stratify=labels, random_state=42
    )
    pep_train, pep_val, y_train, y_val = train_test_split(
        pep_train_val, y_tv, test_size=0.15, stratify=y_tv, random_state=42
    )

    df_train = df_unique[df_unique["peptide"].isin(pep_train)].copy()
    df_val = df_unique[df_unique["peptide"].isin(pep_val)].copy()
    df_test = df_unique[df_unique["peptide"].isin(pep_test)].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_final = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    df_final.to_parquet(OUT_PATH, index=False)

    print("DONE. No peptide overlap between splits.")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()
