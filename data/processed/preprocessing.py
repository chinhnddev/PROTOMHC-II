# data/processed/prepare_peptide_only.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data\processed\mhc2_cancer_train_final.csv")  # 162163 mẫu
print("Columns:", df.columns.tolist())
print("Label distribution:\n", df['label'].value_counts())

# Đổi tên cho chuẩn
df = df.rename(columns={'sequence': 'peptide'})

# Tạo split nếu chưa có
if 'split' not in df.columns:
    train_val, test = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, stratify=train_val['label'], random_state=42)
    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'
    df = pd.concat([train, val, test]).reset_index(drop=True)

df.to_parquet("15mer_antigenicity_dataset.parquet", index=False)
print("Đã tạo file chuẩn: 15mer_antigenicity_dataset.parquet")
print(df['split'].value_counts())