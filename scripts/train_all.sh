#!/bin/bash
# Train all predefined experiments (Hydra overrides exp=...)
set -e

echo "Starting training for all experiments..."

python run.py exp=exp01_esm2_transformer
echo "Done 01 - ESM-2 frozen + Transformer"

python run.py exp=exp02_prototype
echo "Done 02 - ProtoMHC-II (ours)"

python run.py exp=exp03_mlp
echo "Done 03 - ESM-2 frozen + MLP"

python run.py exp=exp04_protbert
echo "Done 04 - ProtBERT + Transformer"

python run.py exp=exp05_cnn_bilstm
echo "Done 05 - CNN+BiLSTM from scratch"

# Skipping ESM-3 8B (too heavy for Colab). Run manually if you have resources:
# python run.py exp=exp06_esm3
# echo "Done 06 - ESM-3 8B linear probe (optional)"

echo "All experiments finished!"
echo "Now run: python scripts/make_table.py to generate Table 1"
