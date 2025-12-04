# ProtoMHC-II

Minimal instructions to train and evaluate on Colab:

1. Install deps
```
pip install -r requirements.txt
```

2. Prepare data  
Place `15mer_antigenicity_dataset.parquet` at repo root (or run `python data/preprocessing.py` to regenerate from `data/processed/mhc2_cancer_train_final.csv`).

3. Train one model
```
python run.py exp=exp02_prototype model=esm2_frozen_prototype
```
Outputs go to `checkpoints/exp02_prototype/`.

4. Train all
```
bash scripts/train_all.sh
```
(`exp06_esm3` is commented out because the 8B model is too heavy for Colab; run manually if you have resources.)

5. Ensemble + table
```
python scripts/ensemble_predict.py
python scripts/make_table.py
```

6. Export artifacts
```
python scripts/export_for_paper.py
```
"# PROTOMHC-II" 
