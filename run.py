"""Entry point to train a single experiment via Hydra."""
import os

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Running experiment: {cfg.exp.name}")

    # Seed for reproducibility
    pl.seed_everything(cfg.exp.seed, workers=True)

    # Instantiate datamodule and model from Hydra configs
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Trainer setup
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    callbacks = []
    patience = getattr(cfg.exp, "patience", None)
    if patience:
        callbacks.append(
            EarlyStopping(monitor="val_auroc", mode="max", patience=patience, verbose=True)
        )

    default_root_dir = os.path.join("checkpoints", cfg.exp.name)
    # Make sure checkpoint directory exists and log where artifacts go
    os.makedirs(default_root_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {default_root_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_root_dir,
        filename="best",
        monitor="val_auroc",
        mode="max",
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    trainer = pl.Trainer(
        max_epochs=cfg.exp.epochs,
        min_epochs=getattr(cfg.exp, "min_epochs", 1),
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.exp.log_every_n_steps,
        default_root_dir=default_root_dir,
        enable_checkpointing=True,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    if checkpoint_callback.best_model_path:
        print(f"Training finished. Best model saved to: {checkpoint_callback.best_model_path}")
    else:
        print("Training finished, but no checkpoint was saved (monitor metric may be missing).")


if __name__ == "__main__":
    main()
