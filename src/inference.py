"""Inference helper to score peptides with a trained checkpoint."""
import torch
import hydra
from omegaconf import OmegaConf


def load_model_from_ckpt(ckpt_path: str, model_target: str, **model_kwargs):
    model_cls = hydra.utils.get_class(model_target)
    model = model_cls(**model_kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict(peptides, ckpt_path: str, model_target: str, **model_kwargs):
    model = load_model_from_ckpt(ckpt_path, model_target, **model_kwargs)
    with torch.no_grad():
        logits = model(peptides)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


if __name__ == "__main__":
    # Example usage:
    cfg = OmegaConf.create(
        {
            "ckpt_path": "checkpoints/exp02_prototype/best.ckpt",
            "model_target": "src.models.esm2_frozen_prototype.ProtoMHCII",
            "peptides": ["SYGFQPTNGVGYQPY"],
        }
    )
    preds = predict(cfg.peptides, cfg.ckpt_path, cfg.model_target)
    print(preds)
