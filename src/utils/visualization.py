"""Visualization utilities for prototype attention."""
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_prototype_motifs(model, dataloader, device="cuda"):
    """Plot average attention weights of ProtoMHCII prototypes."""
    model.eval()
    model.to(device)
    all_weights = []

    with torch.no_grad():
        for peptides, _ in dataloader:
            _, attn_weights = model(peptides)
            all_weights.append(attn_weights.cpu())
            if len(all_weights) > 10:  # limit batches for quick viz
                break

    weights = torch.cat(all_weights)  # (batch, heads, prototypes, seq_len)
    weights = weights.mean(dim=(0, 1))  # (prototypes, seq_len)
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights.numpy()[:20], cmap="viridis", cbar=True)
    plt.title("Top 20 Learned Prototype Activation Heatmap (MHC-II 15-mer)")
    plt.ylabel("Prototype Index")
    plt.xlabel("Position in Peptide")
    plt.tight_layout()
    plt.savefig("results/figures/prototype_motifs_heatmap.pdf")
    plt.show()
