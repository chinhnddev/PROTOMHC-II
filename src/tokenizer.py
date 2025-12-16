import torch

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
PAD_TOKEN = "<PAD>"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}
AA_TO_ID[PAD_TOKEN] = len(AA_TO_ID)
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}
VOCAB_SIZE = len(AA_TO_ID)


def encode_window(window: str) -> torch.Tensor:
    """Encode a single 9-mer string to a LongTensor of shape [9]."""
    return torch.tensor([AA_TO_ID.get(ch, AA_TO_ID["X"]) for ch in window], dtype=torch.long)


def batch_encode(windows_batch, device=None) -> torch.Tensor:
    """
    windows_batch: List[List[str]] shape [B, 7], each inner string length 9.
    Returns: LongTensor shape [B, 7, 9].
    """
    encoded = [[encode_window(w) for w in win_list] for win_list in windows_batch]
    tensor = torch.stack([torch.stack(row, dim=0) for row in encoded], dim=0)
    return tensor.to(device) if device else tensor
