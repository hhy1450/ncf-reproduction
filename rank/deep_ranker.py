"""
Deep learning ranker adapter.

Wraps NCF models (GMF / MLP / NeuMF / Wide&Deep) into the BaseRanker
interface so they can be used as the ranking stage in the pipeline.
"""
import os
import torch
import numpy as np
from typing import List

from rank.base import BaseRanker


class DeepRanker(BaseRanker):
    """Adapter that wraps any PyTorch recommendation model into a BaseRanker.

    Usage::

        ranker = DeepRanker(model, device="cpu")
        top_k_items = ranker.rank(user_id, candidates, top_k=10)
    """

    def __init__(self, model: torch.nn.Module, name: str = "DeepRanker",
                 device: str = "cpu"):
        super().__init__(name=name, device=device)
        self.model = model.to(device)
        self.model.eval()

    def predict(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score user-item pairs. Returns scores tensor of shape (N,)."""
        with torch.no_grad():
            return self.model(user, items)

    def save(self, path: str):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "name": self.name,
        }, path)

    @staticmethod
    def load(path: str, model: torch.nn.Module, device: str = "cpu") -> "DeepRanker":
        """Load model weights from disk and wrap into a DeepRanker."""
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        name = checkpoint.get("name", "DeepRanker")
        return DeepRanker(model, name=name, device=device)


def build_ranker(model_name: str, num_users: int, num_items: int,
                 device: str = "cpu") -> torch.nn.Module:
    """Build a ranking model by name.

    Args:
        model_name: one of "GMF", "MLP", "NeuMF", "Wide&Deep"
        num_users, num_items: dataset dimensions
        device: torch device

    Returns:
        Instantiated (but untrained) model.
    """
    if model_name == "GMF":
        from model import GMF
        return GMF(num_users, num_items, embed_dim=16)

    elif model_name == "MLP":
        from model import MLP
        return MLP(num_users, num_items, embed_dim=16, layers=[64, 32, 16])

    elif model_name == "NeuMF":
        from model import NeuMF
        return NeuMF(num_users, num_items, gmf_dim=16, mlp_dim=16,
                     mlp_layers=[64, 32, 16])

    elif model_name == "Wide&Deep":
        from rank.wide_deep import WideAndDeep
        return WideAndDeep(num_users, num_items, embed_dim=32,
                           mlp_layers=[128, 64, 32])

    else:
        raise ValueError(f"Unknown model: {model_name}")


# All available deep ranking models
MODEL_NAMES = ["GMF", "MLP", "NeuMF", "Wide&Deep"]


def load_ranker(weights_path: str, num_users: int, num_items: int,
                device: str = "cpu") -> DeepRanker:
    """Load a trained ranker from a weights file.

    The weights file name is used to infer the model type
    (e.g. ``ranker_Wide&Deep.pt`` → Wide&Deep).

    Args:
        weights_path: path to the .pt weights file
        num_users, num_items: dataset dimensions
        device: torch device

    Returns:
        DeepRanker wrapping the loaded model.
    """
    # Infer model name from filename
    basename = os.path.splitext(os.path.basename(weights_path))[0]
    # Strip common prefixes
    for prefix in ["ranker_", "model_"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]

    # Match against known model names
    model_name = None
    for name in MODEL_NAMES:
        if basename == name or basename.startswith(name):
            model_name = name
            break
    if model_name is None:
        model_name = "Wide&Deep"  # fallback

    model = build_ranker(model_name, num_users, num_items, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return DeepRanker(model, name=model_name, device=device)
