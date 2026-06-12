from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch


class BaseRanker(ABC):
    """Unified interface for ranking models."""

    def __init__(self, name: str = "BaseRanker", device: str = "cpu"):
        self.name = name
        self.device = device

    @abstractmethod
    def predict(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score user-item pairs. Returns scores tensor of shape (N,)."""
        ...

    def rank(self, user_id: int, candidates: List[int], top_k: int = 10) -> np.ndarray:
        """Rank candidates for a user and return top-k items."""
        if not candidates:
            return np.array([], dtype=np.int64)
        user = torch.tensor([user_id] * len(candidates), device=self.device)
        items = torch.tensor(candidates, device=self.device)
        with torch.no_grad():
            scores = self.predict(user, items).cpu().numpy()
        top_k = min(top_k, len(candidates))
        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return np.array(candidates)[top_indices]
