from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class RecallBase(ABC):
    """Base class for recall methods."""

    def __init__(self, name: str = "RecallBase"):
        self.name = name
        self.num_users = 0
        self.num_items = 0

    @abstractmethod
    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        """Train the recall model on user-item interaction data."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        """Return top-k item indices for a given user."""
        ...

    def recommend_batch(self, user_ids: List[int], k: int = 200) -> Dict[int, np.ndarray]:
        """Return {user_id: top-k item array} for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}
