from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from recall.base import RecallBase


class UserCF(RecallBase):
    """User-based Collaborative Filtering."""

    def __init__(self, top_k_neighbors: int = 50):
        super().__init__(name="UserCF")
        self.top_k_neighbors = top_k_neighbors
        self.user_sim = None
        self.user_items = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.user_items = {uid: set(items) for uid, items in train.items()}

        rows, cols, data = [], [], []
        for uid, items in train.items():
            for iid in items:
                rows.append(uid)
                cols.append(iid)
                data.append(1.0)
        user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(num_users, num_items)
        )
        self.user_sim = cosine_similarity(user_item_matrix, dense_output=False)

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        sim_row = self.user_sim[user_id].toarray().ravel()
        sim_row[user_id] = -1
        top_n = min(self.top_k_neighbors, len(sim_row) - 1)
        neighbors = np.argpartition(-sim_row, top_n)[:top_n]
        neighbors = neighbors[sim_row[neighbors] > 0]

        scores = np.zeros(self.num_items)
        interacted = self.user_items.get(user_id, set())
        for nb in neighbors:
            sim = sim_row[nb]
            for iid in self.user_items.get(int(nb), set()):
                if iid not in interacted:
                    scores[iid] += sim

        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items


class ItemCF(RecallBase):
    """Item-based Collaborative Filtering."""

    def __init__(self, top_k_neighbors: int = 50):
        super().__init__(name="ItemCF")
        self.top_k_neighbors = top_k_neighbors
        self.item_sim = None
        self.user_items = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.user_items = {uid: set(items) for uid, items in train.items()}

        rows, cols, data = [], [], []
        for uid, items in train.items():
            for iid in items:
                rows.append(iid)
                cols.append(uid)
                data.append(1.0)
        item_user_matrix = csr_matrix(
            (data, (rows, cols)), shape=(num_items, num_users)
        )
        self.item_sim = cosine_similarity(item_user_matrix, dense_output=False)

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        interacted = self.user_items.get(user_id, set())
        if not interacted:
            return np.array([], dtype=np.int64)

        scores = np.zeros(self.num_items)
        for iid in interacted:
            sim_row = self.item_sim[iid].toarray().ravel()
            sim_row[iid] = -1
            top_n = min(self.top_k_neighbors, len(sim_row) - 1)
            top_neighbors = np.argpartition(-sim_row, top_n)[:top_n]
            for nb in top_neighbors:
                if nb not in interacted:
                    scores[nb] += sim_row[nb]

        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
