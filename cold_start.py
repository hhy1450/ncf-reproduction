from typing import Dict, List
import numpy as np


class ColdStartHandler:
    """Cold-start recommendation strategies for users with few interactions."""

    def __init__(self, num_items: int):
        self.num_items = num_items
        self.item_popularity = None
        self.user_groups = None

    def compute_popularity(self, train: Dict[int, List[int]]):
        """Compute global item popularity from training data."""
        self.item_popularity = {}
        for uid, items in train.items():
            for iid in items:
                self.item_popularity[iid] = self.item_popularity.get(iid, 0) + 1

    def split_users(self, train: Dict[int, List[int]]) -> Dict[str, List[int]]:
        """Group users by interaction count."""
        groups = {"extreme_cold": [], "cold": [], "warm": [], "hot": []}
        for uid, items in train.items():
            n = len(items)
            if n <= 1:
                groups["extreme_cold"].append(uid)
            elif n <= 3:
                groups["cold"].append(uid)
            elif n <= 5:
                groups["warm"].append(uid)
            else:
                groups["hot"].append(uid)
        self.user_groups = groups
        return groups

    def popular_recommend(self, k: int = 10, diversity_weight: float = 0.3) -> np.ndarray:
        """Top-K popular items with diversity injection."""
        if self.item_popularity is None:
            return np.array([], dtype=np.int64)
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: -x[1])
        top_items = [iid for iid, _ in sorted_items[:k * 2]]

        if diversity_weight > 0 and len(top_items) > k:
            scores = np.array([self.item_popularity[i] for i in top_items], dtype=float)
            scores = scores / scores.max()
            noise = np.random.uniform(0, diversity_weight, len(top_items))
            scores = scores + noise
            top_indices = np.argsort(-scores)[:k]
            return np.array(top_items)[top_indices]
        return np.array(top_items[:k])

    def group_recommend(self, user_id: int, train: Dict[int, List[int]],
                        recall_model=None, top_n: int = 10) -> np.ndarray:
        """Recommend based on user interaction group. Returns None if should use full pipeline."""
        user_interactions = train.get(user_id, [])
        n_interact = len(user_interactions)

        if n_interact <= 1:
            return self.popular_recommend(k=top_n, diversity_weight=0.5)
        elif n_interact <= 3:
            return self.popular_recommend(k=top_n, diversity_weight=0.2)
        elif n_interact <= 5:
            if recall_model is not None:
                return recall_model.recommend(user_id, k=top_n)
            return self.popular_recommend(k=top_n, diversity_weight=0.1)
        else:
            return None  # signal to use full pipeline

    def evaluate_groups(self, test: Dict[int, int], train: Dict[int, List[int]],
                        recommend_fn) -> Dict[str, Dict[str, float]]:
        """Evaluate recommendation performance by user group."""
        groups = self.split_users(train)
        results = {}
        for group_name, users in groups.items():
            if not users:
                results[group_name] = {"count": 0, "HR@10": 0, "NDCG@10": 0}
                continue
            hrs, ndcgs = [], []
            for uid in users:
                if uid not in test:
                    continue
                recs = recommend_fn(uid, 10)
                true_item = test[uid]
                hrs.append(1.0 if true_item in recs else 0.0)
                if true_item in recs:
                    rank = list(recs).index(true_item)
                    ndcgs.append(1.0 / np.log2(rank + 2))
                else:
                    ndcgs.append(0.0)
            results[group_name] = {
                "count": len(hrs),
                "HR@10": np.mean(hrs) if hrs else 0,
                "NDCG@10": np.mean(ndcgs) if ndcgs else 0,
            }
        return results
