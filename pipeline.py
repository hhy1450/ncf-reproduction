from typing import Dict, List, Optional
import numpy as np

from recall.base import RecallBase
from rank.base import BaseRanker


class RecommendationPipeline:
    """Two-stage recommendation: multi-recall → rank → Top-N."""

    def __init__(self, recalls: List[RecallBase], ranker: Optional[BaseRanker] = None,
                 recall_k: int = 200, top_n: int = 10):
        self.recalls = recalls
        self.ranker = ranker
        self.recall_k = recall_k
        self.top_n = top_n

    def recall(self, user_id: int) -> np.ndarray:
        """Multi-recall fusion: union of all recall results, deduplicated."""
        all_items = set()
        for rec in self.recalls:
            items = rec.recommend(user_id, k=self.recall_k)
            all_items.update(items.tolist())
        return np.array(sorted(all_items))

    def recommend(self, user_id: int, top_n: int = None) -> Dict:
        """Full pipeline: recall → (optional rank) → top-n."""
        if top_n is None:
            top_n = self.top_n

        candidates = self.recall(user_id)
        if len(candidates) == 0:
            return {"user_id": user_id, "items": [], "scores": None, "source": "empty"}

        if self.ranker is not None:
            ranked = self.ranker.rank(user_id, candidates.tolist(), top_k=top_n)
            return {
                "user_id": user_id,
                "items": ranked.tolist(),
                "scores": None,
                "source": "ranked",
            }
        else:
            result = candidates[:top_n]
            return {
                "user_id": user_id,
                "items": result.tolist(),
                "scores": None,
                "source": "recall-only",
            }

    def recommend_batch(self, user_ids: List[int], top_n: int = None) -> List[Dict]:
        return [self.recommend(uid, top_n) for uid in user_ids]
