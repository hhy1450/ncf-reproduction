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

    def recall_with_sources(self, user_id: int) -> Dict[int, List[str]]:
        """Multi-recall: return {item_id: [source_names]} for each candidate."""
        item_sources = {}
        for rec in self.recalls:
            items = rec.recommend(user_id, k=self.recall_k)
            for iid in items.tolist():
                if iid not in item_sources:
                    item_sources[iid] = []
                item_sources[iid].append(rec.name)
        return item_sources

    def recall(self, user_id: int) -> np.ndarray:
        """Multi-recall fusion: union of all recall results, deduplicated."""
        item_sources = self.recall_with_sources(user_id)
        return np.array(sorted(item_sources.keys()))

    def recommend(self, user_id: int, top_n: int = None) -> Dict:
        """Full pipeline: recall → (optional rank) → top-n."""
        if top_n is None:
            top_n = self.top_n

        item_sources = self.recall_with_sources(user_id)
        candidates = np.array(sorted(item_sources.keys()))

        if len(candidates) == 0:
            return {"user_id": user_id, "items": [], "scores": None,
                    "sources": {}, "source": "empty"}

        if self.ranker is not None:
            ranked = self.ranker.rank(user_id, candidates.tolist(), top_k=top_n)
            return {
                "user_id": user_id,
                "items": ranked.tolist(),
                "scores": None,
                "sources": {iid: item_sources.get(iid, []) for iid in ranked.tolist()},
                "source": "ranked",
            }
        else:
            result = candidates[:top_n]
            return {
                "user_id": user_id,
                "items": result.tolist(),
                "scores": None,
                "sources": {iid: item_sources.get(iid, []) for iid in result.tolist()},
                "source": "recall-only",
            }

    def recommend_batch(self, user_ids: List[int], top_n: int = None) -> List[Dict]:
        return [self.recommend(uid, top_n) for uid in user_ids]
