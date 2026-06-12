"""FastAPI recommendation service."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Personalized Recommendation System API",
    description="Two-stage recommendation system (Recall + Ranking)",
    version="1.0.0",
)

pipeline = None
available_models = {}
cold_start_handler = None


class RecommendResponse(BaseModel):
    user_id: int
    items: List[int]
    scores: Optional[List[float]] = None
    model_used: str


class ColdStartRequest(BaseModel):
    interactions: List[int]
    top_k: int = 10


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return {"models": list(available_models.keys())}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100),
    model: str = Query(default=None),
):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}

    try:
        result = pipeline.recommend(user_id, top_n=top_k)
        return RecommendResponse(
            user_id=user_id,
            items=result["items"],
            scores=result.get("scores"),
            model_used=model or "default",
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/recommend/cold-start", response_model=RecommendResponse)
async def cold_start_recommend(req: ColdStartRequest):
    if cold_start_handler is None:
        return {"error": "Cold-start handler not initialized"}

    items = cold_start_handler.popular_recommend(
        k=req.top_k, diversity_weight=0.5
    )
    return RecommendResponse(
        user_id=-1,
        items=items.tolist(),
        model_used="cold-start-popular",
    )
