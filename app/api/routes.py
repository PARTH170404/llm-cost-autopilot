"""
app/api/routes.py
FastAPI route handlers for the LLM Cost Autopilot API.
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import CompletionRequest, CompletionResponse, ModelInfo, StatsResponse
from app.services.autopilot import AutopilotService

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton service (initialized on first request via dependency)
_service: AutopilotService | None = None


def get_service() -> AutopilotService:
    global _service
    if _service is None:
        _service = AutopilotService()
    return _service


@router.post("/completions", response_model=CompletionResponse, summary="Route and complete a prompt")
async def completions(req: CompletionRequest, svc: AutopilotService = Depends(get_service)):
    """
    Route the prompt to the cheapest capable model, run inference,
    evaluate quality, and return the result with full metadata.
    """
    try:
        result = await svc.complete(
            prompt=req.prompt,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            force_model=req.force_model,
        )
    except Exception as e:
        logger.exception("Completion failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not result.response.success:
        raise HTTPException(
            status_code=503,
            detail=f"Model error: {result.response.error}",
        )

    return CompletionResponse(
        content=result.response.content,
        model_used=result.response.model_name,
        complexity_tier=result.complexity_tier.value,
        prompt_tokens=result.response.prompt_tokens,
        completion_tokens=result.response.completion_tokens,
        total_tokens=result.response.total_tokens,
        latency_ms=result.response.latency_ms,
        cost=result.response.cost,
        estimated_savings=result.estimated_savings,
        request_id=result.request_id,
    )


@router.get("/models", response_model=list[ModelInfo], summary="List available models")
async def list_models(svc: AutopilotService = Depends(get_service)):
    return [ModelInfo(**m) for m in svc.get_available_models()]


@router.get("/stats", response_model=StatsResponse, summary="Aggregated system statistics")
async def stats(svc: AutopilotService = Depends(get_service)):
    return StatsResponse(**svc.get_stats())


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}
