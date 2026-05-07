"""
app/main.py
FastAPI application entry point for the LLM Cost Autopilot system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.utils.config import setup_logging, get_setting

setup_logging()

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes prompts to the cheapest capable local model via Ollama.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "LLM Cost Autopilot",
        "version": "1.0.0",
        "docs": "/docs",
        "api": "/v1",
    }
