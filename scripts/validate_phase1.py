"""
scripts/validate_phase1.py
Phase 1 Validation: sends the same prompt to all 3 models and prints results.
Run from project root: python scripts/validate_phase1.py
"""

import sys
import json
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.registry import ModelRegistry
from app.services.ollama_client import OllamaClient
from app.utils.config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

TEST_PROMPT = "Explain what machine learning is in 2-3 sentences."


def print_separator(char="─", width=60):
    print(char * width)


def validate():
    print_separator("═")
    print("  PHASE 1 VALIDATION: Model Abstraction Layer")
    print_separator("═")

    # 1. Load registry
    registry = ModelRegistry("configs/models.yaml")
    print(f"\n✅ Registry loaded. Models: {registry.names()}\n")

    # 2. Init client
    client = OllamaClient(registry)

    # 3. Check Ollama availability
    available = client.list_available_models()
    print(f"📡 Ollama models available: {available or 'NONE (Ollama not running)'}")
    print()

    # 4. Run prompt on each model
    results = []
    for model_name in registry.names():
        print_separator()
        print(f"  Model: {model_name.upper()}")
        print_separator()
        print(f"  Prompt: {TEST_PROMPT}")
        print()

        resp = client.send_request(
            model_name=model_name,
            prompt=TEST_PROMPT,
            max_tokens=200,
            temperature=0.7,
        )

        if resp.success:
            print(f"  ✅ Response: {resp.content[:200]}...")
        else:
            print(f"  ❌ Error: {resp.error}")

        print(f"\n  📊 Metrics:")
        print(f"     Prompt tokens    : {resp.prompt_tokens}")
        print(f"     Completion tokens: {resp.completion_tokens}")
        print(f"     Total tokens     : {resp.total_tokens}")
        print(f"     Latency          : {resp.latency_ms:.1f}ms")
        print(f"     Cost             : ${resp.cost:.6f}")
        print()

        results.append(resp.to_dict())

    # 5. Summary
    print_separator("═")
    print("  SUMMARY")
    print_separator("═")
    success_count = sum(1 for r in results if r["error"] is None)
    print(f"  Models tested   : {len(results)}")
    print(f"  Successful      : {success_count}")
    print(f"  Failed          : {len(results) - success_count}")
    total_latency = sum(r["latency_ms"] for r in results)
    print(f"  Total latency   : {total_latency:.1f}ms")
    print()

    if success_count == 0:
        print("⚠️  No models responded. Ensure Ollama is running:")
        print("     ollama serve")
        print("     ollama pull phi3")
        print("     ollama pull mistral")
        print("     ollama pull llama3")
    else:
        print("✅ Phase 1 PASSED — Model Abstraction Layer is working.")

    print_separator("═")
    return success_count > 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
