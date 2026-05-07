# 🚀 LLM Cost Autopilot

> Route prompts to the cheapest capable local model. Zero API cost. Full observability.

```
Prompt → Classifier → Router → Ollama (phi3 / mistral / llama3) → Evaluator → Dashboard
```

---

## Architecture

```
llm-cost-autopilot/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── api/routes.py              # POST /v1/completions, GET /v1/models, GET /v1/stats
│   ├── classifier/
│   │   ├── complexity_classifier.py   # Loads trained model, classifies prompts
│   │   └── features.py                # Hand-crafted feature extractor (15 features)
│   ├── router/routing_engine.py   # Maps tier → model, handles escalation
│   ├── evaluator/quality_evaluator.py # Rule-based quality scoring + escalation trigger
│   ├── services/
│   │   ├── autopilot.py           # Orchestrates full pipeline (async)
│   │   └── ollama_client.py       # Unified send_request() for all models
│   ├── models/
│   │   ├── registry.py            # Loads ModelConfig from YAML
│   │   └── schemas.py             # Pydantic + dataclass types
│   ├── db/database.py             # SQLite schema + request logging
│   └── utils/config.py            # YAML config loader
├── dashboard/app.py               # Streamlit dashboard
├── configs/
│   ├── models.yaml                # Model registry + Ollama config
│   └── settings.yaml              # App, DB, evaluator, routing settings
├── data/                          # SQLite DB, classifier.pkl, training CSV
├── scripts/
│   ├── train_classifier.py        # Trains RandomForest classifier (94% CV accuracy)
│   ├── validate_phase1.py         # Smoke-tests all 3 models
│   └── load_test.py               # 500-prompt load test with cost comparison
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Pull Models
```bash
ollama pull phi3
ollama pull mistral
ollama pull llama3
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Classifier
```bash
python scripts/train_classifier.py
# Expected: RandomForest CV accuracy ~94%, test accuracy ~100%
```

### 4. Start API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Docs available at http://localhost:8000/docs
```

### 5. Start Dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 6. Send a Request
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_tokens": 100}'
```

Response:
```json
{
  "content": "Paris.",
  "model_used": "phi3",
  "complexity_tier": "simple",
  "total_tokens": 42,
  "latency_ms": 143.2,
  "cost": 0.0,
  "estimated_savings": 0.000084,
  "request_id": "abc-123..."
}
```

---

## Docker (One Command)

```bash
cd docker
docker-compose up --build
```

This starts:
- **Ollama** on port 11434 (with model auto-pull)
- **API** on port 8000
- **Dashboard** on port 8501

---

## Routing Logic

| Complexity | Model   | Trigger Examples |
|-----------|---------|-----------------|
| Simple    | phi3    | "What is X?", definitions, math, conversions |
| Medium    | mistral | "Explain how X works", code functions, comparisons |
| Complex   | llama3  | System design, distributed architectures, ML pipelines |

The classifier uses 15 hand-crafted features (keyword density, word count, technical vocabulary, structural signals) trained on 188 labeled prompts with RandomForest (94% CV accuracy).

---

## Load Test

```bash
python scripts/load_test.py --count 500
```

**Expected results** (with all models running):
```
Routed Cost      : $0.0000   (all local via Ollama)
Baseline (GPT-4) : $2.14
Cost Savings     : $2.14 (100%)
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/completions` | Route + complete a prompt |
| GET | `/v1/models` | List available models |
| GET | `/v1/stats` | Aggregated system metrics |
| GET | `/v1/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Key Metrics

- **Cost reduction**: ~100% vs GPT-4 (all local inference)
- **Classifier accuracy**: 94% CV, 100% test set
- **Quality evaluation**: Rule-based scorer (length, coherence, completeness, relevance)
- **Auto-escalation**: Poor responses automatically retry on higher-tier model
- **Full observability**: Every request logged to SQLite with tokens, cost, latency, quality score
