"""REST API and a small web page for PhishGuard.

    uvicorn phishguard.api:app --reload
    open http://127.0.0.1:8000        (web page)
    open http://127.0.0.1:8000/docs   (interactive API docs)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .model import MODEL_PATH, PhishGuard

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="PhishGuard", version=__version__,
              description="Explainable phishing URL classification from the URL string alone.")


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=4096,
                     examples=["http://paypal.com.account-verify.xyz/login"])


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=500)


class PredictionOut(BaseModel):
    url: str
    probability: float
    verdict: str
    reasons: list[str]
    suspicious_fragments: list[str]
    model_scores: dict[str, float]
    trusted_domain: bool


@lru_cache(maxsize=1)
def get_model() -> PhishGuard:
    path = Path(os.environ.get("PHISHGUARD_MODEL", MODEL_PATH))
    if not path.exists():
        raise HTTPException(503, "Model not trained yet: run `python -m phishguard.train`")
    return PhishGuard.load(path)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    model = get_model()
    return {"status": "ok", "version": model.version, "threshold": model.threshold}


@app.post("/predict", response_model=PredictionOut)
def predict(req: URLRequest) -> dict:
    return get_model().explain(req.url).to_dict()


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(req: BatchRequest) -> list[dict]:
    model = get_model()
    return [model.explain(u).to_dict() for u in req.urls]
