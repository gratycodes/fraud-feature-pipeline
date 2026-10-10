"""Online scoring: Redis features -> calibrated LightGBM -> cost-based decision.

Every scored request is appended to a Redis stream (the *feature log*) with the
exact feature vector the model saw. ``monitoring.skew_check`` compares that log
with an offline recomputation, which is how train/serve skew gets caught.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ffp.features import MODEL_FEATURES
from ffp.model import ModelBundle
from ffp.online import OnlineStore
from ffp.schema import CATEGORIES, Transaction

FEATURE_LOG_MAXLEN = 500_000


@dataclass(frozen=True)
class ScoreResult:
    txn_id: str
    probability: float
    decision: str
    features: dict[str, float]
    feature_ms: float
    model_ms: float


class ScoringService:
    def __init__(self, store: OnlineStore, model: ModelBundle, log_features: bool = True) -> None:
        self.store = store
        self.model = model
        self.log_features = log_features
        self.log_key = f"{store.p}:feature_log"

    def score(self, txn: Transaction) -> ScoreResult:
        t0 = time.perf_counter()
        feats = self.store.features_for(txn)
        t1 = time.perf_counter()
        x = np.array([[feats[f] for f in MODEL_FEATURES]], dtype=np.float64)
        p = float(self.model.predict_proba(x)[0])
        decision = self.model.decide(p, txn.amount)
        t2 = time.perf_counter()
        if self.log_features:
            self.store.r.xadd(
                self.log_key,
                {
                    "txn_id": txn.txn_id,
                    "ts": txn.ts,
                    "p": repr(p),
                    "f": json.dumps([_json_num(feats[f]) for f in MODEL_FEATURES]),
                },
                maxlen=FEATURE_LOG_MAXLEN,
                approximate=True,
            )
        return ScoreResult(txn.txn_id, p, decision, feats, (t1 - t0) * 1e3, (t2 - t1) * 1e3)

    def feature_log(self, count: int | None = None) -> pd.DataFrame:
        """Read the feature log back as a DataFrame (txn_id, ts, p, <features>)."""
        entries = self.store.r.xrange(self.log_key, count=count) or []
        rows = []
        for _id, fields in entries:
            if not fields:
                continue
            f = {k.decode() if isinstance(k, bytes) else k: v for k, v in fields.items()}
            vals = json.loads(_s(f["f"]))
            row: dict[str, Any] = {
                "txn_id": _s(f["txn_id"]),
                "ts": int(f["ts"]),
                "p": float(f["p"]),
            }
            row.update(
                {
                    name: (math.nan if v is None else float(v))
                    for name, v in zip(MODEL_FEATURES, vals, strict=True)
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=["txn_id", "ts", "p", *MODEL_FEATURES])


def _json_num(v: float) -> float | None:
    return None if math.isnan(v) else v


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class TransactionIn(BaseModel):
    txn_id: str = Field(min_length=1, max_length=64, pattern=r"^[^|]+$")
    ts: int = Field(gt=0, description="epoch milliseconds, UTC")
    card_id: str = Field(min_length=1, max_length=64, pattern=r"^[^|]+$")
    merchant_id: str = Field(min_length=1, max_length=64, pattern=r"^[^|]+$")
    amount: float = Field(gt=0, lt=1_000_000)
    country: str = Field(min_length=2, max_length=2)
    channel: str = Field(pattern=r"^(pos|online)$")
    category: str

    def to_txn(self) -> Transaction:
        if self.category not in CATEGORIES:
            raise HTTPException(422, f"unknown category {self.category!r}")
        return Transaction(**self.model_dump())


class ScoreOut(BaseModel):
    txn_id: str
    fraud_probability: float
    decision: str
    feature_ms: float
    model_ms: float
    features: dict[str, float | None] | None = None


def create_app(service: ScoringService) -> FastAPI:
    app = FastAPI(title="fraud-feature-pipeline", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        service.store.r.ping()
        return {"ok": True}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        m = service.model.metadata
        return {
            "features": m["features"],
            "best_iteration": m.get("best_iteration"),
            "calibration": m.get("calibration"),
            "cost": m.get("cost"),
        }

    @app.post("/score", response_model=ScoreOut)
    def score(body: TransactionIn, explain: bool = False) -> ScoreOut:
        r = service.score(body.to_txn())
        return ScoreOut(
            txn_id=r.txn_id,
            fraud_probability=r.probability,
            decision=r.decision,
            feature_ms=round(r.feature_ms, 3),
            model_ms=round(r.model_ms, 3),
            features={k: _json_num(v) for k, v in r.features.items()} if explain else None,
        )

    return app


def app_from_env() -> FastAPI:
    """Entry point for ``uvicorn ffp.serving:app_from_env --factory``."""
    import redis

    store = OnlineStore(
        redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    )
    model = ModelBundle.load(os.environ.get("FFP_MODEL_DIR", "data/model"))
    return create_app(ScoringService(store, model))
