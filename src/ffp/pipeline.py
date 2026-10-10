"""End-to-end orchestration used by the CLI, the demo and the CI smoke run."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ffp.features import MAX_WINDOW_MS, MODEL_FEATURES
from ffp.generator import Dataset, GeneratorConfig, generate
from ffp.model import ModelBundle, TrainConfig, evaluate, fit, make_splits
from ffp.monitoring import SkewReport, drift_markdown, drift_report, skew_check
from ffp.offline import backfill, merchant_rate_snapshots, write_offline_store
from ffp.online import OnlineStore
from ffp.schema import DAY, HOUR, Transaction
from ffp.serving import ScoringService
from ffp.stream import InProcessSource, StreamProcessor, transactions_from_frame


@dataclass
class OfflineResult:
    dataset: Dataset
    features: pd.DataFrame
    snapshots: pd.DataFrame
    bundle: ModelBundle
    metrics: dict[str, Any]
    leakage: dict[str, Any]
    timings_s: dict[str, float] = field(default_factory=dict)


def run_offline(
    gen_cfg: GeneratorConfig,
    raw_dir: Path,
    offline_dir: Path,
    model_dir: Path,
    *,
    train_cfg: TrainConfig | None = None,
    with_leakage_study: bool = True,
) -> OfflineResult:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    ds = generate(gen_cfg)
    ds.write(raw_dir)
    timings["generate"] = time.perf_counter() - t

    t = time.perf_counter()
    snaps = merchant_rate_snapshots(ds.transactions, ds.labels, ds.merchants["merchant_id"])
    feats = backfill(ds.transactions, ds.cards, ds.labels, snapshots=snaps)
    write_offline_store(feats, snaps, offline_dir)
    timings["backfill"] = time.perf_counter() - t

    start_ts = int(ds.transactions["ts"].min()) // DAY * DAY
    cfg = train_cfg or TrainConfig.for_days(start_ts, gen_cfg.days)
    t = time.perf_counter()
    splits = make_splits(feats, ds.labels, cfg)
    bundle = fit(splits, cfg)
    bundle.save(model_dir)
    metrics = evaluate(bundle, splits.test, ds.ground_truth)
    timings["train_and_evaluate"] = time.perf_counter() - t

    leakage: dict[str, Any] = {}
    if with_leakage_study:
        t = time.perf_counter()
        leaky_feats = backfill(ds.transactions, ds.cards, ds.labels, leaky=True)
        leaky_splits = make_splits(leaky_feats, ds.labels, cfg)
        leaky_bundle = fit(leaky_splits, cfg)
        leaky_eval = evaluate(leaky_bundle, leaky_splits.test, ds.ground_truth)
        # What production would actually see: online features are point-in-time by construction.
        served_eval = evaluate(leaky_bundle, splits.test, ds.ground_truth)
        leakage = {
            "correct": _summary(metrics),
            "leaky_offline_estimate": _summary(leaky_eval),
            "leaky_model_as_served": _summary(served_eval),
        }
        timings["leakage_study"] = time.perf_counter() - t
    return OfflineResult(ds, feats, snaps, bundle, metrics, leakage, timings)


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "pr_auc": m["pr_auc_raw"],
        "pr_auc_ci95": m["pr_auc_ci95"],
        "recall_by_pattern": {
            k: v["recall_bayes_rule"] for k, v in m.get("recall_by_pattern", {}).items()
        },
        "cost_usd_bayes_rule": m["decisions"]["model_bayes_cost_rule"]["cost_usd"],
    }


@dataclass
class ReplayResult:
    scored: int
    warmup_events: int
    seconds: float
    served: pd.DataFrame
    feature_ms_p50: float
    feature_ms_p99: float


def replay(
    service: ScoringService,
    dataset: Dataset,
    snapshots: pd.DataFrame,
    *,
    start_ts: int,
    end_ts: int,
    processor_lag_events: int = 0,
    bootstrap_last_seen: bool = True,
) -> ReplayResult:
    """Replay a slice of the stream through the online path.

    For each event: score it (features from Redis), then hand it to the stream
    processor. ``processor_lag_events > 0`` makes the processor trail the scorer
    by that many events, like a consumer that has fallen behind.
    """
    store = service.store
    store.load_card_profiles(dataset.cards)
    store.load_merchant_rate_snapshots(snapshots)
    tx = dataset.transactions
    warm_from = start_ts - MAX_WINDOW_MS - HOUR
    if bootstrap_last_seen:
        older = tx[tx["ts"] < warm_from]
        store.load_card_last_seen(older.groupby("card_id")["ts"].max())
    warm = tx[(tx["ts"] >= warm_from) & (tx["ts"] < start_ts)]
    processor = StreamProcessor(store, batch_size=1000)
    processor.run(InProcessSource(transactions_from_frame(warm)))

    window = tx[(tx["ts"] >= start_ts) & (tx["ts"] < end_ts)]
    pending: deque[Transaction] = deque()
    feature_ms: list[float] = []
    t0 = time.perf_counter()
    for txn in transactions_from_frame(window):
        r = service.score(txn)
        feature_ms.append(r.feature_ms)
        pending.append(txn)
        while len(pending) > processor_lag_events:
            store.ingest([pending.popleft()])
    store.ingest(list(pending))
    seconds = time.perf_counter() - t0
    served = service.feature_log()
    served = served[served["txn_id"].isin(window["txn_id"])]
    return ReplayResult(
        scored=len(window),
        warmup_events=len(warm),
        seconds=seconds,
        served=served,
        feature_ms_p50=float(np.percentile(feature_ms, 50)) if feature_ms else math.nan,
        feature_ms_p99=float(np.percentile(feature_ms, 99)) if feature_ms else math.nan,
    )


def monitoring_reports(
    bundle: ModelBundle,
    features: pd.DataFrame,
    served: pd.DataFrame,
    *,
    current_start_ts: int,
) -> tuple[SkewReport, str]:
    skew = skew_check(served, features[["txn_id", *MODEL_FEATURES]])
    current = features[features["ts"] >= current_start_ts]
    rows_period = drift_report(bundle.reference, current)
    rows_served = drift_report(bundle.reference, served)
    md = "\n\n".join(
        [
            drift_markdown(
                rows_period,
                "Training reference vs. the last 7 days (offline features)",
            ),
            drift_markdown(
                rows_served,
                "Training reference vs. one replayed day of online traffic (served features)",
            ),
        ]
    )
    return skew, md


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default, allow_nan=True) + "\n")


def _default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    raise TypeError(type(o))


def online_store_from_url(url: str | None, prefix: str = "ffp") -> OnlineStore:
    """Real Redis when a URL is given, otherwise an in-process fakeredis (no-Docker demo)."""
    if url:
        import redis

        client = redis.Redis.from_url(url)
        client.ping()
        return OnlineStore(client, prefix)
    try:
        import fakeredis
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "No --redis-url given and fakeredis is not installed (pip install fakeredis)"
        ) from exc
    return OnlineStore(fakeredis.FakeRedis(), prefix)
