"""Latency benchmark for the scoring API against a real Redis.

Loads the online store (profiles, merchant snapshots, last-seen, 25h of events
before the benchmark window), starts uvicorn in a subprocess, then sends real
transactions from the held-out period:

* sequentially over one keep-alive connection (latency), and
* from N client threads (throughput and latency under concurrency).

Usage: python scripts/bench_api.py --redis-url redis://localhost:6379/0 --n 5000
Needs a trained model and data from `ffp demo` (defaults: data/, data/model).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from ffp.features import MAX_WINDOW_MS
from ffp.generator import Dataset
from ffp.pipeline import online_store_from_url
from ffp.schema import DAY, HOUR
from ffp.stream import InProcessSource, StreamProcessor, transactions_from_frame


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {
        k: round(float(np.percentile(a, q)), 3) for k, q in [("p50", 50), ("p95", 95), ("p99", 99)]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", required=True)
    ap.add_argument("--workdir", default="data")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--day", type=int, default=140)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="reports/latency.json")
    a = ap.parse_args()

    work = Path(a.workdir)
    ds = Dataset.read(work / "raw")
    snaps = pd.read_parquet(work / "offline" / "merchant_rate_snapshots.parquet")
    prefix = "ffp-bench"
    store = online_store_from_url(a.redis_url, prefix=prefix)
    store.flush()
    tx = ds.transactions
    start = int(tx["ts"].min()) // DAY * DAY + a.day * DAY
    warm_from = start - MAX_WINDOW_MS - HOUR
    store.load_card_profiles(ds.cards)
    store.load_merchant_rate_snapshots(snaps)
    store.load_card_last_seen(tx[tx["ts"] < warm_from].groupby("card_id")["ts"].max())
    warm = tx[(tx["ts"] >= warm_from) & (tx["ts"] < start)]
    StreamProcessor(store, 1000).run(InProcessSource(transactions_from_frame(warm)))
    reqs = [t.to_dict() for t in transactions_from_frame(tx[tx["ts"] >= start].head(a.n))]

    env = {
        **os.environ,
        "REDIS_URL": a.redis_url,
        "FFP_MODEL_DIR": str(work / "model"),
        "FFP_REDIS_PREFIX": prefix,
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ffp.serving:app_from_env",
            "--factory",
            "--port",
            str(a.port),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        env=env,
    )
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/healthz").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        with httpx.Client(base_url=base) as c:
            for r in reqs[:200]:  # warm-up: JIT-free, but warms caches and connections
                c.post("/score", json=r)
            client_ms, feat_ms, model_ms = [], [], []
            t0 = time.perf_counter()
            for r in reqs:
                s = time.perf_counter()
                body = c.post("/score", json=r).json()
                client_ms.append((time.perf_counter() - s) * 1e3)
                feat_ms.append(body["feature_ms"])
                model_ms.append(body["model_ms"])
            seq_s = time.perf_counter() - t0

        def worker(chunk: list[dict]) -> list[float]:
            out = []
            with httpx.Client(base_url=base) as c:
                for r in chunk:
                    s = time.perf_counter()
                    c.post("/score", json=r).raise_for_status()
                    out.append((time.perf_counter() - s) * 1e3)
            return out

        chunks = [reqs[i :: a.threads] for i in range(a.threads)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.threads) as ex:
            conc_ms = [x for part in ex.map(worker, chunks) for x in part]
        conc_s = time.perf_counter() - t0
    finally:
        proc.terminate()
        proc.wait(10)
        store.flush()

    result = {
        "requests": len(reqs),
        "machine": f"{platform.machine()}, {os.cpu_count()} vCPU, Python {platform.python_version()}",
        "server": "uvicorn, 1 worker; Redis on localhost",
        "sequential": {
            "client_ms": pct(client_ms),
            "server_feature_fetch_ms": pct(feat_ms),
            "server_model_ms": pct(model_ms),
            "rps": round(len(reqs) / seq_s, 1),
        },
        f"concurrent_{a.threads}_threads": {
            "client_ms": pct(conc_ms),
            "rps": round(len(reqs) / conc_s, 1),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
