"""Command line entry point: ``ffp <command>``. Run ``ffp -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ffp.generator import Dataset, GeneratorConfig
from ffp.schema import DAY


def _gen_cfg(a: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(seed=a.seed, n_cards=a.cards, n_merchants=a.merchants, days=a.days)


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cards", type=int, default=3000)
    p.add_argument("--merchants", type=int, default=500)
    p.add_argument("--days", type=int, default=120)


def cmd_generate(a: argparse.Namespace) -> None:
    from ffp.generator import generate

    ds = generate(_gen_cfg(a))
    ds.write(a.out)
    n_f = int(ds.ground_truth["is_fraud"].sum())
    print(
        f"wrote {len(ds.transactions):,} transactions ({n_f:,} fraud, {len(ds.labels):,} reported) to {a.out}"
    )


def cmd_demo(a: argparse.Namespace) -> None:
    """Everything, no Docker: generate -> backfill -> train -> replay online -> skew + drift."""
    from ffp.monitoring import skew_check
    from ffp.pipeline import (
        monitoring_reports,
        online_store_from_url,
        replay,
        run_offline,
        write_json,
    )
    from ffp.serving import ScoringService

    work = Path(a.workdir)
    reports = Path(a.reports)
    res = run_offline(_gen_cfg(a), work / "raw", work / "offline", work / "model")
    m = res.metrics
    print(f"[offline] {len(res.features):,} feature rows; test PR-AUC {m['pr_auc_raw']:.3f} (raw)")

    store = online_store_from_url(a.redis_url, prefix=a.prefix)
    start_ts = int(res.dataset.transactions["ts"].min()) // DAY * DAY
    r_start = start_ts + a.replay_day * DAY
    r_end = r_start + int(a.replay_hours * 3600_000)
    scenarios = [
        ("in_sync", "Processor in sync with the scorer", 0, True),
        ("no_last_seen_bootstrap", "Store started without bootstrapping last-seen", 0, False),
        (
            f"processor_lag_{a.lag_events}",
            f"Processor lagging {a.lag_events} events behind",
            a.lag_events,
            True,
        ),
    ]
    skew_json: dict[str, object] = {}
    skew_md = ["# Train/serve skew", ""]
    replay_json: dict[str, object] = {}
    for key, title, lag, boot in scenarios:
        store.flush()
        service = ScoringService(store, res.bundle)
        rep = replay(
            service,
            res.dataset,
            res.snapshots,
            start_ts=r_start,
            end_ts=r_end,
            processor_lag_events=lag,
            bootstrap_last_seen=boot,
        )
        skew = skew_check(rep.served, res.features)
        status = "OK" if skew.ok else "SKEW DETECTED"
        print(f"[online:{key}] scored {rep.scored:,} events in {rep.seconds:.1f}s -> skew {status}")
        skew_json[key] = {
            "compared": skew.compared_events,
            "ok": skew.ok,
            "mismatches": {f.feature: f.mismatches for f in skew.features if f.mismatches},
        }
        skew_md += [f"## {title}", "", skew.to_markdown(), ""]
        if key == "in_sync":
            test_from = start_ts + int(res.bundle.metadata["train_config"]["as_of_day"]) * DAY
            _, drift_md = monitoring_reports(
                res.bundle, res.features, rep.served, current_start_ts=test_from
            )
            replay_json = {
                "start_day": a.replay_day,
                "hours": a.replay_hours,
                "scored": rep.scored,
                "warmup_events": rep.warmup_events,
                "seconds": rep.seconds,
                "feature_ms_p50": rep.feature_ms_p50,
                "feature_ms_p99": rep.feature_ms_p99,
                "redis": "real" if a.redis_url else "fakeredis",
            }

    reports.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "generator": asdict(_gen_cfg(a)),
            "model": res.bundle.metadata,
            "test_metrics": m,
            "leakage_study": res.leakage,
            "replay": replay_json,
            "skew": skew_json,
            "timings_s": res.timings_s,
        },
        reports / "results.json",
    )
    (reports / "skew.md").write_text("\n".join(skew_md))
    (reports / "drift.md").write_text("# Feature drift (PSI)\n\n" + drift_md + "\n")
    print(f"[reports] {reports}/results.json, skew.md, drift.md")
    if a.fail_on_skew and not skew_json["in_sync"]["ok"]:  # type: ignore[index]
        sys.exit("in-sync replay shows train/serve skew")


def cmd_build(a: argparse.Namespace) -> None:
    """generate -> backfill -> train, without the online replay (used by docker compose)."""
    from ffp.pipeline import run_offline

    work = Path(a.workdir)
    res = run_offline(
        _gen_cfg(a), work / "raw", work / "offline", work / "model", with_leakage_study=False
    )
    print(
        f"built {work}: {len(res.features):,} feature rows, test PR-AUC {res.metrics['pr_auc_raw']:.3f}"
    )


def cmd_load_online(a: argparse.Namespace) -> None:
    """Load card profiles, merchant-rate snapshots and last-seen times into Redis.

    This is the batch side of the online store. ``--as-of-day`` is where the stream
    will start; last-seen is bootstrapped from events older than its 25h warm-up.
    """
    from ffp.features import MAX_WINDOW_MS
    from ffp.pipeline import online_store_from_url

    store = online_store_from_url(a.redis_url, prefix=a.prefix)
    ds = Dataset.read(a.raw)
    store.load_card_profiles(ds.cards)
    n = store.load_merchant_rate_snapshots(
        pd.read_parquet(Path(a.offline) / "merchant_rate_snapshots.parquet")
    )
    tx = ds.transactions
    start = int(tx["ts"].min()) // DAY * DAY
    cutoff = start + int(a.as_of_day * DAY) - MAX_WINDOW_MS - 3600_000
    store.load_card_last_seen(tx[tx["ts"] < cutoff].groupby("card_id")["ts"].max())
    print(f"loaded {len(ds.cards):,} card profiles, {n} merchant-rate snapshots, last-seen times")


def cmd_produce(a: argparse.Namespace) -> None:
    from ffp.stream import publish, transactions_from_frame

    tx = pd.read_parquet(Path(a.raw) / "transactions.parquet")
    start = int(tx["ts"].min()) // DAY * DAY
    sel = tx[(tx["ts"] >= start + int(a.from_day * DAY)) & (tx["ts"] < start + int(a.to_day * DAY))]
    n = publish(transactions_from_frame(sel), a.bootstrap, a.topic)
    print(f"produced {n:,} events to {a.topic}")


def cmd_process(a: argparse.Namespace) -> None:
    from ffp.pipeline import online_store_from_url
    from ffp.stream import KafkaSource, StreamProcessor

    store = online_store_from_url(a.redis_url, prefix=a.prefix)
    src = KafkaSource(a.bootstrap, a.topic, a.group, idle_timeout_s=a.idle_timeout)
    try:
        stats = StreamProcessor(store, batch_size=a.batch_size).run(src, max_events=a.max_events)
    finally:
        src.close()
    print(
        f"processed {stats.events:,} events in {stats.seconds:.1f}s ({stats.events_per_s:,.0f}/s)"
    )


def cmd_replay(a: argparse.Namespace) -> None:
    """Compose demo: score each event via the HTTP API, then publish it to Kafka.

    The stream processor consumes asynchronously, so if it falls behind, the API
    scores with slightly stale state. ``ffp monitor`` measures exactly that.
    """
    import httpx

    from ffp.stream import KafkaPublisher, publish, transactions_from_frame

    tx = pd.read_parquet(Path(a.raw) / "transactions.parquet")
    start = int(tx["ts"].min()) // DAY * DAY
    lo = start + int(a.from_day * DAY)
    warm = tx[(tx["ts"] >= lo - DAY - 3600_000) & (tx["ts"] < lo)]
    publish(transactions_from_frame(warm), a.bootstrap, a.topic)
    if a.redis_url and len(warm):
        _wait_for_processor(a.redis_url, a.prefix, warm.tail(100), timeout_s=a.warmup_wait)
    else:
        time.sleep(a.warmup_wait)
    sel = tx[(tx["ts"] >= lo) & (tx["ts"] < start + int(a.to_day * DAY))]
    pub = KafkaPublisher(a.bootstrap, a.topic)
    blocked = 0
    with httpx.Client(base_url=a.api, timeout=10) as client:
        for txn in transactions_from_frame(sel):
            r = client.post("/score", json=txn.to_dict())
            r.raise_for_status()
            blocked += r.json()["decision"] == "block"
            pub.send(txn)
            if a.pace_ms:
                time.sleep(a.pace_ms / 1000)
    pub.flush()
    print(f"warm-up {len(warm):,} events; scored {len(sel):,}; blocked {blocked:,}")


def _wait_for_processor(redis_url: str, prefix: str, tail: pd.DataFrame, timeout_s: float) -> None:
    """Block until the stream processor has applied the given (last) warm-up events.

    Partitions are consumed independently, so one event landing proves nothing about
    the others; the tail of the warm-up spans every partition.
    """
    from ffp.pipeline import online_store_from_url

    store = online_store_from_url(redis_url, prefix=prefix)
    t0 = time.monotonic()
    pending = list(tail[["txn_id", "card_id", "ts"]].itertuples(index=False, name=None))
    while time.monotonic() - t0 < timeout_s:
        pending = [
            (txn_id, card_id, ts)
            for txn_id, card_id, ts in pending
            if not any(
                str(m.decode() if isinstance(m, bytes) else m).startswith(f"{txn_id}|")
                for m in store.r.zrangebyscore(store._card_key(str(card_id)), int(ts), int(ts))
            )
        ]
        if not pending:
            print(f"processor caught up with warm-up after {time.monotonic() - t0:.1f}s")
            return
        time.sleep(0.2)
    sys.exit("timed out waiting for the stream processor to apply the warm-up events")


def cmd_monitor(a: argparse.Namespace) -> None:
    """Skew (feature log vs offline recomputation) + drift report from a live Redis."""
    from ffp.model import ModelBundle
    from ffp.monitoring import drift_markdown, drift_report, skew_check
    from ffp.pipeline import online_store_from_url
    from ffp.serving import ScoringService

    store = online_store_from_url(a.redis_url, prefix=a.prefix)
    bundle = ModelBundle.load(a.model)
    served = ScoringService(store, bundle).feature_log()
    if served.empty:
        sys.exit("feature log is empty: score some traffic first")
    offline = pd.read_parquet(Path(a.offline) / "features.parquet")
    skew = skew_check(served, offline, sample=a.sample)
    print(skew.to_markdown())
    print()
    print(
        drift_markdown(
            drift_report(bundle.reference, served), "Training reference vs. served features"
        )
    )
    if a.fail_on_skew and not skew.ok:
        sys.exit(1)


def cmd_backfill(a: argparse.Namespace) -> None:
    from ffp.offline import backfill, merchant_rate_snapshots, write_offline_store

    ds = Dataset.read(a.raw)
    snaps = merchant_rate_snapshots(ds.transactions, ds.labels, ds.merchants["merchant_id"])
    feats = backfill(ds.transactions, ds.cards, ds.labels, snapshots=snaps)
    write_offline_store(feats, snaps, a.out)
    print(f"wrote {len(feats):,} feature rows to {a.out}")


def cmd_train(a: argparse.Namespace) -> None:
    from ffp.model import TrainConfig, evaluate, fit, make_splits

    ds = Dataset.read(a.raw)
    feats = pd.read_parquet(Path(a.offline) / "features.parquet")
    start = int(ds.transactions["ts"].min()) // DAY * DAY
    days = int((int(ds.transactions["ts"].max()) - start) // DAY) + 1
    cfg = TrainConfig.for_days(start, days)
    splits = make_splits(feats, ds.labels, cfg)
    bundle = fit(splits, cfg)
    bundle.save(a.model)
    print(json.dumps(evaluate(bundle, splits.test, ds.ground_truth), indent=2, default=str))


def cmd_serve(a: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ.setdefault("REDIS_URL", a.redis_url)
    os.environ.setdefault("FFP_MODEL_DIR", a.model)
    uvicorn.run(
        "ffp.serving:app_from_env", factory=True, host=a.host, port=a.port, log_level="warning"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ffp", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="write synthetic Parquet data")
    _add_gen_args(p)
    p.add_argument("--out", default="data/raw")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("backfill", help="offline point-in-time features -> Parquet")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/offline")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("train", help="train + evaluate the model")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--offline", default="data/offline")
    p.add_argument("--model", default="data/model")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("demo", help="full pipeline without Docker (fakeredis unless --redis-url)")
    _add_gen_args(p)
    p.add_argument("--workdir", default="data")
    p.add_argument("--reports", default="reports")
    p.add_argument("--redis-url", default=None)
    p.add_argument("--prefix", default="ffp-demo")
    p.add_argument("--replay-day", type=int, default=110)
    p.add_argument("--replay-hours", type=float, default=24)
    p.add_argument("--lag-events", type=int, default=50)
    p.add_argument(
        "--fail-on-skew", action="store_true", help="exit 1 if the in-sync replay shows skew"
    )
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("build", help="generate + backfill + train (no replay)")
    _add_gen_args(p)
    p.add_argument("--workdir", default="data")
    p.set_defaults(fn=cmd_build)

    for name, fn, help_ in [
        ("load-online", cmd_load_online, "load profiles, merchant snapshots, last-seen into Redis"),
        ("monitor", cmd_monitor, "skew + drift report from the live feature log"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--raw", default="data/raw")
        p.add_argument("--offline", default="data/offline")
        p.add_argument("--model", default="data/model")
        p.add_argument("--redis-url", default="redis://localhost:6379/0")
        p.add_argument("--prefix", default="ffp")
        p.add_argument("--sample", type=int, default=5000)
        p.add_argument("--fail-on-skew", action="store_true")
        p.add_argument("--as-of-day", type=float, default=110)
        p.set_defaults(fn=fn)

    p = sub.add_parser("produce", help="publish transactions to Kafka")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--from-day", type=float, default=0)
    p.add_argument("--to-day", type=float, default=1)
    p.set_defaults(fn=cmd_produce)

    p = sub.add_parser("process", help="Kafka -> Redis stream processor")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--group", default="ffp-feature-processor")
    p.add_argument("--redis-url", default="redis://localhost:6379/0")
    p.add_argument("--prefix", default="ffp")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--idle-timeout", type=float, default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("replay", help="score via the HTTP API, then publish to Kafka")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--from-day", type=float, default=110)
    p.add_argument("--to-day", type=float, default=110.25)
    p.add_argument(
        "--pace-ms",
        type=float,
        default=150,
        help="sleep between events; lower it to watch processor lag show up as skew",
    )
    p.add_argument("--warmup-wait", type=float, default=60, help="max seconds for warm-up to land")
    p.add_argument("--redis-url", default=None, help="wait on Redis until warm-up is applied")
    p.add_argument("--prefix", default="ffp")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("serve", help="run the FastAPI scoring service")
    p.add_argument("--model", default="data/model")
    p.add_argument("--redis-url", default="redis://localhost:6379/0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
