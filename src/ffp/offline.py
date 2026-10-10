"""Offline path: point-in-time-correct feature backfill over Parquet with DuckDB.

Three things make this point-in-time correct, and ``tests/test_point_in_time.py``
checks each one:

1. Window features use RANGE frames ``[t - w, t)`` with ``EXCLUDE GROUP``, so the
   event itself and anything at the same or a later timestamp is never included.
2. The merchant chargeback rate is a daily snapshot built only from labels whose
   ``label_ts`` is at or before the snapshot time, and it is attached with an
   ``ASOF JOIN`` that picks the latest snapshot at or before each event.
3. Training labels are what was known on the training date (``label_training_frame``),
   not the eventual ground truth.

``leaky=True`` reproduces the usual mistake (a merchant fraud rate computed from all
labels, including future ones) so its effect on the metrics can be measured.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ffp.features import (
    MERCHANT_RATE_FEATURE,
    MERCHANT_RATE_WINDOW_MS,
    MODEL_FEATURES,
    RECENCY_FEATURE,
    WINDOW_FEATURES,
    WindowAgg,
    request_features,
    smoothed_rate,
)
from ffp.schema import DAY, SECOND

_PARTITION = {"card": "card_id", "merchant": "merchant_id"}


def _window_sql(f: WindowAgg) -> str:
    cond = f"amount < {f.amount_lt!r}" if f.amount_lt is not None else "TRUE"
    if f.agg == "count":
        agg = f"sum(CASE WHEN {cond} THEN 1 ELSE 0 END)"
    elif f.agg == "sum":
        agg = f"sum(CASE WHEN {cond} THEN amount ELSE 0 END)"
    else:
        agg = f"count(DISTINCT CASE WHEN {cond} THEN {f.column} END)"
    frame = (
        f"PARTITION BY {_PARTITION[f.entity]} ORDER BY ts "
        f"RANGE BETWEEN {f.window_ms} PRECEDING AND CURRENT ROW EXCLUDE GROUP"
    )
    expr = f"coalesce({agg} OVER ({frame}), 0)"
    if f.agg == "sum":
        expr = f"round({expr}, 6)"
    return f"CAST({expr} AS DOUBLE) AS {f.name}"


def window_features_sql(table: str = "txns") -> str:
    """The generated DuckDB query. Exposed so the README/tests can show it."""
    cols = ",\n  ".join(_window_sql(f) for f in WINDOW_FEATURES)
    recency = (
        f"(ts - max(ts) OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN UNBOUNDED "
        f"PRECEDING AND CURRENT ROW EXCLUDE GROUP)) / {SECOND}.0 AS {RECENCY_FEATURE}"
    )
    return f"SELECT txn_id,\n  {cols},\n  {recency}\nFROM {table}"


def merchant_rate_snapshots(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    merchants: pd.Series | None = None,
    *,
    start_ts: int | None = None,
    n_days: int | None = None,
) -> pd.DataFrame:
    """Daily ``merchant_fraud_rate_30d`` snapshots: (snapshot_ts, merchant_id, value).

    The snapshot at day boundary ``D`` covers events in ``[D - 30d, D)`` and counts
    only fraud labels with ``label_ts <= D``. The table is dense (every merchant,
    every day) so an ASOF join can never pick up a stale row.
    """
    if start_ts is None:
        start_ts = int(transactions["ts"].min()) // DAY * DAY
    if n_days is None:
        n_days = int((int(transactions["ts"].max()) - start_ts) // DAY) + 1
    m_ids = (
        pd.Index(sorted(merchants.unique()))
        if merchants is not None
        else pd.Index(sorted(transactions["merchant_id"].unique()))
    )
    m_pos = pd.Series(np.arange(len(m_ids)), index=m_ids)
    window_days = MERCHANT_RATE_WINDOW_MS // DAY

    day = ((transactions["ts"].to_numpy() - start_ts) // DAY).astype(np.int64)
    mi = m_pos.reindex(transactions["merchant_id"]).to_numpy()
    ok = ~np.isnan(mi) & (day >= 0) & (day < n_days)
    counts = np.zeros((len(m_ids), n_days + 1))
    np.add.at(counts, (mi[ok].astype(np.int64), day[ok]), 1.0)
    csum = np.concatenate([np.zeros((len(m_ids), 1)), np.cumsum(counts, axis=1)], axis=1)

    lab = labels[labels["is_fraud"] == 1].merge(
        transactions[["txn_id", "ts", "merchant_id"]], on="txn_id", how="inner"
    )
    l_day = ((lab["ts"].to_numpy() - start_ts) // DAY).astype(np.int64)
    l_mi = m_pos.reindex(lab["merchant_id"]).to_numpy()
    l_known = lab["label_ts"].to_numpy()

    frames = []
    for d in range(n_days + 1):
        snap = start_ts + d * DAY
        lo = max(0, d - window_days)
        n = csum[:, d] - csum[:, lo]
        sel = (l_day >= lo) & (l_day < d) & (l_known <= snap) & ~np.isnan(l_mi)
        fraud = np.bincount(l_mi[sel].astype(np.int64), minlength=len(m_ids)).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "snapshot_ts": np.int64(snap),
                    "merchant_id": m_ids,
                    "value": [smoothed_rate(f, c) for f, c in zip(fraud, n, strict=True)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def leaky_merchant_rate(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The classic mistake: one fraud rate per merchant from *all* labels, past and future."""
    lab = labels[labels["is_fraud"] == 1][["txn_id"]].assign(fraud=1)
    t = transactions[["txn_id", "merchant_id"]].merge(lab, on="txn_id", how="left").fillna(0)
    g = t.groupby("merchant_id").agg(fraud=("fraud", "sum"), n=("fraud", "size")).reset_index()
    g["value"] = [smoothed_rate(f, n) for f, n in zip(g["fraud"], g["n"], strict=True)]
    return g[["merchant_id", "value"]]


def backfill(
    transactions: pd.DataFrame,
    cards: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    snapshots: pd.DataFrame | None = None,
    leaky: bool = False,
) -> pd.DataFrame:
    """Compute every model feature for every transaction, as of that transaction's time."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    try:
        con.register("txns", transactions)
        win = con.execute(window_features_sql("txns")).df()
        if leaky:
            rate = leaky_merchant_rate(transactions, labels)
            con.register("rate", rate)
            mr = con.execute(
                f"SELECT t.txn_id, coalesce(r.value, NULL) AS {MERCHANT_RATE_FEATURE} "
                "FROM txns t LEFT JOIN rate r USING (merchant_id)"
            ).df()
        else:
            snaps = (
                snapshots
                if snapshots is not None
                else merchant_rate_snapshots(transactions, labels)
            )
            con.register("snaps", snaps)
            mr = con.execute(
                f"SELECT t.txn_id, s.value AS {MERCHANT_RATE_FEATURE} "
                "FROM txns t ASOF LEFT JOIN snaps s "
                "ON t.merchant_id = s.merchant_id AND t.ts >= s.snapshot_ts"
            ).df()
    finally:
        con.close()

    df = transactions.merge(win, on="txn_id").merge(mr, on="txn_id")
    df = df.merge(cards[["card_id", "home_country"]], on="card_id", how="left")
    home = df["home_country"].astype(object).where(df["home_country"].notna(), None)
    req = [
        request_features(
            ts=ts,
            amount=a,
            country=c,
            channel=ch,
            category=cat,
            home_country=h,
            card_count_24h=n24,
            card_sum_24h=s24,
        )
        for ts, a, c, ch, cat, h, n24, s24 in zip(
            df["ts"].to_numpy(),
            df["amount"].to_numpy(),
            df["country"].to_numpy(),
            df["channel"].to_numpy(),
            df["category"].to_numpy(),
            home.to_numpy(),
            df["card_txn_count_24h"].to_numpy(),
            df["card_amount_sum_24h"].to_numpy(),
            strict=True,
        )
    ]
    req_df = pd.DataFrame(req, index=df.index).drop(columns=["amount"])
    df = pd.concat([df.drop(columns=["home_country"]), req_df], axis=1)
    df = df.sort_values(["ts", "txn_id"]).reset_index(drop=True)
    keys = ["txn_id", "ts", "card_id", "merchant_id"]
    return df[keys + list(MODEL_FEATURES)]


def label_training_frame(
    features: pd.DataFrame, labels: pd.DataFrame, *, as_of_ts: int, maturity_ms: int
) -> pd.DataFrame:
    """Attach labels *as they were known at ``as_of_ts``*.

    * fraud: a fraud label with ``label_ts <= as_of_ts`` exists;
    * legit: no such label and the event is at least ``maturity_ms`` old;
    * otherwise the outcome is still unknown, so the row is dropped.
    """
    known = labels[(labels["is_fraud"] == 1) & (labels["label_ts"] <= as_of_ts)]
    df = features[features["ts"] <= as_of_ts].copy()
    df["label"] = df["txn_id"].isin(known["txn_id"]).astype(np.int8)
    mature = df["ts"] <= as_of_ts - maturity_ms
    return df[(df["label"] == 1) | mature].reset_index(drop=True)


def eventual_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Labels as eventually reported (used only to evaluate the held-out period)."""
    df = features.copy()
    df["label"] = df["txn_id"].isin(labels.loc[labels["is_fraud"] == 1, "txn_id"]).astype(np.int8)
    return df


def write_offline_store(features: pd.DataFrame, snapshots: pd.DataFrame, out: str | Path) -> None:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    features.to_parquet(p / "features.parquet", index=False)
    snapshots.to_parquet(p / "merchant_rate_snapshots.parquet", index=False)
