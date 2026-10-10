"""Point-in-time correctness of the offline store: no future events, no future labels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ffp.features import (
    MERCHANT_RATE_FEATURE,
    MODEL_FEATURES,
    RECENCY_FEATURE,
    WINDOW_FEATURES,
    smoothed_rate,
)
from ffp.generator import Dataset
from ffp.offline import backfill, label_training_frame, merchant_rate_snapshots
from ffp.schema import DAY, EPOCH_START_MS, HOUR

_KEY = {"card": "card_id", "merchant": "merchant_id"}


def _brute_force(t: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    """Independent, obviously-correct implementation of the window contract."""
    out: dict[str, float] = {}
    for f in WINDOW_FEATURES:
        key = _KEY[f.entity]
        w = t[(t[key] == row[key]) & (t["ts"] >= row["ts"] - f.window_ms) & (t["ts"] < row["ts"])]
        if f.amount_lt is not None:
            w = w[w["amount"] < f.amount_lt]
        if f.agg == "count":
            out[f.name] = float(len(w))
        elif f.agg == "sum":
            out[f.name] = float(w["amount"].sum())
        else:
            out[f.name] = float(w[f.column].nunique())
    prev = t[(t["card_id"] == row["card_id"]) & (t["ts"] < row["ts"])]["ts"]
    out[RECENCY_FEATURE] = (row["ts"] - prev.max()) / 1000 if len(prev) else math.nan
    return out


def test_window_features_match_brute_force(small: Dataset, small_features: pd.DataFrame) -> None:
    t = small.transactions
    # Bias the sample toward busy moments (fraud bursts) where windows are non-trivial.
    busy = small_features[small_features["card_txn_count_1h"] > 1]
    sample = pd.concat([small_features.sample(150, random_state=0), busy.head(150)])
    for _, feat in sample.iterrows():
        row = t[t["txn_id"] == feat["txn_id"]].iloc[0]
        ref = _brute_force(t, row)
        for name, expected in ref.items():
            got = feat[name]
            if math.isnan(expected):
                assert math.isnan(got), (name, feat["txn_id"])
            else:
                assert got == pytest.approx(expected, abs=1e-6), (name, feat["txn_id"])


def test_changing_the_future_does_not_change_past_features(small: Dataset) -> None:
    t, labels = small.transactions, small.labels
    cutoff = EPOCH_START_MS + 12 * DAY
    base = backfill(t, small.cards, labels)

    future = t["ts"] >= cutoff
    rng = np.random.default_rng(0)
    t2 = t.copy()
    t2.loc[future, "amount"] = t2.loc[future, "amount"] * 7
    t2.loc[future, "merchant_id"] = rng.permutation(t2.loc[future, "merchant_id"].to_numpy())
    # Extra events exactly at the cutoff and a burst right after it.
    extra = t2[~future].tail(50).copy()
    extra["txn_id"] = [f"x{i}" for i in range(len(extra))]
    extra["ts"] = cutoff + np.arange(len(extra))
    t2 = pd.concat([t2, extra]).sort_values("ts").reset_index(drop=True)
    # Every past transaction gets a fraud label that only arrives after the cutoff.
    past_ids = t.loc[~future, "txn_id"]
    late = pd.DataFrame({"txn_id": past_ids, "label_ts": cutoff + 1, "is_fraud": 1})
    labels2 = pd.concat([labels[labels["label_ts"] <= cutoff], late], ignore_index=True)

    changed = backfill(t2, small.cards, labels2)
    a = base[base["ts"] < cutoff].set_index("txn_id")[list(MODEL_FEATURES)]
    b = changed.set_index("txn_id").loc[a.index, list(MODEL_FEATURES)]
    pd.testing.assert_frame_equal(a, b)


def _tiny(rows: list[tuple[str, int, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "txn_id": [r[0] for r in rows],
            "ts": np.array([r[1] for r in rows], dtype=np.int64),
            "card_id": [r[2] for r in rows],
            "merchant_id": [r[3] for r in rows],
            "amount": [r[4] for r in rows],
            "country": "IN",
            "channel": "online",
            "category": "retail",
        }
    )


CARDS = pd.DataFrame({"card_id": ["c1", "c2"], "home_country": ["IN", "IN"]})
NO_LABELS = pd.DataFrame(
    {"txn_id": pd.Series([], dtype=str), "label_ts": pd.Series([], dtype=np.int64), "is_fraud": 1}
)


def test_event_and_simultaneous_events_are_excluded() -> None:
    t0 = EPOCH_START_MS + DAY
    t = _tiny(
        [("a", t0, "c1", "m1", 1.0), ("b", t0, "c1", "m2", 2.0), ("c", t0 + 1, "c1", "m1", 4.0)]
    )
    f = backfill(t, CARDS, NO_LABELS).set_index("txn_id")
    assert f.loc["a", "card_txn_count_1h"] == 0
    assert f.loc["b", "card_txn_count_1h"] == 0
    assert f.loc["c", "card_txn_count_1h"] == 2
    assert f.loc["c", "card_amount_sum_1h"] == 3.0
    assert f.loc["c", "card_distinct_merchants_1h"] == 2
    assert f.loc["c", RECENCY_FEATURE] == 0.001


def test_window_lower_bound_is_inclusive() -> None:
    t0 = EPOCH_START_MS + DAY
    t = _tiny(
        [
            ("a", t0, "c1", "m1", 1.0),
            ("b", t0 + HOUR, "c1", "m1", 1.0),
            ("c", t0 + HOUR + 1, "c1", "m1", 1.0),
        ]
    )
    f = backfill(t, CARDS, NO_LABELS).set_index("txn_id")
    assert f.loc["b", "card_txn_count_1h"] == 1  # a is exactly 1h earlier: included
    assert f.loc["c", "card_txn_count_1h"] == 1  # a is 1h + 1ms earlier: excluded


def test_merchant_rate_only_uses_labels_known_at_snapshot_time() -> None:
    d = EPOCH_START_MS
    t = _tiny(
        [
            ("fraud", d + 1 * DAY + 5, "c1", "m1", 50.0),
            ("before_label", d + 3 * DAY + 5, "c2", "m1", 10.0),
            ("after_label", d + 6 * DAY + 5, "c2", "m1", 10.0),
        ]
    )
    labels = pd.DataFrame({"txn_id": ["fraud"], "label_ts": [d + 5 * DAY + 7], "is_fraud": [1]})
    f = backfill(t, CARDS, labels).set_index("txn_id")
    # Label arrives during day 5, so only the day-6 snapshot can see it.
    assert f.loc["before_label", MERCHANT_RATE_FEATURE] == pytest.approx(smoothed_rate(0, 1))
    assert f.loc["after_label", MERCHANT_RATE_FEATURE] == pytest.approx(smoothed_rate(1, 2))

    leaky = backfill(t, CARDS, labels, leaky=True).set_index("txn_id")
    assert leaky.loc["fraud", MERCHANT_RATE_FEATURE] == pytest.approx(smoothed_rate(1, 3))


def test_snapshots_are_dense(small: Dataset, small_snapshots: pd.DataFrame) -> None:
    per_day = small_snapshots.groupby("snapshot_ts")["merchant_id"].nunique()
    assert (per_day == small.merchants["merchant_id"].nunique()).all()


def test_snapshot_ignores_labels_from_the_future(small: Dataset) -> None:
    snaps = merchant_rate_snapshots(small.transactions, small.labels)
    no_future = small.labels.copy()
    no_future["label_ts"] = np.iinfo(np.int64).max
    blind = merchant_rate_snapshots(small.transactions, no_future)
    assert (snaps["value"] >= blind["value"] - 1e-12).all()
    assert (snaps["value"] > blind["value"] + 1e-12).any()


def test_training_labels_are_as_known_at_training_time() -> None:
    d = EPOCH_START_MS
    feats = pd.DataFrame(
        {
            "txn_id": ["old_legit", "old_fraud_known", "old_fraud_late", "recent", "future"],
            "ts": [d, d, d, d + 9 * DAY, d + 11 * DAY],
        }
    )
    labels = pd.DataFrame(
        {
            "txn_id": ["old_fraud_known", "old_fraud_late"],
            "label_ts": [d + 2 * DAY, d + 20 * DAY],
            "is_fraud": [1, 1],
        }
    )
    out = label_training_frame(feats, labels, as_of_ts=d + 10 * DAY, maturity_ms=5 * DAY)
    got = dict(zip(out["txn_id"], out["label"], strict=True))
    # The late label is not known yet, so that fraud is (wrongly but honestly) a negative.
    assert got == {"old_legit": 0, "old_fraud_known": 1, "old_fraud_late": 0}
