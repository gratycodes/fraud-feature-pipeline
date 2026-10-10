from __future__ import annotations

import pandas as pd
from conftest import SMALL

from ffp.generator import Dataset, generate
from ffp.schema import CATEGORIES


def test_same_seed_same_data(small: Dataset) -> None:
    again = generate(SMALL)
    pd.testing.assert_frame_equal(small.transactions, again.transactions)
    pd.testing.assert_frame_equal(small.labels, again.labels)


def test_different_seed_different_data(small: Dataset) -> None:
    from dataclasses import replace

    other = generate(replace(SMALL, seed=12))
    assert not small.transactions["amount"].head(100).equals(other.transactions["amount"].head(100))


def test_events_carry_no_label(small: Dataset) -> None:
    assert not {"is_fraud", "label", "pattern"} & set(small.transactions.columns)


def test_schema_and_ordering(small: Dataset) -> None:
    t = small.transactions
    assert t["txn_id"].is_unique
    assert t["ts"].is_monotonic_increasing
    assert set(t["category"]) <= set(CATEGORIES)
    assert (t["amount"] > 0).all()
    assert set(t["channel"]) <= {"pos", "online"}


def test_all_fraud_patterns_present(small: Dataset) -> None:
    patterns = set(small.ground_truth["pattern"])
    assert {"legit", "card_testing", "account_takeover", "merchant_ring"} <= patterns


def test_labels_arrive_after_the_event_and_not_all_fraud_is_reported(small: Dataset) -> None:
    lab = small.labels.merge(small.transactions[["txn_id", "ts"]], on="txn_id")
    delay_days = (lab["label_ts"] - lab["ts"]) / 86_400_000
    assert (delay_days >= 1).all()
    assert delay_days.median() > 5
    n_fraud = int(small.ground_truth["is_fraud"].sum())
    assert 0 < len(small.labels) < n_fraud


def test_card_testing_is_a_burst_of_small_amounts(small: Dataset) -> None:
    t = small.transactions.merge(small.ground_truth, on="txn_id")
    ct = t[t["pattern"] == "card_testing"]
    assert (ct["amount"] < 5).mean() > 0.6
    assert (ct["channel"] == "online").all()
