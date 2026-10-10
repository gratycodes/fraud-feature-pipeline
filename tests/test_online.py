"""Online store + stream processor: parity with the offline backfill."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pandas as pd
import pytest

from ffp.features import MAX_WINDOW_MS, MODEL_FEATURES
from ffp.generator import Dataset
from ffp.online import OnlineStore
from ffp.schema import DAY, EPOCH_START_MS, Transaction
from ffp.stream import InProcessSource, StreamProcessor, transactions_from_frame


def _assert_same(online: dict[str, float], offline: pd.Series, txn_id: str) -> None:
    for name in MODEL_FEATURES:
        a, b = online[name], float(offline[name])
        if math.isnan(b):
            assert math.isnan(a), (txn_id, name, a, b)
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-6), (txn_id, name, a, b)


def test_online_features_equal_offline_backfill(
    store: OnlineStore,
    small: Dataset,
    small_features: pd.DataFrame,
    small_snapshots: pd.DataFrame,
) -> None:
    """Replay the stream; score each event *before* ingesting it, like production."""
    store.load_card_profiles(small.cards)
    store.load_merchant_rate_snapshots(small_snapshots)
    offline = small_features.set_index("txn_id")
    checked = 0
    for i, txn in enumerate(transactions_from_frame(small.transactions)):
        if i % 3 == 0:  # every third event keeps the test fast
            _assert_same(store.features_for(txn), offline.loc[txn.txn_id], txn.txn_id)
            checked += 1
        store.ingest([txn])
    assert checked > 1000


def test_processor_batches_and_commits(store: OnlineStore, small: Dataset) -> None:
    source = InProcessSource(transactions_from_frame(small.transactions.head(1234)))
    stats = StreamProcessor(store, batch_size=100).run(source)
    assert stats.events == 1234
    assert stats.batches == 13
    assert source.committed == 1234


def test_redelivery_is_a_no_op(store: OnlineStore, small: Dataset) -> None:
    events = list(transactions_from_frame(small.transactions.head(2000)))
    probe = events[-1]
    store.ingest(events[:-1])
    before = store.features_for(probe)
    store.ingest(events[1000:-1])  # at-least-once: a batch arrives twice
    after = store.features_for(probe)
    assert before.keys() == after.keys()
    for k in before:
        assert before[k] == after[k] or (math.isnan(before[k]) and math.isnan(after[k]))


def test_out_of_order_ingest_gives_the_same_state(store: OnlineStore, small: Dataset) -> None:
    events = list(transactions_from_frame(small.transactions.head(3000)))
    probe = events[-1]
    store.ingest(reversed(events[:-1]))
    shuffled = store.features_for(probe)
    store.flush()
    store.ingest(events[:-1])
    in_order = store.features_for(probe)
    for k in in_order:
        assert shuffled[k] == in_order[k] or (math.isnan(shuffled[k]) and math.isnan(in_order[k]))


def test_event_log_is_trimmed(store: OnlineStore) -> None:
    def ev(i: int, ts: int) -> Transaction:
        return Transaction(f"t{i}", ts, "c1", "m1", 1.0, "IN", "pos", "grocery")

    t0 = EPOCH_START_MS
    store.ingest([ev(0, t0), ev(1, t0 + DAY // 2)])
    store.ingest([ev(2, t0 + 3 * DAY)])
    assert store.r.zcard(store._card_key("c1")) == 1
    st = store.fetch("c1", "m1", t0 + 3 * DAY + 1)
    assert [e.txn_id for e in st.card_events] == ["t2"]
    # "seconds since last" still works past the 24h log via card_last.
    assert st.card_last_ts == t0 + 3 * DAY
    assert MAX_WINDOW_MS == DAY


def test_unknown_card_and_merchant(store: OnlineStore) -> None:
    txn = Transaction("t", EPOCH_START_MS, "nobody", "nowhere", 10.0, "IN", "pos", "grocery")
    f = store.features_for(txn)
    assert f["card_txn_count_24h"] == 0
    assert math.isnan(f["card_seconds_since_last_txn"])
    assert math.isnan(f["merchant_fraud_rate_30d"])
    assert f["is_foreign"] == 0.0


def _gen(n: int) -> Iterator[Transaction]:
    for i in range(n):
        yield Transaction(f"t{i}", EPOCH_START_MS + i, "c", "m", 1.0, "IN", "pos", "grocery")


def test_processor_respects_max_events(store: OnlineStore) -> None:
    stats = StreamProcessor(store, batch_size=7).run(InProcessSource(_gen(100)), max_events=30)
    assert stats.events == 30
