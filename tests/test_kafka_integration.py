"""Kafka -> processor -> Redis, end to end. Needs KAFKA_BOOTSTRAP and REDIS_URL."""

from __future__ import annotations

import os
import uuid

import pandas as pd
import pytest
import redis

from ffp.features import WINDOW_FEATURES
from ffp.generator import Dataset
from ffp.online import OnlineStore
from ffp.schema import HOUR
from ffp.stream import KafkaSource, StreamProcessor, publish, transactions_from_frame

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP")
REDIS_URL = os.environ.get("REDIS_URL")


@pytest.fixture
def kafka_store() -> OnlineStore:
    if not (BOOTSTRAP and REDIS_URL):
        if os.environ.get("FFP_REQUIRE_INTEGRATION"):
            pytest.fail("FFP_REQUIRE_INTEGRATION is set but KAFKA_BOOTSTRAP/REDIS_URL are not")
        pytest.skip("KAFKA_BOOTSTRAP and REDIS_URL not set")
    s = OnlineStore(redis.Redis.from_url(REDIS_URL), prefix=f"kit-{uuid.uuid4().hex[:8]}")
    yield s
    s.flush()


def _consume(store: OnlineStore, topic: str, group: str) -> int:
    assert BOOTSTRAP
    src = KafkaSource(BOOTSTRAP, topic, group, idle_timeout_s=8)
    try:
        return StreamProcessor(store, batch_size=200).run(src).events
    finally:
        src.close()


def _window_values(store: OnlineStore, probes: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        t.txn_id: {f.name: store.features_for(t)[f.name] for f in WINDOW_FEATURES}
        for t in transactions_from_frame(probes)
    }


def test_kafka_stream_builds_the_same_features_as_offline(
    kafka_store: OnlineStore, small: Dataset, small_features: pd.DataFrame
) -> None:
    assert BOOTSTRAP
    topic = f"ffp-it-{uuid.uuid4().hex[:8]}"
    tx = small.transactions
    sent = publish(transactions_from_frame(tx), BOOTSTRAP, topic)
    assert sent == len(tx)

    assert _consume(kafka_store, topic, f"g-{uuid.uuid4().hex[:6]}") == len(tx)

    # Events in the last hour still have their full 24h history in the trimmed log.
    probes = tx[tx["ts"] >= tx["ts"].max() - HOUR]
    assert len(probes) > 5
    online = _window_values(kafka_store, probes)
    offline = small_features.set_index("txn_id")
    for txn_id, vals in online.items():
        for name, v in vals.items():
            assert v == pytest.approx(float(offline.loc[txn_id, name]), abs=1e-6), (txn_id, name)

    # At-least-once: a second consumer group replays the whole topic. Nothing changes.
    assert _consume(kafka_store, topic, f"g-{uuid.uuid4().hex[:6]}") == len(tx)
    assert _window_values(kafka_store, probes) == online
