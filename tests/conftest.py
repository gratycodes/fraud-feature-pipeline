from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import fakeredis
import pandas as pd
import pytest
import redis

from ffp.generator import Dataset, GeneratorConfig, generate
from ffp.offline import backfill, merchant_rate_snapshots
from ffp.online import OnlineStore

SMALL = GeneratorConfig(
    seed=11,
    n_cards=150,
    n_merchants=60,
    days=21,
    card_testing_episodes=12,
    ato_episodes=12,
    ring_merchants=1,
    ring_txns_per_merchant=40,
    drift_start_day=16,
)


@pytest.fixture(scope="session")
def small() -> Dataset:
    return generate(SMALL)


@pytest.fixture(scope="session")
def small_snapshots(small: Dataset) -> pd.DataFrame:
    return merchant_rate_snapshots(small.transactions, small.labels, small.merchants["merchant_id"])


@pytest.fixture(scope="session")
def small_features(small: Dataset, small_snapshots: pd.DataFrame) -> pd.DataFrame:
    return backfill(small.transactions, small.cards, small.labels, snapshots=small_snapshots)


_RUN = uuid.uuid4().hex[:8]


def _real_redis() -> redis.Redis:
    url = os.environ.get("REDIS_URL")
    if not url:
        if os.environ.get("FFP_REQUIRE_INTEGRATION"):
            pytest.fail("FFP_REQUIRE_INTEGRATION is set but REDIS_URL is not")
        pytest.skip("REDIS_URL not set")
    client = redis.Redis.from_url(url)
    client.ping()
    return client


@pytest.fixture(
    params=[
        pytest.param("fake", id="fakeredis"),
        pytest.param("real", id="redis", marks=pytest.mark.integration),
    ]
)
def redis_client(request: pytest.FixtureRequest) -> redis.Redis:
    if request.param == "fake":
        return fakeredis.FakeRedis()
    return _real_redis()


@pytest.fixture
def store(redis_client: redis.Redis) -> Iterator[OnlineStore]:
    s = OnlineStore(redis_client, prefix=f"test-{_RUN}-{uuid.uuid4().hex[:6]}")
    yield s
    s.flush()


SMALL_TRAIN = dict(
    train_start_day=3,
    train_end_day=10,
    early_stop_end_day=13,
    calib_end_day=16,
    as_of_day=18,
    test_end_day=21,
    maturity_days=2,
)


@pytest.fixture(scope="session")
def small_bundle(small: Dataset, small_features: pd.DataFrame):  # type: ignore[no-untyped-def]
    from ffp.model import TrainConfig, fit, make_splits
    from ffp.schema import EPOCH_START_MS

    cfg = TrainConfig(start_ts=EPOCH_START_MS, **SMALL_TRAIN)  # type: ignore[arg-type]
    splits = make_splits(small_features, small.labels, cfg)
    return fit(splits, cfg, rounds=200), splits, cfg
