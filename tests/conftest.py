from __future__ import annotations

import pandas as pd
import pytest

from ffp.generator import Dataset, GeneratorConfig, generate
from ffp.offline import backfill, merchant_rate_snapshots

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
