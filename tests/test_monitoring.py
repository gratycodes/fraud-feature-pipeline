from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffp.features import MODEL_FEATURES
from ffp.generator import Dataset
from ffp.monitoring import build_reference, drift_report, psi, reference_bins, skew_check
from ffp.online import OnlineStore
from ffp.pipeline import replay
from ffp.schema import DAY, EPOCH_START_MS
from ffp.serving import ScoringService


def _frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in MODEL_FEATURES})
    df["category_code"] = rng.integers(0, 5, n).astype(float)
    df.insert(0, "txn_id", [f"t{i}" for i in range(n)])
    return df


def test_identical_values_pass() -> None:
    df = _frame()
    rep = skew_check(df, df.copy())
    assert rep.ok and rep.compared_events == 1000 and not rep.skewed


def test_a_single_skewed_feature_is_named() -> None:
    served, offline = _frame(), _frame()
    served.loc[:19, "card_txn_count_1h"] += 1  # 2% of events
    rep = skew_check(served, offline)
    assert not rep.ok
    assert [f.feature for f in rep.skewed] == ["card_txn_count_1h"]
    assert rep.skewed[0].mismatches == 20
    assert "SKEW DETECTED" in rep.to_markdown()


def test_nan_versus_value_is_a_mismatch_but_nan_versus_nan_is_not() -> None:
    served, offline = _frame(10), _frame(10)
    served.loc[0, "amount"] = np.nan
    served.loc[1, "card_seconds_since_last_txn"] = np.nan
    offline.loc[1, "card_seconds_since_last_txn"] = np.nan
    rep = skew_check(served, offline, max_mismatch_rate=0.0)
    assert {f.feature: f.mismatches for f in rep.skewed} == {"amount": 1}


def test_missing_offline_rows_are_counted() -> None:
    served = _frame(10)
    rep = skew_check(served, served.iloc[:7])
    assert rep.missing_offline == 3 and rep.compared_events == 7


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = reference_bins(pd.Series(rng.normal(size=20_000)))
    assert psi(ref, pd.Series(rng.normal(size=20_000))) < 0.01
    assert psi(ref, pd.Series(rng.normal(1.0, 1.0, size=20_000))) > 0.25


def test_psi_counts_a_change_in_missing_values() -> None:
    rng = np.random.default_rng(0)
    ref = reference_bins(pd.Series(rng.normal(size=10_000)))
    cur = pd.Series(rng.normal(size=10_000))
    cur[:3000] = np.nan
    assert psi(ref, cur) > 0.25


def test_psi_categorical_with_an_unseen_category() -> None:
    ref = reference_bins(pd.Series([0.0, 1.0] * 500), categorical=True)
    assert psi(ref, pd.Series([0.0, 1.0] * 500)) < 1e-6
    assert psi(ref, pd.Series([0.0, 1.0, 7.0, 7.0] * 250)) > 0.25


def test_drift_report_sorts_by_psi() -> None:
    train, cur = _frame(5000, 0), _frame(5000, 1)
    cur["amount"] += 2
    rows = drift_report(build_reference(train), cur)
    assert rows[0].feature == "amount" and rows[0].status == "major"
    assert rows[-1].status == "stable"


def _replay(
    store: OnlineStore, small: Dataset, snapshots: pd.DataFrame, bundle, **kw
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    service = ScoringService(store, bundle)
    start = EPOCH_START_MS + 18 * DAY
    rep = replay(service, small, snapshots, start_ts=start, end_ts=start + DAY, **kw)
    return rep.served


def test_replay_in_sync_has_no_skew(
    store, small, small_snapshots, small_features, small_bundle
) -> None:  # type: ignore[no-untyped-def]
    served = _replay(store, small, small_snapshots, small_bundle[0])
    rep = skew_check(served, small_features)
    assert rep.compared_events > 100
    assert rep.ok, rep.to_markdown()


def test_processor_lag_is_detected(
    store, small, small_snapshots, small_features, small_bundle
) -> None:  # type: ignore[no-untyped-def]
    served = _replay(store, small, small_snapshots, small_bundle[0], processor_lag_events=30)
    rep = skew_check(served, small_features)
    assert not rep.ok
    assert "merchant_txn_count_1h" in {f.feature for f in rep.skewed}


def test_missing_last_seen_bootstrap_is_detected(
    store, small, small_snapshots, small_features, small_bundle
) -> None:  # type: ignore[no-untyped-def]
    served = _replay(store, small, small_snapshots, small_bundle[0], bootstrap_last_seen=False)
    rep = skew_check(served, small_features)
    assert [f.feature for f in rep.skewed] == ["card_seconds_since_last_txn"]


def test_a_timezone_bug_in_one_path_is_caught(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, store, small, small_snapshots, small_features, small_bundle
) -> None:
    """Classic skew: the serving path computes hour_of_day in local time (UTC+5:30)."""
    import ffp.online as online

    real = online.request_features

    def buggy(**kw):  # type: ignore[no-untyped-def]
        kw["ts"] = kw["ts"] + int(5.5 * 3600_000)
        return real(**kw)

    monkeypatch.setattr(online, "request_features", buggy)
    served = _replay(store, small, small_snapshots, small_bundle[0])
    rep = skew_check(served, small_features)
    assert [f.feature for f in rep.skewed] == ["hour_of_day"]
