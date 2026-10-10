from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffp.features import MODEL_FEATURES
from ffp.model import (
    Calibrator,
    CostModel,
    ModelBundle,
    best_threshold,
    bootstrap_pr_auc_ci,
    evaluate,
    expected_calibration_error,
)


def test_splits_are_time_ordered_and_disjoint(small_bundle) -> None:  # type: ignore[no-untyped-def]
    _, s, cfg = small_bundle
    assert s.train["ts"].max() < s.early_stop["ts"].min()
    assert s.early_stop["ts"].max() < s.calib["ts"].min()
    assert s.calib["ts"].max() < cfg.ts(cfg.calib_end_day)
    assert s.test["ts"].min() >= cfg.ts(cfg.as_of_day)
    ids = [set(df["txn_id"]) for df in (s.train, s.early_stop, s.calib, s.test)]
    assert sum(len(i) for i in ids) == len(set().union(*ids))
    assert s.train["label"].sum() > 0


def test_save_load_round_trip(small_bundle, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bundle, s, _ = small_bundle
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    x = s.test[list(MODEL_FEATURES)].to_numpy(dtype=float)
    np.testing.assert_allclose(bundle.predict_proba(x), loaded.predict_proba(x), rtol=1e-12)
    assert loaded.threshold == bundle.threshold
    assert loaded.calibrator == bundle.calibrator
    # No pickle anywhere in the artifact.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "calibration.json",
        "metadata.json",
        "model.txt",
        "reference.json",
    ]


def test_load_rejects_a_different_feature_list(small_bundle, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bundle, _, _ = small_bundle
    bundle.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["features"] = meta["features"][::-1]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature list"):
        ModelBundle.load(tmp_path)


def test_evaluate_reports_the_expected_fields(small_bundle, small) -> None:  # type: ignore[no-untyped-def]
    bundle, s, _ = small_bundle
    m = evaluate(bundle, s.test, small.ground_truth)
    assert 0 <= m["pr_auc"] <= 1
    lo, hi = m["pr_auc_ci95"]
    assert lo <= m["pr_auc_raw"] <= hi
    d = m["decisions"]
    assert d["approve_all"]["blocked"] == 0
    assert d["approve_all"]["cost_usd"] >= d["model_bayes_cost_rule"]["cost_usd"]
    assert set(m["recall_by_pattern"]) <= {"card_testing", "account_takeover", "merchant_ring"}


@pytest.mark.parametrize("kind", ["none", "platt", "isotonic"])
def test_calibrators_are_monotone(kind: str) -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < raw**2).astype(int)
    cal = Calibrator.fit(kind, raw, y)
    grid = np.linspace(0, 1, 101)
    out = cal(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1))
    if kind == "none":
        np.testing.assert_array_equal(out, grid)


def test_bayes_rule_is_amount_aware() -> None:
    cost = CostModel(false_negative_fixed=15, false_positive=5)
    p = np.array([0.02, 0.02])
    amount = np.array([10.0, 1000.0])
    # 2% on a $10 txn: expected loss 0.5 < 4.9 friction -> approve. On $1000: 20.3 > 4.9 -> block.
    assert cost.bayes_block(p, amount).tolist() == [False, True]


def test_best_threshold_minimises_cost() -> None:
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.6, 0.9, 10)])
    amount = np.full(100, 100.0)
    t = best_threshold(y, p, amount, CostModel())
    assert 0.5 < t <= 0.6


def test_ece_is_zero_for_a_perfectly_calibrated_constant() -> None:
    y = np.array([0] * 90 + [1] * 10)
    assert expected_calibration_error(y, np.full(100, 0.1)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.4)


def test_bootstrap_ci_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=3000) < 0.05).astype(int)
    s = y * 0.5 + rng.uniform(size=3000)
    lo, hi = bootstrap_pr_auc_ci(y, s, n_boot=100)
    assert lo < hi
    assert 0 < lo < 1


def test_rule_baseline_flags_small_bursts() -> None:
    from ffp.model import rule_baseline
    from ffp.schema import CATEGORY_CODE

    df = pd.DataFrame(
        {
            "card_txn_count_1h": [5, 0, 0],
            "amount": [1.0, 800.0, 20.0],
            "is_foreign": [0.0, 1.0, 1.0],
            "category_code": [
                CATEGORY_CODE["digital_goods"],
                CATEGORY_CODE["electronics"],
                CATEGORY_CODE["grocery"],
            ],
        }
    )
    assert rule_baseline(df).tolist() == [True, True, False]
