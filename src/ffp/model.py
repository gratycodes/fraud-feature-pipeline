"""Fraud model: LightGBM, time-based split, isotonic calibration, cost-based decisions.

Timeline (days since the start of the synthetic data, defaults):

    [0 ........ 70)[70 .. 78)[78 .. 85)[85 ..... 100)[100 ....... 120)
        train        early     calib.    label         held-out test
                     stopping            maturity gap  (deployment period)

The model is "trained on day 100" with the labels known on day 100. Events after
day 85 are excluded from training because their chargebacks may not have arrived
yet. The test period is scored with point-in-time features and evaluated against
the labels as eventually reported.

Artifacts are plain files (LightGBM text model + JSON). No pickle, so loading a
model never executes code.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ffp.features import CATEGORICAL_FEATURES, MODEL_FEATURES, SMALL_AMOUNT
from ffp.monitoring import build_reference
from ffp.offline import eventual_labels, label_training_frame
from ffp.schema import CATEGORY_CODE, DAY


@dataclass(frozen=True)
class CostModel:
    """Business cost of each outcome, in USD (synthetic, but the shape is standard).

    Missing a fraud costs the transaction amount plus chargeback handling. Blocking
    a good customer costs a fixed amount of friction and support.
    """

    false_negative_fixed: float = 15.0
    false_positive: float = 5.0

    def fn_cost(self, amount: np.ndarray) -> np.ndarray:
        return amount + self.false_negative_fixed

    def total(self, y: np.ndarray, block: np.ndarray, amount: np.ndarray) -> float:
        fn = (y == 1) & ~block
        fp = (y == 0) & block
        return float(self.fn_cost(amount[fn]).sum() + self.false_positive * fp.sum())

    def bayes_block(self, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """Block when the expected loss of approving exceeds the expected cost of blocking."""
        return p * self.fn_cost(amount) > (1 - p) * self.false_positive


@dataclass
class Calibrator:
    """Maps raw booster scores to probabilities. ``kind`` is none | platt | isotonic."""

    kind: str = "none"
    a: float = 1.0
    b: float = 0.0
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if self.kind == "platt":
            return 1.0 / (1.0 + np.exp(-(self.a * _logit(raw) + self.b)))
        if self.kind == "isotonic":
            return np.interp(raw, self.x, self.y)
        return np.asarray(raw, dtype=float)

    @classmethod
    def fit(cls, kind: str, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if kind == "platt":
            lr = LogisticRegression(C=1e6).fit(_logit(raw)[:, None], y)
            return cls("platt", a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))
        if kind == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
            return cls("isotonic", x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())
        return cls("none")


ISOTONIC_MIN_POSITIVES = 500


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def choose_calibrator(
    raw_a: np.ndarray, y_a: np.ndarray, raw_b: np.ndarray, y_b: np.ndarray
) -> tuple[Calibrator, dict[str, float]]:
    """Fit each candidate on window A, score Brier on the later window B, refit the winner on A+B.

    "none" is a candidate: a booster trained on log-loss is often already calibrated,
    and a calibrator fitted on a short window can make things worse. Isotonic is only
    considered with enough positives: on a few hundred it overfits, and its step
    function turns distinct scores into ties, which costs ranking quality (PR-AUC).
    """
    kinds = ["none", "platt"]
    if min(y_a.sum(), y_b.sum()) >= ISOTONIC_MIN_POSITIVES:
        kinds.append("isotonic")
    scores = {
        kind: float(brier_score_loss(y_b, Calibrator.fit(kind, raw_a, y_a)(raw_b)))
        for kind in kinds
    }
    best = min(scores, key=lambda k: scores[k])
    return Calibrator.fit(best, np.concatenate([raw_a, raw_b]), np.concatenate([y_a, y_b])), scores


@dataclass(frozen=True)
class TrainConfig:
    start_ts: int
    train_end_day: int = 70
    early_stop_end_day: int = 78
    calib_end_day: int = 85
    as_of_day: int = 100
    test_end_day: int = 120
    maturity_days: int = 15
    seed: int = 7
    cost: CostModel = field(default_factory=CostModel)

    def ts(self, day: int) -> int:
        return self.start_ts + day * DAY

    @classmethod
    def for_days(cls, start_ts: int, days: int) -> TrainConfig:
        """The default 120-day timeline, scaled to a dataset of ``days`` days."""
        if days == 120:
            return cls(start_ts=start_ts)
        f = days / 120
        return cls(
            start_ts=start_ts,
            train_end_day=round(70 * f),
            early_stop_end_day=round(78 * f),
            calib_end_day=round(85 * f),
            as_of_day=round(100 * f),
            test_end_day=days,
            maturity_days=max(1, round(15 * f)),
        )


LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "binary_logloss",
    "verbose": -1,
    "num_threads": 4,
}


@dataclass
class Splits:
    train: pd.DataFrame
    early_stop: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame


def make_splits(
    features: pd.DataFrame, labels: pd.DataFrame, cfg: TrainConfig, *, naive_labels: bool = False
) -> Splits:
    """``naive_labels=True`` trains on labels as *eventually* reported (a leak, for comparison)."""
    if naive_labels:
        known = eventual_labels(features, labels)
    else:
        known = label_training_frame(
            features, labels, as_of_ts=cfg.ts(cfg.as_of_day), maturity_ms=cfg.maturity_days * DAY
        )
    known = known[known["ts"] < cfg.ts(cfg.calib_end_day)]
    test = eventual_labels(features, labels)
    test = test[(test["ts"] >= cfg.ts(cfg.as_of_day)) & (test["ts"] < cfg.ts(cfg.test_end_day))]

    def between(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
        return df[(df["ts"] >= cfg.ts(a)) & (df["ts"] < cfg.ts(b))].reset_index(drop=True)

    return Splits(
        train=between(known, 0, cfg.train_end_day),
        early_stop=between(known, cfg.train_end_day, cfg.early_stop_end_day),
        calib=between(known, cfg.early_stop_end_day, cfg.calib_end_day),
        test=test.reset_index(drop=True),
    )


def _x(df: pd.DataFrame) -> np.ndarray:
    return df[list(MODEL_FEATURES)].to_numpy(dtype=np.float64)


@dataclass
class ModelBundle:
    booster: lgb.Booster
    calibrator: Calibrator
    threshold: float
    cost: CostModel
    metadata: dict[str, Any]
    reference: dict[str, Any]

    def raw_scores(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(x, num_threads=1))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator(raw)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrate(self.raw_scores(x))

    def decide(self, p: float, amount: float) -> str:
        block = bool(self.cost.bayes_block(np.array([p]), np.array([amount]))[0])
        return "block" if block else "approve"

    def save(self, out: str | Path) -> None:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "model.txt"))
        (d / "calibration.json").write_text(json.dumps(asdict(self.calibrator)))
        meta = {**self.metadata, "threshold": self.threshold, "cost": asdict(self.cost)}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        (d / "reference.json").write_text(json.dumps(self.reference))

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        d = Path(path)
        meta = json.loads((d / "metadata.json").read_text())
        if tuple(meta["features"]) != MODEL_FEATURES:
            raise ValueError("model was trained with a different feature list")
        cal = json.loads((d / "calibration.json").read_text())
        return cls(
            booster=lgb.Booster(model_file=str(d / "model.txt")),
            calibrator=Calibrator(**cal),
            threshold=float(meta["threshold"]),
            cost=CostModel(**meta["cost"]),
            metadata=meta,
            reference=json.loads((d / "reference.json").read_text()),
        )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    raise TypeError(type(o))


def fit(splits: Splits, cfg: TrainConfig, *, rounds: int = 2000) -> ModelBundle:
    cat_idx = [MODEL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    dtrain = lgb.Dataset(
        _x(splits.train),
        splits.train["label"],
        feature_name=list(MODEL_FEATURES),
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    dval = lgb.Dataset(_x(splits.early_stop), splits.early_stop["label"], reference=dtrain)
    booster = lgb.train(
        {**LGB_PARAMS, "seed": cfg.seed},
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    # Keep only the best iteration so serving matches evaluation.
    booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    raw_es = np.asarray(booster.predict(_x(splits.early_stop)))
    raw_cal = np.asarray(booster.predict(_x(splits.calib)))
    calibrator, cal_scores = choose_calibrator(
        raw_es, splits.early_stop["label"].to_numpy(), raw_cal, splits.calib["label"].to_numpy()
    )
    threshold = best_threshold(
        splits.calib["label"].to_numpy(),
        calibrator(raw_cal),
        splits.calib["amount"].to_numpy(),
        cfg.cost,
    )
    importance = dict(
        zip(
            MODEL_FEATURES, booster.feature_importance(importance_type="gain").tolist(), strict=True
        )
    )
    meta = {
        "features": list(MODEL_FEATURES),
        "best_iteration": booster.current_iteration(),
        "lgb_params": LGB_PARAMS,
        "train_config": {k: v for k, v in asdict(cfg).items() if k != "cost"},
        "rows": {k: len(getattr(splits, k)) for k in ("train", "early_stop", "calib", "test")},
        "positives": {
            k: int(getattr(splits, k)["label"].sum())
            for k in ("train", "early_stop", "calib", "test")
        },
        "gain_importance": importance,
        "calibration": {"chosen": calibrator.kind, "brier_on_later_window": cal_scores},
    }
    return ModelBundle(
        booster=booster,
        calibrator=calibrator,
        threshold=threshold,
        cost=cfg.cost,
        metadata=meta,
        reference=build_reference(splits.train),
    )


def best_threshold(y: np.ndarray, p: np.ndarray, amount: np.ndarray, cost: CostModel) -> float:
    """The single probability cut-off with the lowest total cost on held-out data."""
    candidates = np.unique(np.quantile(p, np.linspace(0.9, 1.0, 400)))
    costs = [cost.total(y, p >= t, amount) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE over quantile bins (fraud is rare, so equal-width bins would be mostly empty).

    Ties always land in the same bin, so a constant prediction is one bin.
    """
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx = np.searchsorted(edges, p, side="right")
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_pr_auc_ci(
    y: np.ndarray, score: np.ndarray, n_boot: int = 300, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for PR-AUC. With a few hundred positives
    it is wide, which matters when comparing two pipelines."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """What a hand-written rules engine might start with."""
    risky = df["category_code"].isin(
        [CATEGORY_CODE[c] for c in ("electronics", "gift_cards", "jewelry")]
    )
    return (
        (df["card_txn_count_1h"].to_numpy() >= 4) & (df["amount"].to_numpy() < SMALL_AMOUNT)
    ) | ((df["is_foreign"] == 1) & risky & (df["amount"] > 400)).to_numpy()


def evaluate(
    bundle: ModelBundle, test: pd.DataFrame, ground_truth: pd.DataFrame | None = None
) -> dict[str, Any]:
    y = test["label"].to_numpy()
    amount = test["amount"].to_numpy()
    raw = bundle.raw_scores(_x(test))
    p = bundle.calibrate(raw)
    cost = bundle.cost
    fixed = p >= bundle.threshold
    bayes = cost.bayes_block(p, amount)
    rules = rule_baseline(test)

    def outcome(block: np.ndarray) -> dict[str, float]:
        tp = int(((y == 1) & block).sum())
        fp = int(((y == 0) & block).sum())
        fn = int(((y == 1) & ~block).sum())
        return {
            "blocked": int(block.sum()),
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan,
            "fraud_usd_caught": float(amount[(y == 1) & block].sum()),
            "cost_usd": cost.total(y, block, amount),
        }

    res: dict[str, Any] = {
        "test_rows": len(test),
        "test_positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_raw": float(average_precision_score(y, raw)),
        "pr_auc_ci95": bootstrap_pr_auc_ci(y, raw),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_raw": float(brier_score_loss(y, raw)),
        "brier_calibrated": float(brier_score_loss(y, p)),
        "ece_raw": expected_calibration_error(y, raw),
        "ece_calibrated": expected_calibration_error(y, p),
        "threshold": bundle.threshold,
        "decisions": {
            "approve_all": outcome(np.zeros_like(fixed)),
            "rules_baseline": outcome(rules),
            "model_fixed_threshold": outcome(fixed),
            "model_bayes_cost_rule": outcome(bayes),
        },
    }
    if ground_truth is not None:
        g = test[["txn_id"]].merge(ground_truth, on="txn_id", how="left")
        per = {}
        for pattern, idx in g.groupby("pattern").groups.items():
            if pattern == "legit":
                continue
            ii = g.index.get_indexer(idx)
            per[str(pattern)] = {
                "events": len(ii),
                "recall_bayes_rule": float(bayes[ii].mean()),
                "recall_rules_baseline": float(rules[ii].mean()),
            }
        res["recall_by_pattern"] = per
    return res
