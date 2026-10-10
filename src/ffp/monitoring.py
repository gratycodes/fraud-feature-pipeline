"""Train/serve skew and feature drift.

* **Skew**: the serving path logs the exact feature vector it scored. Later, for a
  sample of those events, recompute the features offline from the event log and
  compare value by value. Any difference is a bug (or lag) in one of the paths,
  and the report names the feature.
* **Drift**: Population Stability Index per feature between the training
  reference and a recent window. PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major
  (the usual rule of thumb, not a law).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffp.features import CATEGORICAL_FEATURES, MODEL_FEATURES

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
_EPS = 1e-4


# --------------------------------------------------------------------------- skew
@dataclass
class FeatureSkew:
    feature: str
    compared: int
    mismatches: int
    max_abs_diff: float
    example_txn_ids: list[str] = field(default_factory=list)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / self.compared if self.compared else 0.0


@dataclass
class SkewReport:
    compared_events: int
    missing_offline: int
    features: list[FeatureSkew]
    max_mismatch_rate: float

    @property
    def ok(self) -> bool:
        return all(f.mismatch_rate <= self.max_mismatch_rate for f in self.features)

    @property
    def skewed(self) -> list[FeatureSkew]:
        return [f for f in self.features if f.mismatches]

    def to_markdown(self) -> str:
        lines = [
            f"Compared **{self.compared_events}** served events against offline recomputation "
            f"({self.missing_offline} not found offline). "
            f"Status: **{'OK' if self.ok else 'SKEW DETECTED'}** "
            f"(threshold {self.max_mismatch_rate:.2%} mismatches per feature).",
            "",
            "| feature | mismatches | rate | max abs diff | examples |",
            "|---|---:|---:|---:|---|",
        ]
        for f in sorted(self.features, key=lambda x: -x.mismatches):
            lines.append(
                f"| {f.feature} | {f.mismatches} | {f.mismatch_rate:.2%} | {f.max_abs_diff:.4g} | "
                f"{', '.join(f.example_txn_ids[:3])} |"
            )
        return "\n".join(lines)


def _equal(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=rtol, atol=atol)
    return both_nan | close


def skew_check(
    served: pd.DataFrame,
    offline: pd.DataFrame,
    *,
    sample: int | None = None,
    seed: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_mismatch_rate: float = 0.001,
    features: tuple[str, ...] = MODEL_FEATURES,
) -> SkewReport:
    """Compare served feature vectors (one row per txn_id) with offline recomputation."""
    s = (
        served
        if sample is None or len(served) <= sample
        else served.sample(sample, random_state=seed)
    )
    joined = s.merge(
        offline, on="txn_id", how="left", suffixes=("_served", "_offline"), indicator=True
    )
    missing = int((joined["_merge"] == "left_only").sum())
    joined = joined[joined["_merge"] == "both"]
    out = []
    for f in features:
        a = joined[f"{f}_served"].to_numpy(dtype=float)
        b = joined[f"{f}_offline"].to_numpy(dtype=float)
        eq = _equal(a, b, rtol, atol)
        diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
        diff = np.where(np.isnan(a) != np.isnan(b), np.inf, diff)
        bad = ~eq
        out.append(
            FeatureSkew(
                feature=f,
                compared=len(joined),
                mismatches=int(bad.sum()),
                max_abs_diff=float(diff[bad].max()) if bad.any() else 0.0,
                example_txn_ids=joined.loc[bad, "txn_id"].head(5).tolist(),
            )
        )
    return SkewReport(len(joined), missing, out, max_mismatch_rate)


# -------------------------------------------------------------------------- drift
def reference_bins(values: pd.Series, n_bins: int = 10, categorical: bool = False) -> dict:
    """Bin edges + proportions from the training data; NaN gets its own bin."""
    v = values.to_numpy(dtype=float)
    nan_share = float(np.isnan(v).mean()) if len(v) else 0.0
    v = v[~np.isnan(v)]
    if categorical:
        cats = sorted({float(x) for x in np.unique(v)})
        props = [float((v == c).mean()) * (1 - nan_share) for c in cats] if len(v) else []
        return {"kind": "categorical", "categories": cats, "props": props, "nan": nan_share}
    edges = (
        np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(v) else np.array([])
    )
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
    props = (counts / max(len(v), 1) * (1 - nan_share)).tolist()
    return {"kind": "numeric", "edges": edges.tolist(), "props": props, "nan": nan_share}


def _current_props(ref: dict, values: pd.Series) -> tuple[list[float], float]:
    v = values.to_numpy(dtype=float)
    nan_share = float(np.isnan(v).mean()) if len(v) else 0.0
    v = v[~np.isnan(v)]
    n = max(len(v), 1)
    if ref["kind"] == "categorical":
        cats = ref["categories"]
        props = [float((v == c).mean()) * (1 - nan_share) for c in cats]
        # Unseen categories count against the distribution as one extra bucket.
        props.append(float((~np.isin(v, cats)).sum() / n) * (1 - nan_share))
        return props, nan_share
    edges = np.asarray(ref["edges"])
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
    return (counts / n * (1 - nan_share)).tolist(), nan_share


def psi(ref: dict, values: pd.Series) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected)) over the reference bins."""
    cur, cur_nan = _current_props(ref, values)
    unseen = [0.0] if ref["kind"] == "categorical" else []
    e = np.clip(np.asarray([*ref["props"], *unseen, ref["nan"]]), _EPS, None)
    a = np.clip(np.asarray([*cur, cur_nan]), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(train: pd.DataFrame, features: tuple[str, ...] = MODEL_FEATURES) -> dict:
    return {f: reference_bins(train[f], categorical=f in CATEGORICAL_FEATURES) for f in features}


@dataclass
class DriftRow:
    feature: str
    psi: float

    @property
    def status(self) -> str:
        if math.isnan(self.psi):
            return "n/a"
        if self.psi >= PSI_MAJOR:
            return "major"
        if self.psi >= PSI_MODERATE:
            return "moderate"
        return "stable"


def drift_report(reference: dict, current: pd.DataFrame) -> list[DriftRow]:
    rows = [DriftRow(f, psi(ref, current[f])) for f, ref in reference.items() if f in current]
    return sorted(rows, key=lambda r: -r.psi)


def drift_markdown(rows: list[DriftRow], title: str) -> str:
    lines = [f"**{title}**", "", "| feature | PSI | status |", "|---|---:|---|"]
    lines += [f"| {r.feature} | {r.psi:.3f} | {r.status} |" for r in rows]
    return "\n".join(lines)
