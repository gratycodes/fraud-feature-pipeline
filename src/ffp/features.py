"""Feature definitions, declared once.

Every feature the model sees is declared in this module. Two engines evaluate the
same declarations:

* ``offline.backfill`` compiles the ``WindowAgg`` specs to DuckDB window SQL over
  Parquet history (for training).
* ``online`` evaluates them in Python over the per-entity event log kept in Redis
  (for serving).

Window semantics are the contract both engines must honour, and the parity and
skew tests check it: a window feature for an event at time ``t`` aggregates the
same entity's events with ``t - window <= ts < t``. The event itself, and any
other event at exactly ``t``, is excluded, so a feature never depends on
something that has not happened before the decision is made.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from ffp.schema import CATEGORY_CODE, DAY, HOUR, MINUTE, SECOND

Entity = Literal["card", "merchant"]
Agg = Literal["count", "sum", "distinct"]


@dataclass(frozen=True, slots=True)
class WindowAgg:
    """An aggregate over one entity's events in a trailing window ``[t - window, t)``."""

    name: str
    entity: Entity
    agg: Agg
    window_ms: int
    column: str | None = None  # "amount" for sum; the counted key for distinct
    amount_lt: float | None = None  # optional filter: only events with amount < this

    def __post_init__(self) -> None:
        if self.agg == "sum" and self.column != "amount":
            raise ValueError(f"{self.name}: sum is only defined over 'amount'")
        if self.agg == "distinct" and self.column not in _DISTINCT_COLUMNS[self.entity]:
            raise ValueError(
                f"{self.name}: cannot count distinct {self.column!r} per {self.entity}"
            )


# Which columns can be counted distinct per entity, and what the online event log
# stores for them ("other" = the counter-party id: merchant for a card, card for a merchant).
_DISTINCT_COLUMNS: dict[str, dict[str, str]] = {
    "card": {"merchant_id": "other", "country": "country"},
    "merchant": {"card_id": "other", "country": "country"},
}

SMALL_AMOUNT = 5.0

WINDOW_FEATURES: tuple[WindowAgg, ...] = (
    WindowAgg("card_txn_count_1m", "card", "count", 1 * MINUTE),
    WindowAgg("card_txn_count_1h", "card", "count", 1 * HOUR),
    WindowAgg("card_txn_count_24h", "card", "count", 24 * HOUR),
    WindowAgg("card_small_txn_count_1h", "card", "count", 1 * HOUR, amount_lt=SMALL_AMOUNT),
    WindowAgg("card_amount_sum_1h", "card", "sum", 1 * HOUR, column="amount"),
    WindowAgg("card_amount_sum_24h", "card", "sum", 24 * HOUR, column="amount"),
    WindowAgg("card_distinct_merchants_1h", "card", "distinct", 1 * HOUR, column="merchant_id"),
    WindowAgg("card_distinct_merchants_24h", "card", "distinct", 24 * HOUR, column="merchant_id"),
    WindowAgg("card_distinct_countries_24h", "card", "distinct", 24 * HOUR, column="country"),
    WindowAgg("merchant_txn_count_1h", "merchant", "count", 1 * HOUR),
    WindowAgg("merchant_txn_count_24h", "merchant", "count", 24 * HOUR),
    WindowAgg("merchant_small_txn_count_1h", "merchant", "count", 1 * HOUR, amount_lt=SMALL_AMOUNT),
    WindowAgg("merchant_distinct_cards_1h", "merchant", "distinct", 1 * HOUR, column="card_id"),
)

# Longest window: the online event log only needs to keep this much history.
MAX_WINDOW_MS = max(f.window_ms for f in WINDOW_FEATURES)

RECENCY_FEATURE = "card_seconds_since_last_txn"

# Merchant chargeback rate over the previous 30 days, as known from labels that had
# *arrived* by the daily snapshot time. Computed by a batch job, materialised daily.
MERCHANT_RATE_FEATURE = "merchant_fraud_rate_30d"
MERCHANT_RATE_WINDOW_MS = 30 * DAY
MERCHANT_RATE_PRIOR = 0.005
MERCHANT_RATE_PRIOR_STRENGTH = 50.0

REQUEST_FEATURES: tuple[str, ...] = (
    "amount",
    "is_foreign",
    "is_online",
    "hour_of_day",
    "category_code",
    "card_amount_ratio_24h",
)

MODEL_FEATURES: tuple[str, ...] = (
    *(f.name for f in WINDOW_FEATURES),
    RECENCY_FEATURE,
    MERCHANT_RATE_FEATURE,
    *REQUEST_FEATURES,
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("category_code",)


def smoothed_rate(fraud: float, n: float) -> float:
    """Beta-prior smoothed fraud rate so a merchant with 2 txns and 1 chargeback isn't 50%."""
    return (fraud + MERCHANT_RATE_PRIOR * MERCHANT_RATE_PRIOR_STRENGTH) / (
        n + MERCHANT_RATE_PRIOR_STRENGTH
    )


@dataclass(frozen=True, slots=True)
class StateEvent:
    """One entry of an entity's online event log."""

    ts: int
    txn_id: str
    amount: float
    other: str  # merchant_id for card logs, card_id for merchant logs
    country: str


def evaluate_windows(
    events_by_entity: Mapping[str, Iterable[StateEvent]], ts: int
) -> dict[str, float]:
    """Online engine: evaluate every ``WindowAgg`` at time ``ts`` from raw event logs.

    ``events_by_entity`` maps "card"/"merchant" to that entity's events. Events at or
    after ``ts`` are ignored, matching the offline ``[t - w, t)`` semantics.
    """
    logs = {k: [e for e in v if e.ts < ts] for k, v in events_by_entity.items()}
    out: dict[str, float] = {}
    for f in WINDOW_FEATURES:
        lo = ts - f.window_ms
        evs = [e for e in logs.get(f.entity, []) if e.ts >= lo]
        if f.amount_lt is not None:
            evs = [e for e in evs if e.amount < f.amount_lt]
        if f.agg == "count":
            out[f.name] = float(len(evs))
        elif f.agg == "sum":
            out[f.name] = float(round(sum(e.amount for e in evs), 6))
        else:
            assert f.column is not None
            attr = _DISTINCT_COLUMNS[f.entity][f.column]
            out[f.name] = float(len({getattr(e, attr) for e in evs}))
    return out


def recency_seconds(last_ts: int | None, ts: int) -> float:
    if last_ts is None or last_ts >= ts:
        return math.nan
    return (ts - last_ts) / SECOND


def request_features(
    *,
    ts: int,
    amount: float,
    country: str,
    channel: str,
    category: str,
    home_country: str | None,
    card_count_24h: float,
    card_sum_24h: float,
) -> dict[str, float]:
    """Features derived from the request itself plus a couple of aggregates.

    Both engines call this exact function, so these can't drift apart.
    """
    avg = card_sum_24h / card_count_24h if card_count_24h > 0 else math.nan
    return {
        "amount": float(amount),
        "is_foreign": float(home_country is not None and country != home_country),
        "is_online": float(channel == "online"),
        "hour_of_day": float((ts // HOUR) % 24),
        "category_code": float(CATEGORY_CODE.get(category, -1)),
        "card_amount_ratio_24h": float(amount) / avg if avg and not math.isnan(avg) else math.nan,
    }
