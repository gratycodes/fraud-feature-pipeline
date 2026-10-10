"""Online store on Redis.

Layout (all keys under a prefix, default ``ffp``):

* ``{p}:card:{id}:ev`` / ``{p}:merchant:{id}:ev`` - sorted set, score = event ts,
  member = ``txn_id|amount|counterparty|country``. Trimmed to the longest window
  plus slack, so memory is bounded by traffic in the last ~25 hours.
* ``{p}:card_last`` - sorted set card_id -> latest ts seen (``ZADD GT``), for
  "seconds since last transaction" beyond the 24h log.
* ``{p}:card_profile`` - hash card_id -> home country.
* ``{p}:mrate:snapshots`` + ``{p}:mrate:{snapshot_ts}`` - daily merchant chargeback
  rate snapshots written by the batch job.

Window aggregates are evaluated at read time over the event log rather than
pre-aggregated, so a sliding window is exact at any request time and ingestion is
order-independent and idempotent (re-delivering an event is a no-op ``ZADD``).
The trade-off is read cost proportional to an entity's recent traffic; see the
README for when you would switch to time-bucketed tiles instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import redis

from ffp.features import (
    MAX_WINDOW_MS,
    MERCHANT_RATE_FEATURE,
    MODEL_FEATURES,
    RECENCY_FEATURE,
    StateEvent,
    evaluate_windows,
    recency_seconds,
    request_features,
)
from ffp.schema import HOUR, Transaction

RETENTION_MS = MAX_WINDOW_MS + 1 * HOUR
KEY_TTL_S = 3 * 24 * 3600


def _encode(txn_id: str, amount: float, other: str, country: str) -> str:
    return f"{txn_id}|{amount!r}|{other}|{country}"


def _decode(member: bytes | str, score: float) -> StateEvent:
    s = member.decode() if isinstance(member, bytes) else member
    txn_id, amount, other, country = s.split("|")
    return StateEvent(int(score), txn_id, float(amount), other, country)


@dataclass(frozen=True)
class OnlineState:
    card_events: list[StateEvent]
    merchant_events: list[StateEvent]
    card_last_ts: int | None
    home_country: str | None
    merchant_rate: float | None


class OnlineStore:
    def __init__(self, client: redis.Redis, prefix: str = "ffp") -> None:
        self.r = client
        self.p = prefix

    # -- keys ---------------------------------------------------------------
    def _card_key(self, card_id: str) -> str:
        return f"{self.p}:card:{card_id}:ev"

    def _merchant_key(self, merchant_id: str) -> str:
        return f"{self.p}:merchant:{merchant_id}:ev"

    # -- writes (stream processor / batch jobs) -----------------------------
    def ingest(self, txns: Iterable[Transaction]) -> int:
        """Apply a batch of events. Idempotent and safe to apply out of order."""
        pipe = self.r.pipeline(transaction=False)
        n = 0
        for t in txns:
            ck, mk = self._card_key(t.card_id), self._merchant_key(t.merchant_id)
            pipe.zadd(ck, {_encode(t.txn_id, t.amount, t.merchant_id, t.country): t.ts})
            pipe.zadd(mk, {_encode(t.txn_id, t.amount, t.card_id, t.country): t.ts})
            pipe.zremrangebyscore(ck, "-inf", f"({t.ts - RETENTION_MS}")
            pipe.zremrangebyscore(mk, "-inf", f"({t.ts - RETENTION_MS}")
            pipe.expire(ck, KEY_TTL_S)
            pipe.expire(mk, KEY_TTL_S)
            pipe.zadd(f"{self.p}:card_last", {t.card_id: t.ts}, gt=True)
            n += 1
        pipe.execute()
        return n

    def load_card_profiles(self, cards: pd.DataFrame) -> None:
        mapping = dict(zip(cards["card_id"], cards["home_country"], strict=True))
        for chunk in _chunks(list(mapping.items()), 5_000):
            self.r.hset(f"{self.p}:card_profile", mapping=dict(chunk))

    def load_merchant_rate_snapshots(self, snapshots: pd.DataFrame) -> int:
        """Materialise daily snapshots (what a nightly batch job would do)."""
        n = 0
        for _, g in snapshots.groupby("snapshot_ts"):
            snap_ts = int(g["snapshot_ts"].iloc[0])
            key = f"{self.p}:mrate:{snap_ts}"
            pipe = self.r.pipeline(transaction=False)
            pipe.hset(
                key, mapping=dict(zip(g["merchant_id"], g["value"].astype(float), strict=True))
            )
            pipe.zadd(f"{self.p}:mrate:snapshots", {str(snap_ts): snap_ts})
            pipe.execute()
            n += 1
        return n

    # -- reads (serving) ----------------------------------------------------
    def fetch(self, card_id: str, merchant_id: str, ts: int) -> OnlineState:
        lo = ts - MAX_WINDOW_MS
        pipe = self.r.pipeline(transaction=False)
        pipe.zrangebyscore(self._card_key(card_id), lo, f"({ts}", withscores=True)
        pipe.zrangebyscore(self._merchant_key(merchant_id), lo, f"({ts}", withscores=True)
        pipe.zscore(f"{self.p}:card_last", card_id)
        pipe.hget(f"{self.p}:card_profile", card_id)
        pipe.zrevrangebyscore(f"{self.p}:mrate:snapshots", ts, "-inf", start=0, num=1)
        card_raw, merch_raw, last, home, snap = pipe.execute()

        card_events = [_decode(m, s) for m, s in card_raw]
        last_ts: int | None = int(last) if last is not None else None
        if last_ts is None or last_ts >= ts:
            # Events at/after ts were already ingested; fall back to the log.
            last_ts = max((e.ts for e in card_events), default=None)
        rate: float | None = None
        if snap:
            raw = self.r.hget(f"{self.p}:mrate:{_s(snap[0])}", merchant_id)
            rate = float(raw) if raw is not None else None
        return OnlineState(
            card_events=card_events,
            merchant_events=[_decode(m, s) for m, s in merch_raw],
            card_last_ts=last_ts,
            home_country=_s(home) if home is not None else None,
            merchant_rate=rate,
        )

    def features_for(self, txn: Transaction) -> dict[str, float]:
        """All model features for ``txn`` as of ``txn.ts``, in ``MODEL_FEATURES`` order."""
        st = self.fetch(txn.card_id, txn.merchant_id, txn.ts)
        return compute_features(txn, st)

    def flush(self) -> None:
        """Delete every key under this store's prefix (tests and demos only)."""
        for chunk in _chunks(list(self.r.scan_iter(f"{self.p}:*", count=1000)), 1000):
            self.r.delete(*chunk)


def compute_features(txn: Transaction, st: OnlineState) -> dict[str, float]:
    win = evaluate_windows({"card": st.card_events, "merchant": st.merchant_events}, txn.ts)
    feats: dict[str, float] = dict(win)
    feats[RECENCY_FEATURE] = recency_seconds(st.card_last_ts, txn.ts)
    feats[MERCHANT_RATE_FEATURE] = st.merchant_rate if st.merchant_rate is not None else math.nan
    feats.update(
        request_features(
            ts=txn.ts,
            amount=txn.amount,
            country=txn.country,
            channel=txn.channel,
            category=txn.category,
            home_country=st.home_country,
            card_count_24h=win["card_txn_count_24h"],
            card_sum_24h=win["card_amount_sum_24h"],
        )
    )
    return {k: feats[k] for k in MODEL_FEATURES}


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
