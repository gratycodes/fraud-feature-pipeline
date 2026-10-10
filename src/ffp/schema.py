"""Shared constants and the transaction record used by every component."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MS = 1
SECOND = 1_000 * MS
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# 2026-01-05T00:00:00Z. Synthetic data starts here; the date itself carries no meaning.
EPOCH_START_MS = 1_767_571_200_000

CATEGORIES: tuple[str, ...] = (
    "grocery",
    "restaurant",
    "fuel",
    "transit",
    "retail",
    "travel",
    "utilities",
    "electronics",
    "digital_goods",
    "gift_cards",
    "jewelry",
)
CATEGORY_CODE: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

COUNTRIES: tuple[str, ...] = ("IN", "AE", "GB", "DE", "NL", "FR", "US", "SG", "BR", "NG")


@dataclass(frozen=True, slots=True)
class Transaction:
    """A card authorization event as it arrives on the stream.

    ``ts`` is epoch milliseconds (UTC). Labels are *not* part of the event:
    they arrive separately and late (see ``generator``).
    """

    txn_id: str
    ts: int
    card_id: str
    merchant_id: str
    amount: float
    country: str
    channel: str  # "pos" | "online"
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transaction:
        return cls(
            txn_id=str(d["txn_id"]),
            ts=int(d["ts"]),
            card_id=str(d["card_id"]),
            merchant_id=str(d["merchant_id"]),
            amount=float(d["amount"]),
            country=str(d["country"]),
            channel=str(d["channel"]),
            category=str(d["category"]),
        )
