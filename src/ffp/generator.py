"""Seedable synthetic card-transaction generator with delayed fraud labels.

Nothing here is derived from real data. Volumes, amounts and fraud mechanics are
plausible shapes chosen so the features have something real to find:

* ``card_testing`` - a burst of tiny online authorizations at a handful of
  low-friction merchants within minutes, often followed by a large cash-out.
* ``account_takeover`` - several high-value purchases in risky categories over a
  few hours, usually in a country the card has never used.
* ``merchant_ring`` - a compromised online merchant that charges many unrelated
  cards for a week or two (only visible through delayed chargeback labels).

Legitimate traffic includes look-alikes on purpose: transit taps arrive in small
bursts, people travel abroad, and some people buy a laptop.

Labels arrive late (chargeback delay, median ~12 days) and ~8% of fraud is never
reported, so "no label yet" is not the same as "legitimate".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ffp.schema import CATEGORIES, COUNTRIES, DAY, EPOCH_START_MS, HOUR, MINUTE, SECOND

_CATEGORY_WEIGHTS = {
    "grocery": 0.18,
    "restaurant": 0.17,
    "fuel": 0.08,
    "transit": 0.08,
    "retail": 0.16,
    "travel": 0.04,
    "utilities": 0.05,
    "electronics": 0.06,
    "digital_goods": 0.10,
    "gift_cards": 0.04,
    "jewelry": 0.04,
}
_BASE_AMOUNT = {
    "grocery": 40.0,
    "restaurant": 30.0,
    "fuel": 50.0,
    "transit": 2.5,
    "retail": 60.0,
    "travel": 280.0,
    "utilities": 80.0,
    "electronics": 220.0,
    "digital_goods": 12.0,
    "gift_cards": 50.0,
    "jewelry": 350.0,
}
_ONLINE_PROB = {
    "grocery": 0.10,
    "restaurant": 0.05,
    "fuel": 0.0,
    "transit": 0.0,
    "retail": 0.30,
    "travel": 0.60,
    "utilities": 0.70,
    "electronics": 0.50,
    "digital_goods": 0.95,
    "gift_cards": 0.85,
    "jewelry": 0.25,
}
_HOME_COUNTRY_WEIGHTS = np.array([0.35, 0.15, 0.12, 0.08, 0.05, 0.05, 0.10, 0.05, 0.03, 0.02])
_HOUR_WEIGHTS = np.array(
    [1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3, 3.5, 4, 4.5, 5, 5, 4.5, 4.5, 5, 5.5, 6, 6, 5.5, 4, 3, 2]
)
_RISKY = ("electronics", "gift_cards", "jewelry", "digital_goods")


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 7
    n_cards: int = 3_000
    n_merchants: int = 500
    days: int = 150
    mean_txn_per_card_day: float = 1.3
    card_testing_episodes: int = 165
    ato_episodes: int = 200
    ring_merchants: int = 4
    ring_txns_per_merchant: int = 130
    label_delay_median_days: float = 12.0
    unreported_rate: float = 0.08
    # "Holiday season" from this day on: legit spend goes up and moves online.
    # Gives the drift report something real to detect. ``None`` disables it.
    drift_start_day: int | None = 138
    drift_amount_mult: float = 1.35


@dataclass
class Dataset:
    transactions: pd.DataFrame
    cards: pd.DataFrame
    merchants: pd.DataFrame
    labels: pd.DataFrame  # txn_id, label_ts, is_fraud (only reported chargebacks)
    ground_truth: pd.DataFrame  # txn_id, is_fraud, pattern (for analysis only)

    def write(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.transactions.to_parquet(out / "transactions.parquet", index=False)
        self.cards.to_parquet(out / "cards.parquet", index=False)
        self.merchants.to_parquet(out / "merchants.parquet", index=False)
        self.labels.to_parquet(out / "labels.parquet", index=False)
        self.ground_truth.to_parquet(out / "ground_truth.parquet", index=False)

    @classmethod
    def read(cls, in_dir: str | Path) -> Dataset:
        d = Path(in_dir)
        return cls(
            transactions=pd.read_parquet(d / "transactions.parquet"),
            cards=pd.read_parquet(d / "cards.parquet"),
            merchants=pd.read_parquet(d / "merchants.parquet"),
            labels=pd.read_parquet(d / "labels.parquet"),
            ground_truth=pd.read_parquet(d / "ground_truth.parquet"),
        )


def _merchants(rng: np.random.Generator, cfg: GeneratorConfig) -> pd.DataFrame:
    cats = np.array(list(_CATEGORY_WEIGHTS))
    w = np.array(list(_CATEGORY_WEIGHTS.values()))
    category = rng.choice(cats, size=cfg.n_merchants, p=w / w.sum())
    country = rng.choice(np.array(COUNTRIES), size=cfg.n_merchants, p=_HOME_COUNTRY_WEIGHTS)
    online = np.array([rng.random() < _ONLINE_PROB[c] for c in category])
    popularity = 1.0 / (rng.permutation(cfg.n_merchants) + 1.0) ** 0.8
    df = pd.DataFrame(
        {
            "merchant_id": [f"m{i:04d}" for i in range(cfg.n_merchants)],
            "category": category,
            "country": country,
            "channel": np.where(online, "online", "pos"),
            "popularity": popularity,
        }
    )
    # A few low-friction online merchants that fraudsters use to test cards.
    cand = df.index[df["category"].isin(["digital_goods", "gift_cards"]) & online].to_numpy()
    n_test = min(8, len(cand))
    testing = rng.choice(cand, size=n_test, replace=False) if n_test else np.array([], dtype=int)
    df["testing_target"] = df.index.isin(testing)
    # Merchants that will be compromised by a fraud ring at some point.
    cand = df.index[online & ~df["testing_target"].to_numpy()].to_numpy()
    ring = rng.choice(cand, size=min(cfg.ring_merchants, len(cand)), replace=False)
    df["ring_target"] = df.index.isin(ring)
    return df


def _cards(rng: np.random.Generator, cfg: GeneratorConfig) -> pd.DataFrame:
    sigma = 0.6
    rate = rng.lognormal(np.log(cfg.mean_txn_per_card_day) - sigma**2 / 2, sigma, cfg.n_cards)
    return pd.DataFrame(
        {
            "card_id": [f"c{i:05d}" for i in range(cfg.n_cards)],
            "home_country": rng.choice(
                np.array(COUNTRIES), size=cfg.n_cards, p=_HOME_COUNTRY_WEIGHTS
            ),
            "rate": rate,
            "amount_scale": rng.lognormal(0.0, 0.45, cfg.n_cards),
        }
    )


def _amounts(rng: np.random.Generator, categories: np.ndarray, scale: float) -> np.ndarray:
    base = np.array([_BASE_AMOUNT[c] for c in categories])
    return np.maximum(0.5, base * scale * rng.lognormal(-0.18, 0.6, len(categories)))


def generate(cfg: GeneratorConfig | None = None) -> Dataset:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    merchants = _merchants(rng, cfg)
    cards = _cards(rng, cfg)

    m_ids = merchants["merchant_id"].to_numpy()
    m_cat = merchants["category"].to_numpy()
    m_country = merchants["country"].to_numpy()
    m_online = (merchants["channel"] == "online").to_numpy()
    m_pop = merchants["popularity"].to_numpy()

    def pool(country: str) -> tuple[np.ndarray, np.ndarray]:
        # Mostly domestic merchants; foreign online shops get a small share.
        idx = np.flatnonzero((m_country == country) | m_online)
        p = m_pop[idx] * np.where(m_country[idx] == country, 1.0, 0.006)
        return idx, p / p.sum()

    pools = {c: pool(c) for c in COUNTRIES}
    hour_p = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()
    dow = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.2, 1.15])
    day_factor = dow[np.arange(cfg.days) % 7]

    rows_card: list[np.ndarray] = []
    rows_merchant: list[np.ndarray] = []
    rows_ts: list[np.ndarray] = []
    rows_amount: list[np.ndarray] = []

    c_rate = cards["rate"].to_numpy(dtype=float)
    c_home = cards["home_country"].astype(str).tolist()
    c_scale = cards["amount_scale"].to_numpy(dtype=float)
    for ci in range(len(cards)):
        home, scale = c_home[ci], float(c_scale[ci])
        counts = rng.poisson(c_rate[ci] * day_factor)
        n = int(counts.sum())
        if n == 0:
            continue
        day = np.repeat(np.arange(cfg.days), counts)
        ts = (
            EPOCH_START_MS
            + day * DAY
            + rng.choice(24, size=n, p=hour_p) * HOUR
            + rng.integers(0, HOUR, size=n)
        )
        home_idx, home_p = pools[home]
        pref = rng.choice(home_idx, size=min(15, len(home_idx)), replace=False, p=home_p)
        pref_p = m_pop[pref] / m_pop[pref].sum()
        use_pref = rng.random(n) < 0.8
        m = np.where(
            use_pref,
            rng.choice(pref, size=n, p=pref_p),
            rng.choice(home_idx, size=n, p=home_p),
        )
        # 25% of cards take one trip abroad.
        if rng.random() < 0.25:
            start = int(rng.integers(0, cfg.days - 3))
            length = int(rng.integers(3, 11))
            trip_country = rng.choice([c for c in COUNTRIES if c != home])
            on_trip = (day >= start) & (day < start + length)
            if on_trip.any():
                t_idx, t_p = pools[str(trip_country)]
                m[on_trip] = rng.choice(t_idx, size=int(on_trip.sum()), p=t_p)
        amount = _amounts(rng, m_cat[m], scale)
        if cfg.drift_start_day is not None:
            drifted = day >= cfg.drift_start_day
            amount[drifted] *= cfg.drift_amount_mult
            # Holiday shoppers move online: re-draw 30% of drifted purchases at online merchants.
            move = drifted & (rng.random(n) < 0.3)
            if move.any():
                online_idx = np.flatnonzero(m_online & (m_cat != "transit"))
                op = m_pop[online_idx] / m_pop[online_idx].sum()
                m[move] = rng.choice(online_idx, size=int(move.sum()), p=op)
                amount[move] = _amounts(rng, m_cat[m[move]], scale) * cfg.drift_amount_mult
        # Legit look-alike of card testing: transit taps come in small bursts.
        transit = np.flatnonzero(m_cat[m] == "transit")
        extra = transit[rng.random(len(transit)) < 0.4]
        if len(extra):
            k = rng.integers(1, 4, size=len(extra))
            src = np.repeat(extra, k)
            gaps = rng.integers(3 * MINUTE, 40 * MINUTE, size=len(src))
            m = np.concatenate([m, m[src]])
            ts = np.concatenate([ts, ts[src] + gaps])
            amount = np.concatenate([amount, rng.uniform(1.0, 4.0, len(src))])
        rows_card.append(np.full(len(m), ci))
        rows_merchant.append(m)
        rows_ts.append(ts)
        rows_amount.append(amount)

    legit = pd.DataFrame(
        {
            "card_idx": np.concatenate(rows_card),
            "merchant_idx": np.concatenate(rows_merchant),
            "ts": np.concatenate(rows_ts).astype(np.int64),
            "amount": np.concatenate(rows_amount),
        }
    )
    legit["pattern"] = "legit"

    fraud = _fraud(rng, cfg, cards, merchants)
    allt = pd.concat([legit, fraud], ignore_index=True)
    end = EPOCH_START_MS + cfg.days * DAY
    allt = allt[(allt["ts"] >= EPOCH_START_MS) & (allt["ts"] < end)]
    allt = allt.sort_values(["ts", "card_idx"], kind="stable").reset_index(drop=True)
    allt["txn_id"] = [f"t{i:08d}" for i in range(len(allt))]

    mi = allt["merchant_idx"].to_numpy()
    transactions = pd.DataFrame(
        {
            "txn_id": allt["txn_id"],
            "ts": allt["ts"].astype(np.int64),
            "card_id": cards["card_id"].to_numpy()[allt["card_idx"].to_numpy()],
            "merchant_id": m_ids[mi],
            "amount": allt["amount"].round(2),
            "country": m_country[mi],
            "channel": np.where(m_online[mi], "online", "pos"),
            "category": m_cat[mi],
        }
    )
    is_fraud = (allt["pattern"] != "legit").to_numpy()
    ground_truth = pd.DataFrame(
        {"txn_id": allt["txn_id"], "is_fraud": is_fraud.astype(np.int8), "pattern": allt["pattern"]}
    )

    f_idx = np.flatnonzero(is_fraud)
    reported = f_idx[rng.random(len(f_idx)) >= cfg.unreported_rate]
    delay_days = np.clip(
        rng.lognormal(np.log(cfg.label_delay_median_days), 0.6, len(reported)), 1, 90
    )
    labels = pd.DataFrame(
        {
            "txn_id": allt["txn_id"].to_numpy()[reported],
            "label_ts": (allt["ts"].to_numpy()[reported] + delay_days * DAY).astype(np.int64),
            "is_fraud": np.ones(len(reported), dtype=np.int8),
        }
    )
    return Dataset(
        transactions=transactions,
        cards=cards[["card_id", "home_country"]].copy(),
        merchants=merchants[["merchant_id", "category", "country", "channel"]].copy(),
        labels=labels,
        ground_truth=ground_truth,
    )


def _fraud(
    rng: np.random.Generator, cfg: GeneratorConfig, cards: pd.DataFrame, merchants: pd.DataFrame
) -> pd.DataFrame:
    m_cat = merchants["category"].to_numpy()
    m_online = (merchants["channel"] == "online").to_numpy()
    m_country = merchants["country"].to_numpy()
    home = cards["home_country"].to_numpy()
    testing = np.flatnonzero(merchants["testing_target"].to_numpy())
    cashout = np.flatnonzero(np.isin(m_cat, ["electronics", "gift_cards", "jewelry"]) & m_online)
    risky = np.flatnonzero(np.isin(m_cat, list(_RISKY)))
    span = cfg.days * DAY
    out: list[tuple[int, int, int, float, str]] = []

    for _ in range(cfg.card_testing_episodes):
        c = int(rng.integers(len(cards)))
        t = EPOCH_START_MS + int(rng.integers(DAY, span - DAY))
        n_test = int(rng.integers(3, 16))
        ms = rng.choice(testing, size=int(rng.integers(1, 4)))
        times = t + np.cumsum(rng.exponential(25 * SECOND, n_test)).astype(np.int64)
        for k in range(n_test):
            out.append(
                (
                    c,
                    int(rng.choice(ms)),
                    int(times[k]),
                    float(rng.uniform(0.5, 3.0)),
                    "card_testing",
                )
            )
        if rng.random() < 0.6:
            t2 = int(times[-1]) + int(rng.integers(10 * MINUTE, 6 * HOUR))
            for _k in range(int(rng.integers(1, 4))):
                out.append(
                    (c, int(rng.choice(cashout)), t2, float(rng.uniform(200, 1500)), "card_testing")
                )
                t2 += int(rng.integers(2 * MINUTE, 30 * MINUTE))

    for _ in range(cfg.ato_episodes):
        c = int(rng.integers(len(cards)))
        t = EPOCH_START_MS + int(rng.integers(DAY, span - DAY))
        foreign = rng.random() < 0.6
        domestic = m_country[risky] == home[c]
        cand = risky[~domestic if foreign else domestic]
        if len(cand) == 0:
            cand = risky
        for _k in range(int(rng.integers(2, 7))):
            amt = float(np.clip(rng.lognormal(np.log(500), 0.7), 80, 3000))
            out.append((c, int(rng.choice(cand)), t, amt, "account_takeover"))
            t += int(rng.exponential(50 * MINUTE))

    lo = min(10, cfg.days // 4)
    hi = max(lo + 1, cfg.days - 15)
    ring_ids = np.flatnonzero(merchants["ring_target"].to_numpy())
    for k, mi in enumerate(ring_ids):
        # The last ring starts 15 days before the end, so the most recent period
        # always contains a merchant compromise that no label has reported yet.
        day0 = (
            cfg.days - 15 if k == len(ring_ids) - 1 and cfg.days > 20 else int(rng.integers(lo, hi))
        )
        start = EPOCH_START_MS + day0 * DAY
        length = int(rng.integers(6, 13)) * DAY
        for _k in range(cfg.ring_txns_per_merchant):
            out.append(
                (
                    int(rng.integers(len(cards))),
                    int(mi),
                    start + int(rng.integers(0, length)),
                    float(rng.uniform(40, 400)),
                    "merchant_ring",
                )
            )

    return pd.DataFrame(
        out, columns=["card_idx", "merchant_idx", "ts", "amount", "pattern"]
    ).astype({"ts": np.int64})


__all__ = ["CATEGORIES", "Dataset", "GeneratorConfig", "generate"]
