from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ffp.online import OnlineStore
from ffp.schema import EPOCH_START_MS, Transaction
from ffp.serving import ScoringService, create_app


@pytest.fixture
def client(
    store: OnlineStore, small, small_snapshots, small_bundle
) -> Iterator[tuple[TestClient, ScoringService]]:  # type: ignore[no-untyped-def]
    store.load_card_profiles(small.cards)
    store.load_merchant_rate_snapshots(small_snapshots)
    service = ScoringService(store, small_bundle[0])
    with TestClient(create_app(service)) as c:
        yield c, service


def _txn(**kw: object) -> dict[str, object]:
    base = {
        "txn_id": "api-1",
        "ts": EPOCH_START_MS + 19 * 86_400_000,
        "card_id": "c00001",
        "merchant_id": "m0001",
        "amount": 42.5,
        "country": "IN",
        "channel": "pos",
        "category": "grocery",
    }
    base.update(kw)
    return base


def test_score_returns_probability_and_decision(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    r = c.post("/score", json=_txn())
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "block"}
    assert body["features"] is None
    assert body["feature_ms"] >= 0 and body["model_ms"] >= 0


def test_explain_returns_every_feature(client) -> None:  # type: ignore[no-untyped-def]
    from ffp.features import MODEL_FEATURES

    c, _ = client
    body = c.post("/score?explain=true", json=_txn()).json()
    assert list(body["features"]) == list(MODEL_FEATURES)
    assert body["features"]["amount"] == 42.5


def test_scores_are_logged_for_skew_checks(client) -> None:  # type: ignore[no-untyped-def]
    c, service = client
    for i in range(3):
        c.post("/score", json=_txn(txn_id=f"log-{i}"))
    log = service.feature_log()
    assert log["txn_id"].tolist() == ["log-0", "log-1", "log-2"]


def test_burst_of_small_txns_changes_the_features(client) -> None:  # type: ignore[no-untyped-def]
    c, service = client
    t0 = EPOCH_START_MS + 19 * 86_400_000
    service.store.ingest(
        Transaction(
            f"b{i}", t0 + i * 20_000, "c00002", "m0002", 1.0, "IN", "online", "digital_goods"
        )
        for i in range(8)
    )
    f = c.post("/score?explain=true", json=_txn(card_id="c00002", ts=t0 + 150_000)).json()[
        "features"
    ]
    assert f["card_small_txn_count_1h"] == 8
    assert f["card_txn_count_1m"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"amount": -1},
        {"category": "weapons"},
        {"channel": "fax"},
        {"card_id": "c1|evil"},
        {"country": "IND"},
        {"ts": 0},
    ],
)
def test_invalid_requests_are_rejected(client, bad: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    assert c.post("/score", json=_txn(**bad)).status_code == 422


def test_health_and_model_info(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    assert c.get("/healthz").json() == {"ok": True}
    info = c.get("/model").json()
    assert info["calibration"]["chosen"] in {"none", "platt", "isotonic"}
