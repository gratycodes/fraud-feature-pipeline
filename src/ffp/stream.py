"""Streaming path: consume transactions and keep the online store current.

The processor only depends on an ``EventSource``: an in-process iterator for tests
and the no-Docker demo, or a Kafka/Redpanda consumer. Offsets are committed after
the Redis write succeeds (at-least-once); because ``OnlineStore.ingest`` is
idempotent, a redelivered batch changes nothing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from ffp.online import OnlineStore
from ffp.schema import Transaction


class EventSource(Protocol):
    """Yields events; may yield ``None`` as an idle tick so the processor can flush."""

    def __iter__(self) -> Iterator[Transaction | None]: ...

    def commit(self) -> None: ...


class InProcessSource:
    """Feeds events from any iterable. ``commit`` records how far we got."""

    def __init__(self, events: Iterable[Transaction]) -> None:
        self._events = events
        self.committed = 0
        self._seen = 0

    def __iter__(self) -> Iterator[Transaction | None]:
        for e in self._events:
            self._seen += 1
            yield e

    def commit(self) -> None:
        self.committed = self._seen


class KafkaSource:
    """Kafka/Redpanda consumer yielding ``Transaction`` objects from JSON values.

    ``idle_timeout_s`` stops iteration after that long without a message (useful
    for batch backfills and tests); ``None`` runs forever.
    """

    def __init__(
        self,
        bootstrap: str,
        topic: str = "transactions",
        group_id: str = "ffp-feature-processor",
        idle_timeout_s: float | None = None,
    ) -> None:
        from confluent_kafka import Consumer

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        self.consumer.subscribe([topic])
        self.idle_timeout_s = idle_timeout_s

    def __iter__(self) -> Iterator[Transaction]:
        last = time.monotonic()
        while True:
            msg = self.consumer.poll(0.5)
            if msg is None:
                if (
                    self.idle_timeout_s is not None
                    and time.monotonic() - last > self.idle_timeout_s
                ):
                    return
                continue
            if msg.error():
                raise RuntimeError(str(msg.error()))
            last = time.monotonic()
            yield Transaction.from_dict(json.loads(msg.value() or b"{}"))

    def commit(self) -> None:
        try:
            self.consumer.commit(asynchronous=False)
        except Exception as exc:  # nothing consumed yet -> "no offset to commit"
            if "NO_OFFSET" not in str(exc):
                raise

    def close(self) -> None:
        self.consumer.close()


@dataclass
class ProcessorStats:
    events: int = 0
    batches: int = 0
    seconds: float = 0.0

    @property
    def events_per_s(self) -> float:
        return self.events / self.seconds if self.seconds else 0.0


class StreamProcessor:
    def __init__(self, store: OnlineStore, batch_size: int = 500) -> None:
        self.store = store
        self.batch_size = batch_size

    def run(self, source: EventSource, max_events: int | None = None) -> ProcessorStats:
        stats = ProcessorStats()
        start = time.perf_counter()
        batch: list[Transaction] = []
        for e in source:
            if e is None:
                self._flush(batch, source, stats)
                continue
            batch.append(e)
            if len(batch) >= self.batch_size:
                self._flush(batch, source, stats)
            if max_events is not None and stats.events + len(batch) >= max_events:
                break
        self._flush(batch, source, stats)
        stats.seconds = time.perf_counter() - start
        return stats

    def _flush(self, batch: list[Transaction], source: EventSource, stats: ProcessorStats) -> None:
        if not batch:
            return
        stats.events += self.store.ingest(batch)
        stats.batches += 1
        source.commit()
        batch.clear()


class KafkaPublisher:
    """Produces events keyed by card_id, so one card's events stay ordered in a partition."""

    def __init__(self, bootstrap: str, topic: str = "transactions") -> None:
        from confluent_kafka import Producer

        self.topic = topic
        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5, "acks": "all"})
        self.errors: list[Any] = []
        self.sent = 0

    def _on_delivery(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self.errors.append(err)

    def send(self, e: Transaction) -> None:
        self.producer.produce(
            self.topic, key=e.card_id, value=json.dumps(e.to_dict()), on_delivery=self._on_delivery
        )
        self.sent += 1
        self.producer.poll(0)

    def flush(self, timeout_s: float = 30) -> None:
        remaining = self.producer.flush(timeout_s)
        if remaining or self.errors:
            raise RuntimeError(
                f"{remaining} undelivered, {len(self.errors)} errors: {self.errors[:1]}"
            )


def publish(events: Iterable[Transaction], bootstrap: str, topic: str = "transactions") -> int:
    pub = KafkaPublisher(bootstrap, topic)
    for e in events:
        pub.send(e)
    pub.flush()
    return pub.sent


def transactions_from_frame(df: Any) -> Iterator[Transaction]:
    cols = ["txn_id", "ts", "card_id", "merchant_id", "amount", "country", "channel", "category"]
    for row in df[cols].itertuples(index=False, name=None):
        yield Transaction(
            str(row[0]),
            int(row[1]),
            str(row[2]),
            str(row[3]),
            float(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
        )
