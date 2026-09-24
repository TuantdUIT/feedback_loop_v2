"""Kiểm tra broker SQLite: FIFO theo topic, ack nguyên tử, retry, dead-letter, khôi phục và tranh chấp luồng."""

import threading
import time

import pytest

from kafka_simulation.broker import Broker


@pytest.fixture()
def broker(tmp_path):
    b = Broker(tmp_path / "q.db", max_attempts=3, backoff_sec=(0, 0, 0))
    b.conn.execute("CREATE TABLE side (k TEXT PRIMARY KEY, v TEXT)")
    yield b
    b.close()


def test_fifo_per_topic_and_empty(broker: Broker) -> None:
    for i in range(3):
        broker.publish("a", "r1", f"c{i}")
    broker.publish("b", "r1", "other")
    assert [broker.claim("a").case_id for _ in range(3)] == ["c0", "c1", "c2"]
    assert broker.claim("a") is None
    assert broker.claim("b").case_id == "other"


def test_ack_moves_to_next_topic_with_side_write(broker: Broker) -> None:
    broker.publish("a", "r1", "c0")
    msg = broker.claim("a")
    broker.ack(msg, "b", [("INSERT INTO side VALUES (?, ?)", ("c0", "x"))])
    assert broker.counts("r1") == {"a": {"done": 1}, "b": {"pending": 1}}
    assert broker.query("SELECT v FROM side")[0]["v"] == "x"


def test_failed_side_write_rolls_back_ack(broker: Broker) -> None:
    broker.publish("a", "r1", "c0")
    msg = broker.claim("a")
    with pytest.raises(Exception):
        broker.ack(msg, "b", [("INSERT INTO no_such_table VALUES (1)", ())])
    assert broker.counts("r1") == {"a": {"processing": 1}}


def test_retry_then_dead_letter(broker: Broker) -> None:
    broker.publish("a", "r1", "c0")
    for attempt in (1, 2):
        msg = broker.claim("a")
        assert msg.attempts == attempt
        assert broker.nack(msg, "tạm thời") is True
    msg = broker.claim("a")
    assert broker.nack(msg, "vẫn lỗi") is False
    assert broker.claim("a") is None
    assert broker.counts() == {"a": {"failed": 1}}


def test_backoff_delays_redelivery(tmp_path) -> None:
    b = Broker(tmp_path / "q.db", backoff_sec=(60,))
    b.publish("a", "r1", "c0")
    b.nack(b.claim("a"), "429")
    assert b.claim("a") is None
    b.close()


def test_non_retryable_goes_straight_to_failed(broker: Broker) -> None:
    broker.publish("a", "r1", "c0")
    assert broker.nack(broker.claim("a"), "401", retry=False) is False
    assert broker.counts() == {"a": {"failed": 1}}


def test_recover_after_crash(tmp_path) -> None:
    path = tmp_path / "q.db"
    first = Broker(path)
    first.publish("a", "r1", "c0")
    assert first.claim("a") is not None
    first.close()                                   # "chết" khi message đang processing
    second = Broker(path)
    assert second.recover() == 1
    assert second.claim("a").case_id == "c0"
    second.close()


def test_concurrent_claims_never_duplicate(broker: Broker) -> None:
    broker.publish_many("a", [("r1", f"c{i}") for i in range(200)])
    seen: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        while (msg := broker.claim("a")) is not None:
            broker.ack(msg)
            with lock:
                seen.append(msg.case_id)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == sorted(f"c{i}" for i in range(200))


def test_publish_wakes_waiting_worker(broker: Broker) -> None:
    woke = threading.Event()

    def waiter() -> None:
        broker.wait("a", timeout=5)
        woke.set()

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.1)
    broker.publish("a", "r1", "c0")
    assert woke.wait(2)
    t.join()
