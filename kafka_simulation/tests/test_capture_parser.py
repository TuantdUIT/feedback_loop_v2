"""Kiểm tra ghép cặp request/response và từ chối capture lệch số lượng."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kafka_simulation.capture_parser import parse_capture
from kafka_simulation.message import validate
from kafka_simulation.message_queue import MessageQueue
from kafka_simulation.producer import DEFAULT_TEMPLATE, run_producer


FIXTURE = Path(__file__).parent / "fixtures/template_52.json"


def test_capture_has_52_pairs_and_70_entities() -> None:
    payloads = parse_capture(DEFAULT_TEMPLATE)
    assert len(payloads) == 52
    assert sum(len(entities) for item in payloads for entities in item["result"].values()) == 70
    assert [item["source"]["batch_index"] for item in payloads] == list(range(52))
    assert payloads[0]["text"] == "Thành phố Hồ Chí Minh"
    assert payloads[0]["source"]["endpoint"] == "/ner/predict-batch"


def test_capture_preserves_literal_escape_in_source() -> None:
    raw = parse_capture(DEFAULT_TEMPLATE)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "<0303>" in raw[22]["text"]
    assert "<0323>" in raw[22]["text"]
    assert "<0303>" not in fixture[22]["text"]
    assert fixture[22]["text"] != raw[22]["text"]


def test_mismatched_lengths_raise() -> None:
    capture = (
        "curl --location 'http://example.test/ner/predict-batch' --data '{\"texts\":[\"a\",\"b\"]}'\n"
        '{"results":[{"result":{"L1":[]},"confidence":0.9}]}\n'
    )
    with patch.object(Path, "read_text", return_value=capture):
        with pytest.raises(ValueError, match="texts .* khác results"):
            parse_capture("synthetic.txt")


def test_validate_rejects_bad_message_without_losing_batch() -> None:
    payload = parse_capture(DEFAULT_TEMPLATE)[0]
    message = {"msg_id": "template#00000", **payload,
               "produced_at": "2026-09-23T15:40:00+07:00"}
    assert validate(message) == []
    message["result"]["L2"] = "không phải list"
    assert any("result.L2" in error for error in validate(message))


def test_producer_skips_invalid_message_and_keeps_valid_one() -> None:
    result = {f"L{i}": [] for i in range(1, 8)}
    good = {"text": "abc", "result": result, "confidence": 0.9,
            "source": {"capture": "sample.txt", "endpoint": "/ner/predict-batch", "batch_index": 0}}
    bad = {"text": "def", "result": {**result, "L2": "invalid"}, "confidence": 0.8,
           "source": {"capture": "sample.txt", "endpoint": "/ner/predict-batch", "batch_index": 1}}
    queue = MessageQueue()
    with patch("kafka_simulation.producer.parse_capture", return_value=[good, bad]):
        stats = run_producer(queue, capture_paths=["sample.txt"])
    assert stats == {"sent": 1, "skipped": 1}
    assert len(queue) == 1
    assert queue.consume()["text"] == "abc"
