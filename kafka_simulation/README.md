# Message queue: nạp output NER vào pipeline đánh giá

Component message queue của pipeline trong [BUILD_PIPELINE.md](../BUILD_PIPELINE.md). Đọc capture curl
(`texts[]` + `results[]`), kiểm tra từng message, dựng `feedback.core.schemas.CaseRecord`, và luân
chuyển case giữa các giai đoạn qua các topic. Không gọi endpoint staging.

```
capture ─► capture_parser ─► message.validate ─► bridge ─► CaseRecord ─► broker [ingested] ─► …worker…
```

## Hai hàng đợi

| | `broker.py` — dùng trong pipeline | `message_queue.py` — mô phỏng tối giản |
|---|---|---|
| Lưu trữ | SQLite trên đĩa (`feedback/store/pipeline.db`) | Bộ nhớ, mất khi tắt tiến trình |
| Topic | Nhiều topic theo giai đoạn | Một hàng FIFO |
| Ack / retry | Ack nguyên tử cùng ghi dữ liệu; retry có backoff; dead-letter sau 3 lần | Không |
| Crash | `recover()` trả message đang xử lý về hàng (at-least-once) | Chạy lại từ đầu |

Message của broker chỉ mang `run_id` + `case_id`; dữ liệu case nằm ở bảng `cases`
(`feedback/pipeline/store.py`). `run_simulation.py` + `MessageQueue` giữ lại cho test và chạy nhanh:

```powershell
.venv-layer1/Scripts/python.exe -m kafka_simulation.run_simulation
.venv-layer1/Scripts/python.exe -m pytest kafka_simulation/tests -v
```

## Định dạng capture

Như `template.txt`: một lệnh `curl … --data '{"texts": [...]}'` rồi response `{"results": [{"result": {"input": …, "L1": […], …, "L7": […]}}]}`.
**Không có `confidence`** (bỏ từ 2026-09-24); capture cũ còn trường này vẫn đọc được, trường bị bỏ qua.

`template.txt` hiện là **dữ liệu mô phỏng** 46 case với `results` là đáp án đúng. Capture thật trước đó
(52 case, còn chữ dính như `Bến NghéQuận 1` và marker `<0303>`) được giữ ở
`tests/fixtures/capture_52.txt` để test thuật toán định vị span trên dữ liệu bẩn.

## Thành phần

- `capture_parser.py`: bóc `texts[]` và `results[]`; báo lỗi nếu hai mảng lệch độ dài.
- `message.py`: schema và `validate()` cho một message.
- `bridge.py`: NFC text, định vị entity dài trước, khớp nguyên văn rồi nới khoảng trắng; tokens/BIO sinh bằng
  `feedback.core.bio.tokens_and_bio`. Span không định vị được ghi ở `meta.unlocated`; `meta.truncated_unknown=True`.
- `broker.py`: broker SQLite ở trên.
- `producer.py` / `consumer.py` / `message_queue.py` / `run_simulation.py`: luồng mô phỏng tối giản.
