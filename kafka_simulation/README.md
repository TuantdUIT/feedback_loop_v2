# Message queue mô phỏng: nạp output NER vào CaseRecord

Pipeline này mô phỏng một **message queue FIFO trong cùng tiến trình Python** để luân chuyển dữ liệu từ output của hệ thống NER sang feedback loop. Nó đọc capture curl, ghép `texts[i]` với `results[i]`, đẩy từng địa chỉ vào hàng đợi, rồi consumer lấy ra và chuyển thành `feedback.core.schemas.CaseRecord`. Không gọi endpoint staging và chưa gọi cascade của `feedback/`.

```
template.txt ─► capture_parser ─► producer ─► MessageQueue ─► consumer ─► bridge ─► out/cases.json
                                  (publish)     (FIFO)       (consume)
```

## Chạy

Từ thư mục gốc dự án:

```powershell
.venv/Scripts/python.exe -m kafka_simulation.run_simulation
.venv/Scripts/python.exe -m pytest kafka_simulation/tests -v
```

Mặc định lệnh đọc `kafka_simulation/template.txt` và ghi `kafka_simulation/out/cases.json`. Muốn dữ liệu mới thì thay `template.txt` hoặc truyền một hay nhiều capture khác:

```powershell
.venv/Scripts/python.exe -m kafka_simulation.run_simulation kafka_simulation/template.txt kafka_simulation/template_2026-10-01.txt --output kafka_simulation/out/multi_capture.json
```

`out/` được bỏ qua bởi `.gitignore` trong thư mục này. JSON xuất ra là danh sách `CaseRecord.to_dict()`, **giữ đúng thứ tự** trong capture; `case_id` (`stem#NNNNN`) giữ chỉ số gốc để truy vết. `meta.api_confidence` giữ nguyên float của API, `meta.truncated_unknown=True` cho biết API không cung cấp tín hiệu cắt cụt. Span không tìm được bị bỏ và ghi ở `meta.unlocated`.

## Các thành phần

- `capture_parser.py`: đọc request `texts[]` và response `results[]`; báo lỗi nếu hai mảng lệch độ dài.
- `message.py`: schema và `validate()` cho một message.
- `producer.py`: gắn nguồn và thời gian, kiểm tra message, bỏ message hỏng, `publish()` vào hàng đợi.
- `message_queue.py`: `MessageQueue` — `publish()` vào cuối, `consume()` lấy từ đầu, trả `None` khi rỗng; đếm `published` / `consumed`.
- `consumer.py`: `consume()` tới khi hàng đợi rỗng, chuyển qua `bridge`, ghi JSON một lần.
- `bridge.py`: NFC text, định vị entity dài trước, khớp nguyên văn rồi nới khoảng trắng, cắt token ở biên span và dấu câu, sinh BIO và `CaseRecord`.

## Giới hạn mô phỏng

Đây là hàng đợi trong bộ nhớ, **không phải broker thật**: không có partition, offset, consumer group, commit/ack, retry hay DLQ. Hàng đợi chỉ tồn tại trong phiên chạy; nếu consumer lỗi giữa chừng thì chạy lại toàn bộ. Cascade hiện là stub; pipeline dừng ở file JSON của `CaseRecord`.

Capture gốc hiện còn chữ `<0303>` và `<0323>` trong text của `template#00022`, thay cho dấu Unicode rời. Pipeline giữ nguyên dữ liệu này, nên replay thực tế cho **52 CaseRecord, 69 span định vị, 1 unlocated**. Fixture `tests/fixtures/template_52.json` chỉ sửa hai marker Unicode trong bản fixture để kiểm tra thuật toán trên 52 mẫu: **70/70 span**. `template.txt` không bị sửa. Chi tiết ở [REPORT.md](REPORT.md).
