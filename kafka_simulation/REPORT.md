# Tổng kết build pipeline Kafka mô phỏng

Ngày thực hiện: 2026-09-23.

> **Cập nhật cùng ngày — đơn giản hoá thành message queue.** `bus.py` (topic 3 partition, offset, consumer group, commit) đã được thay bằng `message_queue.py`: một hàng đợi FIFO với `publish()` / `consume()`. Producer không còn key SHA-1; consumer lấy hết hàng đợi rồi ghi JSON một lần, không còn commit/tiếp tục. Kết quả dữ liệu không đổi (52 CaseRecord, 69 span, 1 unlocated), output nay giữ đúng thứ tự capture. Các dòng về partition/commit bên dưới là lịch sử của bản trước.

## Đã hoàn thành

| Thành phần | Kết quả |
|---|---|
| Capture parser | Đọc file capture theo đường dẫn; ghép 52 `texts` với 52 `results` một lần tại nguồn; từ chối mảng lệch độ dài. |
| Message và producer | Giữ text nguyên văn, độ tin cậy float và chỉ số batch; kiểm tra schema trước khi gửi; ~~dùng SHA-1 của text làm key~~ (đã bỏ). |
| Message queue | `MessageQueue` FIFO trong bộ nhớ: `publish()`, `consume()` trả `None` khi rỗng, đếm `published`/`consumed`. *(Bản trước: topic 3 partition, offset, consumer group, commit.)* |
| Bridge | Dùng lại `CaseRecord`, `ParseResult`, `Span` của `feedback.core.schemas`; NFC, định vị span dài trước, fallback chỉ nới khoảng trắng, token/BIO căn ở biên span. |
| Consumer | Lấy tới khi hàng đợi rỗng, chuyển qua bridge, ghi JSON một lần. |
| Entry point | `python -m kafka_simulation.run_simulation` xuất `kafka_simulation/out/cases.json`; nhận nhiều đường dẫn capture. |

Không sửa `feedback/`, `frontend/` hoặc `template.txt`. Không gọi API staging, không dựng broker thật, không cài thư viện Kafka. Chỉ cài `pytest` vào `.venv` để chạy kiểm thử; không đổi `requirements.txt`.

## Kết quả kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Đọc capture gốc | 52 cặp input/output; response chứa 70 entity. |
| Replay `template.txt` | **52 message → 52 CaseRecord, 69 span, 1 unlocated** (`template#00022`). Không bỏ message. |
| Metadata file output | Mọi record có `api_confidence`, `truncated_unknown=True`; 1 record giữ `confidence=0.0`. |
| Queue | `publish 52 | consume 52 | còn lại 0`; thứ tự output trùng thứ tự capture. |
| Fixture 52 mẫu với dấu Unicode đúng | **70/70 span**, 0 unlocated; 1 `nfc_shift`, 1 `loose_ws_match`. |
| Token/BIO | Hai ví dụ trong tài liệu khớp hoàn toàn; không có token cắt ngang span trong 52 mẫu. |
| Schema | Mọi record trong fixture và output thật round-trip qua `CaseRecord.from_dict(record.to_dict())`. |
| Test | `pytest kafka_simulation/tests -v`: **14 passed** (bỏ 4 test partition/commit, thêm 2 test hàng đợi). |

### Sai lệch của capture gốc

Trong `template.txt` mình vừa đọc, địa chỉ thứ 23 còn chứa literal `<0303>` và `<0323>`, ví dụ `Nguyê<0303>n Huê<0323>`. Đây là các ký tự chữ trong file, không phải dấu tổ hợp Unicode. Response lại trả entity với dấu tiếng Việt đúng. Vì vậy bridge không thể định vị một span bằng phép khớp nguyên văn hoặc nới khoảng trắng; nó gắn cờ `meta.unlocated` và tiếp tục xử lý 51 message còn lại.

Theo lựa chọn giữ capture gốc, pipeline **không** unescape trong đường chạy chính và **không** sửa `template.txt`. Chỉ fixture kiểm thử thay hai marker bằng dấu tổ hợp thật. Điều này giải thích vì sao mốc 70/70 đạt trong kiểm thử bridge nhưng lệnh replay capture gốc trả 69/70. Muốn replay thực tế đạt 70/70 mà vẫn giữ quy tắc hiện tại, cần thêm một capture mới chứa ký tự Unicode thật.

## Phạm vi còn lại

Pipeline dừng ở file `CaseRecord`. `feedback/core/cascade.py` và `feedback/scripts/run_loop.py` vẫn là stub nên chưa nối vào Lớp 0→3. Hàng đợi không có ack/retry/DLQ; consumer lỗi giữa chừng thì chạy lại toàn bộ.
