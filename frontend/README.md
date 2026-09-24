# Giao diện so sánh parser địa chỉ

Trang nội bộ gồm hai tab: thử một địa chỉ tiếng Việt với PhoBERT PyTorch FP32, PhoBERT ONNX FP32 và Qwen qua OpenRouter; và pipeline đánh giá output parser (xem cuối file). Một lần bấm **Parse cả 3 engine** sẽ gọi cả ba endpoint song song; kết quả nào xong sẽ hiện ngay.

## Chạy local

Từ thư mục gốc dự án:

```powershell
.venv-layer1/Scripts/python.exe -m uvicorn frontend.server:app --port 8000
```

Mở <http://localhost:8000>. Nếu môi trường chưa có web framework:

```powershell
.venv-layer1/Scripts/python.exe -m pip install fastapi uvicorn
```

Phiên bản đã kiểm tra: **fastapi 0.141.1**, **uvicorn 0.53.0**.

## Chi phí và hành vi

- Mỗi lần bấm Parse chủ động gọi Qwen qua OpenRouter và có thể tốn khoảng **$0.005** hoặc hơn. Nếu Qwen trả rỗng, hệ thống có thể retry tối đa ba lần; chi phí cộng dồn được hiển thị trong kết quả.
- Tải trang, nhập chữ và xem trạng thái không gọi OpenRouter. Nút bị khoá cho đến khi cả ba yêu cầu hoàn tất.
- Cấu hình Qwen lấy từ `.env` ở gốc dự án: `OPENROUTER_API` và `OPENROUTER_MODEL`. Prompt lấy từ `prompt/v2/compiled/system_prompt_v2_with_partial_input.txt`.
- Hai model local được nạp lười và giữ trong bộ nhớ sau lần gọi đầu. Mỗi model có thể chiếm hàng trăm MB RAM; trang hiển thị RAM của tiến trình server và RAM còn trống.
- Span của PhoBERT được chuẩn hoá qua `scripts/layer1_adapter.py`. PhoBERT không cung cấp offset ký tự nên cột này hiển thị “—”.

## Endpoint

`GET /api/status` cho biết model nào đã nạp, RAM và tình trạng cấu hình Qwen. Ba endpoint parse nhận JSON `{"text":"..."}`: `POST /api/parse/pytorch`, `POST /api/parse/onnx`, `POST /api/parse/qwen`.

## Tab "Mô phỏng Production" — pipeline đánh giá

Worker của pipeline ([BUILD_PIPELINE.md](../BUILD_PIPELINE.md)) chạy cùng tiến trình server, khởi động khi server bật. Đặt `PIPELINE_DISABLED=1` để chỉ bật trang so sánh.

- **Upload:** chọn file capture (định dạng `kafka_simulation/template.txt`) → xem số case, message bị loại, chi phí ước tính và trần $2 → bấm **Chạy pipeline**. Tick **Lượt hiệu chuẩn** và chọn thêm file gold (định dạng `golden_dataset`) để đo judge.
- **Run:** sơ đồ số case ở từng giai đoạn (tự làm mới mỗi 2 giây khi đang chạy), chi phí so với trần, khuyến nghị retrain, metrics đồng thuận / sức khoẻ judge / chất lượng model / chi phí.
- **Case:** lọc theo quyết định hoặc audit; bấm một dòng để xem hai bản parse tô màu theo level, khác biệt, cả hai lần chấm của judge kèm lỗi judge nêu, Lớp 0 và CaseRecord đầy đủ.
- **Tải hàng chờ gán nhãn:** file JSON định dạng golden (chỉ có text, result để trống) gồm case random audit và case judge lật — gán nhãn xong dùng làm gold cho lượt hiệu chuẩn.

Không chạy `feedback.scripts.run_pipeline` cùng lúc với server trên cùng DB: hai tiến trình sẽ cùng giành message.

API: `POST /api/runs/preview`, `POST /api/runs`, `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/cases`, `GET /api/runs/{id}/cases/{case_id}`, `GET /api/runs/{id}/label-queue`, `GET /api/pipeline/config`.
