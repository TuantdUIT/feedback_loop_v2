# Giao diện so sánh parser địa chỉ

Trang nội bộ để thử một địa chỉ tiếng Việt với PhoBERT PyTorch FP32, PhoBERT ONNX FP32 và Qwen qua OpenRouter. Một lần bấm **Parse cả 3 engine** sẽ gọi cả ba endpoint song song; kết quả nào xong sẽ hiện ngay.

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
