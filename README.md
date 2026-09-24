# Feedback loop cho NER địa chỉ tiếng Việt

Subsystem `feedback/` đánh giá output của parser NER L1–L7 đang triển khai: phát hiện case parse chưa tốt và đo xem có đủ căn cứ để retrain không. Thiết kế đầy đủ: [BUILD_PIPELINE.md](BUILD_PIPELINE.md).

```
upload (template.txt) ─► [ingested] ─► DeepSeek parse ─► [parsed] ─► so sánh ─┬─► [agreed] ───────────────┐
                                        + Lớp 0                               └─► [to_judge] ─► Qwen judge ─┤
                                                                                                 (A/B, B/A)  ▼
                                                                                     gate ─► [decided] ─► metrics + khuyến nghị
```

| Giai đoạn | Công cụ | Vai trò |
|---|---|---|
| Lớp 0 | Python thuần | Kiểm hình thức span/BIO của cả model lẫn DeepSeek |
| Parse | DeepSeek (`deepseek-flash`) | Parse lại cùng input bằng prompt v2 — giả thuyết cạnh tranh, không phải đáp án |
| Judge | Qwen (`qwen3.6-35b-a3b`, OpenRouter) | Chọn model / DeepSeek / hoà; chấm 2 lần đảo thứ tự, ẩn nguồn |
| Gate + metrics | Python thuần | Quyết định từng case, tỉ lệ lật, báo động giả, thiên vị, khuyến nghị retrain |

Mỗi giai đoạn là worker đọc một topic của message queue (SQLite, `kafka_simulation/broker.py`) và publish sang topic kế tiếp khi xong. Trần chi phí $2 / lần upload.

## Chạy

```powershell
# giao diện + worker (tab "Mô phỏng Production")
.venv-layer1/Scripts/python.exe -m uvicorn frontend.server:app --port 8000

# không cần giao diện
.venv-layer1/Scripts/python.exe -m feedback.scripts.run_pipeline kafka_simulation/template.txt

# test (mọi lời gọi API đi vào mock server, không tốn tiền)
.venv-layer1/Scripts/python.exe -m pytest feedback/tests kafka_simulation/tests
```

Sao chép `.env.example` thành `.env` và điền `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `OPENROUTER_API`, `OPENROUTER_MODEL`.

Khuyến nghị retrain chỉ bật sau khi judge được hiệu chuẩn trên gold cùng phân bố dữ liệu thật — xem [BUILD_PIPELINE.md §7.3–7.4](BUILD_PIPELINE.md).
