# Giao diện thử nghiệm 3 cách parse địa chỉ

Tài liệu dành cho agent thực thi. Đọc hết trước khi tạo file đầu tiên.

Nhiệm vụ: dựng một giao diện web chạy local trong thư mục **`frontend/`**, cho phép gõ một
địa chỉ tiếng Việt rồi bấm **một nút duy nhất** để chạy cả 3 engine, kết quả hiện cạnh nhau
để so sánh.

**Thực hiện tuần tự, không dừng lại hỏi xác nhận giữa chừng.**

---

## 1. Bối cảnh và vị trí trong dự án

Đây là **công cụ hỗ trợ**, không phải một lớp trong feedback loop. Nó không nằm trong bảng
12 bước ở `BUILD.md` §7 và không được đụng vào `feedback/`.

Mục đích: thay vì phải sửa script rồi chạy lại hàng loạt mỗi lần muốn thử một địa chỉ,
gõ thẳng vào ô nhập và thấy ngay cả 3 kết quả — phục vụ việc soi lỗi thủ công và kiểm
chứng 6 quy tắc chuẩn hoá.

Ba engine cần chạy:

| Engine | Nguồn | Ghi chú |
|---|---|---|
| **PyTorch FP32** | `models/phobert-ner-address-pytorch/` | chạy local, miễn phí |
| **ONNX FP32** | `models/phobert-ner-address-onnx/` | chạy local, miễn phí |
| **Qwen** | OpenRouter, xem `.env` | **tốn tiền mỗi lần gọi (~$0,005)** |

Đã đo được: **ONNX FP32 và PyTorch FP32 cho kết quả giống hệt nhau** (100/100 mẫu, và
5261/5261 token ở `models/REPORT.md`). Giao diện vẫn phải hiện cả hai để người dùng tự
kiểm chứng, nhưng đừng ngạc nhiên nếu hai cột luôn trùng nhau.

---

## 2. Quyết định đã chốt — KHÔNG được làm khác

1. **Không viết lại logic chuẩn hoá.** Bắt buộc `import` từ `scripts/layer1_adapter.py`
   (6 quy tắc R1–R6, đã calibrate trên dữ liệu thật). Nếu thấy cần sửa quy tắc, dừng lại
   và báo — không tự sửa, không tự chép sang file khác.

2. **Chỉ gọi Qwen khi người dùng chủ động bấm nút.** Nút Parse chạy cả 3 engine, trong đó
   có Qwen, nên **mỗi lần bấm tốn tiền thật (~$0,005)**. Vì vậy: không gọi khi tải trang,
   không gọi khi gõ (debounce/autocomplete), không tự gọi lại khi lỗi, và nút phải ghi rõ
   ước tính chi phí ngay trên mặt nút.

3. **Không tạo bước build.** Frontend là HTML + CSS + JavaScript thuần trong một vài file
   tĩnh. Không npm, không webpack, không React/Vue. Đây là công cụ nội bộ cho một người
   dùng, thêm toolchain chỉ tạo gánh nặng.

4. **Không đụng vào `feedback/`, `prompt/`, `models/`, `clean_data/`, `results/`.**
   Chỉ tạo file mới trong `frontend/`.

5. **Không commit gì trong `models/`.** Thư mục đó đã gitignore, giữ nguyên.

---

## 3. Môi trường

Dùng lại virtualenv đã có: **`.venv-layer1/`** — nó đã có sẵn `torch`, `transformers`,
`onnxruntime`. Chỉ cần cài thêm web framework:

```
.venv-layer1/Scripts/pip install fastapi uvicorn
```

Ghi lại version của `fastapi`, `uvicorn` vào `frontend/README.md`.

**Cảnh báo RAM:** PyTorch FP32 chiếm ~373 MB sau khi nạp (đỉnh 744 MB), ONNX FP32 chiếm
~917 MB (đỉnh 1.110 MB). Nạp cả hai cùng lúc là **~1,3 GB thường trực**. Máy đích có 16 GB
nhưng thường chỉ còn ~5 GB trống. Vì vậy:

- Nạp model **một lần lúc khởi động server**, không nạp lại mỗi request
- Nạp **lười (lazy)**: chỉ nạp engine nào được gọi lần đầu, không nạp sẵn cả hai
- Hiện trạng thái đã nạp / chưa nạp của từng engine trên giao diện

---

## 4. Cấu trúc thư mục cần tạo

```
frontend/
├── README.md              cách chạy, version package, cảnh báo chi phí
├── server.py              FastAPI: nạp model, 3 endpoint parse, phục vụ file tĩnh
├── engines.py             bọc 3 engine về cùng một giao diện hàm
└── static/
    ├── index.html         giao diện một trang
    ├── app.js             gọi API, dựng bảng so sánh
    └── style.css
```

---

## 5. Backend — `server.py` và `engines.py`

### 5.1. `engines.py` — bọc 3 engine về cùng một dạng trả về

Mỗi engine trả về **cùng một cấu trúc**, để frontend không phải biết engine nào khác engine nào:

```python
{
  "engine": "pytorch_fp32" | "onnx_fp32" | "qwen",
  "ok": bool,
  "error": str | None,
  "spans": [{"level": "L5", "text": "...", "start": int|None, "end": int|None,
             "truncated": bool|None}],
  "level_chain": "L6>L5>L3",
  "raw_labels": [{"token": "th@@", "label": "B_STREET"}] | None,   # chỉ PhoBERT
  "elapsed_sec": float,
  "cost_usd": float,          # 0 với engine local
  "meta": {...}               # token count, reasoning_tokens... tuỳ engine
}
```

**Quan trọng — không được giả vờ có dữ liệu không tồn tại:**

- PhoBERT (cả PyTorch lẫn ONNX) **không có offset ký tự** — tokenizer là loại chậm,
  `return_offsets_mapping` ném `NotImplementedError`. Để `start`/`end`/`truncated` là
  `null`, **không được bịa** bằng cách dò chuỗi.
- Chỉ Qwen mới có `start`/`end`/`truncated` thật, vì nó sinh thẳng theo
  `prompt/v2/compiled/system_prompt_v2_with_partial_input.txt`.

**`raw_labels` chỉ có ở PhoBERT** và là nhãn **trước khi qua adapter** (`B_STREET`,
`STREET_TYPE`, …). Đây là dữ liệu quan trọng nhất để soi lỗi — xem §6.3.

### 5.2. Engine Qwen — bắt buộc dùng lại cấu hình đã có

Sao chép đúng 4 biện pháp chống kẹt vòng lặp suy luận từ `scripts/external_api/qwen/run_qwen_on_file.py`:

```python
max_tokens = 12000
reasoning  = {"max_tokens": 10000}
response_format = {"type": "json_object"}   # tự tắt nếu provider trả HTTP 400
# + retry khi trả về RỖNG dù đã tốn phí, thang giảm dần [10000, 5000, 3000]
```

Lý do (đã đo trên 100 mẫu): mẫu thành công có `reasoning_tokens` tối đa 7.733, mẫu kẹt
lặp tối thiểu 17.021 — ngưỡng 10.000 chặn hết ca lặp mà không cắt ca hợp lệ.

**Xử lý lỗi bắt buộc hiện rõ trên giao diện, không nuốt:**

| Tình huống | Hiện gì |
|---|---|
| HTTP 403 `Key limit exceeded` | *"Key OpenRouter đã hết hạn mức chi tiêu"* — nêu rõ, đừng chỉ ghi "lỗi" |
| Trả về rỗng dù tốn phí (kẹt lặp) | Nêu rõ đã retry mấy lần, tổng chi phí đã mất |
| Timeout | Nêu rõ, kèm thời gian đã chờ |

### 5.3. Endpoint

| Method | Đường dẫn | Việc |
|---|---|---|
| `GET` | `/` | Trả `static/index.html` |
| `GET` | `/api/status` | Engine nào đã nạp, RAM hiện tại, key OpenRouter có cấu hình chưa |
| `POST` | `/api/parse/pytorch` | Chạy riêng PyTorch FP32 |
| `POST` | `/api/parse/onnx` | Chạy riêng ONNX FP32 |
| `POST` | `/api/parse/qwen` | Chạy riêng Qwen |

**Ba endpoint riêng, nhưng giao diện chỉ có MỘT nút** — nút đó gọi cả 3 endpoint.

Tách endpoint thay vì gộp thành một là chủ ý, vì ba engine có đặc tính rất khác nhau:

- **Chạy song song, hiện kết quả ngay khi có** — không chờ đủ cả 3. Engine local trả về
  sau ~25–73 ms, Qwen mất **15–90 giây**. Nếu gộp một endpoint thì người dùng phải ngồi
  chờ Qwen mới thấy được kết quả PhoBERT, dù nó đã xong từ lâu.
- **Lỗi độc lập** — Qwen hỏng (hết hạn mức, kẹt vòng lặp) không được làm mất kết quả của
  hai engine local.
- **Timeout riêng** — local vài giây là quá đủ, Qwen cần tới 300 giây.

---

## 6. Giao diện — `static/`

### 6.1. Bố cục

```
┌──────────────────────────────────────────────────────────────────┐
│ [ô nhập địa chỉ                                                 ] │
│                                                                  │
│          [▶ Parse cả 3 engine — tốn ~$0.005 cho Qwen]            │
│                                                                  │
│ Trạng thái: PyTorch ✓ · ONNX ✓ · Qwen ⏳ đang chạy (12s)          │
├──────────────────────────────────────────────────────────────────┤
│ CHUỖI LEVEL                                                      │
│   PyTorch FP32   L6>L5>L3     ┐                                  │
│   ONNX FP32      L6>L5>L3     ├─ tô màu giống/khác nhau          │
│   Qwen           ⏳ đang chờ… ┘                                  │
├──────────────────────────────────────────────────────────────────┤
│ BẢNG SPAN CHI TIẾT (3 cột cạnh nhau)                             │
│   Level | Text | start:end | truncated                           │
└──────────────────────────────────────────────────────────────────┘
```

**Một nút duy nhất, gọi cả 3 endpoint song song.** Không có nút riêng cho từng engine.

Vì Qwen chậm hơn engine local khoảng **500–3.000 lần** (25–73 ms so với 15–90 giây), giao
diện **không được chờ đủ cả 3 rồi mới vẽ**. Hai cột local phải hiện gần như tức thì, cột
Qwen hiện `⏳ đang chờ…` kèm đồng hồ đếm giây rồi tự cập nhật khi có kết quả.

Nút ghi rõ chi phí ngay trên mặt nút. Trong lúc đang chạy phải **khoá nút** lại, tránh bấm
chồng làm gọi Qwen nhiều lần.

### 6.2. Quy tắc hiển thị

- **Chuỗi level** dạng `L6>L5>L3`, theo thứ tự xuất hiện trong địa chỉ (sắp theo `start`
  nếu có, còn không thì giữ nguyên thứ tự engine trả về).
- **Tô màu so sánh**: engine nào trùng nhau thì cùng màu; engine lệch thì tô khác để mắt
  bắt được ngay. Không dùng màu đỏ/xanh kiểu đúng/sai — **chưa có gold set**, không engine
  nào là chuẩn.
- Ô nào không có dữ liệu (offset của PhoBERT) hiện `—`, không hiện `0` hay `null`.
- Hiện `elapsed_sec` và `cost_usd` của từng engine.
- Bốn trạng thái phải phân biệt rõ, không được lẫn: `(chưa chạy)` · `⏳ đang chờ…` ·
  `(rỗng)` (chạy xong, model không tìm thấy thực thể nào) · `⚠ lỗi`. Đặc biệt `(rỗng)` và
  `⚠ lỗi` mang ý nghĩa hoàn toàn khác nhau — một bên là kết quả hợp lệ, một bên là hỏng.
- Khi người dùng sửa ô nhập, các kết quả cũ phải được đánh dấu là **đã cũ** (làm mờ hoặc
  ghi rõ "kết quả của địa chỉ trước"), không được để người dùng tưởng đó là kết quả của
  địa chỉ đang gõ. Không tự động chạy lại.

### 6.3. Khung "soi quy tắc chuẩn hoá" — phần giá trị nhất

Với PhoBERT, thêm một khu vực có thể mở/đóng, hiện **trước và sau khi qua adapter**:

```
Nhãn thô từ model          →  Sau 6 quy tắc
  đường  STREET_TYPE            L5  "đường thạnh xuân"
  th@@   B_STREET
  ạnh    B_STREET
  xuân   I_STREET
```

Đây là công cụ soi lỗi quan trọng nhất của cả giao diện: nó cho thấy quy tắc nào đã gộp
cái gì. Ví dụ trên là ca thật — R6 gộp `th@@`+`ạnh` (BPE cắt đôi từ "thạnh") thành một span
thay vì hai.

Nếu làm được, đánh dấu **quy tắc nào đã thay đổi nhãn nào** (R1/R4/R5/R6). Không bắt buộc,
nhưng rất hữu ích.

---

## 7. Nghiệm thu

Chạy và xác nhận từng mục:

```bash
# 1. Server khởi động được
.venv-layer1/Scripts/python.exe -m uvicorn frontend.server:app --port 8000

# 2. Mở http://localhost:8000 — trang hiện ra, không lỗi console
```

| # | Kiểm tra | Kết quả mong đợi |
|---|---|---|
| 1 | Gõ `501 đường nơ trang long, bình thạnh`, bấm Parse | **Hai cột local hiện trong vòng ~1 giây**, không chờ Qwen; cột Qwen hiện `⏳ đang chờ…` |
| 2 | Chờ Qwen xong | Cột Qwen tự cập nhật, không phải bấm lại hay tải lại trang |
| 3 | So PyTorch với ONNX | Trùng nhau (`L6>L5>L3`) — đã đo 100/100 mẫu |
| 4 | Gõ `đường thạnh xuân`, Parse → mở khung soi quy tắc | Thấy `th@@`/`ạnh` gộp thành **một** span L5 |
| 5 | Gõ `Thành phố Đà Nẵng`, Parse | Thấy nhãn thô `B_CITY`/`I_DIST`… và sau adapter gộp lại |
| 6 | Cột offset của PyTorch/ONNX | Hiện `—`, KHÔNG hiện số bịa |
| 7 | Bấm Parse liên tục nhiều lần | Nút bị khoá khi đang chạy — không gọi Qwen chồng lên nhau |
| 8 | Key OpenRouter hết hạn mức | Hai cột local vẫn hiện bình thường; chỉ cột Qwen báo lỗi rõ ràng |
| 9 | Sửa ô nhập sau khi đã có kết quả | Kết quả cũ bị đánh dấu là cũ; KHÔNG tự chạy lại |
| 10 | Tải lại trang, chưa bấm gì | Tab Network: KHÔNG có request nào tới OpenRouter |
| 11 | `git status` | Chỉ có file mới trong `frontend/`, không có gì trong `models/` |

---

## 8. Không làm

- Không sửa file ngoài `frontend/` (trừ `.gitignore` nếu cần thêm `__pycache__`)
- Không viết lại 6 quy tắc chuẩn hoá — import từ `scripts/layer1_adapter.py`
- Không gọi Qwen tự động ở bất kỳ tình huống nào
- Không thêm INT8 vào giao diện — đã bị bác bỏ ở `models/REPORT.md` (trùng nhãn 98,25% /
  94,93%, dưới ngưỡng 99,5%)
- Không tô màu kiểu "đúng/sai" giữa các engine — chưa có gold set để phán xử
- Không dùng npm/bundler/framework frontend
