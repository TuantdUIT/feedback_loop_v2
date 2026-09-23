# Đánh giá PhoBERT FP32 trên golden dataset (v2)

Model: `kiendt/phobert-ner-address`, hai bản **PyTorch FP32** và **ONNX FP32**, output đi qua
`scripts/layer1_adapter.py` (6 quy tắc chuẩn hoá về L1–L7). Chạy ngày 2026-09-23 trên CPU
(AMD Ryzen 5 5700U), ONNX `intra_op_num_threads=4`. Tái tạo:

```powershell
.venv-layer1/Scripts/python.exe scripts/eval_golden.py
```

| File | Mẫu | Nội dung |
|---|---:|---|
| `golden_full.json` | 150 | Địa chỉ **đầy đủ**, gold có L1–L7 |
| `golden_uncomplete.json` | 600 | Tiền tố **đang gõ dở**, gold chỉ có L4–L7 |

Kết quả trên bộ dữ liệu trước khi cập nhật nằm ở [`v1_old/`](v1_old/README.md).

Cách tính giữ nguyên như v1:
- **Accuracy:** mẫu phải khớp hoàn toàn cả 7 level.
- **P/R/F1:** tính micro trên các span `(level, text)`, span phải khớp tuyệt đối. Text được đưa
  về NFC, chữ thường, gộp khoảng trắng.
- **Latency:** tính theo từng node, không gồm thời gian nạp model và 5 mẫu warm-up.

## Kết quả v2

PyTorch FP32 và ONNX FP32 **giống hệt nhau trên 750/750 mẫu**, nên các chỉ số chất lượng
dùng chung cho cả hai bản.

| Chỉ số | `golden_full` (150) | `golden_uncomplete` (600) |
|---|---:|---:|
| Accuracy | 49,3% (74/150) | 45,5% (273/600) |
| Precision | 83,9% | 57,5% |
| Recall | 87,9% | 62,5% |
| F1 | **85,9%** | **59,9%** |
| Latency ONNX · mean / p95 | 33,8 / 47,0 ms | 21,3 / 31,0 ms |
| Latency PyTorch · mean / p95 | 120,7 / 187,2 ms | 75,8 / 114,6 ms |
| Nạp model | ONNX 2,9 s · PyTorch 12,4 s | |

Latency chi tiết theo từng node (tokenize / inference / adapter) nằm trong `metrics.json`.
Giống v1, hơn 98% thời gian nằm ở inference.

## So sánh v1 (cũ) → v2 (mới)

### Chỉ số tổng

| Chỉ số | full · v1 | full · v2 | Δ | uncomplete · v1 | uncomplete · v2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 49,3% | 49,3% | 0 | 44,3% | 45,5% | +1,2 |
| Precision | 79,4% | 83,9% | **+4,5** | 49,8% | 57,5% | **+7,7** |
| Recall | 87,8% | 87,9% | +0,1 | 61,1% | 62,5% | +1,4 |
| F1 | 83,4% | 85,9% | **+2,5** | 54,9% | 59,9% | **+5,0** |
| Latency ONNX · mean | 31,5 ms | 33,8 ms | +2,3 | 20,9 ms | 21,3 ms | +0,4 |
| Latency PyTorch · mean | 100,3 ms | 120,7 ms | +20,4 | 68,1 ms | 75,8 ms | +7,7 |

### Chỉ số tăng không phải vì model tốt lên

**Model không đổi.** Trên các mẫu có mặt ở cả hai bộ dữ liệu (107 mẫu `full`, 212 mẫu
`uncomplete`), dự đoán **giống hệt nhau từng mẫu**, và F1 trên phần chung giữ nguyên
(89,4% và 73,7%). Mọi chênh lệch đều đến từ việc **thay mẫu**: bạn đã thay 43/150 mẫu
`full` và 388/600 mẫu `uncomplete`. Gold của các mẫu giữ lại không bị sửa.

Thay đổi quan trọng nhất là **tỉ lệ POI có chữ "dự án"** giảm mạnh, trong khi tổng số mẫu
có L7 gần như giữ nguyên:

| | full · v1 | full · v2 | uncomplete · v1 | uncomplete · v2 |
|---|---:|---:|---:|---:|
| Mẫu có L7 | 69 | 69 | 272 | 267 |
| … trong đó có "dự án" | 65 | **22** | 256 | **87** |
| … POI không có "dự án" (`tower`, `tòa nhà`, `trường`, `bệnh viện`…) | 4 | **47** | 16 | **180** |

Hai loại POI này bị model phạt theo hai cách khác nhau:

```
có "dự án"   : dự án icon 56  →  L2 "dự" + L2 "án icon 56"   = 2 span thừa + 1 bỏ sót
không "dự án": gold star tower →  L2 "gold star tower"        = 1 span thừa + 1 bỏ sót
```

Ở cả hai trường hợp, L7 vẫn **không bao giờ được nhận ra** (F1 L7 = 0% ở cả v1 lẫn v2). Loại
thứ hai chỉ "rẻ" hơn một span thừa, vì không bị lỗi R1 cắt đôi. Vì vậy khi thay bớt POI
"dự án" bằng POI không có tiền tố, số span dự đoán thừa giảm (L2 thừa ở `uncomplete`:
531 → 321) và **precision tăng một cách cơ học**.

Kiểm chứng: xét riêng các mẫu bị thay, F1 của 43 mẫu mới là 77,6% so với 69,3% của 43 mẫu
cũ bị bỏ (`full`). Ở `uncomplete`, con số tương ứng là 54,1% so với 47,1%.

### Theo từng level

| Level | full F1 · v1 → v2 | uncomplete F1 · v1 → v2 | Ghi chú |
|---|---:|---:|---|
| L1 | 80,0% → 80,0% | — | |
| L2 | 64,1% → 72,3% | 0% → 0% (thừa 531 → 321) | Ít span `dự` + `án …` gán nhầm L2 hơn |
| L3 | 99,3% → 99,3% | 0% → 0% (thừa 37 → 51) | |
| L4 | 98,2% → 97,3% | 76,2% → **68,8%** | v2 có nhiều mẫu cắt ngay tiền tố `Phư`/`Phườ`/`Phườn` hơn |
| L5 | 96,8% → 95,8% | 88,2% → **83,1%** | Precision L5 giảm 88,5% → 76,2%, xem dưới |
| L6 | 92,7% → 93,9% | 90,1% → 90,3% | |
| L7 | 0% → 0% | 0% → 0% | Model không có nhãn POI |

Ở `uncomplete` v2 có **85 span L5 báo thừa**. Phân loại theo phần gold mà chúng chồng lên:

| Chồng lên gold | Số span | Ví dụ |
|---|---:|---|
| L7 (POI) | 49 | tên POI dở dang bị gán là tên đường |
| L6 | 14 | `số 22` → L5, gold là số nhà L6 |
| L4 | 12 | `phư` cuối chuỗi bị nuốt vào tên đường: `đường sông thao phư` |
| L5 (lệch biên) | 10 | `đường lê quang` thay vì `Đường Lê Quang Định` |

### Chỉ số chẩn đoán (bỏ mẫu có L7 / "dự án")

| | Mẫu · v1 → v2 | F1 · v1 → v2 |
|---|---:|---:|
| `golden_full` | 81 → 81 | 96,2% → 96,2% |
| `golden_uncomplete` | 328 → 333 | 86,6% → 87,2% |

Khi loại hết POI, model cho **kết quả gần như không đổi** giữa hai bộ dữ liệu. Điều này xác
nhận POI (L7) vẫn là lỗi chi phối toàn bộ chênh lệch.

### Latency

Các con số chất lượng không đổi, nhưng latency lần này cao hơn: PyTorch mean +20% ở `full`,
ONNX chỉ +2–7%. Đây là dao động do tải máy tại thời điểm đo, không phải do dữ liệu. Độ dài
chuỗi hai bộ gần như nhau, và ONNX (ít nhạy với tải máy hơn) gần như giữ nguyên. **Tỉ lệ
ONNX nhanh hơn PyTorch khoảng 3–3,5× không đổi.**

## Kết luận

1. Bộ v2 **đa dạng POI hơn** (nhiều `tower`, `tòa nhà`, `trường`, `bệnh viện`), phản ánh
   dữ liệu thật tốt hơn v1 vốn gần như chỉ có "dự án".
2. F1 tăng 2,5–5 điểm **chỉ vì thành phần dữ liệu thay đổi, không phải vì model tốt hơn**.
   Khi so sánh các lần sửa adapter sau này, phải dùng **cùng một bộ golden** thì con số
   mới có ý nghĩa.
3. Với v2, R7 kiểu "span bắt đầu bằng `dự án` → L7" chỉ giải quyết được khoảng 1/3 số mẫu
   POI (22/69 và 87/267). Phần còn lại (`gold star tower`, `Peridot Building`, `trường …`)
   cần danh sách từ khoá POI rộng hơn (`tower`, `building`, `tòa nhà`, `trường`,
   `bệnh viện`, `khu dân cư`, `residence`…), hoặc cần một model có nhãn POI.

## File trong thư mục này

| File | Nội dung |
|---|---|
| `metrics.json` | Toàn bộ số liệu v2: micro, theo level, latency từng node, nạp model, chẩn đoán |
| `pred_{pytorch,onnx}_fp32_golden_{full,uncomplete}.jsonl` | Dự đoán từng mẫu v2 |
| `v1_old/` | README, metrics và dự đoán trên bộ golden trước khi cập nhật |
