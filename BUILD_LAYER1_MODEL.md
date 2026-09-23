# Chuẩn bị model NER địa chỉ chạy ONNX INT8 cho Lớp 1

Tài liệu dành cho agent thực thi. Đọc hết trước khi chạy lệnh đầu tiên.

Bối cảnh: model này đóng vai Lớp 1 trong feedback loop (xem `BUILD.md` §1) — đọc lại output của ML pipeline chính và nêu nghi vấn.

Mục tiêu tối ưu: **giảm RAM khi chạy**. KHÔNG tối ưu tốc độ. Máy đích còn khoảng 5 GB RAM trống trên tổng 16 GB.

Phần cứng: AMD Ryzen 5 5700U (Zen 2, 8 nhân, có AVX2, KHÔNG có AVX-512/VNNI).

**Thực hiện tuần tự, không dừng lại hỏi xác nhận giữa chừng.**

---

## 1. Vị trí trong quy trình feedback loop

Đây là bước **chuẩn bị cho Bước 7** trong bảng 12 bước ở `BUILD.md` §7 — bước "Lớp 1: detector". Nhiệm vụ này CHUẨN BỊ model, KHÔNG tích hợp.

| # | Bước | Trạng thái |
|---|---|---|
| 1 | `core/`: schemas, normalize, prompt_loader | ✅ xong |
| 2 | `core/bio.py` — chuyển đổi span ↔ BIO | ⬜ |
| 3 | `core/diff_metrics.py` — thước đo so sánh 2 bản parse | ⬜ |
| 4 | Lớp 0 validator + `run_layer0_only.py` | ⬜ |
| 5 | Gold set 300–500 mẫu label tay | ⬜ |
| 6 | `core/llm_client.py` | ⬜ |
| **7** | **Lớp 1 — detector** | ⬜ **← nhiệm vụ này chuẩn bị model cho bước này** |
| 8–12 | Lớp 2, Lớp 3, cascade, calibrate, export silver | ⬜ |

Nhiệm vụ này chạy VƯỢT thứ tự. Chấp nhận được, vì export và đo đạc một model là việc độc lập, không cần bước 2–6.

Nhưng vì bước 2–5 chưa có, **TUYỆT ĐỐI KHÔNG tích hợp model này vào `feedback/`**: chưa có `bio.py` để chuyển nhãn thành span, chưa có `diff_metrics.py` để so sánh, chưa có gold set để biết model tốt hay dở.

Kết quả cần có: các thư mục model trong `models/`, và file `models/REPORT.md`.

Ghi chú phụ thuộc: Lớp 1 dùng model ONNX chạy local, không gọi API, nên Bước 7 **không còn phụ thuộc Bước 6** (`llm_client.py`). File đó giờ chỉ là tiền đề của Bước 8–9.

---

## 2. Khảo sát model

Tải config + tokenizer của `kiendt/phobert-ner-address` và ghi vào báo cáo:

1. `config.id2label` đầy đủ, không rút gọn
2. `type(tokenizer).__name__` và `tokenizer.is_fast`
3. Tokenize câu mẫu với `return_offsets_mapping=True`, in từng token kèm `(start, end)` và chuỗi con `text[start:end]` để đối chiếu
4. Model card của repo: yêu cầu văn bản ĐÃ tách từ (dạng `Hà_Nội`) hay thô?

Sau đó lập bảng ánh xạ từ nhãn của model sang L1–L7 (L1=quốc gia, L2=tỉnh/TP trực thuộc TƯ, L3=quận/huyện, L4=phường/xã, L5=đường, L6=số nhà, L7=POI). Nhãn nào không ánh xạ được thì ghi **"KHÔNG ÁNH XẠ ĐƯỢC"**, tuyệt đối không đoán.

Ghi nhận rồi **đi tiếp**, kể cả khi gặp một trong các vấn đề sau — chỉ cần nêu rõ trong báo cáo:

- `tokenizer.is_fast` là False, hoặc không dùng được `return_offsets_mapping`
- `text[start:end]` không khớp token in ra
- Model yêu cầu văn bản đã tách từ

Ba điều này ảnh hưởng tới việc ghép model vào schema `Span(start, end)` ở Bước 7, nên phải xuất hiện rõ trong `REPORT.md`.

---

## 3. Môi trường

Tạo virtualenv riêng, không cài vào môi trường chính của dự án.

```
pip install "optimum[onnxruntime]" transformers onnxruntime torch psutil
```

**Đặt `HF_HOME` tường minh** trỏ vào một thư mục trên ổ chứa dự án. Mặc định Hugging Face tải về `C:\Users\DELL\.cache`, dễ làm đầy ổ hệ thống mà không ai để ý.

In và ghi lại version của: optimum, transformers, onnxruntime, torch. API `ORTQuantizer` thay đổi giữa các bản optimum nên version là một phần của kết quả.

---

## 4. Export ONNX FP32

Export sang `models/phobert-ner-address-onnx/`, kèm tokenizer (`save_pretrained` cùng thư mục) và `config.json` để giữ `id2label`.

---

## 5. Quantize INT8

`ORTQuantizer` + `AutoQuantizationConfig.avx2(is_static=False, ...)`. KHÔNG dùng `avx512_vnni` — Zen 2 không có.

Tạo hai bản:

- `models/phobert-ner-address-int8/` (`per_channel=False`)
- `models/phobert-ner-address-int8-perchannel/` (`per_channel=True`)

---

## 6. Kiểm chứng độ chính xác — TIÊU CHÍ QUYẾT ĐỊNH

INT8 chỉ đáng dùng nếu gần như không giảm độ chính xác, vì thứ đổi lại chỉ là ~1 GB RAM.

### 6a. Câu mẫu

Chạy cả 3 bản (PyTorch FP32, INT8, INT8-perchannel) trên:

```
32 -34 đường Nguyễn Văn Linh, PHƯỜNG PHÚC ĐỒNG, QUẬN LONG BIÊN, THÀNH PHỐ HÀ NỘI
```

In bảng: token | offset | nhãn FP32 | nhãn INT8 | nhãn INT8-pc | lệch?

### 6b. Dữ liệu thật

Chạy cả 3 bản trên **548 bản ghi** lấy từ dữ liệu đã làm sạch:

- 500 dòng đầu của `clean_data/VAER_test_fix.txt` (địa chỉ đầy đủ)
- toàn bộ 48 dòng của `clean_data/VAER_train_partial_input_50.txt` (địa chỉ gõ dở)

Lấy 500 dòng ĐẦU chứ không lấy ngẫu nhiên, để lần chạy sau tái tạo được đúng kết quả.

Hai nhóm này khác nhau về bản chất — nhóm gõ dở có chuỗi bị cắt giữa chừng
(`đườn. g `, `daresco residence (đức hòa iii resco) - xã đức lập thượ`) nên dễ
làm lộ khác biệt giữa FP32 và INT8 hơn. **Báo cáo tỉ lệ trùng nhãn tách riêng
cho từng nhóm**, đừng gộp một con số.

Báo cáo:

- Tỉ lệ token trùng nhãn giữa FP32 và mỗi bản INT8 — **ngưỡng chấp nhận ≥ 99,5%**
- Danh sách CỤ THỂ mọi token lệch, kèm câu chứa nó
- Chênh lệch trung bình và lớn nhất của xác suất softmax tại nhãn được chọn

Dữ liệu này gõ dở, viết hoa/thường lộn xộn — đúng loại đầu vào thật, nên lệch ở đây quan trọng hơn lệch ở câu mẫu sạch.

### 6c. Đối chiếu với model chính

Với 24 bản ghi của `output_model.json` (file DUY NHẤT còn lại có sẵn nhãn L1–L7), áp bảng ánh xạ ở §2 rồi báo tỉ lệ đồng thuận theo từng level.

24 bản ghi là quá ít để kết luận chắc chắn — ghi rõ trong báo cáo rằng đây chỉ
là ước lượng thô, và nêu số bản ghi thực tế dùng được cho từng level (có level
sẽ không xuất hiện lần nào trong 24 bản ghi đó).

KHÔNG phải để chấm ai đúng ai sai — chưa có gold set. Mục đích là ước lượng Lớp 1 sẽ gắn cờ bao nhiêu % bản ghi, tức ước lượng chi phí Lớp 2 trước khi tiêu tiền. Bất đồng > 40% nghĩa là hai model hiểu schema khác nhau; nêu rõ trong báo cáo.

---

## 7. Đo RAM

**Bắt buộc: mỗi bản model đo trong MỘT TIẾN TRÌNH RIÊNG.** Đo nhiều model trong cùng tiến trình cho số sai, vì bộ nhớ của model nạp trước không được trả lại hệ điều hành và allocator giữ lại vùng đã cấp.

Với mỗi bản (PyTorch FP32, ONNX FP32, INT8, INT8-perchannel), chạy một subprocess riêng: nạp model → inference toàn bộ 548 bản ghi ở §6b → in ra:

- RAM đỉnh: `psutil.Process().memory_info().peak_wset` (Windows). Không có thì lấy `rss` lớn nhất bằng cách lấy mẫu định kỳ trong một thread.
- RAM sau khi nạp model nhưng trước khi inference (phần trọng số chiếm)
- Dung lượng thư mục model trên đĩa (MB)

Ghi thêm tổng thời gian chạy 548 bản ghi — **chỉ để tham khảo, không phải tiêu chí**, không cần warm-up hay thống kê p95.

Lưu ý: onnxruntime dùng memory arena tự mở rộng nên RAM đỉnh có thể cao hơn kích thước trọng số nhiều. Nếu chênh lệch FP32 với INT8 nhỏ hơn kỳ vọng, thử `SessionOptions.enable_cpu_mem_arena = False` rồi đo lại, báo cáo cả hai cấu hình.

---

## 8. Dọn dẹp và báo cáo

Thêm `models/` vào `.gitignore` ở project root. Không commit model.

Ghi `models/REPORT.md` gồm:

1. Version 4 package ở §3
2. `id2label` đầy đủ + bảng ánh xạ sang L1–L7, ghi rõ nhãn nào không ánh xạ được
3. Kết quả khảo sát tokenizer ở §2: có offset ký tự không, có yêu cầu tách từ không
4. Bảng RAM: 4 bản model × (RAM sau khi nạp, RAM đỉnh, dung lượng đĩa)
5. Kết quả 6a/6b/6c, kèm danh sách token lệch
6. Thời gian chạy 548 bản ghi (tham khảo)
7. **Khuyến nghị dùng bản nào.** Nếu INT8 tụt dưới 99,5% trùng nhãn thì nói rõ "không nên dùng INT8, phần RAM tiết kiệm không bù được".

---

## 9. Không làm

- Không sửa file nào trong `feedback/` — đây là bước chuẩn bị model
- Không commit model
- Không tối ưu tốc độ, không tinh chỉnh số luồng
