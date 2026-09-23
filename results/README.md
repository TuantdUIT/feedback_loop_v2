# Đánh giá PhoBERT (`kiendt/phobert-ner-address`) và Qwen làm Lớp 1

Tổng hợp số liệu từ quá trình chuẩn bị và đánh giá model cho **Lớp 1** của feedback loop
(xem `BUILD.md` §7). Toàn bộ số liệu ở đây đo được, không suy đoán — script nguồn nằm ở
`scripts/`, dữ liệu thô ở `models/testdata/` và `models/REPORT.md`.

---

## 1. Model dùng là gì

**`kiendt/phobert-ner-address`** — checkpoint đã fine-tune công khai trên Hugging Face,
**không phải** `phobert-base`. Xác nhận từ `config.json`:

```
architectures : RobertaForTokenClassification   (có đầu phân loại 21 nhãn — phobert-base không có)
hidden_size    : 768, 12 layer, 12 head           (backbone = phobert-base)
vocab_size     : 64001                            (tokenizer PhobertTokenizer, dùng chung phobert-base)
```

`phobert-base` chỉ là **phần lõi bên trong** — `kiendt` lấy backbone đó, gắn thêm đầu phân
loại token 21 nhãn, rồi fine-tune tiếp trên dữ liệu địa chỉ của riêng họ.

**Tên 21 nhãn không chính thức có trong model.** `config.json` gốc chỉ ghi `LABEL_0`…`LABEL_20`.
Tên `B_STREET`, `STREET_TYPE`, … được suy ra từ ví dụ trong model card, **chưa được tác giả
xác nhận** — nếu sai thứ tự, toàn bộ bảng ánh xạ bên dưới sai từ gốc.

### Dễ dùng: `kiendt` áp đảo, nhưng đổi lại "thuế adapter"

| | Sẵn sàng dùng ngay | Khớp đúng schema L1–L7 | Chi phí duy trì |
|---|---|---|---|
| `kiendt/phobert-ner-address` | ✅ `from_pretrained()` là chạy | ❌ qua adapter, trần 88,4%, thiếu L1/L7 | thấp (đã xong) |
| Tự train `phobert-base` trên L1–L7 | ❌ chưa có dữ liệu nhãn đủ lớn (`output_model.json` chỉ 24 bản ghi) | ✅ khớp tuyệt đối nếu train đúng | cao lúc đầu, thấp về sau |

---

## 2. Bảng 6 quy tắc chuẩn hoá (`scripts/layer1_adapter.py`)

Output thô của `kiendt` không khớp thẳng với schema L1–L7 của dự án. Đo tăng dần trên
**24 bản ghi / 69 span** của `output_model.json`:

| Quy tắc | Vì sao **phải** có | Model gốc hoạt động thế nào mà thiếu nó | Khớp |
|---|---|---|---:|
| *(thô, chưa chuẩn hoá)* | — | — | 49,3% |
| **R1** gộp `*_TYPE` | Bộ nhãn tách tiền tố ra một lớp nhãn riêng; `03_boundary_and_bio.txt` §3.1 quy định tiền tố **thuộc về** thực thể | `"đường phùng hưng"` → `đường=STREET_TYPE` (riêng) + `phùng hưng=B/I_STREET` (riêng) → 2 span thay vì 1 | 76,8% |
| **R4** sửa IOB2 sai | Bộ gộp span chỉ nối `I_X` khi khớp đúng loại thực thể đang mở; `I_X` sai chỗ rơi vào nhánh `else`, bị coi **như `O`** | Token mang `I_X` lạc chỗ **biến mất khỏi kết quả hoàn toàn**, không lỗi, không cảnh báo — mất dữ liệu âm thầm | 76,8%¹ |
| **R2** gazetteer quốc gia | Bộ nhãn **không có nhãn L1/COUNTRY** — model buộc gán bừa vào nhãn gần nhất | `"việt nam"` → `việt=B_CITY, nam=I_CITY` (đã kiểm chứng) | 79,7% |
| **R3** gazetteer tỉnh/thành | Ranh giới `CITY`/`DIST` của model không theo danh sách đóng 5 thành phố trực thuộc TW (`02_label_definitions.txt`) | `"đồng tháp"` (tỉnh, đáng lẽ L2) → model gán L3 như huyện/quận thường | 82,6% |
| **R5** gộp tiền tố theo từ vựng | R1 chỉ cứu được khi model **có phát** nhãn `*_TYPE`; nhiều lúc model bỏ qua bước đó | `"Thành phố Đà Nẵng"` → `Thành=B_CITY, phố=I_DIST, Đà=B_CITY, Nẵng=I_CITY` — không có `CITY_TYPE` nào để R1 bắt, còn tự phạm luôn lỗi IOB2 | 87,0% |
| **R6** sửa BPE cắt đôi từ | Tokenizer cắt từ hiếm gặp thành nhiều mảnh; đầu phân loại đôi lúc đoán `B_X` (không phải `I_X`) cho mảnh sau | `"thạnh"` → `th@@=B_STREET, ạnh=B_STREET` → tách thành 2 span rời từ 1 địa danh | **88,4%** |

¹ R4 không đổi điểm trên đúng 24 mẫu này (không mẫu nào rơi trúng ca), nhưng vẫn giữ vì nó
phòng kiểu mất dữ liệu không lộ ra qua phép đo `%khớp`.

### Gom theo nguyên nhân gốc

| Nhóm | Quy tắc | Bản chất |
|---|---|---|
| Lược đồ nhãn khác thiết kế | R1, R5 | `kiendt` tách tiền tố ra riêng; schema dự án gộp chung |
| Thiếu/lệch định nghĩa nhãn | R2, R3 | Không có L1; ranh giới L2/L3 không theo danh sách 5 thành phố |
| Lỗi cơ chế (tokenizer + model) | R4, R6 | BPE cắt từ, đầu phân loại đoán sai ở ranh giới mảnh từ |

### R6 — sửa lỗi của `kiendt`, không phải của `phobert-base`

Đo tại **mọi điểm nối bị BPE cắt đôi** trên 100 mẫu (`clean_data/VAER_test_v2_100.txt`),
loại bỏ trường hợp nối với dấu câu thuần:

```
tổng điểm nối kiểm được         : 83
model tự đoán ĐÚNG (I_X liên tục): 79/83  (95,2%)
model đoán SAI                   : 4/83   (4,8%)
```

`phobert-base` (qua `PhobertTokenizer`) chỉ cung cấp **tiền đề đáng tin** — dấu `@@` luôn
đúng nghĩa "còn nối tiếp". Việc gán nhãn tại điểm nối là việc của đầu phân loại do `kiendt`
tự huấn luyện, và nó **tự đúng 95,2%** — chứng tỏ lỗi không nằm ở cơ chế tokenize (nếu vậy
mọi điểm cắt phải sai như nhau), mà ở việc huấn luyện của `kiendt` thỉnh thoảng không ép
được ràng buộc "mảnh phụ luôn phải là `I_`".

**Cảnh báo tự ghi nhận:** bản đầu của R6 giả định mọi `@@` đều là ranh giới thực thể, kéo
điểm từ 87,0% xuống còn 62,3% (ăn nhầm cả dấu phẩy vào span, vd `"long@@" + ","`). Chỉ lộ ra
khi so trên 100 mẫu — 24 mẫu không đủ để bắt được lỗi này. Bản sửa chỉ ép gộp khi token theo
sau có chữ/số thật, bỏ qua dấu câu thuần.

---

## 3. So ba chiều: Qwen (qua OpenRouter) vs ONNX FP32 vs PyTorch FP32

Tập `clean_data/VAER_test_v2_100.txt` (seed `20260923`): 50 mẫu đầy đủ + 50 mẫu gõ dở
(cắt còn 40–95% độ dài gốc), độc lập nhau. Model dùng ở `.env`: `qwen/qwen3.6-35b-a3b`.

**ONNX FP32 = PyTorch FP32 tuyệt đối: 100/100 mẫu giống hệt nhau** (0 khác biệt). Dùng
bản nào cũng cho cùng kết quả — xác nhận cả trên tập 24 mẫu ban đầu (5261/5261 token).

### 50 mẫu ĐẦY ĐỦ (`models/testdata/test_v2/full_50.md`)

| Chỉ số | Giá trị |
|---|---:|
| Qwen không có kết quả (kẹt vòng lặp suy luận) | 3/50 |
| Số mẫu đưa vào so sánh | 47 |
| Cả 3 cho cùng chuỗi level | **31/47 (66%)** |
| ONNX FP32 khác PyTorch FP32 | 0/47 |

### 50 mẫu GÕ DỞ (`models/testdata/test_v2/partial_50.md`)

Chỉ giữ mẫu **cả 3 model đều sinh ra kết quả** (đã bỏ hẳn mẫu Qwen lỗi/chưa chạy).

| Chỉ số | Giá trị |
|---|---:|
| Qwen không có kết quả | 4/50 |
| Số mẫu đưa vào so sánh | 46 |
| Cả 3 cho cùng chuỗi level | **24/46 (52%)** |
| ONNX FP32 khác PyTorch FP32 | 0/46 |

Nhóm gõ dở có tỷ lệ đồng thuận thấp hơn hẳn nhóm đầy đủ (52% so với 66%) — đúng dự đoán,
chuỗi cụt là chỗ khó nhất cho cả hai hướng tiếp cận.

### So trên tập 24 mẫu gốc (`output_model.json`) — Qwen vs PhoBERT (đã chuẩn hoá)

| Level | span | Qwen | PhoBERT |
|---|---:|---:|---:|
| L1 quốc gia | 3 | **100%** | 66,7% |
| L2 tỉnh/thành | 8 | **100%** | 75,0% |
| L3 quận/huyện | 6 | 100% | 100% |
| L4 phường/xã | 8 | **100%** | 75,0% |
| L5 đường | 22 | 86,4% | 81,8% |
| L6 số nhà | 22 | 81,8% | **100%** |

Hai model **bù trừ nhau**: Qwen mạnh cấp hành chính (L1–L4), PhoBERT mạnh số nhà (L6).
Khi cả hai cùng đồng ý (56 span), **100% khớp đúng model chính** — không có ngoại lệ nào
trên mẫu đo được, gợi ý dùng "cả hai đồng ý → chấp nhận thẳng, không cần Lớp 2" cho case đó.

---

## 4. Chi phí Qwen (qua OpenRouter, `qwen/qwen3.6-35b-a3b`)

Đo trên 100 mẫu thật (`models/testdata/test_v2/qwen.jsonl`):

| Chỉ số | Giá trị |
|---|---:|
| Parse thành công | 93/100 |
| Tổng chi phí (kể cả mẫu lỗi) | $0,8361 |
| **Chi phí trung bình / địa chỉ (mẫu thành công)** | **$0,00491** |
| Token vào mỗi lần (system prompt) | ~14.945 |

**7/100 mẫu (7%) bị kẹt vòng lặp suy luận** — model không chốt được câu trả lời, chạy tới
khi chạm trần token (có ca tới 131.072 completion token, cost $0,12 một mẫu) rồi bị cắt,
response rỗng. 7 mẫu lỗi này chiếm tỷ trọng chi phí bất thường so với số lượng — rủi ro
thật khi triển khai quy mô lớn nếu không giới hạn `max_tokens`/`reasoning.max_tokens`.

Quy mô ước tính (nếu chạy toàn bộ, theo giá $0,00491/địa chỉ):
`VAER_test_fix.txt` (3.217 dòng) ≈ **$15,8** · `VAER_train_fix.txt` (9.501 dòng) ≈ **$46,7**.
92% chi phí là system prompt tĩnh — gộp lô `{"texts": [...]}` (`06_io_format.txt` §6.1) là
cách giảm chi phí lớn nhất chưa thử nghiệm.

---

## 5. RAM và dung lượng (PhoBERT, đo trên 548 bản ghi, mỗi bản một tiến trình riêng)

| Bản model | RAM sau nạp (MB) | RAM đỉnh (MB) | Đĩa (MB) |
|---|---:|---:|---:|
| PyTorch FP32 | 373,2 | 744,3 | 539,8 |
| ONNX FP32 | 917,3 | **1.110,1** | 540,0 |
| ONNX INT8 | 516,6 | 564,0 | 137,3 |

ONNX FP32 **tốn RAM hơn PyTorch FP32** dù cùng kết quả — chỉ export ONNX mà không quantize
thì phản tác dụng với mục tiêu tiết kiệm RAM.

**INT8 bị bác bỏ** (chi tiết `models/REPORT.md`): trùng nhãn với FP32 chỉ 98,25% (nhóm đầy
đủ) / 94,93% (nhóm gõ dở), dưới ngưỡng chấp nhận 99,5% đã đặt.

---

## 6. Khuyến nghị

1. **Dùng `kiendt/phobert-ner-address` bản PyTorch FP32 hoặc ONNX FP32** (giống hệt nhau) làm
   một trong hai nguồn tín hiệu của Lớp 1 — không dùng INT8.
2. **Bắt buộc đi qua `scripts/layer1_adapter.py`** (6 quy tắc) trước khi so sánh với schema
   L1–L7 — dùng thẳng output thô chỉ đạt 49,3%.
3. **Cân nhắc chạy song song cả Qwen lẫn PhoBERT** cho Lớp 1: hai bên bù trừ theo level, và
   "cả hai đồng ý" là tín hiệu chấp nhận mạnh trên mẫu đã đo (100% đúng, tuy mẫu còn nhỏ).
4. Nếu dùng Qwen ở quy mô lớn, **phải giới hạn `max_tokens`/`reasoning.max_tokens`** — 7%
   mẫu bị kẹt vòng lặp suy luận là rủi ro chi phí thật, không phải lý thuyết.
5. Cả hai con số 88,4% (PhoBERT) và các tỷ lệ Qwen đều đo trên mẫu nhỏ (24–100 bản ghi),
   **chưa có gold set**. Phải kiểm lại trên gold set (`BUILD.md` §7 bước 5) trước khi đưa
   bất kỳ ngưỡng nào ở đây vào `feedback/config/thresholds.yaml`.

---

*Nguồn dữ liệu thô: `models/testdata/test_v2/{full_50,partial_50}.md`, `models/testdata/test_v2/qwen.jsonl`,
`models/testdata/phobert_vs_main_24.json`, `models/REPORT.md`. Script tái tạo: `scripts/layer1_adapter.py`,
`scripts/report_test_v2.py`, `scripts/run_qwen_on_file.py`, `scripts/run_local_on_file.py`.*
