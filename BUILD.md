# BUILD.md — Dựng khung thư mục cho hệ thống feedback loop 3 lớp

Tài liệu này dành cho agent thực thi. Đọc hết trước khi tạo file đầu tiên.

Nhiệm vụ: tạo **khung thư mục rỗng** cho subsystem `feedback/`. Đây là bước scaffold — **không implement logic nghiệp vụ**. Phần implement bắt đầu từ §7.

Tuân thủ `AGENT.md` của dự án: chỉ thao tác bên trong project root.

---

## 1. Bối cảnh

Dự án hiện có một ML pipeline NER địa chỉ tiếng Việt, gán nhãn `L1`–`L7` theo BIO/IOB2. Output của nó nằm ở `output_model.json` và `result/`.

`feedback/` là hệ thống chạy **sau** pipeline đó: đọc output, phát hiện parse sai, sửa, và đo xem việc sửa có thật sự làm dữ liệu tốt lên không. Kiến trúc là một cascade 4 tầng lọc:

| Tầng | Công cụ | Vai trò | Tối ưu cho |
|---|---|---|---|
| Lớp 0 | Python thuần, không LLM | Bắt lỗi hình thức (offset, BIO, enum) | Chi phí 0 |
| Lớp 1 | Model NER chuyên dụng (PhoBERT, local) | **Detector** — parse độc lập rồi so bất đồng | **Recall cao** |
| Lớp 2 | DeepSeek | **Comparator** — chấm điểm parse cũ vs mới | Precision |
| Lớp 3 | OpenAI | **Arbiter** — chốt / viết lại / đẩy cho người | Đúng tuyệt đối |

Mỗi lớp **lọc bớt** case cho lớp sau, không tự quyết tất cả. Chỉ case mà lớp trước không kết luận được mới đi tiếp — đó là lý do chi phí không nổ.

---

## 2. Quyết định đã chốt — KHÔNG được làm khác

Những điểm sau đã được thống nhất. Nếu thấy có lý do kỹ thuật để làm khác, **dừng lại và hỏi**, đừng tự đổi.

1. **Bộ nhãn BIO là hệ L, không phải tên loại thực thể.** Chỉ dùng `O`, `B-L1`…`B-L7`, `I-L1`…`I-L7`. Không có `B-STREET`, không có bảng map 2 chiều, không có module convert nhãn. Tên loại thực thể (COUNTRY, PROVINCE_OR_CITY, DISTRICT, WARD, STREET, HOUSE_NUMBER, POI) chỉ là bước suy luận trung gian mô tả ở `prompt/v2/02_label_definitions.txt` §2.0 — không bao giờ xuất hiện trong dữ liệu.

2. **Một nguồn sự thật duy nhất về lý thuyết BIO.** Lớp 2 và Lớp 3 đều nạp `prompt/v2/compiled/system_prompt_v2_with_partial_input.txt`. Không sao chép, không viết lại, không tóm tắt phần lý thuyết đó vào `feedback/prompts/`. File trong `feedback/prompts/` chỉ chứa phần *nhiệm vụ riêng của từng lớp* và được nối vào sau prompt lõi lúc runtime.

   Lớp 1 KHÔNG dùng prompt — nó là model NER chạy local, xem quyết định 7.

3. **Không sửa file trong `compiled/`.** Muốn đổi lý thuyết BIO thì sửa module nguồn `prompt/v2/0X_*.txt` rồi chạy lại `python build.py --with-partial-input`.

4. **Lớp 0 chạy trước và chạy lại sau Lớp 3.** Lớp 3 là LLM, vẫn có thể trả offset sai; output của nó phải qua validator lần nữa.

5. **Mọi ngưỡng nằm trong `config/thresholds.yaml`**, không hardcode trong code. Ngưỡng sẽ được calibrate bằng dữ liệu, sẽ thay đổi nhiều lần.

7. **Lớp 1 là model NER chuyên dụng chạy local, không phải SLM qua API.** Quyết định đổi ngày 2026-09-23 sau khi đo `kiendt/phobert-ner-address` (xem `BUILD_LAYER1_MODEL.md` và `models/REPORT.md`). Hệ quả: Lớp 1 không nạp prompt, không cần constrained decoding, không cần mẹo self-consistency k lần — độ tự tin lấy thẳng từ xác suất softmax. Lớp 1 **không còn phụ thuộc Bước 6** (`llm_client.py`); file đó giờ chỉ là tiền đề của Bước 8–9.

8. **Output của Lớp 1 phải qua bước chuẩn hoá schema trước khi so sánh.** Model NER dùng lược đồ nhãn KHÁC với L1–L7: nó tách tiền tố thành nhãn `*_TYPE` riêng, trong khi `03_boundary_and_bio.txt` §3.1 quy định tiền tố THUỘC thực thể.

   ```
   "đường phùng hưng"
   Model chính : đường=B-L5     | phùng=I-L5   | hưng=I-L5
   PhoBERT     : đường=STREET_TYPE | phùng=B_STREET | hưng=I_STREET
   ```

   **Quy tắc gộp bắt buộc:** token `X_TYPE` đứng ngay trước `B_X` cùng loại thì `X_TYPE` trở thành điểm bắt đầu span, `B_X` hạ xuống `I_X`. `X_TYPE` đứng trơ trọi (không có `B_X` theo sau) vẫn là một span — khớp với §3.5 về tiền tố cuối chuỗi bị cắt dở.

   Đo trên 24 bản ghi `output_model.json`: không gộp thì khớp 34/69 span (49,3%), có gộp thì 53/69 (76,8%) — quy tắc này cứu 19 span. **Bỏ qua nó sẽ khiến mọi địa chỉ có tiền tố bị báo bất đồng giả.**

6. **`compiled/` KHÔNG được commit; thay vào đó mọi output phải đóng dấu `prompt_sha`.** `build.py` là phép nối deterministic (đã kiểm chứng: build 2 lần cho cùng sha256), nên lịch sử git của 9 module nguồn đã đủ để tái tạo chính xác prompt tại bất kỳ commit nào — commit thêm file compiled chỉ tạo ra nguồn sự thật thứ hai phải giữ đồng bộ. Khả năng truy vết đến từ `_meta.prompt_sha` trong file kết quả, xem §5.8.

---

## 3. Bước 0 — Xử lý git trước khi tạo file

Hiện `.env` **đang được git theo dõi** và đang rỗng. Ngay khi điền API key vào, key sẽ bị commit. Gỡ khỏi index trước (file trên đĩa vẫn còn nguyên):

```bash
git rm --cached .env
git rm --cached -r __pycache__
git rm --cached -r prompt/v2/compiled
```

Ba lệnh này chỉ gỡ khỏi index, **file trên đĩa vẫn còn nguyên**. Lý do gỡ `compiled/`: xem §2 quyết định 6.

Tạo `.gitignore` ở project root với nội dung:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/

# Bí mật
.env

# Dữ liệu sinh ra khi chạy feedback loop — không commit
feedback/store/cases/*
feedback/store/silver/*
feedback/store/human_queue/*
!feedback/store/**/.gitkeep

# Prompt đã build — sinh lại được từ module nguồn
prompt/v2/compiled/
```

Tạo `.env.example` ở project root (file này **được** commit, làm tài liệu):

```dotenv
# Lớp 2 — DeepSeek
DEEPSEEK_BASE_URL=
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=

# Lớp 3 — OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=
```

> `.gitignore` không tự gỡ file đã được theo dõi — đó là lý do phải có `git rm --cached` ở trên. Sau khi gỡ, một bản clone mới sẽ không có `compiled/`; `core/prompt_loader.py` (§5.8) chịu trách nhiệm build lại tự động nên code vẫn chạy được ngay.

---

## 4. Cây thư mục đích

Chỉ tạo phần đánh dấu `+`. Phần đánh dấu `✓` đã tồn tại — **không đụng vào**.

```
feedback_loop_v2/
│
├── .env                                  ✓ (gỡ khỏi git index ở Bước 0)
├── .env.example                          +
├── .gitignore                            +
├── AGENT.md                              ✓
├── BUILD.md                              ✓ (chính là file này)
├── README.md                             +
├── requirements.txt                      +
│
├── data/                                 ✓ không đụng
├── clean_data/                           ✓ không đụng
├── result/                               ✓ không đụng
├── output_model.json                     ✓ input của feedback loop
├── output_model.txt                      ✓
│
├── prompt/v2/                            ✓ NGUỒN SỰ THẬT về BIO — không đụng
│   ├── 00_manifest.txt … 09_*.txt
│   ├── build.py
│   └── compiled/
│       └── system_prompt_v2_with_partial_input.txt
│
└── feedback/                             + TOÀN BỘ HỆ THỐNG 3 LỚP
    ├── __init__.py
    │
    ├── config/
    │   ├── labels.yaml                   L1–L7 + trọng số nghiêm trọng
    │   ├── models.yaml                   model_id + giá token từng lớp
    │   └── thresholds.yaml               mọi ngưỡng gate
    │
    ├── prompts/
    │   ├── l2_judge.txt                  rubric chấm A/B của Lớp 2
    │   └── l3_arbiter.txt                nhiệm vụ chốt của Lớp 3
    │
    ├── core/
    │   ├── __init__.py
    │   ├── schemas.py                    Span, ParseResult, CaseRecord, Decision, RunMeta
    │   ├── normalize.py                  NFC + adapter text↔input + giữ metadata
    │   ├── prompt_loader.py              nạp prompt lõi + sha, tự build lại khi stale
    │   ├── bio.py                        spans↔bio, kiểm transition IOB2
    │   ├── diff_metrics.py               COR/INC/PAR/MIS/SPU + truncated + abstain
    │   ├── llm_client.py                 retry, ép JSON schema, đếm cost
    │   └── cascade.py                    orchestrator + toàn bộ hàm gate
    │
    ├── layers/
    │   ├── __init__.py
    │   ├── layer0_validator.py
    │   ├── layer1_ner.py
    │   ├── layer2_judge.py
    │   └── layer3_arbiter.py
    │
    ├── eval/
    │   ├── __init__.py
    │   ├── gold/
    │   │   ├── ANNOTATION_GUIDE.md
    │   │   └── .gitkeep
    │   ├── make_gold_sample.py
    │   ├── calibrate.py
    │   └── report.py
    │
    ├── store/
    │   ├── cases/.gitkeep
    │   ├── silver/.gitkeep
    │   └── human_queue/.gitkeep
    │
    ├── scripts/
    │   ├── run_layer0_only.py
    │   ├── run_loop.py
    │   └── export_silver.py
    │
    └── tests/
        ├── __init__.py
        ├── test_normalize.py
        ├── test_prompt_loader.py
        ├── test_bio.py
        ├── test_diff_metrics.py
        └── test_layer0.py
```

---

## 5. Nội dung khởi tạo cho từng file

### 5.1. File `.py` — quy ước stub

**Tất cả** file `.py` trong `feedback/` (trừ `__init__.py`) được tạo ở dạng stub: chỉ có docstring mô tả trách nhiệm, không có hàm, không có class, không có import thừa.

Lý do: thiết kế chi tiết của từng module chưa chốt. Viết sẵn signature sẽ khoá thiết kế vào một hướng chưa được duyệt, và agent implement sau sẽ tưởng đó là đặc tả.

Mẫu:

```python
"""<Tên module> — <trách nhiệm trong một câu>.

<2-4 dòng mô tả: module này nhận gì, trả gì, và nó KHÔNG làm gì.>

Chưa implement. Xem BUILD.md §7 để biết thứ tự triển khai.
"""
```

`__init__.py` để rỗng hoàn toàn (0 byte).

Docstring cụ thể cho từng file:

| File | Docstring dòng đầu | Ghi chú bắt buộc trong docstring |
|---|---|---|
| `core/schemas.py` | Định nghĩa kiểu dữ liệu dùng chung cho cả 4 lớp | `CaseRecord` chỉ được **append** field của lớp mới, không sửa field lớp trước — giữ trace đầy đủ để audit. Chứa cả `RunMeta` (khối `_meta` ở §5.8) |
| `core/prompt_loader.py` | Nạp prompt lõi từ `prompt/v2/compiled/` và trả về kèm sha | Tự chạy `build.py` khi file thiếu hoặc cũ hơn module nguồn; **không** sửa nội dung prompt, chỉ nạp |
| `core/normalize.py` | Chuẩn hoá bản ghi đầu vào về một dạng duy nhất | NFC trước mọi phép so offset; `output_model.json` dùng key `text` còn spec `06_io_format.txt` dùng `input` — adapter phải nhận cả hai; giữ nguyên metadata `trunc`/`source_id`/`group`/`is_full`/`is_long` để sau này stratify metric |
| `core/bio.py` | Chuyển đổi hai chiều giữa chuỗi BIO và danh sách span | Chỉ chấp nhận tag hệ L; `I-Lx` sau `O` hoặc sau `B-Ly` khác loại là không hợp lệ |
| `core/diff_metrics.py` | So sánh hai bản parse và phân loại từng điểm khác biệt | Phân loại COR/INC/PAR/MIS/SPU; tách riêng lỗi biên với lỗi nhãn; `truncated` là chiều đúng/sai thứ ba; `spans` rỗng có thể là đáp án đúng nên cần metric abstain riêng |
| `core/llm_client.py` | Lớp bọc chung để gọi 3 nhà cung cấp LLM | Retry, ép output đúng JSON schema, ghi lại token + chi phí mỗi lần gọi |
| `core/cascade.py` | Điều phối luồng qua Lớp 0→3 và chứa toàn bộ hàm gate | Mọi ngưỡng đọc từ `config/thresholds.yaml`, không hardcode |
| `layers/layer0_validator.py` | Kiểm tra hình thức bằng Python thuần, không gọi LLM | Phân biệt vi phạm `hard` (chắc chắn sai) và `soft` (đáng ngờ); chạy cả trước Lớp 1 lẫn sau Lớp 3 |
| `layers/layer1_ner.py` | Model NER chuyên dụng parse độc lập rồi so bất đồng | Ưu tiên recall; phải áp quy tắc gộp `*_TYPE` ở §2 quyết định 8 TRƯỚC khi so sánh; parse mới chỉ là *giả thuyết cạnh tranh*, không phải đáp án |
| `layers/layer2_judge.py` | Chấm điểm so sánh parse cũ vs parse mới | Phải chạy 2 lần đảo thứ tự A/B để phát hiện position bias; ẩn nguồn gốc của mỗi parse khỏi model |
| `layers/layer3_arbiter.py` | Ra quyết định cuối cùng cho case còn tranh cãi | 4 kết quả: giữ cũ / nhận mới / tự viết lại / đẩy cho người; output phải qua Lớp 0 lần nữa |
| `eval/make_gold_sample.py` | Lấy mẫu phân tầng từ output ML để người label tay | Phân tầng theo `group` và `is_full` để gold set phản ánh đúng phân bố thật |
| `eval/calibrate.py` | Quét ngưỡng trên gold set, vẽ đường đánh đổi chất lượng ↔ chi phí | Chạy cả 4 lớp nhưng **không** áp gate, log toàn bộ score thô |
| `eval/report.py` | Tổng hợp chỉ số sức khoẻ của cả loop | `net_gain` và `break_rate` là hai chỉ số quyết định loop có đáng chạy không |
| `scripts/run_layer0_only.py` | Chạy riêng Lớp 0 trên toàn bộ output ML | Không tốn API; chạy đầu tiên để biết tỷ lệ lỗi hình thức thật |
| `scripts/run_loop.py` | Chạy full cascade trên một file input | |
| `scripts/export_silver.py` | Xuất case đã chốt với độ tin cậy cao thành dữ liệu retrain | |
| `tests/test_*.py` | Test cho module tương ứng | Để rỗng ngoài docstring; test viết cùng lúc với module |

### 5.2. `feedback/config/labels.yaml`

File này có nội dung thật ngay từ đầu (không stub):

```yaml
# Bộ nhãn của ML pipeline. Hệ L, không dùng tên loại thực thể.
# Xem prompt/v2/02_label_definitions.txt §2.0 để biết ý nghĩa từng level.
label_set: [L1, L2, L3, L4, L5, L6, L7]

# Trọng số mức nghiêm trọng khi một span bị parse sai level.
# Dùng trong diff_metrics để tính severity — sai L2 nặng hơn sai L7 nhiều.
# GIÁ TRỊ KHỞI ĐIỂM, phải chỉnh lại sau khi calibrate trên gold set.
severity_weight:
  L1: 0.3   # quốc gia, hiếm xuất hiện, sai ít ảnh hưởng
  L2: 1.0   # tỉnh/thành — sai là hỏng toàn bộ định vị
  L3: 1.0   # quận/huyện — tương tự
  L4: 0.7   # phường/xã
  L5: 0.8   # đường
  L6: 0.9   # số nhà
  L7: 0.5   # POI
```

`bio_tags_valid` **không** liệt kê trong file này — sinh tự động từ `label_set` lúc load, để không bao giờ lệch nhau.

### 5.3. `feedback/config/models.yaml` và `thresholds.yaml`

Tạo file chỉ chứa comment mô tả những khoá sẽ có, **chưa điền giá trị**. Ngưỡng phải đến từ calibrate chứ không phải từ phỏng đoán, và điền số sẵn sẽ khiến người sau tưởng đó là số đã kiểm chứng.

`models.yaml`:

```yaml
# Cấu hình model từng lớp. Biến môi trường đọc từ .env, xem .env.example.
# Mỗi lớp cần: provider, model_id, base_url, temperature, max_tokens,
# và đơn giá input/output (USD / 1M token) để llm_client tính chi phí.
#
# layer1: ...
# layer2: ...
# layer3: ...
```

`thresholds.yaml`:

```yaml
# Toàn bộ ngưỡng gate của cascade. KHÔNG hardcode ngưỡng trong code.
# Mọi giá trị ở đây phải do eval/calibrate.py sinh ra trên gold set,
# không được đoán. File này có version và được commit theo từng lần calibrate.
#
# layer0: ngưỡng phân loại vi phạm hard/soft
# layer1: ngưỡng self-consistency, severity tối thiểu để escalate
# layer2: margin tối thiểu, confidence tối thiểu, xử lý khi position flip
# layer3: ngưỡng confidence để chốt / gắn cờ audit / đẩy cho người
# sampling: tỉ lệ random audit (2-5%)
```

### 5.4. `feedback/prompts/*.txt`

Mỗi file chứa một comment header mô tả vai trò và nhắc rằng phần lý thuyết BIO đến từ nơi khác:

```
# <Tên lớp> — <vai trò>
#
# File này CHỈ chứa phần nhiệm vụ riêng của lớp này.
# Phần lý thuyết BIO/IOB2 và định nghĩa nhãn L1-L7 được nạp từ
# prompt/v2/compiled/system_prompt_v2_with_partial_input.txt lúc runtime.
# KHÔNG sao chép lại nội dung đó vào đây.
#
# Chưa viết. Xem BUILD.md §7.
```

### 5.5. `feedback/eval/gold/ANNOTATION_GUIDE.md`

Tạo file với phần khung sau, **chưa điền nội dung chi tiết**:

```markdown
# Hướng dẫn label tay cho gold set

Chưa viết. Phải chốt TRƯỚC khi bắt đầu label — sửa quy ước giữa chừng
sẽ làm hỏng toàn bộ gold set đã label.

## 1. Quy mô và cách lấy mẫu
## 2. Quy ước label (tham chiếu prompt/v2, không viết lại)
## 3. Các trường hợp biên đã biết
## 4. Khi nào để spans rỗng thay vì đoán
## 5. Quy ước đánh dấu truncated
## 6. Cách xử lý khi người label không chắc
```

### 5.6. `requirements.txt`

```
# Lõi
pyyaml
python-dotenv

# Gọi LLM
openai              # dùng cho cả Lớp 3 và các endpoint tương thích OpenAI
httpx

# Tính toán metric
numpy

# Test
pytest

# Lớp 1 — model NER local (cài trong venv riêng, xem BUILD_LAYER1_MODEL.md)
# torch
# optimum[onnxruntime]
# onnxruntime
```

Không pin version ở bước scaffold. Pin sau khi môi trường chạy được.

### 5.7. `README.md`

Ngắn gọn, chỉ gồm: mô tả một đoạn về feedback loop, bảng 4 lớp ở §1 của file này, cách cài đặt, và trỏ sang `BUILD.md` cho thứ tự triển khai. Không lặp lại nội dung BUILD.md.

### 5.8. Truy vết prompt version — `core/prompt_loader.py` và khối `_meta`

Vì `compiled/` không nằm trong git, khả năng truy vết phải đến từ chính file kết quả.

`prompt_loader.py` có hai trách nhiệm, viết rõ trong docstring stub:

1. **Tự build lại khi cần.** So mtime của `compiled/system_prompt_v2_with_partial_input.txt` với 9 module nguồn `prompt/v2/0X_*.txt`. Thiếu file, hoặc bất kỳ module nào mới hơn → gọi `build.py` rồi mới nạp. Nhờ vậy clone mới chạy được ngay, và không bao giờ có chuyện chạy nhầm prompt cũ sau khi sửa module.
2. **Trả về sha.** `sha256` của nội dung prompt đã nạp, cắt 8 ký tự đầu.

Chữ ký (đây là **ngoại lệ duy nhất** của quy ước stub ở §5.1 — hai module khác sẽ gọi hàm này nên chữ ký cần chốt trước):

```python
def load_prompt(variant: str = "v2_with_partial_input") -> tuple[str, str]:
    """Trả về (nội dung prompt, sha256 8 ký tự đầu)."""
```

Mọi file output của feedback loop — kết quả chạy, `store/cases/*.jsonl`, báo cáo của `eval/report.py` — phải mang khối `_meta` ở cấp ngoài cùng:

```json
{
  "_meta": {
    "prompt_sha": "28e6ed90",
    "prompt_variant": "v2_with_partial_input",
    "git_commit": "ac9bb7b",
    "run_at": "2026-09-23T02:40:00+07:00"
  },
  "records": [ ... ]
}
```

Tác dụng: cho phép **nhóm kết quả theo prompt version** khi so sánh hiệu suất giữa các lần sửa prompt. Đây là dữ liệu đầu vào trực tiếp của `eval/calibrate.py` ở bước 11 — không có nó thì không phân biệt được "model tốt lên" với "prompt vừa đổi".

Lưu ý: `result/*.json` hiện có **chưa** có khối này, nên không biết chúng sinh ra bởi prompt version nào. Không truy ngược được; chấp nhận và áp dụng từ các lần chạy mới trở đi.

---

## 6. Nghiệm thu bước scaffold

Chạy và xác nhận cả bốn điều sau:

```bash
# 1. Cây thư mục khớp §4
find feedback -type f | sort

# 2. Import được, không lỗi cú pháp
python -c "import feedback, feedback.core, feedback.layers, feedback.eval; print('ok')"

# 3. pytest chạy được (0 test là đúng ở bước này, miễn không có error)
pytest feedback/tests -q

# 4. .env và compiled/ không còn trong index
git status --short
git ls-files | grep -E '^\.env$' && echo "LỖI: .env vẫn bị theo dõi" || echo "ok"
git ls-files | grep -E '^prompt/v2/compiled/' && echo "LỖI: compiled/ vẫn bị theo dõi" || echo "ok"
```

Không chạy `prompt/v2/build.py` ở bước này — prompt đã được build lại rồi.

> Nếu chạy `build.py` trên console Windows, thêm `PYTHONIOENCODING=utf-8` phía trước: script ghi file đúng nhưng crash ở dòng `print` cuối vì console dùng cp1252.

---

## 7. Sau scaffold — thứ tự triển khai

Làm **đúng thứ tự** này. Mỗi bước phải có test xanh trước khi sang bước sau. Cột "Thư mục" cho biết bước đó đụng vào đâu.

| # | Bước | Thư mục / file | Vì sao ở vị trí này |
|---|---|---|---|
| **1** | **`schemas.py` + `normalize.py` + `prompt_loader.py`** ← **bước kế tiếp** | `feedback/core/` | Mọi module khác đều import từ đây. `normalize.py` phải xong trước vì NFC và chuyện key `text` vs `input` ảnh hưởng tới mọi phép so sánh offset phía sau. `prompt_loader.py` đi cùng vì khối `_meta` (§5.8) là một phần của schema, và vì mọi lớp LLM đều cần nó. Test đi kèm: `feedback/tests/test_normalize.py`, `feedback/tests/test_prompt_loader.py` |
| 2 | `bio.py` | `feedback/core/` | Chuyển đổi span↔BIO. Test biên: `"62 - 64"`, `"125/ 84"`, `"q."` tách 2 token. Test: `feedback/tests/test_bio.py` |
| 3 | `diff_metrics.py` | `feedback/core/` | **Module quan trọng nhất.** Gate của Lớp 1→2 và 2→3 đều dựa vào nó; sai ở đây thì mọi ngưỡng phía trên vô nghĩa. Test kỹ nhất: `feedback/tests/test_diff_metrics.py` |
| 4 | `layer0_validator.py` + `run_layer0_only.py` | `feedback/layers/`, `feedback/scripts/` | Chạy được ngay trên `output_model.json`, **0 đồng API**. Kết quả có thể cho thấy phần lớn lỗi nằm ở post-processing của ML pipeline — sửa chỗ đó rẻ hơn xây 3 lớp LLM rất nhiều |
| 5 | Gold set | `feedback/eval/gold/` + `make_gold_sample.py` | Label tay 300–500 mẫu. Tốn 1–2 ngày công và **không bỏ qua được**: không có gold set thì mọi ngưỡng ở bước 7–9 chỉ là phỏng đoán |
| 6 | `llm_client.py` | `feedback/core/` | Hạ tầng chung cho Lớp 2 và Lớp 3. Lớp 1 chạy local nên KHÔNG phụ thuộc bước này |
| 7 | `layer1_ner.py` | `feedback/layers/` | Model đã chuẩn bị ở `BUILD_LAYER1_MODEL.md`. Phải giải quyết 3 hạn chế đã đo trong `models/REPORT.md` trước: tokenizer không có offset, `id2label` chỉ là `LABEL_n` (bảng nhãn là giả định), và thiếu hẳn L1/L7 trong khi ~15% dữ liệu có POI. Đo trên gold set: **detection recall ≥ 0.90** là chỉ tiêu số một |
| 8 | `layer2_judge.py` + `l2_judge.txt` | `feedback/layers/`, `feedback/prompts/` | **Validate độc lập trước khi nối vào cascade**: agreement với người (Cohen κ) phải ≥ 0.6, nếu không thì judge không dùng được |
| 9 | `layer3_arbiter.py` + `l3_arbiter.txt` | `feedback/layers/`, `feedback/prompts/` | Chỉ làm khi Lớp 2 thật sự để lại > 15% case không kết luận được. Nếu ít hơn, Lớp 3 không đáng chi phí |
| 10 | `cascade.py` + `run_loop.py` | `feedback/core/`, `feedback/scripts/` | Ghép các lớp, đọc ngưỡng từ config |
| 11 | `calibrate.py` + `report.py` | `feedback/eval/` | Chốt ngưỡng bằng dữ liệu, thay cho giá trị phỏng đoán |
| 12 | `export_silver.py` | `feedback/scripts/` | Đóng vòng lặp: đẩy dữ liệu đã sửa về cho lần train sau |

**Không nhảy cóc sang bước 7.** Bước 4 và 5 quyết định các bước sau có ý nghĩa hay không.

---

## 8. Chỉ số phải theo dõi ngay khi loop chạy được

Ghi ở đây để bước 11 không quên. Hai chỉ số đầu là chỉ số phanh gấp:

- **`break_rate`** — tỉ lệ case đang ĐÚNG bị loop làm hỏng.
- **`net_gain = n_fixed − n_broken`** — nếu ≤ 0 thì **tắt loop**.

Ví dụ vì sao chỉ nhìn `fix_rate` là bẫy: base error rate 10%, `fix_rate = 0.6`, `break_rate = 0.08` → sửa được `0.6 × 10% = 6%`, làm hỏng `0.08 × 90% = 7.2%`. Loop lỗ ròng, dù nghe "sửa được 60% lỗi" rất kêu.

Các chỉ số còn lại: `escalation_rate` từng tầng (mục tiêu L1→L2 khoảng 10–25%, L2→L3 khoảng 3–8% tổng traffic), `cost_per_fixed_error`, và toàn bộ chỉ số trên **stratify theo `group` / `is_full` / `is_long`** — tỉ lệ lỗi trên input cắt dở khác hẳn input đầy đủ, gộp chung sẽ che mất chỗ model thật sự yếu.
