# BUILD_KAFKA.md — Luồng Kafka mô phỏng nạp output pipeline vào feedback loop

> **Ghi chú 2026-09-24:** trường `confidence` đã bị bỏ khỏi định dạng đầu vào theo quyết định của người
> dùng; `meta.api_confidence` không còn. Hàng đợi dùng trong pipeline là `broker.py` (SQLite) — xem
> [README.md](README.md) và [BUILD_PIPELINE.md](../BUILD_PIPELINE.md). Nội dung dưới đây là tài liệu build
> ban đầu, giữ nguyên để truy vết; các số đo `confidence` ở §3.3 thuộc capture cũ.

Tài liệu này dành cho agent thực thi. Đọc hết trước khi tạo file đầu tiên.

Nhiệm vụ: dựng `kafka_simulation/` — một đường ống mô phỏng Kafka đọc output thật của
pipeline NER trên staging và nạp vào `feedback/` dưới dạng `CaseRecord` hợp lệ.

Tuân thủ `AGENT.md` của dự án: chỉ thao tác bên trong project root.

> **Đã đơn giản hoá (2026-09-23):** lớp mô phỏng Kafka (quyết định 8 ở §2, toàn bộ §6, test 11–12 ở §8, mục partition ở §9) được thay bằng hàng đợi FIFO `message_queue.py`. Các phần đó giữ lại làm lịch sử thiết kế; code hiện tại xem `README.md`.

---

## 1. Bối cảnh và vị trí trong dự án

`feedback/` hiện chỉ ăn được `output_model.json` — file do script nội bộ sinh ra, có sẵn
`tokens`, `bio`, `spans`. Nhưng hệ thống thật ngoài staging **không trả về dạng đó**.

`kafka_simulation/template.txt` là bản capture `curl` gọi endpoint thật:

```
POST http://search-ltr.staging-osm.gsm-api.net/ner/predict-batch
request : {"texts": [ ...52 chuỗi... ]}
response: {"results": [{"result": {"L1": [], ..., "L7": []}, "confidence": 0.99}, ...]}
```

Đây là **công cụ nạp dữ liệu**, không phải một lớp trong feedback loop. Nó không nằm trong
bảng 12 bước ở `BUILD.md` §7 và **không được đụng vào `feedback/`**. Quan hệ đúng là:

```
[capture / API staging]  →  kafka_simulation/  →  CaseRecord  →  feedback/ (cascade)
                            ↑ tài liệu này                        ↑ BUILD.md §7 lo phần này
```

### 1.1. Khoảng cách phải lấp

| | API staging trả về | `feedback/core/normalize.py` bắt buộc có |
|---|---|---|
| Thực thể | list chuỗi theo level | `spans` có `start`/`end`/`level`/`text`/`truncated` |
| Token | **không có** | `tokens[]` |
| Nhãn BIO | **không có** | `bio[]` |
| Text gốc | chỉ có trong *request*, không có trong *response* | `text` |
| Độ tin | một số `confidence` cho cả câu | pipeline cũ không có trường này |

`normalize_record()` sẽ ném `KeyError` ngay tại `raw["tokens"]`. Không lấp khoảng cách này
thì không có gì để chạy Lớp 0, và `diff_metrics` không có đầu vào.

---

## 2. Quyết định đã chốt — KHÔNG được làm khác

Những điểm sau đã thống nhất ngày 2026-09-23. Thấy có lý do kỹ thuật để làm khác thì
**dừng lại và hỏi**, đừng tự đổi.

1. **Kafka là mô phỏng in-process bằng Python thuần, KHÔNG dựng broker thật.**
   Không Docker, không ZooKeeper/KRaft, không `kafka-python`, không `confluent-kafka`.
   Lý do: mục tiêu hiện tại là kiểm chứng *phép biến đổi dữ liệu*, không phải kiểm chứng
   hạ tầng. Thêm broker vào lúc này chỉ đổi một bài toán schema thành một bài toán vận hành.

   **Hệ quả phải ghi rõ trong `kafka_simulation/README.md`:** consumer group rebalance,
   replication, retention, backpressure và lỗi mạng **không được kiểm chứng** ở đây. Đừng
   viết tài liệu như thể đã chạy Kafka thật.

2. **Luồng một chiều.** Chỉ có topic nạp vào. Kết quả cascade ghi ra file JSON như
   `feedback/scripts/run_loop.py` vẫn làm, **không** có topic trả về. Không tạo
   `ner.reviewed`, không tạo DLQ.

3. **Producer ghép `texts[i]` ↔ `results[i]` một lần duy nhất, tại producer.**
   Mỗi message mang **một** địa chỉ và tự chứa cả input lẫn output. Response của API không
   mang text gốc, nên nếu để việc ghép cặp trôi xuống hạ nguồn thì chỉ cần một lần lệch thứ
   tự là toàn bộ offset sai mà không có cách nào phát hiện.

4. **Không bịa `truncated`.** API không trả tín hiệu cắt cụt. Đặt `truncated=False` cho mọi
   span và ghi `meta["truncated_unknown"]=True`. **Không** suy diễn từ việc span chạm cuối
   chuỗi. Đây là cùng nguyên tắc với `BUILD_FRONTEND.md` §5.1 về offset của PhoBERT.

5. **Span không định vị được thì bỏ span đó, gắn cờ, chạy tiếp.** Không ném lỗi, không bỏ
   cả message. Ghi vào `meta["unlocated"]` để Lớp 0 coi đó là dấu hiệu nghi ngờ. `spans`
   rỗng là trạng thái hợp lệ của schema — xem `feedback/eval/gold/ANNOTATION_GUIDE.md` §4.

6. **Bridge KHÔNG sửa lỗi parse.** Nhiệm vụ của nó là *chuyển dạng* trung thực. Model trả
   `L7: ["Nam", "Khởi NghĩaBến NghéQuận"]` cho `"Nam Kỳ Khởi NghĩaBến NghéQuận 1"` — tức
   bỏ rơi chữ "Kỳ" ở giữa — thì bridge phải sinh ra đúng BIO có lỗ hổng đó. Phát hiện và
   sửa là việc của Lớp 0→3. Bridge tự ý vá sẽ giấu mất chính thứ hệ thống sinh ra để tìm.

7. **Không sửa `template.txt`.** Nó là bằng chứng gốc. Mọi thứ phái sinh ghi ra file khác.
   Có dữ liệu mới (capture một đợt gọi API khác) thì **thêm file capture mới** cạnh nó
   (ví dụ `template_2026-10-01.txt`), **không** ghi đè hay nối thêm vào `template.txt`.
   Đây là lý do `capture_parser.py` phải nhận đường dẫn làm tham số (§7.2) thay vì hardcode
   tên file — thêm dữ liệu chỉ là thêm file, không phải sửa code.

8. **Đặt tên API của lớp mô phỏng theo `kafka-python`** (`send`, `poll`, `commit`,
   `topic`, `partition`, `offset`, `consumer_group`). Sau này thay bằng broker thật thì chỉ
   đổi phần khởi tạo, không phải viết lại producer/consumer.

---

## 3. Dữ liệu vào — số đo thực tế, đọc trước khi viết code

Đã đo trên toàn bộ 52 mẫu của `template.txt`. **Đây là số đo, không phải ước lượng** —
dùng chúng làm mốc nghiệm thu, đừng đo lại bằng mắt.

### 3.1. Tái dựng offset có khả thi không

Thử khớp ngược chuỗi thực thể vào text gốc để tìm `start`/`end`, chỉ dùng các bước **được
phép** trong bridge (NFC hoá + fallback khớp lỏng khoảng trắng — §7.3), **không** unescape:

| Cách khớp | Kết quả |
|---|---|
| Khớp nguyên văn trên text thô | 68/70 span (97,1%) — 2 miss |
| + NFC hoá text trước khi khớp | 68/70 — ca NFD vẫn miss, xem lý do bên dưới |
| + fallback khớp lỏng khoảng trắng | **69/70 (98,6%)** — còn đúng 1 miss, **cố định, biết nguyên nhân** |

Hai ca gốc lộ hai lỗi hệ thống khác nhau, nhưng chỉ một ca bridge sửa được:

- **`"Nguyễn Thái BìnhQuận 1Đức Chính"`** — model trả L3 = `"1 Đức Chính"` (có dấu cách),
  text gốc là `"1Đức Chính"` (không dấu cách). Model detokenize bằng cách nối token bằng
  dấu cách, nên **text thực thể không phải lúc nào cũng là substring của input**. Fallback
  khớp lỏng khoảng trắng ở §7.3 xử lý được ca này.

- **`"Nguyê<0303>n Huê<0323> Hồ Chí Minh Quận 1"`** — **miss vĩnh viễn, không phải lỗi bridge.**
  Đoạn text vào không phải Unicode NFD thật; nó chứa **chuỗi ký tự escape theo nghĩa đen**
  `<0303>`/`<0323>` (dấu `<`, chữ số, dấu `>` — sáu ký tự ASCII), do phiên terminal lúc lưu
  `template.txt` in dấu tổ hợp không hiển thị được ra dạng hex-notation thay vì giữ ký tự
  U+0303/U+0323 thật. Model trả về `"Nguyễn Huệ Hồ Chí"` bằng ký tự tổ hợp thật (NFC), nên
  không thể khớp với chuỗi `<0303>` nghĩa đen bằng bất kỳ phép biến đổi Unicode nào — NFC
  hoá không tác dụng lên một chuỗi vốn dĩ là text thường.

  **Đây là lỗi capture, không phải lỗi model hay lỗi bridge.** Sửa nó cần một hàm
  un-escape (`<XXXX>` → `chr(0xXXXX)`) — bị cấm trong đường chạy chính theo §2.6/§2.7: bridge
  không được âm thầm vá dữ liệu nguồn, dù nguồn lỗi ở khâu capture hay ở khâu model. Ca này
  **phải** rơi vào `meta["unlocated"]`, không được che bằng cách sửa `template.txt` hay chèn
  unescape vào bridge. Xem §8 test #4 — test khẳng định đúng hành vi "miss có chủ đích" này.

### 3.2. Span so với biên token

Nếu tách token bằng khoảng trắng thuần thì **18/70 span (25,7%) cắt ngang giữa một token** —
BIO không gán được. Hai nguyên nhân, cả hai đều phổ biến trong tập này:

```
dấu câu dính   : "Thành phố Hồ Chí Minh, Quận 1"  → span L2 kết thúc ở 21, token "Minh," hết ở 22
địa chỉ dính   : "Quận 1Bến Nghé"                 → token khoảng-trắng là "1Bến", span cắt giữa
```

Cách xử lý ở §7.4 đưa con số này về **0/52 câu có xung đột**.

### 3.3. Phân bố đáng chú ý

| Quan sát | Số |
|---|---|
| Câu không ra thực thể nào | 7/52 |
| `confidence` < 0,7 | 5/52 |
| `confidence` < 0,9 | 17/52 |
| `confidence` đúng bằng `0.0` | 1 (câu `"Bến Nghé Quận 1Hồ Chí Minh"`) |

`confidence` cấp câu là tín hiệu gate **miễn phí** — chuyển thẳng vào `meta` để Lớp 0 dùng.
Nhưng **không đặt ngưỡng trong code**: theo `BUILD.md` §2.5 mọi ngưỡng nằm ở
`feedback/config/thresholds.yaml` và phải do `eval/calibrate.py` sinh ra.

---

## 4. Môi trường

Dùng lại **`.venv/`** (venv của `feedback/`, không phải `.venv-layer1/`). Không cần cài
thêm gì: toàn bộ việc này làm được bằng `json`, `re`, `unicodedata`, `dataclasses`,
`hashlib` trong thư viện chuẩn.

Không thêm dependency vào `requirements.txt`.

---

## 5. Cấu trúc thư mục cần tạo

```
kafka_simulation/
├── README.md                cách chạy + ghi rõ giới hạn của mô phỏng (§2.1)
├── template.txt             ĐÃ CÓ — capture gốc, không sửa
├── __init__.py
├── bus.py                   lớp mô phỏng Kafka in-process
├── message.py               schema message + hàm validate
├── capture_parser.py        template.txt (capture curl) → list[payload]
├── producer.py              payload → bus
├── bridge.py                payload → CaseRecord          ★ phần khó nhất
├── consumer.py              bus → bridge → ghi ra file
├── run_simulation.py        entry point
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── template_52.json  52 payload trích sẵn, sinh bằng capture_parser
    ├── test_capture_parser.py
    ├── test_bus.py
    └── test_bridge.py        ★ test nặng nhất
```

Quy ước stub `.py` theo `BUILD.md` §5.1. Docstring tiếng Việt.

---

## 6. `bus.py` — mô phỏng Kafka

Giữ đúng những ngữ nghĩa Kafka **thật sự ảnh hưởng đến tính đúng đắn của dữ liệu**:

| Giữ | Bỏ |
|---|---|
| topic có N partition | replication |
| key → `partition = hash(key) % N` | retention / compaction |
| partition là log append-only, `offset` = chỉ số | rebalance |
| consumer group giữ offset đã commit theo từng partition | lỗi mạng, timeout |
| `poll(max_records)` trả theo lô, `commit()` tường minh | broker, serialization qua dây |

Ba tính chất **bắt buộc** phải mô phỏng đúng, vì chúng là thứ dễ gây sai khi lên thật:

1. **Thứ tự chỉ được bảo đảm trong một partition**, không phải toàn topic.
2. **`poll()` không tự commit.** Chưa `commit()` mà tạo consumer mới cùng group thì phải
   đọc lại từ offset cũ — đây chính là ngữ nghĩa at-least-once.
3. **Key quyết định partition.** Cùng một địa chỉ phải luôn rơi vào cùng partition. Dùng
   `hashlib.sha1` chứ **không** dùng `hash()` dựng sẵn của Python — `hash()` của `str` bị
   randomize theo tiến trình (`PYTHONHASHSEED`), chạy hai lần cho hai kết quả khác nhau.

Mặc định: một topic `ner.raw`, **3 partition**.

---

## 7. Các bước triển khai

### 7.1. `message.py` — định dạng message

```json
{
  "msg_id": "template#00019",
  "text": "Quận 1Bến Nghé",
  "result": {"L1": [], "L2": [], "L3": ["Quận 1", "Bến Nghé"],
             "L4": [], "L5": [], "L6": [], "L7": []},
  "confidence": 0.9832422584295273,
  "source": {"capture": "template.txt", "endpoint": "/ner/predict-batch",
             "batch_index": 19},
  "produced_at": "2026-09-23T15:40:00+07:00"
}
```

- `text` giữ **nguyên văn, chưa NFC hoá**. Chuẩn hoá là việc của bridge, và phải thấy được
  text trước khi chuẩn hoá thì mới gắn cờ `nfc_shift` trung thực được.
- `msg_id` theo quy ước `make_case_id` ở `feedback/core/normalize.py:41` (`stem#NNNNN`) để
  truy ngược về đúng dòng trong capture.
- `confidence` giữ nguyên độ chính xác float của API, không làm tròn.

Viết `validate(msg) -> list[str]` trả danh sách lỗi (rỗng = hợp lệ). Producer gọi nó trước
khi gửi; message hỏng thì bỏ qua và log, không làm hỏng cả lô.

### 7.2. `capture_parser.py` — đọc capture curl

Hàm vào chính nhận **đường dẫn**, không hardcode tên file:

```python
def parse_capture(path: str | Path) -> list[dict]:
    """Bóc texts[] + results[] từ một file capture curl, trả list payload đã ghép cặp."""
```

Lý do nhận tham số thay vì cố định `"template.txt"`: theo §2.7, có dữ liệu mới nghĩa là
**thêm một file capture khác**, gọi `parse_capture()` với đường dẫn khác — không sửa hàm,
không sửa `template.txt`.

Nội dung một file capture không phải JSON thuần. Nó là một phiên terminal: dòng lệnh
`curl`, bảng tiến độ của curl, rồi mới tới JSON. Phải bóc hai phần:

1. **`texts`** nằm trong `--data '{...}'` của dòng lệnh.
2. **`results`** nằm trong khối JSON, bắt đầu từ dòng `{` đứng ngay trước `"results"`.

Bắt buộc kiểm tra `len(texts) == len(results)` và **ném lỗi nếu lệch** — đây là bất biến duy
nhất giữ cho việc ghép cặp đúng, và một khi đã ghép sai thì không có cách nào phát hiện ở
hạ nguồn. Với `template.txt` con số phải là **52**; file capture khác thì số khác, hàm
không được giả định con số cố định.

Lưu ý ký tự: đọc bằng `encoding="utf-8"`. Nếu capture hiện `<0303>` dạng chữ thay vì ký tự
tổ hợp thật thì file đã bị terminal escape lúc lưu — báo lại, **không** tự viết hàm un-escape
vào đường chạy chính (nó chỉ được phép nằm trong test fixture).

### 7.3. `bridge.py` — định vị span

Đây là phần khó nhất và là nơi mọi lỗi im lặng sẽ nằm. Thuật toán chốt:

```
B1. text_nfc = NFC(text)
    nếu text_nfc != text  →  meta["nfc_shift"] = True

B2. Gom mọi (level, entity) thành một danh sách, sắp xếp ENTITY DÀI TRƯỚC.
    Với từng entity e (đã NFC hoá), tìm vị trí đầu tiên KHÔNG chồng lấn span đã nhận:
      A. khớp nguyên văn       : re.escape(e)
      B. khớp lỏng khoảng trắng: r'\s*'.join(re.escape(p) for p in e.split())
      C. vẫn không thấy        : BỎ span, thêm vào meta["unlocated"]

B3. Sắp span theo start tăng dần.
```

Vì sao **dài trước**: entity ngắn khớp trước có thể chiếm mất ký tự của entity dài bao quanh
nó. Trên 52 mẫu này hai thứ tự (dài-trước và L1→L7) cho **kết quả giống hệt nhau** — tập
hiện tại không phân biệt được — nhưng dài-trước là greedy an toàn hơn nên chốt dùng nó.

Bước B chỉ nới **khoảng trắng**, không nới gì khác. Không viết fuzzy match theo khoảng cách
Levenshtein: nó sẽ gán bừa cho những entity mà model bịa ra, mà đó đúng là loại lỗi feedback
loop cần bắt.

### 7.4. `bridge.py` — sinh `tokens` và `bio`

Tách token bằng khoảng trắng thuần **hỏng 25,7% span** (§3.2). Cách đúng là lấy hợp của bốn
tập điểm cắt:

```
điểm cắt = {0, len(text)}
         ∪ mọi biên span (start và end)      ← điều này bảo đảm căn khớp 100%
         ∪ biên các cụm khoảng trắng
         ∪ biên các ký tự không phải chữ-số (dấu phẩy, gạch ngang…)

token = các đoạn giữa hai điểm cắt liên tiếp, bỏ đoạn toàn khoảng trắng
```

Đưa biên span vào tập điểm cắt khiến việc căn khớp đúng **theo cấu trúc**, không phải nhờ
may mắn. Gán nhãn:

```
token nằm trọn trong một span  →  "B-" + level nếu token.start == span.start, ngược lại "I-" + level
còn lại                        →  "O"
```

Nhãn dùng hệ `L1`–`L7`, theo `BUILD.md` §2.1. **Không** có `B-STREET`, không bảng map.

Kết quả mong đợi trên 52 mẫu — dùng làm test:

```
text  : 'Thành phố Hồ Chí Minh, Quận 1'
tokens: ['Thành', 'phố', 'Hồ', 'Chí', 'Minh', ',', 'Quận', '1']
bio   : ['B-L2', 'I-L2', 'I-L2', 'I-L2', 'I-L2', 'O', 'B-L3', 'I-L3']

text  : '74, Phường 12Nơ Trang Long'
tokens: ['74', ',', 'Phường', '12', 'Nơ', 'Trang', 'Long']
bio   : ['B-L6', 'O', 'B-L4', 'I-L4', 'O', 'O', 'O']
         ↑ "Nơ Trang Long" là tên đường nhưng model không trả L5 nào.
           Bridge PHẢI để nguyên O — đây là ca parse sai, để Lớp 0→3 xử lý (§2.6).
```

### 7.5. `bridge.py` — dựng `CaseRecord`

Trả thẳng `CaseRecord` của `feedback/core/schemas.py`, **không** định nghĩa dataclass song
song, **không** ghi file JSON trung gian rồi gọi `load_records()` — làm vậy là tạo nguồn sự
thật thứ hai cho cùng một schema.

```python
ParseResult(tokens=..., bio=..., spans=..., source="gsm_api")
```

`source="gsm_api"` để phân biệt với `"ml_model"` mà `normalize_record()` đang dùng cho
`output_model.json`.

`meta` mang đúng những gì bridge biết, không thêm:

| Khoá | Khi nào | Ý nghĩa |
|---|---|---|
| `api_confidence` | luôn luôn | `confidence` cấp câu từ API |
| `truncated_unknown` | luôn luôn | API không cho tín hiệu cắt cụt (§2.4) |
| `source_msg_id` | luôn luôn | truy ngược về message |
| `nfc_shift` | khi text vào không phải NFC | cùng nghĩa với `normalize.py:56` |
| `unlocated` | khi có span bị bỏ | `[{"level": ..., "text": ...}]` |
| `loose_ws_match` | khi phải dùng fallback B | span nào khớp nhờ nới khoảng trắng |

### 7.6. `producer.py` / `consumer.py` / `run_simulation.py`

Producer nhận **danh sách đường dẫn capture**, mặc định `["kafka_simulation/template.txt"]`
nếu không truyền gì — không hardcode chỉ một file trong thân hàm:

```python
def run_producer(bus, topic, capture_paths: list[str | Path] = (DEFAULT_TEMPLATE,)):
    for path in capture_paths:
        for payload in parse_capture(path):
            ...
```

Có capture mới thì gọi `run_producer(bus, topic, capture_paths=["template.txt", "template_2026-10-01.txt"])`
— không sửa hàm, không sửa `template.txt` (§2.7).

Producer: `capture_parser` → `validate` → `bus.send(topic, key=sha1(text), value=msg)`.

Consumer: `poll` → `bridge` → gom `CaseRecord` → `commit()`. **Commit sau khi xử lý xong,
không phải trước** — đúng ngữ nghĩa at-least-once.

`run_simulation.py` nhận capture path(s) qua tham số dòng lệnh (mặc định `template.txt`),
chạy cả hai in-process và ghi ra `kafka_simulation/out/` (nhớ thêm vào `.gitignore`). Bàn
giao cho cascade dừng ở ranh giới này: **chưa gọi `feedback/`**, vì
`core/cascade.py` và `scripts/run_loop.py` hiện vẫn là stub chưa implement. In ra số
`CaseRecord` dựng được và đường dẫn file. Nối vào cascade là việc của `BUILD.md` §7.

---

## 8. Test bắt buộc

`test_bridge.py` là test có giá trị nhất — nó phải chạy trên **cả 52 mẫu thật**, không phải
vài ca tự bịa. Sinh `tests/fixtures/template_52.json` bằng chính `capture_parser`.

| # | Test | Khẳng định |
|---|---|---|
| 1 | `capture_parser` trên `template.txt` | đúng 52 texts và 52 results |
| 2 | lệch độ dài texts/results | ném lỗi, không âm thầm zip cụt |
| 3 | định vị span trên cả 52 mẫu | **69/70 span**, đúng 1 `unlocated` — xem test #4 |
| 4 | ca `"Nguyê<0303>n Huê<0323> …"` | **KHÔNG** định vị được (miss có chủ đích) — `meta["unlocated"]` chứa đúng entity L7 này. Đây là lỗi terminal-escape ở capture (§3.1), không phải lỗi bridge; bridge không được unescape để né test này |
| 5 | ca `"Nguyễn Thái BìnhQuận 1Đức Chính"` | L3 khớp `(21, 31)` = `"1Đức Chính"`, có cờ `loose_ws_match` |
| 6 | căn khớp token/span trên 52 mẫu | **0 ca** token chồng lấn một phần span |
| 7 | hai ca mẫu ở §7.4 | `tokens` và `bio` trùng khít chuỗi mong đợi |
| 8 | `span.text == raw_text[span.start:span.end]` | đúng với **mọi** span — đây là bất biến Lớp 0 sẽ kiểm |
| 9 | 7 câu không có thực thể | `spans == []`, không ném lỗi |
| 10 | `truncated` | luôn `False` và có `meta["truncated_unknown"]` |
| 11 | `bus`: `poll` rồi không `commit`, tạo consumer mới cùng group | đọc lại từ offset cũ |
| 12 | `bus`: cùng key gửi 2 lần | về cùng partition, qua 2 lần chạy tiến trình khác nhau |
| 13 | `CaseRecord.from_dict(record.to_dict())` | bằng bản gốc (round-trip) |

---

## 9. Nghiệm thu

```bash
.venv/Scripts/python.exe -m pytest kafka_simulation/tests -v
.venv/Scripts/python.exe -m kafka_simulation.run_simulation
```

| # | Kiểm tra | Kết quả mong đợi |
|---|---|---|
| 1 | Toàn bộ test §8 | xanh hết |
| 2 | `run_simulation` | in `52 message → 52 CaseRecord, 69/70 span định vị được, 1 unlocated (đã biết: escape NFD)` |
| 3 | Chạy lại lần hai | phân bố partition **giống hệt** lần đầu |
| 4 | Mở file output | thấy `api_confidence`, `truncated_unknown` ở mọi record |
| 5 | `git status` | chỉ có file mới trong `kafka_simulation/`, `template.txt` không đổi |
| 6 | `git diff feedback/` | **rỗng** |

---

## 10. Không làm

- Không sửa bất cứ thứ gì trong `feedback/`, `prompt/`, `models/`, `clean_data/`, `results/`,
  `frontend/`
- Không sửa `template.txt`
- Không cài Docker, không cài `kafka-python`/`confluent-kafka`, không thêm dependency
- Không tạo topic trả về, không tạo DLQ (§2.2)
- Không bịa `truncated` (§2.4)
- Không sửa lỗi parse trong bridge (§2.6)
- Không fuzzy match ngoài việc nới khoảng trắng (§7.3)
- Không hardcode ngưỡng `confidence` — ngưỡng thuộc `feedback/config/thresholds.yaml`
- Không gọi endpoint staging thật; tài liệu này chỉ replay capture
- Không định nghĩa lại `CaseRecord`/`Span`/`ParseResult` — import từ `feedback.core.schemas`
