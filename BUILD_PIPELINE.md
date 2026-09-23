# BUILD_PIPELINE.md — Pipeline đánh giá output parser: upload → DeepSeek → Qwen judge → metrics

> **Trạng thái: bản thiết kế chờ duyệt (2026-09-24).** Chưa có dòng code nào theo tài liệu này.
> Khi được duyệt, §1–2 và §7 của `BUILD.md` sẽ được sửa để trỏ sang đây (xem §11).

---

## 1. Mục tiêu

Người dùng upload một file output của parser đang triển khai (định dạng `kafka_simulation/template.txt`).
Hệ thống tự chạy theo dây chuyền:

1. Đưa từng case vào message queue.
2. DeepSeek parse lại cùng input bằng prompt v2.
3. Qwen làm LLM-as-a-judge: bản parse của **model đang triển khai** hay của **DeepSeek** tốt hơn.
4. Tổng hợp metrics để trả lời: case nào model parse chưa tốt, và dữ liệu có đủ căn cứ để **khuyến nghị retrain** không.

Người dùng theo dõi toàn bộ trên giao diện.

**Ngoài phạm vi:** chạy retrain; gọi endpoint staging; sửa prompt v2 (Q1–Q3 trong
`scripts/external_api/nhuocdiem.md` chốt sau, chỉ cần build lại prompt, không sửa code).

---

## 2. Quyết định đã chốt (người dùng, 2026-09-23/24)

| # | Quyết định |
|---|---|
| 1 | **Bỏ PhoBERT** (Lớp 1 cũ). **Bỏ OpenAI**; vai trò chốt kết luận chuyển cho Qwen |
| 2 | **DeepSeek parse** (`deepseek-flash`, API chính hãng). **Qwen làm judge** (`qwen/qwen3.6-35b-a3b` qua OpenRouter) |
| 3 | Judge trả kết luận cấp case: **model tốt hơn / DeepSeek tốt hơn / hoà** |
| 4 | Đầu vào là file định dạng `template.txt`, **không có trường `confidence`** |
| 5 | Trigger dây chuyền: giai đoạn trước xong thì kích hoạt giai đoạn sau |
| 6 | **Trần chi phí $2 / lần upload** |
| 7 | Giữ prompt v2 hiện tại; chốt nguyên tắc gán nhãn đầu vào gõ dở sau |
| 8 | `template.txt` chỉ là dữ liệu mô phỏng; dữ liệu thật trải trên nhiều trường hợp hơn |

Giữ nguyên từ `BUILD.md` §2: bộ nhãn hệ L (quyết định 1), prompt lõi là nguồn sự thật duy nhất và
prompt nhiệm vụ nối sau nó (2), không sửa `compiled/` (3), mọi ngưỡng nằm trong
`config/thresholds.yaml` (5), `compiled/` không commit + khối `_meta.prompt_sha` (6).

---

## 3. Luồng tổng thể

```
 UI upload ──► ingest ──► [ingested] ──► deepseek_parse ──► [parsed] ──► compare ──┬─► [agreed] ─────────────┐
 (template)    capture_parser             DeepSeek API        Lớp 0         so span │                         │
               + bridge + Lớp 0           + Lớp 0                                   └─► [to_judge] ─► qwen_judge ─► [judged] ─┤
                                                                                                    Qwen × 2 (A/B, B/A)        │
                                                                                                                               ▼
                                                                                                   gate ──► [decided] ──► run_report
                                                                                                   (quyết định case)      (metrics, khuyến nghị)
```

`[tên]` là một topic của queue. Mỗi giai đoạn là một worker chỉ đọc topic của mình và publish sang topic
kế tiếp khi xong — đó chính là trigger dây chuyền. Không có orchestrator trung tâm gọi tuần tự.

**Rẽ nhánh `agreed`:** nếu span của model và DeepSeek **giống hệt nhau** (sau chuẩn hoá ở §6.1) thì
không gọi Qwen. Trên dữ liệu sạch, đây là phần lớn case, nên tiết kiệm phần lớn chi phí judge.
Giống nhau không có nghĩa là đúng (cả hai có thể cùng sai) — kênh random audit (§7.4) lấy mẫu cả nhánh này.

---

## 4. Message queue

### 4.1. Broker: SQLite, lưu trên đĩa

Hàng đợi hiện tại (`kafka_simulation/message_queue.py`) nằm trong bộ nhớ và mất khi tắt tiến trình.
Upload từ giao diện chạy bất đồng bộ, có thể kéo dài nhiều phút (Qwen có case > 10 phút), nên cần lưu bền.

Chọn **SQLite** (thư viện chuẩn, không cài thêm):
- Lưu bền, xem lại được sau khi tắt server.
- Giao diện đọc tiến độ bằng truy vấn trực tiếp.
- Claim message nguyên tử bằng một câu `UPDATE … RETURNING`, an toàn khi nhiều worker cùng chạy.

File: `feedback/store/pipeline.db` (bỏ qua bởi `.gitignore`).

### 4.2. Bảng

| Bảng | Cột chính | Vai trò |
|---|---|---|
| `runs` | `run_id`, `filename`, `kind` (`normal`/`calibration`), `status`, `n_cases`, `budget_usd`, `cost_usd`, `prompt_sha`, `git_commit`, `created_at` | Một lần upload |
| `cases` | `run_id`, `case_id`, `stage`, `status`, `record` (JSON của `CaseRecord`), `cost_usd`, `updated_at` | Nguồn sự thật của từng case |
| `messages` | `id`, `topic`, `run_id`, `case_id`, `status` (`pending`/`processing`/`done`/`failed`), `attempts`, `available_at`, `last_error` | Hàng đợi |

**Message chỉ mang `run_id` + `case_id`**, không mang dữ liệu. Dữ liệu nằm ở `cases.record`. Như vậy
không có hai bản sao phải giữ đồng bộ, và giao diện chỉ cần đọc một bảng.

### 4.3. Ngữ nghĩa

- **Claim:** worker lấy message `pending` cũ nhất có `available_at ≤ now`, chuyển sang `processing`.
- **Ack:** xử lý xong → trong **cùng một transaction**: ghi `cases.record`, đánh `done`, publish message
  cho topic kế tiếp. Không có trạng thái "đã ghi kết quả nhưng chưa chuyển tiếp".
- **Retry:** lỗi tạm thời (HTTP 429/5xx, timeout) → `pending` lại với backoff, tối đa 3 lần (config).
  Quá 3 lần → `failed` (dead-letter), case hiện lỗi trên giao diện.
- **Crash giữa chừng:** lúc khởi động, message `processing` bị trả về `pending`. Nghĩa là
  *at-least-once*: một lần gọi API có thể bị lặp lại sau crash. Chấp nhận, vì hiếm và chi phí mỗi lần
  nhỏ; cost của lần lặp vẫn được cộng vào run.
- **Đánh thức:** publish xong thì bật `threading.Event` của topic đích để worker chạy ngay; worker
  vẫn poll mỗi 1 giây phòng khi lỡ tín hiệu.

### 4.4. Worker

Chạy trong cùng tiến trình với server FastAPI, khởi động trong `lifespan`. Số luồng mỗi giai đoạn đặt
trong `config/models.yaml`: mặc định DeepSeek 4, Qwen 4, các giai đoạn còn lại 1.

`kafka_simulation/` giữ vai trò component message queue: thêm `broker.py` (SQLite). `MessageQueue`
trong bộ nhớ và CLI `run_simulation.py` giữ lại cho test và chạy nhanh không cần server.

---

## 5. Các giai đoạn

### 5.1. `ingest` (đồng bộ, lúc upload)

1. `capture_parser.parse_capture` đọc `texts[]` + `results[]`; từ chối file lệch độ dài.
2. `message.validate` — **bỏ kiểm `confidence`**. Nếu file vẫn có trường này thì bỏ qua, không lỗi.
3. `bridge.to_case_record` → `CaseRecord.old` (parse của model, `source="model"`), định vị span, sinh
   tokens/BIO. Span không định vị được ghi vào `meta.unlocated`.
4. **Lớp 0** trên `old` (offset, BIO, level). Vi phạm ghi vào `layers.layer0_old`, **không chặn**
   case: lỗi hình thức của model cũng là tín hiệu model parse chưa tốt.
5. Ước tính chi phí (§8) và trả về cho giao diện **trước khi chạy**; người dùng bấm "Chạy" mới publish
   các case vào `[ingested]`.

### 5.2. `deepseek_parse`

- Gọi DeepSeek với prompt v2 qua `core/llm_client.py`. Logic lấy từ
  `scripts/external_api/deepseek/run_deepseek_on_file.py`: retry khi rỗng, dừng ngay khi gặp 401/402/422,
  ghi `utc` và token cache hit/miss từng lần thử, tính cost theo khung giờ cao/thấp điểm.
- **Lớp 0** trên output DeepSeek. Vi phạm hard (offset sai, BIO sai) → gọi lại 1 lần; vẫn sai → case kết
  thúc với quyết định `deepseek_invalid`, không đưa sang judge.
- Ghi `CaseRecord.new` (`source="deepseek"`) và `layers.deepseek` (usage, cost, attempts).

### 5.3. `compare`

- Chuẩn hoá hai bản về dạng so sánh (§6.1), chạy `diff_metrics` (COR/INC/PAR/MIS/SPU) → `CaseRecord.diff`.
- Giống hệt → `[agreed]`. Khác → `[to_judge]`.

### 5.4. `qwen_judge`

Xem §6. Ghi `layers.judge`.

### 5.5. `gate`

Xem §7.1. Ghi `CaseRecord.decision` và `layers.gate`. Case nào xong thì kiểm tra run đã xong chưa;
xong thì tạo báo cáo run (§7.3).

---

## 6. Qwen judge

### 6.1. Dạng chuẩn đưa cho judge

Hai bản parse được đưa về **đúng một dạng** trước khi đưa cho Qwen:

```json
{"spans": [{"level": "L5", "text": "Nơ Trang Long"}, {"level": "L4", "text": "Phường 12"}]}
```

- Chỉ giữ `level` và `text`, sắp theo vị trí xuất hiện trong input.
- **Bỏ `start`/`end`**: offset đã được Lớp 0 kiểm; giữ lại chỉ thêm khác biệt hình thức.
- **Bỏ `truncated`**: model đang triển khai không có tín hiệu này (`meta.truncated_unknown`), DeepSeek
  thì có. Giữ lại sẽ **để lộ bản nào của DeepSeek** và làm bản DeepSeek trông "đầy đủ hơn".
- **Bỏ `tokens`/`bio`** của DeepSeek, cùng lý do.

### 6.2. Chống thiên kiến

| Thiên kiến | Biện pháp | Đo bằng |
|---|---|---|
| **Position bias** — thích phương án ở một vị trí cố định | Chấm **2 lần**: (model, DeepSeek) và (DeepSeek, model). Chỉ chấp nhận khi hai lần cùng chỉ ra một bên. Lật → `inconclusive` | **Tỉ lệ lật**; probe "hai bản giống hệt" phải ra hoà (§7.5) |
| **Verbosity bias** — thích bản dài hơn, nhiều span hơn | Dạng chuẩn §6.1; rubric ghi rõ *"nhiều span hơn không phải là tốt hơn; span thừa bị phạt như span thiếu"*; judge phải nêu từng span sai kèm quy tắc trong prompt v2 | Trong các case có kết luận: tỉ lệ bên thắng có nhiều span hơn, so với tỉ lệ nền; probe "thêm span thừa" (§7.5) |
| **Self-enhancement bias** — thích output của chính mình hoặc cùng họ | Qwen **không parse ở bất kỳ đâu** trong pipeline. Ẩn nguồn: gọi là "Phương án 1/2", không nhắc model hay DeepSeek | Tỉ lệ báo động giả khi model đúng (§7.5); độ lệch giữa hai chiều sai trên gold |

### 6.3. Prompt

- System: prompt lõi v2 (`prompt_loader.load_prompt`) + `feedback/prompts/judge.txt` nối sau
  (`BUILD.md` quyết định 2). `judge.txt` chỉ chứa nhiệm vụ, rubric và định dạng output.
- User: `{"input": "...", "phuong_an_1": {...}, "phuong_an_2": {...}}`.
- Output (JSON, `response_format=json_object`):

```json
{
  "winner": "1" | "2" | "tie",
  "issues_1": [{"text": "...", "level": "L4", "problem": "sai level | thừa | thiếu | sai ranh giới", "rule": "02 §L4"}],
  "issues_2": [ ... ]
}
```

Không hỏi judge tự chấm độ tin cậy: số tự báo của LLM không đáng tin. Độ tin cậy đến từ việc hai lần
chấm có nhất quán không, và từ hiệu chuẩn.

### 6.4. Tham số

- Giữ 4 biện pháp chống vòng lặp suy luận của `run_qwen_on_file.py`.
- `temperature` và `presence_penalty` **đo trước khi chốt** (bước 4 của §10): so `temperature=0` hiện tại
  với khuyến nghị của trang model (`1.0`, `presence_penalty=1.5`) trên cùng một bộ nhỏ — theo tỉ lệ lật,
  tỉ lệ kẹt/retry, reasoning token, độ trễ. Giá trị chốt ghi vào `models.yaml`.

---

## 7. Quyết định và metrics

### 7.1. Quyết định cho từng case

| Quyết định | Điều kiện | Ý nghĩa |
|---|---|---|
| `agree` | Hai bản giống hệt | Không gọi judge |
| `model_better` | Cả 2 lần chấm chọn model | Model ổn |
| `tie` | Cả 2 lần ra hoà | Khác nhau nhưng tương đương |
| `deepseek_better` | Cả 2 lần chấm chọn DeepSeek | **Ứng viên "model parse chưa tốt"** |
| `inconclusive` | Hai lần chấm không nhất quán | Đưa vào hàng chờ người xem |
| `deepseek_invalid` | Output DeepSeek sai hình thức sau 1 lần gọi lại | Không so được |
| `failed` / `budget_exceeded` | Lỗi API quá số lần retry / chạm trần chi phí | Hiện lỗi trên giao diện |

`deepseek_better` chỉ được tính là **lỗi đã xác nhận của model** khi thêm điều kiện severity (theo
`diff_metrics` × `labels.yaml.severity_weight`) ≥ ngưỡng trong `thresholds.yaml`. Trước khi hiệu chuẩn,
ngưỡng để trống và mọi kết luận mang nhãn **"chưa hiệu chuẩn"**.

`CaseRecord.decision` dùng tập giá trị trên thay cho `KEEP_OLD`/`ACCEPT_NEW`/`REWRITE`/`ESCALATE_HUMAN`
(chưa có code nào dùng tập cũ; `REWRITE` không còn vì không còn arbiter).

### 7.2. Metrics của một run

**Luồng:** số case theo từng giai đoạn/quyết định; lỗi; cost DeepSeek, cost Qwen, tổng so với trần;
độ trễ mỗi giai đoạn.

**Mức đồng thuận model ↔ DeepSeek** (không cần gold): tỉ lệ `agree`; span-level P/R/F1 của model so
với DeepSeek, theo từng level.

**Sức khoẻ judge** (không cần gold):
- Tỉ lệ lật khi đảo thứ tự — chỉ báo trực tiếp position bias.
- Phân bố kết luận (model / DeepSeek / hoà).
- Chỉ báo verbosity: trong các case có bên thắng, tỉ lệ bên thắng có nhiều span hơn.

**Ước lượng chất lượng model:**
- Tỉ lệ `deepseek_better` (và lỗi đã xác nhận) trên tổng case, kèm **khoảng tin cậy Wilson 95%**.
- Chia theo level, theo `is_full`/độ dài, theo kiểu lỗi (thiếu/thừa/sai ranh giới/sai level từ `diff_metrics`).
- Danh sách case lỗi, kèm `issues` judge nêu, để người xem.

### 7.3. Khuyến nghị retrain

Khuyến nghị retrain **chỉ được bật** khi đủ cả ba:

1. **Judge đã hiệu chuẩn** (§7.4) và đạt chuẩn: Cohen κ với người ≥ ngưỡng (khởi điểm 0,6 như `BUILD.md`
   bước 8), tỉ lệ báo động giả ≤ ngưỡng, tỉ lệ lật ≤ ngưỡng.
2. **Đủ dữ liệu:** số lỗi đã xác nhận ≥ `min_cases`.
3. **Tỉ lệ lỗi đủ cao:** cận dưới khoảng tin cậy của tỉ lệ lỗi đã xác nhận ≥ `max_error_rate`
   (tổng thể, hoặc trên một lát cắt như một level/kiểu lỗi — lỗi tập trung đáng retrain hơn lỗi rải rác).

Mọi con số đều nằm trong `thresholds.yaml`. Khi thiếu điều kiện 1, giao diện ghi rõ
**"Chưa hiệu chuẩn — chưa có căn cứ khuyến nghị retrain"**, vẫn hiện đầy đủ metrics để tham khảo.

Case lỗi đã xác nhận được xuất thành **ứng viên dữ liệu retrain** (`export_silver.py`), mang nhãn silver
(nhãn đề xuất là bản DeepSeek, được judge chọn, chưa qua người). **Case từ run `calibration` không bao
giờ được xuất**, để không nhiễm tập dùng đo judge.

### 7.4. Hiệu chuẩn judge

Tập hiệu chuẩn phải **cùng phân bố với dữ liệu thật** (golden hiện có khác xa template: 0% vs 24–29% POI,
27% vs 1% thành phần dính liền). Vì vậy:

- **Random audit:** mỗi run, `audit_rate` (khởi điểm 5%) số case — lấy ở **mọi** nhánh, kể cả `agree` —
  vào hàng chờ gán nhãn tay. Nhãn tích luỹ dần thành gold thật.
- **Run `calibration`:** upload file template kèm file gold. Pipeline chạy như thường; gold cho biết bên
  nào thật sự tốt hơn (span F1 so với gold; bằng nhau → hoà), từ đó tính precision/recall bắt lỗi, κ,
  tỉ lệ báo động giả, tỉ lệ lật. `calibrate.py` quét ngưỡng và ghi `thresholds.yaml`.
- Golden hiện có (`golden_full`/`golden_uncomplete`) chỉ dùng kiểm tra phụ các kiểu POI/địa chỉ đầy đủ.

Gán nhãn **không được** dựa trên bản DeepSeek làm sẵn — gold sẽ nghiêng về DeepSeek.

### 7.5. Probe thiên kiến (chạy trong run `calibration`, không cần gold)

| Probe | Cách tạo | Kết quả đúng |
|---|---|---|
| Position | Hai phương án **giống hệt** | `tie` ở cả hai thứ tự |
| Verbosity | Bản đúng vs bản đúng **+ một span thừa** | Bản đúng thắng |
| Báo động giả | Case có `results` đã biết là đúng (như `template.txt` hiện tại) | Không chọn DeepSeek khi DeepSeek khác |

---

## 8. Chi phí

- Ước tính trước khi chạy: `n × 0,0015` (DeepSeek) + `n_khác_nhau × 2 × c_qwen`. Số case khác nhau chưa
  biết trước, nên ước tính trần giả định mọi case đi qua judge. Với `c_qwen ≈ $0,008`: `template.txt`
  46 case ≈ **$0,80 trần**, thực tế thấp hơn nhiều nhờ nhánh `agree`.
- **Trần $2/run** (`thresholds.yaml → budget.per_run_usd`). Trước mỗi lần gọi API, worker kiểm
  `runs.cost_usd + ước tính lần gọi` so với trần; vượt → case nhận `budget_exceeded`, không gọi API.
  Case đang chạy dở vẫn được hoàn tất, nên chi phí thực có thể vượt trần tối đa một vòng gọi.
- Cost DeepSeek tính từ token × bảng giá theo khung giờ; cost Qwen lấy từ `usage.cost` của OpenRouter.
  Đơn giá nằm trong `models.yaml`.

---

## 9. Giao diện

Thêm trang **`/pipeline`** vào `frontend/server.py`. Trang so sánh parser hiện có ở `/` giữ nguyên.
Server và worker chạy bằng `.venv-layer1` (đã có FastAPI 0.141.1; `.venv` chưa có). Pipeline không cần
torch, nên sau khi bỏ PhoBERT có thể chuyển sang `.venv` bằng cách cài `fastapi uvicorn`.

| Khu vực | Nội dung |
|---|---|
| **Upload** | Chọn file → hiển thị số case, lỗi định dạng, **ước tính chi phí**, trần $2 → nút "Chạy". Tuỳ chọn "Lượt hiệu chuẩn" + file gold |
| **Danh sách run** | Trạng thái, thanh tiến độ theo giai đoạn, cost / trần, thời gian |
| **Chi tiết run** | Bảng case: input, span model, span DeepSeek, kết luận từng lần chấm, quyết định; lọc theo quyết định |
| **Chi tiết case** | Hai bản parse cạnh nhau, chỗ khác nhau được tô, `issues` judge nêu ở cả 2 lần chấm, cost, độ trễ |
| **Metrics** | §7.2 + khuyến nghị retrain §7.3, với nhãn "chưa hiệu chuẩn" khi chưa có hiệu chuẩn |

Giao diện poll `/api/runs/{id}` mỗi 2 giây khi run đang chạy. API mới:
`POST /api/runs/preview` (upload, trả ước tính), `POST /api/runs` (bắt đầu chạy), `GET /api/runs`,
`GET /api/runs/{id}`, `GET /api/runs/{id}/cases`, `GET /api/runs/{id}/cases/{case_id}`, `GET /api/runs/{id}/metrics`.

---

## 10. Thứ tự triển khai

Mỗi bước có test xanh trước khi sang bước sau, và xem được trên giao diện từ bước 6.

| # | Bước | File chính | Nghiệm thu |
|---|---|---|---|
| 1 | **Tài liệu này** | `BUILD_PIPELINE.md` | Người dùng duyệt |
| 2 | Bỏ `confidence`; broker SQLite + topic + ack/retry | `kafka_simulation/{capture_parser,message,producer,bridge,broker}.py`, `template.txt`, fixture, test | 46 case của `template.txt` vào `[ingested]`; test retry, dead-letter, khôi phục sau crash |
| 3 | Lớp 0, `bio`, `diff_metrics`, `llm_client`, worker DeepSeek | `feedback/core/`, `feedback/stages/` | Run thật trên `template.txt` tới `[parsed]`; cost khớp tính tay |
| 4 | Judge Qwen + đo tham số | `feedback/stages/qwen_judge.py`, `feedback/prompts/judge.txt` | Hai lần chấm đảo thứ tự; bảng so `temperature`; probe position/verbosity chạy được |
| 5 | Gate, metrics, báo cáo, khuyến nghị | `feedback/stages/gate.py`, `feedback/eval/` | Metrics §7.2 đúng trên dữ liệu dựng tay; khuyến nghị tắt khi chưa hiệu chuẩn |
| 6 | Giao diện `/pipeline` | `frontend/` | Upload → theo dõi → xem case → xem metrics, trên trình duyệt |
| 7 | Hiệu chuẩn, random audit, export silver | `feedback/eval/calibrate.py`, `feedback/scripts/export_silver.py` | Run `calibration` ra κ, báo động giả, tỉ lệ lật; run `calibration` không bị export |
| 8 | Sửa `BUILD.md`, README, test tổng | | |

---

## 11. Thay đổi so với cấu trúc hiện có

```
feedback/
├── config/
│   ├── labels.yaml              giữ
│   ├── models.yaml              điền: deepseek, qwen, đơn giá, số worker, tham số
│   └── thresholds.yaml          viết lại: budget, gate, audit_rate, retrain (để trống tới khi hiệu chuẩn)
├── prompts/
│   ├── judge.txt                + (thay l2_judge.txt)
│   ├── l1_detector.txt          − xoá
│   └── l3_arbiter.txt           − xoá
├── core/
│   ├── schemas.py               sửa: Decision theo §7.1
│   ├── bio.py, diff_metrics.py, llm_client.py     implement (đang là stub)
│   ├── normalize.py, prompt_loader.py             giữ
│   └── cascade.py               − xoá (thay bằng worker theo topic)
├── layers/ → stages/            đổi tên
│   ├── validator.py             (layer0_validator.py)
│   ├── deepseek_parser.py       +
│   ├── compare.py               +
│   ├── qwen_judge.py            +
│   ├── gate.py                  +
│   └── layer1_slm.py, layer2_judge.py, layer3_arbiter.py   − xoá
├── pipeline/                    + đăng ký worker, topic, lifespan
├── eval/                        calibrate.py, report.py implement; + judge_metrics.py, probes.py
├── store/                       + pipeline.db (gitignore)
└── scripts/                     run_pipeline.py (chạy không cần UI), export_silver.py

kafka_simulation/                + broker.py; bỏ confidence ở capture_parser/message/producer/bridge
frontend/                        + trang /pipeline và API §9
BUILD.md                         §1–2, §7 trỏ sang tài liệu này; đánh dấu Lớp 1–3 cũ đã thay
```

Các stub bị xoá chưa có code (chỉ docstring), nên không mất logic nào.

---

## 12. Điểm còn mở (không chặn triển khai)

- **Ai gán nhãn** cho random audit và run `calibration`. Cần trước bước 7.
- **Ảnh định nghĩa thiên kiến** người dùng nhắc tới chưa được gửi; §6.2 dùng định nghĩa phổ biến.
- **Q1–Q3** (`scripts/external_api/nhuocdiem.md`): chốt sau; ảnh hưởng cả DeepSeek lẫn judge vì cùng prompt v2.
