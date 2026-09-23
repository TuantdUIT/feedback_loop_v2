# Nhược điểm: nguyên tắc gán nhãn cho đầu vào gõ dở

Ghi lại từ lần eval LLM qua API trên golden test (2026-09-23). Chưa chốt — cần quyết định thiết kế
trước khi sửa prompt hoặc gold.

## Nguồn

| Bộ | Model | Accuracy | Mẫu sai |
|---|---|---:|---:|
| `golden_test` (10 full + 10 uncomplete, seed 42) | `deepseek-flash` | 95% | 1 |
| `golden_test2` (50 full + 50 uncomplete, seed 43, không trùng test 1) | `deepseek-flash` | 84% (full 96%, uncomplete 72%) | 16 |

Prompt: `prompt/v2/compiled/system_prompt_v2_with_partial_input.txt`. Chi tiết từng mẫu:
`golden_dataset/evals/golden_test*/pred_deepseek.jsonl`, chỉ số: `metrics.json` cùng thư mục.

## Phân loại lỗi

Không vá prompt theo từng mẫu sai — làm vậy là overfit vào chính tập eval (mỗi ngoại lệ chỉ dựa trên
1–3 mẫu, ngoại lệ chồng ngoại lệ, và mặc định gold luôn đúng). 17 mẫu sai rơi vào ba loại:

| Loại | Mẫu | Xử lý |
|---|---|---|
| **Xung đột nguyên tắc prompt ↔ gold** | ~11 (Q1, Q2 bên dưới) | Chốt **một** nguyên tắc, sửa prompt hoặc gold cho nhất quán |
| **Gold nghi sai / không nhất quán** | vd `210/87 Số 5, Phường Nghĩa Đô…` (gold gán "Số 5" là L5 cho địa chỉ Hà Nội) | Rà gold, không sửa prompt |
| **Lỗi model ở ca mơ hồ** | `TaniBuilding Sơn Kỳ 1`, `lan phương mhbr tower`, `dự án him lam chợ l`, `Phường Tân Thạnh Tân Phú` | Chấp nhận là noise; chỉ xử lý nếu lặp lại trên dữ liệu mới |

## Câu hỏi thiết kế

**Khi chuỗi chưa gõ xong, model được phép dùng loại bằng chứng nào để gán nhãn?**

| Loại bằng chứng | Ví dụ | Prompt hiện tại | Gold hiện tại |
|---|---|---|---|
| **(a) Văn bản tự chứng minh** — tiền tố, tên địa danh có thật | "Đường", "Quận Li" → "Liên Chiểu" | Chấp nhận | Chấp nhận |
| **(b) Vị trí theo thứ tự địa chỉ Việt Nam** (số nhà → đường → phường → quận → tỉnh) | "123" ở đầu chuỗi = số nhà; "2" sau POI = số nhà | **Chấp nhận nửa vời** | Chấp nhận |
| **(c) Biết trước phần sẽ gõ tiếp** | "2" là số nhà vì bản đầy đủ là "207 Giải Phóng" | Cấm (`03_boundary_and_bio.txt` §3.5) | **Có lúc dùng** — gold uncomplete được cắt từ địa chỉ full |

Prompt đã dùng (b) ở một số chỗ nhưng không ở chỗ khác:

- "Hai Bà Trưng, 40" → "40" là L6 chỉ vì số đứng sau tên đường; "4 38 39 phố Đại Đồng" → chuỗi số
  là L6 chỉ vì đứng trước tên phố. Nhưng số đứng một mình ("123") thì bị cấm gán
  ([`02_label_definitions.txt:106`](../../prompt/v2/02_label_definitions.txt#L106)).
- Tiền tố "Đường" đứng một mình được gán L5, còn "dự án" hay "chợ" đứng một mình thì prompt không nói
  gì ([`03_boundary_and_bio.txt:60`](../../prompt/v2/03_boundary_and_bio.txt#L60)).

Phần cần chốt vì vậy gói lại trong ba câu:

### Q1. Ở chế độ gõ dở, vị trí trong địa chỉ có đủ làm bằng chứng không, kể cả khi không có thành phần nào bên cạnh xác nhận?

- **Có:** "123", "1436" → L6; "2" hay "109" đứng sau một POI hoàn chỉnh → L6.
- **Không:** giữ quy tắc hiện tại, sửa gold các mẫu này.
- Mẫu liên quan (6): `114c`, `210`, `123`, `1436`, `trường đại học kinh tế quốc dân 2`,
  `dự án legend tower 109`.

### Q2. Từ chỉ loại POI ("dự án", "chợ", "trường", "tower", "plaza"…) có được coi như tiền tố hành chính không?

Tức là khi đứng một mình hoặc đang gõ dở thì gán L7 với `truncated: true`, giống "Đường" → L5.

- **Có:** chỉ cần mở rộng câu đã có ở §3.5 — câu đó vốn đã viết "tiền tố … **loại thực thể**".
- **Không:** các mẫu này để trống, sửa gold.
- Mẫu liên quan (5): `dự án`, `dự án hap`, `dự án the`, `chợ`, `central pla`. Riêng `central pla`
  không có từ khoá POI nguyên vẹn, nên dù chọn "có" vẫn ở vùng xám.

### Q3. Gold của mẫu gõ dở có được dựa vào phần chưa gõ không?

- Đề xuất **không**, bất kể Q1 và Q2 chọn gì: gold phải gán được chỉ từ phần tiền tố đang có, nếu không
  thì gold đòi model biết trước tương lai.
- Đây là quy tắc rà gold, không phải sửa prompt. Ví dụ `…quốc dân 2`: gán L6 cho "2" hợp lệ nếu
  Q1 = có; nếu Q1 = không thì đó là rò rỉ thông tin từ bản full.

## Căn cứ để chọn

Căn cứ không nằm trong dữ liệu mà ở phía sau parse:

1. **Bộ phận nhận kết quả parse xử lý hai kiểu lỗi ra sao?** Chưa có mô tả trong repo.
   - Nếu `spans` rỗng khiến hệ thống quay về tìm kiếm toàn văn → để trống là an toàn → nghiêng về "không".
   - Nếu gợi ý địa chỉ cần biết người dùng đang gõ số nhà hay tên POI để lọc → để trống là mất thông
     tin → nghiêng về "có". Khi đó `truncated: true` là cách schema đã có sẵn để báo "chưa chắc".
2. **Prompt này đồng thời là chuẩn cho Lớp 2 và Lớp 3** ([`BUILD.md:34`](../../BUILD.md#L34)). Nguyên
   tắc chốt ở đây cũng là tiêu chí để DeepSeek và OpenAI chấm điểm các bản parse. Nếu prompt và gold
   còn lệch, feedback loop sẽ chấm sai một cách có hệ thống.

## Đề xuất

Q1 = có (chỉ ở chế độ gõ dở), Q2 = có, Q3 = không. Q1 và Q2 chỉ làm nhất quán những gì prompt đã làm
dở dang, không thêm ngoại lệ mới: sửa hai câu trong prompt, cộng một quy tắc rà gold.

## Quy trình sau khi chốt

1. Chốt Q1–Q3 (thiết kế, không phải dữ liệu).
2. Rà gold theo nguyên tắc đã chốt, sửa các mẫu không nhất quán.
3. Chỉ sửa prompt ở những chỗ trái với nguyên tắc đã chốt — sửa ở module nguồn `prompt/v2/0X_*.txt`,
   rồi `python build.py --with-partial-input`.
4. Đo trên một bộ held-out chưa dùng để chẩn đoán (ví dụ `golden_test3`), không đo lại trên
   `golden_test2`.

## Câu hỏi còn mở

- Phía sau parse (trong GSM Map) dùng `spans` ở chế độ autocomplete để làm gì?
