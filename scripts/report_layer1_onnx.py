"""Generate models/REPORT.md from the saved, separate-process measurements."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MEASUREMENTS = MODELS / "measurements"
VARIANTS = ("pytorch", "onnx", "int8", "int8_perchannel")
LEVELS = tuple(f"L{i}" for i in range(1, 8))

# This order is stated explicitly in the model card's Usage example. The
# published config.json itself contains only LABEL_0 ... LABEL_20.
CARD_LABELS = (
    "B_PRO", "B_CITY", "NUMBER_TYPE", "B_DIST", "TO_TYPE", "B_STREET",
    "I_PRO", "I_DIST", "PRO_TYPE", "OTHER", "I_STREET", "B_WARD",
    "STREET_TYPE", "I_CITY", "CITY_TYPE", "O", "NUMBER", "WARD_TYPE",
    "I_WARD", "DIST_TYPE", "TO",
)
LEVEL_BY_CARD_LABEL = {
    "B_PRO": "L2", "I_PRO": "L2", "PRO_TYPE": "L2",
    "B_CITY": "L2", "I_CITY": "L2", "CITY_TYPE": "L2",
    "B_DIST": "L3", "I_DIST": "L3", "DIST_TYPE": "L3",
    "B_WARD": "L4", "I_WARD": "L4", "WARD_TYPE": "L4",
    "B_STREET": "L5", "I_STREET": "L5", "STREET_TYPE": "L5",
    "NUMBER": "L6",
}


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", "")


def label(label_id: int) -> str:
    return f"{CARD_LABELS[label_id]} (#{label_id})"


def align_tokens(record: dict) -> list[tuple[int, int]] | None:
    """Only accept an exact, monotonic BPE-piece alignment to the raw text."""
    text = record["text"]
    cursor = 0
    offsets = []
    for token, special in zip(record["tokens"], record["special_mask"]):
        if special:
            offsets.append((-1, -1))
            continue
        piece = token.removesuffix("@@")
        if piece == "<unk>" or not piece:
            return None
        start = text.find(piece, cursor)
        if start < 0 or any(not char.isspace() for char in text[cursor:start]):
            return None
        end = start + len(piece)
        offsets.append((start, end))
        cursor = end
    if record["truncated"] or any(not char.isspace() for char in text[cursor:]):
        return None
    return offsets


def covered_chars(text: str, spans: list[tuple[int, int]]) -> set[int]:
    return {
        index
        for start, end in spans
        for index in range(start, end)
        if text[index].isalnum()
    }


def agreement_24(pytorch: dict) -> tuple[list[str], int, int, int]:
    source = json.loads((ROOT / "output_model.json").read_text(encoding="utf-8"))
    predictions = pytorch["output_model"]
    count = Counter()
    agree = Counter()
    model_count = Counter()
    source_count = Counter()
    aligned = 0
    flags = 0
    unmapped_predicted = Counter()
    for original, predicted in zip(source, predictions, strict=True):
        offsets = align_tokens(predicted)
        if offsets is None:
            continue
        aligned += 1
        model_spans = {level: [] for level in LEVELS}
        for (start, end), label_id, special in zip(
            offsets, predicted["label_ids"], predicted["special_mask"], strict=True
        ):
            if special:
                continue
            card_label = CARD_LABELS[label_id]
            level = LEVEL_BY_CARD_LABEL.get(card_label)
            if level is None:
                if card_label != "O":
                    unmapped_predicted[card_label] += 1
                continue
            model_spans[level].append((start, end))
        record_flagged = False
        for level in LEVELS:
            model_chars = covered_chars(predicted["text"], model_spans[level])
            main_spans = [
                (span["start"], span["end"])
                for span in original["spans"]
                if span["level"] == level
            ]
            main_chars = covered_chars(original["text"], main_spans)
            if not (model_chars or main_chars):
                continue
            count[level] += 1
            model_count[level] += bool(model_chars)
            source_count[level] += bool(main_chars)
            agree[level] += model_chars == main_chars
            record_flagged |= model_chars != main_chars
        flags += record_flagged
    lines = [
        "So sánh **phủ ký tự chữ/số đúng bằng nhau** theo từng level sau khi ghép "
        "các mảnh BPE vào chuỗi gốc bằng phép tìm khớp chính xác, tuần tự. "
        "Chỉ dùng bản ghi ghép trọn vẹn, không bị cắt ở 128 token. "
        "Đây là ước lượng thô vì tokenizer không cung cấp offset chính thức và "
        "config chỉ có tên nhãn tổng quát.",
        "",
        "| Level | Bản ghi dùng được (có level ở ít nhất một bên) | Đồng thuận | Tỷ lệ | Có ở model chính | Có ở PhoBERT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        total = count[level]
        rate = f"{100 * agree[level] / total:.2f}%" if total else "không xác định"
        lines.append(
            f"| {level} | {total} | {agree[level]} | {rate} | "
            f"{source_count[level]} | {model_count[level]} |"
        )
    lines += [
        "",
        f"Ghép được **{aligned}/{len(source)}** bản ghi. "
        f"Có **{flags}/{aligned}** bản ghi ghép được có ít nhất một level bất đồng "
        f"({100 * flags / aligned:.2f}%); đây là tỷ lệ gắn cờ tham khảo cho Lớp 1."
        if aligned else "Không ghép được bản ghi nào; không tính được tỷ lệ gắn cờ.",
        "",
        "Các nhãn dự đoán không thể ánh xạ (ngoài O): "
        + (", ".join(f"{name}: {n}" for name, n in unmapped_predicted.items())
           if unmapped_predicted else "không có")
        + ".",
        "",
        "Chỉ có 24 bản ghi, không có gold set. Bất đồng không chứng minh model nào sai. "
        + (
            "Tỷ lệ bất đồng **vượt 40%**: hai model có thể đang hiểu schema khác nhau; "
            if aligned and flags / aligned > 0.4
            else "Tỷ lệ bất đồng chưa vượt 40%; "
        )
        + "chưa nên dùng con số này để dự trù chi phí Lớp 2 một cách chắc chắn.",
    ]
    return lines, aligned, flags, len(source)


def main() -> None:
    survey = json.loads((MODELS / "survey.json").read_text(encoding="utf-8"))
    results = {
        name: json.loads((MEASUREMENTS / f"{name}.json").read_text(encoding="utf-8"))
        for name in VARIANTS
    }
    base = results["pytorch"]
    lines = [
        "# Báo cáo chuẩn bị PhoBERT NER địa chỉ cho Lớp 1",
        "",
        f"Nguồn: [kiendt/phobert-ner-address](https://huggingface.co/kiendt/phobert-ner-address), "
        f"revision `{survey['model_revision']}`. Đã export ONNX FP32 và quantize động "
        "AVX2 với `per_channel=False` và `per_channel=True`. Chưa tích hợp vào `feedback/`.",
        "",
        "## Môi trường",
        "",
        "Virtualenv: `.venv-layer1/`; `HF_HOME`: `.hf-cache/` trên ổ chứa dự án.",
        "",
        "| Package | Version |", "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in survey["versions"].items()]
    lines += [
        "", "## Nhãn và tokenizer", "",
        "`config.id2label` gốc đầy đủ nằm dưới đây. File config của cả bốn bản "
        "giữ nguyên nhãn tổng quát này. Cột tên thực thể lấy từ **`label_list` trong "
        "ví dụ Usage của model card**, theo cùng thứ tự ID; đây là giả định khi "
        "đối chiếu nhãn, chưa được xác nhận bằng config của tác giả.",
        "",
        "| ID | config.id2label | Model card `label_list` | Ánh xạ |",
        "|---:|---|---|---|",
    ]
    for index, name in enumerate(CARD_LABELS):
        level = LEVEL_BY_CARD_LABEL.get(name, "KHÔNG ÁNH XẠ ĐƯỢC")
        lines.append(
            f"| {index} | {survey['id2label'][str(index)]} | {name} | {level} |"
        )
    lines += [
        "",
        "L1 (quốc gia) và L7 (POI) không có nhãn tương ứng. `NUMBER_TYPE`, "
        "`TO_TYPE`, `OTHER`, `O`, `TO` không được đoán level.",
        "",
        f"Tokenizer: `{survey['tokenizer_class']}`, `is_fast={survey['tokenizer_is_fast']}`. "
        f"`return_offsets_mapping=True` thất bại: `{survey['offset_error']}`. "
        "Vì không có offset, không thể kiểm tra `text[start:end]` hay xuất "
        "`Span(start, end)` trực tiếp. Dấu `@@` trong token là mảnh BPE, "
        "không phải một phần của chuỗi gốc.",
        "",
        "Model card của bản fine-tune **không nói rõ dữ liệu huấn luyện đã tách từ "
        "hay chưa**, nhưng ví dụ Usage truyền câu thô. "
        "[Tài liệu PhoBERT gốc](https://huggingface.co/docs/transformers/model_doc/phobert) "
        "yêu cầu đầu vào đã tách từ. Chưa thể kết luận yêu cầu của bản fine-tune; "
        "toàn bộ phép đo dưới đây dùng nguyên văn địa chỉ thô của dự án.",
        "",
        f"Tokenize câu mẫu: `{survey['sample']}`", "",
        "| Token | Offset ký tự | `text[start:end]` |",
        "|---|---|---|",
    ]
    for row in survey["sample_tokens"]:
        lines.append(f"| `{md(row['token'])}` | — | — |")

    lines += [
        "", "## Dung lượng và RAM", "",
        "Mỗi hàng được đo trong **một tiến trình riêng** trên Windows. RAM sau nạp "
        "là RSS; RAM đỉnh là `peak_wset`. MB = 1.000.000 byte. Thời gian chỉ để tham khảo.",
        "",
        "| Bản model | RAM sau nạp (MB) | RAM đỉnh (MB) | Đĩa (MB) | 548 bản ghi (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    titles = {
        "pytorch": "PyTorch FP32", "onnx": "ONNX FP32",
        "int8": "ONNX INT8", "int8_perchannel": "ONNX INT8 per-channel",
    }
    for name in VARIANTS:
        row = results[name]
        lines.append(
            f"| {titles[name]} | {row['after_load_mb']:.1f} | "
            f"{row['peak_mb']:.1f} | {row['disk_mb']:.1f} | {row['elapsed_548_s']:.1f} |"
        )
    if all((MEASUREMENTS / f"{name}_noarena.json").exists() for name in VARIANTS[1:]):
        lines += [
            "", "Thử lại với `SessionOptions.enable_cpu_mem_arena=False` "
            "(mỗi bản một tiến trình riêng):", "",
            "| Bản model | RAM sau nạp (MB) | RAM đỉnh (MB) | 548 bản ghi (s) |",
            "|---|---:|---:|---:|",
        ]
        for name in VARIANTS[1:]:
            row = json.loads((MEASUREMENTS / f"{name}_noarena.json").read_text(encoding="utf-8"))
            lines.append(
                f"| {titles[name]} | {row['after_load_mb']:.1f} | "
                f"{row['peak_mb']:.1f} | {row['elapsed_548_s']:.1f} |"
            )

    lines += [
        "", "## Câu mẫu (§6a)", "",
        "Offset không có vì tokenizer slow. `*` đánh dấu nhãn lệch so với PyTorch FP32.",
        "",
        "| Token | Offset | PyTorch FP32 | INT8 | INT8 per-channel | Lệch? |",
        "|---|---|---|---|---|---|",
    ]
    sample = {name: results[name]["sample"] for name in VARIANTS}
    for index, token in enumerate(sample["pytorch"]["tokens"]):
        ids = {name: sample[name]["label_ids"][index] for name in ("pytorch", "int8", "int8_perchannel")}
        mismatch = ids["pytorch"] != ids["int8"] or ids["pytorch"] != ids["int8_perchannel"]
        lines.append(
            f"| `{md(token)}` | — | {label(ids['pytorch'])} | {label(ids['int8'])} | "
            f"{label(ids['int8_perchannel'])} | {'*' if mismatch else ''} |"
        )

    lines += [
        "", "## Dữ liệu thật (§6b)", "",
        "500 dòng đầu `VAER_test_fix.txt` và đủ 48 dòng "
        "`VAER_train_partial_input_50.txt`; không lấy ngẫu nhiên. "
        "So sánh theo vị trí token giống hệt giữa các bản, bỏ token đặc biệt. "
        "Trùng nhãn FP32 ONNX với PyTorch FP32: "
        + str(sum(
            x == y
            for row_a, row_b in zip(base["results"], results["onnx"]["results"], strict=True)
            for x, y, special in zip(row_a["label_ids"], row_b["label_ids"], row_a["special_mask"], strict=True)
            if not special
        ))
        + " / "
        + str(sum(not special for row in base["results"] for special in row["special_mask"]))
        + ".",
        "",
        "| Bản INT8 | Nhóm | Token trùng/tổng | Tỷ lệ | Đạt ≥99,5%? | "
        "|ΔP| trung bình tại nhãn FP32 | |ΔP| lớn nhất |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    differences = {}
    for variant in ("int8", "int8_perchannel"):
        differences[variant] = []
        for group in ("full", "partial"):
            total = matches = 0
            prob_diffs = []
            for row_index, (a, b) in enumerate(zip(base["results"], results[variant]["results"], strict=True)):
                if a["group"] != group:
                    continue
                if a["tokens"] != b["tokens"]:
                    raise ValueError(f"Tokenization differs at row {row_index}")
                for token_index, (a_id, b_id, special) in enumerate(zip(a["label_ids"], b["label_ids"], a["special_mask"], strict=True)):
                    if special:
                        continue
                    total += 1
                    matches += a_id == b_id
                    prob_diffs.append(abs(a["probs"][token_index][a_id] - b["probs"][token_index][a_id]))
                    if a_id != b_id:
                        differences[variant].append((row_index, token_index, a, a_id, b_id))
            lines.append(
                f"| {titles[variant]} | {group} | {matches}/{total} | "
                f"{100 * matches / total:.3f}% | {'có' if matches / total >= 0.995 else 'không'} | "
                f"{sum(prob_diffs) / len(prob_diffs):.5f} | {max(prob_diffs):.5f} |"
            )
    lines += [
        "",
        "Xác suất được so sánh tại **nhãn mà PyTorch FP32 chọn** trên cùng token; "
        "|ΔP| là trị tuyệt đối. Các câu bị cắt ở 128 token vẫn được tính trên "
        "phần model thực sự đọc.",
        "",
        f"Số câu bị cắt: full = {sum(row['truncated'] for row in base['results'] if row['group'] == 'full')}, "
        f"partial = {sum(row['truncated'] for row in base['results'] if row['group'] == 'partial')}.",
    ]
    for variant in ("int8", "int8_perchannel"):
        lines += [
            "", f"### Mọi token lệch: {titles[variant]} ({len(differences[variant])})", "",
            "| Dòng trong nhóm (1-based) | Token | Nhãn FP32 → INT8 | Câu chứa token |",
            "|---:|---|---|---|",
        ]
        for row_index, token_index, row, a_id, b_id in differences[variant]:
            group_index = row_index + 1 if row["group"] == "full" else row_index - 499
            lines.append(
                f"| {row['group']} #{group_index} | `{md(row['tokens'][token_index])}` "
                f"(vị trí {token_index}) | {label(a_id)} → {label(b_id)} | "
                f"{md(row['text'])} |"
            )

    lines += ["", "## Đối chiếu 24 bản ghi model chính (§6c)", ""]
    comparison, _, _, _ = agreement_24(base)
    lines += comparison
    lines += [
        "", "## Khuyến nghị", "",
        "**Không nên dùng INT8, phần RAM tiết kiệm không bù được**: cả hai "
        "bản INT8 đều dưới ngưỡng trùng nhãn 99,5% ở cả nhóm đầy đủ lẫn nhóm "
        "gõ dở. Trong các bản đã kiểm tra, dùng **ONNX FP32** nếu cần định dạng "
        "ONNX; PyTorch FP32 cho cùng nhãn trên 548 bản ghi và dùng ít RAM đỉnh "
        "hơn trong phép đo này. Chưa tích hợp vào feedback loop khi chưa có "
        "offset đáng tin cậy và gold set đánh giá chất lượng trên dữ liệu địa chỉ thô.",
        "",
        "Chạy lại: `.venv-layer1/Scripts/python.exe scripts/build_layer1_onnx.py`, "
        "sau đó chạy `scripts/measure_layer1_onnx.py` riêng cho từng "
        "`--variant pytorch|onnx|int8|int8_perchannel` và cuối cùng "
        "`scripts/report_layer1_onnx.py`.",
        "",
    ]
    (MODELS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {MODELS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
