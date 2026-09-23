"""Dung bang so sanh: moi dong la chuoi level L1->L7 (thu tu xuat hien) cua
Qwen / ONNX FP32 / PyTorch FP32 tren cung mot dia chi.

full_50.md    : 50 mau day du, giu nguyen ca mau Qwen loi (danh dau ro).
partial_50.md : 50 mau go do, CHI giu mau ca 3 model deu sinh ra ket qua —
                bo hoan toan mau Qwen loi/chua chay, de dong bo cach dem
                voi bang full_50.md."""
import json
from pathlib import Path

D = Path("models/testdata/test_v2")
local = {o["i"]: o for o in json.load((D / "local.json").open(encoding="utf-8"))}
qwen = {}
for ln in (D / "qwen.jsonl").read_text(encoding="utf-8").splitlines():
    if ln.strip():
        o = json.loads(ln)
        qwen[o["i"]] = o


def level_chain(spans):
    """spans -> 'L6>L5>L3' theo dung thu tu xuat hien trong dia chi."""
    if not spans:
        return "(rỗng)"
    if all("start" in s for s in spans):
        spans = sorted(spans, key=lambda s: s["start"])
    return ">".join(s["level"] for s in spans)


def build(lo, hi, title, note, outfile, only_valid):
    rows = []
    n3 = n_missing = kept = 0
    for i in range(lo, hi):
        entry = qwen.get(i)
        has_qwen = bool(entry and entry.get("parsed"))

        if not has_qwen:
            n_missing += 1
            if only_valid:
                continue

        q_spans = entry["parsed"].get("spans", []) if has_qwen else []
        o_spans = local[i]["onnx_fp32"]
        p_spans = local[i]["pytorch_fp32"]

        cq = level_chain(q_spans) if has_qwen else "—"
        co = level_chain(o_spans)
        cp = level_chain(p_spans)

        if not has_qwen:
            status = "⛔ Qwen lỗi (không parse được)"
        elif cq == co == cp:
            n3 += 1
            status = "✅ cả 3 giống"
        elif co == cp:
            status = "⚠️ ONNX=PyTorch, Qwen khác"
        else:
            status = "❌ ba bên lệch nhau"

        kept += 1
        rows.append("| %d | `%s` | %s | %s | %s | %s |" %
                    (i + 1, local[i]["text"], cq, co, cp, status))

    n = hi - lo
    n_cmp = kept if only_valid else n - n_missing
    pct = 100 * n3 / n_cmp if n_cmp else 0
    onnx_diff = sum(
        1 for i in range(lo, hi)
        if (not only_valid or (qwen.get(i) and qwen[i].get("parsed")))
        and level_chain(local[i]["onnx_fp32"]) != level_chain(local[i]["pytorch_fp32"])
    )

    head = [
        "# %s" % title, "", note, "",
        "Mỗi cột là **chuỗi level theo đúng thứ tự xuất hiện trong địa chỉ** "
        "(vd `L6>L5>L3`), không phải văn bản trích ra.",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---:|",
        "| Số mẫu trong tập gốc | %d |" % n,
        "| Qwen không có kết quả (loại khỏi so sánh) | %d |" % n_missing,
        "| Số mẫu đưa vào so sánh | %d |" % n_cmp,
        "| Cả 3 cho cùng chuỗi level | %d/%d (%.0f%%) |" % (n3, n_cmp, pct),
        "| ONNX FP32 khác PyTorch FP32 (trong %d mẫu so sánh) | %d/%d |"
        % (n_cmp, onnx_diff, n_cmp),
        "",
        "---", "",
        "| # | Đầu vào | Qwen | ONNX FP32 | PyTorch FP32 | Khớp |",
        "|---:|---|---|---|---|---|",
    ]
    (D / outfile).write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    print("  %-16s so sanh %d/%d mau (bo %d) | ca 3 giong: %d/%d (%.0f%%)"
          % (outfile, n_cmp, n, n_missing, n3, n_cmp, pct))


print("Da ghi:")
build(
    0, 50,
    "test_v2 — 50 mẫu ĐẦY ĐỦ (chuỗi level)",
    "Nguồn: `clean_data/VAER_test_v2_100.txt` dòng 1–50 (seed 20260923). "
    "Giữ nguyên toàn bộ 50 mẫu, kể cả mẫu Qwen lỗi kẹt vòng lặp suy luận (đánh dấu ⛔).",
    "full_50.md",
    only_valid=False,
)
build(
    50, 100,
    "test_v2 — 50 mẫu GÕ DỞ, CHỈ MẪU CẢ 3 MODEL ĐỀU CÓ KẾT QUẢ",
    "Nguồn: `clean_data/VAER_test_v2_100.txt` dòng 51–100 (seed 20260923). "
    "Đã bỏ hoàn toàn các mẫu Qwen không sinh ra kết quả (kẹt vòng lặp suy luận, "
    "chạm trần token) — chỉ giữ lại mẫu cả 3 model đều có kết quả, để so sánh "
    "công bằng và đồng bộ cách đếm với bảng 50 mẫu đầy đủ.",
    "partial_50.md",
    only_valid=True,
)
