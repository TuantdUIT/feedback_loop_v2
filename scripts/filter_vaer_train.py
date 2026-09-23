"""Lọc bỏ các dòng địa chỉ kém chất lượng khỏi file VAER (mỗi dòng một địa chỉ).

Năm tiêu chí loại bỏ (một dòng có thể dính nhiều tiêu chí, tất cả đều được ghi lại):

  0  thieu_dau_sai_chinh_ta  thiếu dấu hoặc sai dấu/chính tả
       - luật chính tả: âm tiết không dấu kết thúc bằng c/ch/p/t ("trac"), hoặc
         có vần ie/uo/ye chưa có mũ/móc ("vien", "truong", "nguyen")
       - luật ngữ cảnh: cặp từ liền nhau mà trong corpus một dạng khác (cùng chữ
         cái, khác dấu) phổ biến áp đảo ("ho chi" vs "hồ chí", "đỗ thị" vs "đô thị")
  1  gach_noi_giua_cap       có " - " ngăn cách các thành phần; dải số nhà
                             "62 - 64" (hai phía đều là số) không tính
  2  trung_keyword           từ khoá tiền tố lặp liền nhau ("đường đường",
                             "dự án dự án"), hoặc tên nhiều tiếng lặp lại mà có
                             ít nhất một lần không đứng sau tiền tố hành chính
                             ("thị trấn long thành long thành"); "phường X, quận X"
                             được chấp nhận
  3  keyword_la_toan_so      không còn chữ sau khi bỏ số/dấu câu, chứa HTML entity
                             rác, hoặc toàn từ không dấu mà không có từ khoá địa chỉ
  4  duoi_32_ky_tu           độ dài (sau NFC, bỏ khoảng trắng hai đầu) < 32

Chạy:
  python scripts/filter_vaer_train.py                    # dry-run, chỉ in thống kê
  python scripts/filter_vaer_train.py --apply            # ghi đè file + ghi .removed.tsv
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata as ud
from pathlib import Path

MIN_LEN = 32
TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}

PREFIX_WORDS = {
    "đường", "phố", "ngõ", "ngách", "hẻm", "kiệt", "phường", "quận", "huyện", "xã",
    "tỉnh", "tp", "p", "q", "h", "tx", "tt", "kdc", "kđt", "kcn", "khu",
}
PREFIX_BIGRAMS = {
    ("dự", "án"), ("thị", "trấn"), ("thị", "xã"), ("thành", "phố"), ("chung", "cư"),
    ("khu", "phố"), ("đô", "thị"), ("tổ", "dân"), ("tòa", "nhà"),
}
# Từ khoá tiền tố mà lặp liền nhau là lỗi nhập liệu.
REPEAT_KEYWORDS = {"đường", "phố", "phường", "quận", "huyện", "xã", "tỉnh", "ngõ", "hẻm"}
REPEAT_KEYWORD_BIGRAMS = {("dự", "án"), ("thị", "trấn"), ("thị", "xã"), ("thành", "phố"), ("chung", "cư")}
ADDRESS_KEYWORDS = PREFIX_WORDS | {"số", "sn", "lô", "tổ", "ấp", "thôn", "xóm", "block", "tower"}

VN_SYLLABLE = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnprstvx])?"
    r"([aăâeêioôơuưy]{1,3})"
    r"(ng|nh|ch|[cmnpt])?$"
)


def nfc(s: str) -> str:
    return ud.normalize("NFC", s)


def strip_all(s: str) -> str:
    """Bỏ toàn bộ dấu (thanh + mũ/móc/trăng), đ -> d."""
    s = ud.normalize("NFD", s)
    s = "".join(c for c in s if ud.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def canon(tok: str) -> str:
    """Dạng chuẩn không phụ thuộc vị trí đặt dấu thanh: 'hoà' và 'hòa' -> 'hoa2'."""
    d = ud.normalize("NFD", tok)
    tone = next((c for c in d if c in TONE_MARKS), "")
    base = nfc("".join(c for c in d if c not in TONE_MARKS))
    return base + (str(sorted(TONE_MARKS).index(tone) + 1) if tone else "")


def words(line: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", line.lower())


def has_diacritic(tok: str) -> bool:
    return strip_all(tok) != tok


# ---------------------------------------------------------------- tiêu chí 0
# Cụm thiếu dấu lặp lại nhiều đến mức luật ngữ cảnh không coi là hiếm.
KNOWN_UNACCENTED = {("chung", "cu"), ("dân", "cu"), ("dan", "cu"), ("can", "ho")}


def orthography_errors(line: str) -> list[str]:
    ws = words(line)
    bad = [f"{a} {b}" for a, b in zip(ws, ws[1:]) if (a, b) in KNOWN_UNACCENTED]
    for w in ws:
        if has_diacritic(w) or not VN_SYLLABLE.match(w):
            continue
        m = VN_SYLLABLE.match(w)
        nucleus, final = m.group(2), m.group(3) or ""
        # c/ch/p/t cuối chỉ đi với thanh sắc/nặng -> không dấu là thiếu dấu
        if final in {"c", "ch", "p", "t"}:
            bad.append(w)
        # ie/uo/ye trong vần luôn cần mũ/móc (iê, uô/ươ, yê); "ia/ua" thì không
        elif re.search(r"ie|uo|ye", nucleus) and final:
            bad.append(w)
    return bad


def build_bigram_forms(lines: list[str]) -> dict[tuple[str, str], collections.Counter]:
    forms: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for line in lines:
        ws = words(line)
        for a, b in zip(ws, ws[1:]):
            forms[(strip_all(a), strip_all(b))][(canon(a), canon(b))] += 1
    return forms


def build_unigram_forms(lines: list[str]) -> dict[str, collections.Counter]:
    forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in lines:
        for w in words(line):
            forms[strip_all(w)][canon(w)] += 1
    return forms


def unaccented_is_rare(tok: str, uni) -> bool:
    """Từ không dấu mà các dạng có dấu của nó phổ biến gấp >= 3 lần."""
    cnt = uni[tok]
    mine = cnt[tok]
    accented = sum(v for k, v in cnt.items() if k != tok)
    return accented >= 3 * mine


def context_errors(line: str, forms, uni) -> list[str]:
    """Chỉ bắt thiếu dấu: một từ hoàn toàn không dấu, trong khi cả cặp từ lẫn
    riêng từ đó đều có dạng có dấu phổ biến áp đảo. Hai dạng cùng có dấu
    ("phường ba" / "phường bà") không bị so, vì cả hai đều có thể đúng."""
    bad = []
    ws = words(line)
    for a, b in zip(ws, ws[1:]):
        cnt = forms[(strip_all(a), strip_all(b))]
        mine_key = (canon(a), canon(b))
        mine = cnt[mine_key]
        best, best_n = cnt.most_common(1)[0]
        if best == mine_key or best_n < 5 or best_n < 3 * mine:
            continue
        for tok, best_tok in ((a, best[0]), (b, best[1])):
            if not has_diacritic(tok) and canon(tok) != best_tok and unaccented_is_rare(tok, uni):
                bad.append(f"{a} {b}")
                break
    return bad


# ---------------------------------------------------------------- tiêu chí 1
def has_level_hyphen(line: str) -> bool:
    for m in re.finditer(r"\s*-\s*", line):
        left = line[: m.start()].split()
        right = line[m.end():].split()
        lt = left[-1] if left else ""
        rt = right[0] if right else ""
        if not (re.search(r"\d$", lt) and re.match(r"\d", rt)):
            return True
    return False


# ---------------------------------------------------------------- tiêu chí 2
def is_prefixed(ws: list[str], i: int) -> bool:
    if i >= 1 and ws[i - 1] in PREFIX_WORDS:
        return True
    return i >= 2 and (ws[i - 2], ws[i - 1]) in PREFIX_BIGRAMS


def duplicate_keyword(line: str) -> str | None:
    ws = words(line)
    for i in range(len(ws) - 1):
        if ws[i] == ws[i + 1] and ws[i] in REPEAT_KEYWORDS:
            return f"{ws[i]} {ws[i]}"
    for i in range(len(ws) - 3):
        if (ws[i], ws[i + 1]) == (ws[i + 2], ws[i + 3]) and (ws[i], ws[i + 1]) in REPEAT_KEYWORD_BIGRAMS:
            return " ".join(ws[i:i + 4])
    # tên >= 2 tiếng lặp lại; bỏ qua nếu bản thân nó là từ khoá tiền tố
    for n in (3, 2):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(ws) - n + 1):
            g = tuple(ws[i:i + n])
            if n == 2 and g in PREFIX_BIGRAMS:
                continue
            if g in seen and seen[g] + n <= i:
                if not (is_prefixed(ws, seen[g]) and is_prefixed(ws, i)):
                    return " ".join(g)
            seen.setdefault(g, i)
    return None


# ---------------------------------------------------------------- tiêu chí 3
def strange_or_numeric(line: str) -> bool:
    if not re.search(r"[^\W\d_]", line):
        return True
    if re.search(r"&\s*#\s*\d+\s*;", line):
        return True
    ws = words(line)
    return not any(has_diacritic(w) for w in ws) and not any(w in ADDRESS_KEYWORDS for w in ws)


# ---------------------------------------------------------------- main
def classify(line: str, forms, uni) -> list[str]:
    reasons = []
    ortho = orthography_errors(line)
    ctx = context_errors(line, forms, uni)
    if ortho or ctx:
        reasons.append("0:thieu_dau_sai_chinh_ta[" + ",".join(ortho + ctx) + "]")
    if has_level_hyphen(line):
        reasons.append("1:gach_noi_giua_cap")
    dup = duplicate_keyword(line)
    if dup:
        reasons.append(f"2:trung_keyword[{dup}]")
    if strange_or_numeric(line):
        reasons.append("3:keyword_la_toan_so")
    if len(line.strip()) < MIN_LEN:
        reasons.append("4:duoi_32_ky_tu")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="clean_data/VAER_train_fix.txt")
    ap.add_argument("--apply", action="store_true", help="ghi đè file và ghi <file>.removed.tsv")
    args = ap.parse_args()

    path = Path(args.path)
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = [nfc(l) for l in raw.splitlines()]
    forms = build_bigram_forms(lines)
    uni = build_unigram_forms(lines)

    kept, removed = [], []
    per_reason: collections.Counter = collections.Counter()
    for line in lines:
        reasons = classify(line, forms, uni)
        if reasons:
            removed.append((line, reasons))
            per_reason.update(r.split("[")[0] for r in reasons)
        else:
            kept.append(line)

    print(f"tổng {len(lines)} | giữ {len(kept)} | bỏ {len(removed)}")
    for r, c in sorted(per_reason.items()):
        print(f"  {r:<28} {c}")

    if args.apply:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(eol.join(kept) + eol)
        log = path.with_suffix(".removed.tsv")
        with log.open("w", encoding="utf-8", newline="") as fh:
            fh.write("line\treasons" + eol)
            fh.writelines(f"{l}\t{' | '.join(r)}{eol}" for l, r in removed)
        print(f"đã ghi {path} và {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
