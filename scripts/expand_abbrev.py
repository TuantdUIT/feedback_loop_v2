"""Khai triển từ viết tắt tiền tố hành chính trong file địa chỉ (mỗi dòng một địa chỉ).

  p. / P6 / F. 12 -> phường        q. / Q5   -> quận        h.   -> huyện
  tp / tp.        -> thành phố     kp / 27kp -> khu phố
  kdc -> khu dân cư   kđt / kdt -> khu đô thị   kcn -> khu công nghiệp   sn -> số nhà

Viết tắt một chữ cái (p, q, h) chỉ được khai triển khi có dấu chấm hoặc đứng ngay
trước số, để không đụng vào chữ cái là một phần tên ("phước long b", "đường d4").
Viết hoa giữ theo bản gốc: "P6" -> "Phường 6", "KĐT" -> "Khu Đô Thị".

Chạy:
  python scripts/expand_abbrev.py                 # dry-run, in các dòng sẽ đổi
  python scripts/expand_abbrev.py --apply         # ghi đè file
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata as ud
from pathlib import Path

MULTI = {
    "kp": "khu phố", "tp": "thành phố", "kdc": "khu dân cư", "kđt": "khu đô thị",
    "kdt": "khu đô thị", "kcn": "khu công nghiệp", "sn": "số nhà",
}
SINGLE = {"p": "phường", "f": "phường", "q": "quận", "h": "huyện"}

NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
MULTI_RE = re.compile(NOT_LETTER_BEFORE + r"(" + "|".join(MULTI) + r")(\.?)(?![^\W\d_])", re.I)
# một chữ cái: cần dấu chấm ("p. 9", "q.tân bình") hoặc số đứng ngay/cách một khoảng ("P6", "p 12")
SINGLE_RE = re.compile(r"(?<![^\W_])(" + "|".join(SINGLE) + r")(\.\s*|(?=\s?\d))", re.I)


def styled(abbr: str, full: str) -> str:
    return full.title() if abbr[0].isupper() else full


def expand(line: str) -> str:
    def multi(m: re.Match) -> str:
        return styled(m.group(1), MULTI[m.group(1).lower()]) + " "

    def single(m: re.Match) -> str:
        return styled(m.group(1), SINGLE[m.group(1).lower()]) + " "

    out = MULTI_RE.sub(multi, line)
    out = SINGLE_RE.sub(single, out)
    if out == line:  # không có viết tắt -> giữ nguyên dòng, kể cả khoảng trắng
        return line
    out = re.sub(r"(\d)(?=[^\W\d_])(?=(?:khu|thành|phường|quận|huyện|số)\b)", r"\1 ", out, flags=re.I)
    out = re.sub(r" {2,}", " ", out)
    return re.sub(r" ([,.)])", r"\1", out).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="clean_data/VAER_train_fix.txt")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = [ud.normalize("NFC", l) for l in raw.splitlines()]
    new = [expand(l) for l in lines]

    changed = [(a, b) for a, b in zip(lines, new) if a != b]
    for a, b in changed:
        print(f"- {a}\n+ {b}")
    print(f"\n{len(changed)}/{len(lines)} dòng thay đổi")

    if args.apply:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(eol.join(new) + eol)
        print(f"đã ghi {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
