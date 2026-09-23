"""Bốc ngẫu nhiên mẫu từ golden_full và golden_uncomplete, gộp thành một bộ golden test.

Giữ nguyên định dạng golden (`{"texts": [...], "results": [{"result": {...}}]}`) và thêm
`sources` để biết mỗi mẫu lấy từ file nào, ở chỉ số nào. Ghi kèm `evals/<tên bộ>/input.txt`
(mỗi dòng một địa chỉ) làm đầu vào cho các script trong `scripts/external_api/`.

`--exclude` nhận các bộ test đã tạo trước đó: mẫu đã có trong đó (cùng file nguồn và chỉ số,
hoặc cùng text) không được bốc lại.

Chạy:
  python scripts/make_golden_test.py                                   # golden_test.json, 10+10, seed 42
  python scripts/make_golden_test.py --n 50 --seed 43 --out golden_test2.json --exclude golden_test.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "golden_dataset"
SOURCES = {"full": "golden_full.json", "uncomplete": "golden_uncomplete.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="số mẫu lấy từ mỗi file nguồn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="golden_test.json", help="tên file trong golden_dataset/")
    ap.add_argument("--exclude", nargs="*", default=[], help="các bộ test cũ trong golden_dataset/")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    used_idx, used_text = set(), set()
    for name in args.exclude:
        old = json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))
        used_idx |= {(s["file"], s["index"]) for s in old["sources"]}
        used_text |= set(old["texts"])

    texts, results, sources = [], [], []
    for subset, name in SOURCES.items():
        raw = json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))
        pool = [i for i, t in enumerate(raw["texts"]) if (name, i) not in used_idx and t not in used_text]
        if len(pool) < args.n:
            sys.exit(f"{name}: chỉ còn {len(pool)} mẫu chưa dùng, không đủ {args.n}")
        for i in sorted(rng.sample(pool, args.n)):
            texts.append(raw["texts"][i])
            results.append(raw["results"][i])
            sources.append({"subset": subset, "file": name, "index": i})

    out = GOLDEN_DIR / args.out
    input_txt = GOLDEN_DIR / "evals" / out.stem / "input.txt"
    out.write_text(json.dumps({"seed": args.seed, "exclude": args.exclude, "texts": texts, "results": results,
                               "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    input_txt.parent.mkdir(parents=True, exist_ok=True)
    input_txt.write_text("\n".join(texts) + "\n", encoding="utf-8")
    print(f"{len(texts)} mẫu (seed {args.seed}, loại {len(used_idx)} mẫu cũ) -> "
          f"{out.relative_to(ROOT)} và {input_txt.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
