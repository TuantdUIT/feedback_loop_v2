"""Dung tap test_v2: 50 mau day du + 50 mau gõ dở, doc lap nhau, lay tu 500 dong dau
cua clean_data/VAER_test_fix.txt (dung tap ma PhoBERT da chay o REPORT.md §6b)."""
import random
from pathlib import Path

SEED = 20260923
POOL = [l for l in Path("clean_data/VAER_test_fix.txt").read_text(encoding="utf-8").splitlines() if l.strip()][:500]

rng = random.Random(SEED)
idx = list(range(len(POOL)))
rng.shuffle(idx)
full_idx, part_idx = sorted(idx[:50]), sorted(idx[50:100])     # doc lap, khong giao nhau
assert not (set(full_idx) & set(part_idx))

def cut(s, r):
    """Cat giua chung nhu dang go do: giu 40-95% do dai, toi thieu 5 ky tu."""
    n = len(s)
    k = max(5, int(n * r.uniform(0.40, 0.95)))
    return s[:k]

rng2 = random.Random(SEED + 1)
full = [POOL[i] for i in full_idx]
part = [cut(POOL[i], rng2) for i in part_idx]

out = Path("clean_data/VAER_test_v2_100.txt")
out.write_text("\n".join(full + part) + "\n", encoding="utf-8")

meta = Path("clean_data/VAER_test_v2_100.index.txt")
meta.write_text(
    "# Tap test_v2 — 100 dong, sinh boi scripts/make_test_v2.py (seed=%d)\n"
    "# Dong   1-50 : mau DAY DU, lay nguyen van tu 500 dong dau VAER_test_fix.txt\n"
    "# Dong  51-100: mau GO DO, cat con 40-95%% do dai goc\n"
    "# Chi so goc (0-based) trong 500 dong dau:\n"
    "# full = %s\n# part = %s\n" % (SEED, full_idx, part_idx), encoding="utf-8")

print("da ghi %s (%d dong)" % (out, len(full)+len(part)))
print()
print("--- 3 mau DAY DU ---")
for s in full[:3]: print("   ", s)
print("--- 3 mau GO DO (goc -> cat) ---")
for i, s in zip(part_idx[:3], part[:3]): print("    %-52s -> %s" % (POOL[i][:52], s))
