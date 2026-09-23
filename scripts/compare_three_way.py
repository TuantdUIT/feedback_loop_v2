"""So ba chieu: model chinh (output_model.json) vs PhoBERT ONNX vs Qwen qua OpenRouter."""
import json, sys, unicodedata, re, torch
from collections import Counter
sys.path.insert(0, "scripts")
from layer1_adapter import LABELS, normalize, nfc
from transformers import AutoTokenizer, AutoModelForTokenClassification

recs = json.load(open("output_model.json", encoding="utf-8"))
qwen = {o["i"]: o for o in json.load(open("models/qwen_single.json", encoding="utf-8"))}

M = "models/phobert-ner-address-pytorch"
tk = AutoTokenizer.from_pretrained(M); md = AutoModelForTokenClassification.from_pretrained(M).eval()
def phobert(text):
    e = tk(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad(): ids = md(**e).logits.argmax(-1)[0].tolist()
    toks = tk.convert_ids_to_tokens(e["input_ids"][0])
    return [[t, LABELS[i]] for t, i in zip(toks, ids) if t not in ("<s>", "</s>")]

def spanset(spans): return {(s["level"], nfc(s["text"])) for s in spans}

# --- kiem tra hop le kieu Lop 0 tren output Qwen ---
def layer0(rec, p):
    v = []
    if p is None: return ["khong parse duoc JSON"]
    txt = unicodedata.normalize("NFC", rec["text"])
    if p.get("input") != rec["text"]: v.append("input khong khop")
    if len(p.get("bio", [])) != len(p.get("tokens", [])): v.append("len(bio)!=len(tokens)")
    prev = "O"
    for tg in p.get("bio", []):
        if tg.startswith("I-") and prev[2:] != tg[2:]: v.append("BIO transition sai")
        if tg != "O" and not re.fullmatch(r"[BI]-L[1-7]", tg): v.append("nhan la: "+tg)
        prev = tg
    for s in p.get("spans", []):
        if txt[s["start"]:s["end"]] != s["text"]: v.append("offset lech: "+s.get("text",""))
    return v

gold_n = qw_n = ph_n = 0
qw_hit = ph_hit = both = 0
qw_full = ph_full = 0
per_lv = {}
l0_bad = 0; l0_msgs = Counter()
for i, r in enumerate(recs):
    g = spanset(r["spans"])
    p = qwen[i]["parsed"]
    viol = layer0(r, p)
    if viol: l0_bad += 1; [l0_msgs.update([v.split(":")[0]]) for v in viol]
    q = spanset(p.get("spans", [])) if p else set()
    h = set(map(tuple, normalize(phobert(r["text"]))))
    gold_n += len(g); qw_n += len(q); ph_n += len(h)
    qw_hit += len(g & q); ph_hit += len(g & h); both += len(g & q & h)
    qw_full += (g == q); ph_full += (g == h)
    for lv, tx in g:
        d = per_lv.setdefault(lv, [0, 0, 0]); d[0] += 1
        d[1] += (lv, tx) in q; d[2] += (lv, tx) in h

print("=== SO BA CHIEU tren 24 ban ghi / %d span cua model chinh ===" % gold_n)
print("  %-26s %-18s %s" % ("", "Qwen (OpenRouter)", "PhoBERT ONNX + chuan hoa"))
print("  %-26s %-18s %s" % ("span khop model chinh", "%d/%d (%.1f%%)"%(qw_hit,gold_n,100*qw_hit/gold_n),
                                                     "%d/%d (%.1f%%)"%(ph_hit,gold_n,100*ph_hit/gold_n)))
print("  %-26s %-18s %s" % ("ban ghi khop tron ven", "%d/24"%qw_full, "%d/24"%ph_full))
print("  %-26s %-18s %s" % ("tong span sinh ra",     str(qw_n), str(ph_n)))
print("  ca hai cung dong y voi model chinh: %d/%d span (%.1f%%)" % (both, gold_n, 100*both/gold_n))
print()
print("=== THEO LEVEL ===")
print("  Level  span  Qwen         PhoBERT")
for lv in sorted(per_lv):
    n, q, h = per_lv[lv]
    print("  %-6s %4d  %2d (%5.1f%%)  %2d (%5.1f%%)" % (lv, n, q, 100*q/n, h, 100*h/n))
print()
print("=== KIEM TRA HOP LE (Lop 0) tren output Qwen ===")
print("  ban ghi co vi pham: %d/24" % l0_bad)
for k, v in l0_msgs.most_common(): print("     %-24s %d" % (k, v))
