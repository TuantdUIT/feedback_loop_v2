"""Chay qwen qua OpenRouter tren output_model.json. Ghi tung dong, chay lai duoc tu cho dang do."""
import json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

env = {}
for ln in Path(".env").read_text(encoding="utf-8").splitlines():
    ln = ln.strip()
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"').strip("'")
KEY, MODEL = env["OPENROUTER_API"], env["OPENROUTER_MODEL"]
SYS = Path("prompt/v2/compiled/system_prompt_v2_with_partial_input.txt").read_text(encoding="utf-8")
URL = "https://openrouter.ai/api/v1/chat/completions"

def call(payload):
    body = {"model": MODEL, "temperature": 0, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return r["choices"][0]["message"]["content"], r.get("usage", {}), time.time() - t0
        except Exception as e:
            if attempt == 2:
                return None, {"error": str(e)[:200]}, 0
            time.sleep(3 * (attempt + 1))

def extract(txt):
    if txt is None:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    raw = (m.group(1) if m else txt).strip()
    try:
        return json.loads(raw)
    except Exception:
        m2 = re.search(r"[\{\[].*[\}\]]", raw, re.S)
        if not m2:
            return None
        try:
            return json.loads(m2.group(0))
        except Exception:
            return None

recs = json.load(open("output_model.json", encoding="utf-8"))
JL = Path("models/qwen_single.jsonl")

def one(i):
    txt, usage, dt = call({"text": recs[i]["text"]})
    return {"i": i, "text": recs[i]["text"], "raw": txt,
            "parsed": extract(txt), "usage": usage, "sec": round(dt, 2)}

done = set()
if JL.exists():
    done = {json.loads(l)["i"] for l in JL.read_text(encoding="utf-8").splitlines() if l.strip()}
todo = [i for i in range(len(recs)) if i not in done]
print("da co %d, con lai %d" % (len(done), len(todo)), flush=True)

with ThreadPoolExecutor(max_workers=4) as ex:
    for d in ex.map(one, todo):
        with JL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
        print("  xong #%-3d $%.5f  %.1fs  %s" % (d["i"], d["usage"].get("cost", 0), d["sec"],
              "OK" if d["parsed"] else "PARSE LOI"), flush=True)

out = sorted([json.loads(l) for l in JL.read_text(encoding="utf-8").splitlines() if l.strip()],
             key=lambda d: d["i"])
Path("models/qwen_single.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
cost = sum(o["usage"].get("cost", 0) for o in out)
pt = sum(o["usage"].get("prompt_tokens", 0) for o in out)
ct = sum(o["usage"].get("completion_tokens", 0) for o in out)
print("\n=== TONG KET %d ban ghi ===" % len(out))
print("  parse duoc JSON   : %d/%d" % (sum(1 for o in out if o["parsed"]), len(out)))
print("  tong chi phi      : $%.5f" % cost)
print("  CHI PHI / DIA CHI : $%.5f" % (cost / len(out)))
print("  token vao / ra    : %d / %d  (tb %.0f / %.0f)" % (pt, ct, pt/len(out), ct/len(out)))
