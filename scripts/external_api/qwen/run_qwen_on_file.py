"""Chay qwen qua OpenRouter tren mot file text (moi dong mot dia chi).
Ghi tung dong vao .jsonl, chay lai duoc tu cho dang do.

Dung: python scripts/external_api/qwen/run_qwen_on_file.py <input.txt> <output.jsonl> [workers]

Bon bien phap chong ket vong lap suy luan (do tren 100 mau test_v2):
  1. reasoning.max_tokens = 10000  — mau thanh cong reasoning max 7.733,
                                     mau ket lap min 17.021 -> nguong 10k
                                     chan het ca lap ma khong cat ca hop le
  2. max_tokens = 12000            — chot chan cuoi, phong provider khong
                                     ton trong reasoning.max_tokens
  3. response_format json_object   — ep provider tra JSON hop le, bo bot
                                     phu thuoc vao regex extract()
  4. Retry khi RONG du da ton phi  — truoc day coi la "thanh cong" vi khong
                                     co exception HTTP; da gay $0,357 lang phi
"""
import json, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

MAX_TOKENS = 12000
# Thang giam dan qua tung lan thu: lan dau rong rai, ket lap thi that chat.
# p50 mau thanh cong = 2.852, p90 = 4.606, max = 7.733.
REASONING_BUDGET = [10000, 5000, 3000]

env = {}
for ln in Path(".env").read_text(encoding="utf-8").splitlines():
    ln = ln.strip()
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"').strip("'")
KEY, MODEL = env["OPENROUTER_API"], env["OPENROUTER_MODEL"]
SYS = Path("prompt/v2/compiled/system_prompt_v2_with_partial_input.txt").read_text(encoding="utf-8")
URL = "https://openrouter.ai/api/v1/chat/completions"

# Tu tat neu provider tu choi tham so nay (khong phai model nao cung ho tro).
USE_RESPONSE_FORMAT = [True]


def extract(txt):
    """Doc JSON tu noi dung tra ve. Voi response_format=json_object thi txt
    da la JSON thuan, nhung giu lai cac lop du phong cho truong hop provider
    khong ho tro tham so do."""
    if not txt:
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


def build_body(text, reasoning_budget):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,                      # (2)
        "reasoning": {"max_tokens": reasoning_budget},  # (1)
        "usage": {"include": True},
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": json.dumps({"text": text}, ensure_ascii=False)},
        ],
    }
    if USE_RESPONSE_FORMAT[0]:
        body["response_format"] = {"type": "json_object"}   # (3)
    return body


def post(body):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def call(text):
    """Tra ve (content, usage_tong, giay, attempts).

    Retry khi: (a) loi HTTP/mang, hoac (b) tra ve RONG/khong parse duoc JSON
    du da ton phi — truong hop (b) chinh la ket vong lap suy luan, truoc day
    bi coi nham la thanh cong."""
    total_cost = 0.0
    last_usage = {}
    attempts = []
    t_start = time.time()

    for attempt, budget in enumerate(REASONING_BUDGET):
        try:
            r = post(build_body(text, budget))
            usage = r.get("usage", {}) or {}
            cost = usage.get("cost", 0) or 0
            total_cost += cost
            last_usage = usage
            content = (r.get("choices") or [{}])[0].get("message", {}).get("content")
            parsed = extract(content)

            attempts.append({
                "attempt": attempt, "reasoning_budget": budget, "cost": cost,
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "empty": parsed is None,
            })

            if parsed is not None:                                   # (4) thanh cong that
                last_usage = dict(usage, cost=total_cost, retries=attempt)
                return content, last_usage, time.time() - t_start, attempts

            # Rong du da ton phi -> that chat ngan sach suy luan roi thu lai.
            if attempt < len(REASONING_BUDGET) - 1:
                time.sleep(2)

        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            # Provider tu choi response_format -> tat han va thu lai ngay.
            if e.code == 400 and USE_RESPONSE_FORMAT[0] and "response_format" in detail:
                USE_RESPONSE_FORMAT[0] = False
                attempts.append({"attempt": attempt, "note": "tat response_format", "detail": detail})
                continue
            attempts.append({"attempt": attempt, "http_error": e.code, "detail": detail})
            if attempt == len(REASONING_BUDGET) - 1:
                return None, {"error": "HTTP %s: %s" % (e.code, detail[:150]),
                              "cost": total_cost, "retries": attempt}, time.time() - t_start, attempts
            time.sleep(4 * (attempt + 1))

        except Exception as e:
            attempts.append({"attempt": attempt, "error": str(e)[:200]})
            if attempt == len(REASONING_BUDGET) - 1:
                return None, {"error": str(e)[:200], "cost": total_cost,
                              "retries": attempt}, time.time() - t_start, attempts
            time.sleep(4 * (attempt + 1))

    return None, dict(last_usage, cost=total_cost, retries=len(REASONING_BUDGET) - 1), \
        time.time() - t_start, attempts


lines = [l for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]
done = set()
if OUT.exists():
    done = {json.loads(l)["i"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
todo = [i for i in range(len(lines)) if i not in done]
print("tong %d dong | da co %d | con %d" % (len(lines), len(done), len(todo)), flush=True)
print("max_tokens=%d | reasoning budget=%s | response_format=%s"
      % (MAX_TOKENS, REASONING_BUDGET, "json_object"), flush=True)


def one(i):
    txt, usage, dt, attempts = call(lines[i])
    return {"i": i, "text": lines[i], "raw": txt, "parsed": extract(txt),
            "usage": usage, "sec": round(dt, 2), "attempts": attempts}


n = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for d in ex.map(one, todo):
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False)); f.write("\n")
        n += 1
        retries = d["usage"].get("retries", 0)
        print("  %3d/%d  #%-3d $%.5f %6.1fs %s%s" % (
            n, len(todo), d["i"], d["usage"].get("cost", 0), d["sec"],
            "OK" if d["parsed"] else "LOI",
            "" if not retries else "  (retry x%d)" % retries), flush=True)

all_ = sorted([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()],
              key=lambda d: d["i"])
cost = sum(o["usage"].get("cost", 0) for o in all_)
n_retry = sum(1 for o in all_ if o["usage"].get("retries"))
print("\n=== %d ban ghi | parse OK %d | phai retry %d | tong $%.5f | tb $%.5f/dia chi ===" %
      (len(all_), sum(1 for o in all_ if o["parsed"]), n_retry, cost, cost / len(all_)))
