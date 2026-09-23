"""Chay DeepSeek qua API chinh hang tren mot file text (moi dong mot dia chi).
Ghi tung dong vao .jsonl, chay lai duoc tu cho dang do. Dinh dang dong output
giong het scripts/external_api/qwen/run_qwen_on_file.py de so sanh truc tiep.

Dung: python scripts/external_api/deepseek/run_deepseek_on_file.py <input.txt> <output.jsonl> [workers]

Bien .env:
  DEEPSEEK_API_KEY, DEEPSEEK_MODEL     bat buoc
  DEEPSEEK_BASE_URL                    tuy chon, mac dinh https://api.deepseek.com
  DEEPSEEK_PRICE_PER_1M                tuy chon, "vao_miss,vao_hit,ra" USD/1M token;
                                       API DeepSeek khong tra chi phi, co bien nay
                                       thi script tu tinh, khong co thi chi dem token

Khac voi nhanh Qwen/OpenRouter:
  - Khong co reasoning.max_tokens. Voi model suy luan, max_tokens bao ca phan
    suy luan lan cau tra loi, nen chi giu max_tokens = 12000 lam chot chan.
  - Ket lap -> content rong, finish_reason = "length". Van retry khi RONG du da
    ton phi (bien phap 4 ben Qwen), toi da MAX_ATTEMPTS lan.
  - System prompt giong nhau o moi request nen DeepSeek tu cache phan tien to;
    token cache hit re hon, duoc tach rieng trong usage.
"""
import json, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

MAX_TOKENS = 12000
MAX_ATTEMPTS = 3

env = {}
for ln in Path(".env").read_text(encoding="utf-8").splitlines():
    ln = ln.strip()
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"').strip("'")
KEY, MODEL = env.get("DEEPSEEK_API_KEY"), env.get("DEEPSEEK_MODEL")
if not KEY or not MODEL:
    sys.exit("Thieu DEEPSEEK_API_KEY hoac DEEPSEEK_MODEL trong .env")
URL = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
PRICE = [float(x) for x in env["DEEPSEEK_PRICE_PER_1M"].split(",")] if env.get("DEEPSEEK_PRICE_PER_1M") else None
SYS = Path("prompt/v2/compiled/system_prompt_v2_with_partial_input.txt").read_text(encoding="utf-8")

# Tu tat neu model tu choi tham so nay.
USE_RESPONSE_FORMAT = [True]


def extract(txt):
    """Doc JSON tu noi dung tra ve, giu cac lop du phong khi khong dung
    duoc response_format."""
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


def cost_of(usage):
    if not PRICE:
        return 0.0
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", (usage.get("prompt_tokens", 0) or 0) - hit) or 0
    out = usage.get("completion_tokens", 0) or 0
    return (miss * PRICE[0] + hit * PRICE[1] + out * PRICE[2]) / 1e6


def build_body(text):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": json.dumps({"text": text}, ensure_ascii=False)},
        ],
    }
    if USE_RESPONSE_FORMAT[0]:
        body["response_format"] = {"type": "json_object"}
    return body


def post(body):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def call(text):
    """Tra ve (content, usage_tong, giay, attempts).

    Retry khi: (a) loi HTTP/mang, hoac (b) tra ve RONG/khong parse duoc JSON
    du da ton phi."""
    total_cost = 0.0
    tokens = {"prompt_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
              "completion_tokens": 0, "reasoning_tokens": 0}
    attempts = []
    t_start = time.time()

    def summary(**extra):
        return dict(tokens, cost=total_cost, **extra)

    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        try:
            # Gia DeepSeek doi theo khung gio cao/thap diem (UTC) -> ghi lai de tinh lai sau.
            sent_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            r = post(build_body(text))
            usage = r.get("usage", {}) or {}
            cost = cost_of(usage)
            total_cost += cost
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            for k in tokens:
                tokens[k] += reasoning if k == "reasoning_tokens" else (usage.get(k, 0) or 0)
            choice = (r.get("choices") or [{}])[0]
            content = choice.get("message", {}).get("content")
            parsed = extract(content)

            attempts.append({
                "attempt": attempt, "utc": sent_utc, "cost": cost,
                "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "empty": parsed is None,
            })

            if parsed is not None:
                return content, summary(retries=attempt), time.time() - t_start, attempts
            if not last:
                time.sleep(2)

        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            if e.code == 400 and USE_RESPONSE_FORMAT[0] and "response_format" in detail:
                USE_RESPONSE_FORMAT[0] = False
                attempts.append({"attempt": attempt, "note": "tat response_format", "detail": detail})
                continue
            attempts.append({"attempt": attempt, "http_error": e.code, "detail": detail})
            # 401/402/422: sai key, het tien, sai tham so -> retry cung vo ich.
            if last or e.code in (401, 402, 422):
                return None, summary(error="HTTP %s: %s" % (e.code, detail[:150]), retries=attempt), \
                    time.time() - t_start, attempts
            time.sleep((10 if e.code in (429, 503) else 4) * (attempt + 1))

        except Exception as e:
            attempts.append({"attempt": attempt, "error": str(e)[:200]})
            if last:
                return None, summary(error=str(e)[:200], retries=attempt), time.time() - t_start, attempts
            time.sleep(4 * (attempt + 1))

    return None, summary(retries=MAX_ATTEMPTS - 1), time.time() - t_start, attempts


lines = [l for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]
done = set()
if OUT.exists():
    done = {json.loads(l)["i"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
todo = [i for i in range(len(lines)) if i not in done]
print("tong %d dong | da co %d | con %d" % (len(lines), len(done), len(todo)), flush=True)
print("model=%s | max_tokens=%d | toi da %d lan thu | gia=%s"
      % (MODEL, MAX_TOKENS, MAX_ATTEMPTS, PRICE or "khong dat, chi dem token"), flush=True)


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
        print("  %3d/%d  #%-3d %6d tok $%.5f %6.1fs %s%s" % (
            n, len(todo), d["i"], d["usage"].get("completion_tokens", 0), d["usage"].get("cost", 0), d["sec"],
            "OK" if d["parsed"] else "LOI",
            "" if not retries else "  (retry x%d)" % retries), flush=True)

all_ = sorted([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()],
              key=lambda d: d["i"])
if not all_:
    sys.exit("Khong co ban ghi nao")
cost = sum(o["usage"].get("cost", 0) for o in all_)
n_retry = sum(1 for o in all_ if o["usage"].get("retries"))
tok = {k: sum(o["usage"].get(k, 0) for o in all_)
       for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens", "reasoning_tokens")}
print("\n=== %d ban ghi | parse OK %d | phai retry %d | tong $%.5f | tb $%.5f/dia chi ===" %
      (len(all_), sum(1 for o in all_ if o["parsed"]), n_retry, cost, cost / len(all_)))
print("    token vao: cache hit %d, miss %d | ra %d (suy luan %d)" % (
    tok["prompt_cache_hit_tokens"], tok["prompt_cache_miss_tokens"],
    tok["completion_tokens"], tok["reasoning_tokens"]))
