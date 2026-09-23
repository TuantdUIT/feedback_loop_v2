"""Đánh giá PhoBERT PyTorch FP32 và ONNX FP32 trên golden dataset.

Mỗi model chạy lần lượt qua từng file golden (`{"texts": [...], "results": [{"result": {L1..L7}}]}`),
output đi qua `scripts/layer1_adapter.normalize` (6 quy tắc) rồi so với gold.

Chỉ số:
  Accuracy   mẫu khớp hoàn toàn cả 7 level / tổng số mẫu
  Precision  span dự đoán đúng / tổng span dự đoán       (span = cặp (level, text), khớp tuyệt đối)
  Recall     span dự đoán đúng / tổng span gold
  F1         trung bình điều hòa P và R (micro), kèm bảng theo từng level
  Latency    ms/mẫu cho từng node (tokenize, inference, adapter) và tổng; mean / p50 / p95 / max.
             Thời gian nạp model và warm-up được tách riêng, không tính vào latency.

So khớp text: NFC, chữ thường, gộp khoảng trắng, bỏ dấu câu/khoảng trắng ở hai đầu —
vì adapter trả chữ thường còn gold giữ nguyên chữ hoa của input.

Chạy (cần venv có torch + onnxruntime):
  .venv-layer1/Scripts/python.exe scripts/eval_golden.py
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frontend.engines import _get_onnx_session, _get_tokenizer, _get_torch_model  # noqa: E402
from scripts.layer1_adapter import LABELS, normalize  # noqa: E402

GOLDEN_DIR = ROOT / "golden_dataset"
OUT_DIR = GOLDEN_DIR / "evals"
DATASETS = ["golden_full.json", "golden_uncomplete.json"]
MODELS = ["pytorch_fp32", "onnx_fp32"]
LEVELS = [f"L{i}" for i in range(1, 8)]
WARMUP = 5


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.;:-")


def to_counter(spans: list[tuple[str, str]]) -> collections.Counter:
    return collections.Counter((lv, norm_text(tx)) for lv, tx in spans if norm_text(tx))


def gold_spans(result: dict) -> list[tuple[str, str]]:
    return [(lv, tx) for lv in LEVELS for tx in result.get(lv, [])]


# ------------------------------------------------------------------ inference
class Runner:
    def __init__(self, model: str) -> None:
        self.model = model
        started = time.perf_counter()
        self.tok = _get_tokenizer()
        if model == "pytorch_fp32":
            import torch

            self.torch = torch
            self.net = _get_torch_model()
        else:
            self.sess = _get_onnx_session()
            self.input_names = {i.name for i in self.sess.get_inputs()}
        self.load_sec = time.perf_counter() - started

    def parse(self, text: str) -> tuple[list[tuple[str, str]], dict[str, float]]:
        t0 = time.perf_counter()
        if self.model == "pytorch_fp32":
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=128)
            t1 = time.perf_counter()
            with self.torch.inference_mode():
                ids = self.net(**enc).logits.argmax(-1)[0].tolist()
        else:
            enc = self.tok(text, return_tensors="np", truncation=True, max_length=128)
            t1 = time.perf_counter()
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            ids = self.sess.run(None, feeds)[0].argmax(-1)[0].tolist()
        t2 = time.perf_counter()
        tokens = self.tok.convert_ids_to_tokens(list(enc["input_ids"][0]))
        pairs = [[t, LABELS[i]] for t, i in zip(tokens, ids) if t not in ("<s>", "</s>")]
        spans = [(lv, tx) for lv, tx in normalize(pairs)]
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000  # noqa: E731
        return spans, {"tokenize": ms(t0, t1), "inference": ms(t1, t2),
                       "adapter": ms(t2, t3), "total": ms(t0, t3)}


# ------------------------------------------------------------------ metrics
def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def latency_stats(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    return {"mean": statistics.fmean(v), "p50": v[len(v) // 2],
            "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], "max": v[-1]}


def evaluate(rows: list[dict]) -> dict:
    exact = 0
    tp = n_pred = n_gold = 0
    per_level = {lv: [0, 0, 0] for lv in LEVELS}  # tp, pred, gold
    for row in rows:
        g, p = to_counter(row["gold"]), to_counter(row["pred"])
        hit = g & p
        exact += g == p
        tp += sum(hit.values()); n_pred += sum(p.values()); n_gold += sum(g.values())
        for (lv, _), c in hit.items():
            per_level[lv][0] += c
        for (lv, _), c in p.items():
            per_level[lv][1] += c
        for (lv, _), c in g.items():
            per_level[lv][2] += c
    # node lấy theo dữ liệu: PhoBERT có tokenize/inference/adapter/total, LLM qua API chỉ có total
    lat = {node: latency_stats([r["latency_ms"][node] for r in rows])
           for node in rows[0]["latency_ms"]}
    return {
        "n": len(rows),
        "accuracy": exact / len(rows),
        "exact_match": exact,
        "micro": prf(tp, n_pred, n_gold),
        "per_level": {lv: prf(*v) for lv, v in per_level.items() if v[1] or v[2]},
        "latency_ms": lat,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in DATASETS:
        raw = json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))
        data[name] = list(zip(raw["texts"], (r["result"] for r in raw["results"]), strict=True))

    metrics: dict = {"datasets": {n: len(v) for n, v in data.items()}, "models": {}}
    for model in MODELS:
        runner = Runner(model)
        for text, _ in list(data.values())[0][:WARMUP]:
            runner.parse(text)
        metrics["models"][model] = {"load_sec": runner.load_sec, "results": {}}
        print(f"[{model}] nạp model {runner.load_sec:.1f}s")
        for name, samples in data.items():
            rows = []
            for i, (text, gold) in enumerate(samples):
                pred, lat = runner.parse(text)
                rows.append({"i": i, "text": text, "gold": gold_spans(gold), "pred": pred,
                             "latency_ms": lat})
            for r in rows:
                r["exact"] = to_counter(r["gold"]) == to_counter(r["pred"])
            out = OUT_DIR / f"pred_{model}_{Path(name).stem}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            m = evaluate(rows)
            metrics["models"][model]["results"][name] = m
            print(f"  {name}: acc {m['accuracy']:.3f} | P {m['micro']['precision']:.3f} "
                  f"R {m['micro']['recall']:.3f} F1 {m['micro']['f1']:.3f} | "
                  f"latency mean {m['latency_ms']['total']['mean']:.1f} ms")

    # hai bản FP32 có cho cùng output không
    agree = {}
    for name in DATASETS:
        a = [json.loads(l)["pred"] for l in (OUT_DIR / f"pred_pytorch_fp32_{Path(name).stem}.jsonl").open(encoding="utf-8")]
        b = [json.loads(l)["pred"] for l in (OUT_DIR / f"pred_onnx_fp32_{Path(name).stem}.jsonl").open(encoding="utf-8")]
        agree[name] = sum(x == y for x, y in zip(a, b))
    metrics["pytorch_vs_onnx_identical"] = agree

    # chẩn đoán: bỏ mẫu có POI (L7 / "dự án") — model không có nhãn L7 nên đây là lỗi hệ thống
    diag = {}
    for name in DATASETS:
        rows = [json.loads(l) for l in (OUT_DIR / f"pred_onnx_fp32_{Path(name).stem}.jsonl").open(encoding="utf-8")]
        sub = [r for r in rows
               if not any(lv == "L7" for lv, _ in r["gold"]) and "dự án" not in r["text"].lower()]
        m = evaluate(sub)
        diag[name] = {"n": m["n"], "accuracy": m["accuracy"], "exact": m["exact_match"],
                      **{k: m["micro"][k] for k in ("precision", "recall", "f1")}}
    metrics["diagnostic_without_L7_du_an"] = diag
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PyTorch == ONNX: {agree} | đã ghi {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
