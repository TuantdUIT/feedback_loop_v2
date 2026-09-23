"""Measure one model per process and save its token predictions.

Run this script separately for each --variant. This separation is needed for
meaningful peak memory measurements on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".hf-cache")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from optimum.onnxruntime import ORTModelForTokenClassification  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402


DIRS = {
    "pytorch": "phobert-ner-address-pytorch",
    "onnx": "phobert-ner-address-onnx",
    "int8": "phobert-ner-address-int8",
    "int8_perchannel": "phobert-ner-address-int8-perchannel",
}
SAMPLE = (
    "32 -34 đường Nguyễn Văn Linh, PHƯỜNG PHÚC ĐỒNG, "
    "QUẬN LONG BIÊN, THÀNH PHỐ HÀ NỘI"
)


def input_texts() -> list[tuple[str, str]]:
    full = (ROOT / "clean_data" / "VAER_test_fix.txt").read_text(
        encoding="utf-8"
    ).splitlines()[:500]
    partial = (
        ROOT / "clean_data" / "VAER_train_partial_input_50.txt"
    ).read_text(encoding="utf-8").splitlines()
    if len(full) != 500 or len(partial) != 48:
        raise ValueError(f"Expected 500 full and 48 partial: {len(full)}, {len(partial)}")
    return [("full", text) for text in full] + [
        ("partial", text) for text in partial
    ]


def directory_mb(directory: Path) -> float:
    return round(
        sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        / 1_000_000,
        3,
    )


def predict(model, tokenizer, text: str) -> dict:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        max_length=128,
    )
    special_mask = encoded.pop("special_tokens_mask")[0].tolist()
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    else:
        logits = np.asarray(logits)
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    ids = encoded["input_ids"][0].tolist()
    chosen = probs.argmax(axis=-1)
    return {
        "text": text,
        "tokens": tokenizer.convert_ids_to_tokens(ids),
        "special_mask": special_mask,
        "label_ids": chosen.tolist(),
        "chosen_probs": probs[np.arange(len(chosen)), chosen].tolist(),
        "probs": probs.tolist(),
        "truncated": len(tokenizer(text, add_special_tokens=True)["input_ids"])
        > len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=DIRS, required=True)
    parser.add_argument("--disable-cpu-arena", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "models" / DIRS[args.variant]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    if args.variant == "pytorch":
        model = AutoModelForTokenClassification.from_pretrained(directory).eval()
    else:
        session_options = None
        if args.disable_cpu_arena:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.enable_cpu_mem_arena = False
        model = ORTModelForTokenClassification.from_pretrained(
            directory,
            file_name=(
                "model_quantized.onnx"
                if args.variant in {"int8", "int8_perchannel"}
                else "model.onnx"
            ),
            session_options=session_options,
        )

    process = psutil.Process()
    after_load_mb = process.memory_info().rss / 1_000_000
    inputs = input_texts()
    start = time.perf_counter()
    results = [
        {"group": group, **predict(model, tokenizer, text)}
        for group, text in inputs
    ]
    elapsed = time.perf_counter() - start
    memory = process.memory_info()
    peak_mb = getattr(memory, "peak_wset", memory.rss) / 1_000_000

    payload = {
        "variant": args.variant,
        "disable_cpu_arena": args.disable_cpu_arena,
        "after_load_mb": round(after_load_mb, 3),
        "peak_mb": round(peak_mb, 3),
        "disk_mb": directory_mb(directory),
        "elapsed_548_s": round(elapsed, 3),
        "results": results,
        "sample": predict(model, tokenizer, SAMPLE),
    }
    if args.variant == "pytorch":
        records = json.loads((ROOT / "output_model.json").read_text(encoding="utf-8"))
        payload["output_model"] = [
            predict(model, tokenizer, record["text"]) for record in records
        ]
    output = ROOT / "models" / "measurements"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_noarena" if args.disable_cpu_arena else ""
    path = output / f"{args.variant}{suffix}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"results", "sample", "output_model"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
