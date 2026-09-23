"""Export the address NER model and make two AVX2 dynamic INT8 variants.

Run with .venv-layer1/Scripts/python.exe. All downloads stay under .hf-cache.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".hf-cache")

from optimum.onnxruntime import (  # noqa: E402
    ORTModelForTokenClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import AutoQuantizationConfig  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


MODEL_ID = "kiendt/phobert-ner-address"
SAMPLE = (
    "32 -34 đường Nguyễn Văn Linh, PHƯỜNG PHÚC ĐỒNG, "
    "QUẬN LONG BIÊN, THÀNH PHỐ HÀ NỘI"
)
MODELS = ROOT / "models"


def save_metadata(tokenizer, config) -> None:
    try:
        encoding = tokenizer(SAMPLE, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        offset_error = None
    except Exception as exc:
        encoding = tokenizer(SAMPLE)
        offsets = None
        offset_error = f"{type(exc).__name__}: {exc}"

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    rows = []
    for i, token in enumerate(tokens):
        start, end = offsets[i] if offsets is not None else (None, None)
        rows.append(
            {
                "token": token,
                "start": start,
                "end": end,
                "slice": SAMPLE[start:end] if start is not None else None,
            }
        )

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": getattr(config, "_commit_hash", None),
        "versions": {
            name: version(name)
            for name in ("optimum", "transformers", "onnxruntime", "torch")
        },
        "id2label": config.id2label,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": tokenizer.is_fast,
        "offset_error": offset_error,
        "sample": SAMPLE,
        "sample_tokens": rows,
    }
    (MODELS / "survey.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    save_metadata(tokenizer, config)

    pytorch_dir = MODELS / "phobert-ner-address-pytorch"
    if not (pytorch_dir / "model.safetensors").exists():
        model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        model.save_pretrained(pytorch_dir, safe_serialization=True)
        tokenizer.save_pretrained(pytorch_dir)
        del model

    onnx_dir = MODELS / "phobert-ner-address-onnx"
    if not (onnx_dir / "model.onnx").exists():
        ort_model = ORTModelForTokenClassification.from_pretrained(
            pytorch_dir, export=True
        )
        ort_model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
        del ort_model

    for per_channel, name in (
        (False, "phobert-ner-address-int8"),
        (True, "phobert-ner-address-int8-perchannel"),
    ):
        target = MODELS / name
        if list(target.glob("*.onnx")):
            continue
        quantizer = ORTQuantizer.from_pretrained(onnx_dir)
        qconfig = AutoQuantizationConfig.avx2(
            is_static=False, per_channel=per_channel
        )
        quantizer.quantize(save_dir=target, quantization_config=qconfig)
        config.save_pretrained(target)
        tokenizer.save_pretrained(target)
        del quantizer
        print(f"Created {target}", flush=True)


if __name__ == "__main__":
    main()
