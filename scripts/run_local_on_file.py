"""Chay ONNX FP32 va PyTorch FP32 tren mot file text, luu span da chuan hoa ve L1-L7."""
import json, sys, glob, numpy as np, torch, onnxruntime as ort
from pathlib import Path
sys.path.insert(0, "scripts")
from layer1_adapter import LABELS, normalize
from transformers import AutoTokenizer, AutoModelForTokenClassification

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
lines = [l for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]
tk = AutoTokenizer.from_pretrained("models/phobert-ner-address-onnx")

def spans(pairs): return [{"level": lv, "text": tx} for lv, tx in normalize(pairs)]

def onnx_fp32():
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    path = sorted(glob.glob("models/phobert-ner-address-onnx/*.onnx"))[0]
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    names = {i.name for i in s.get_inputs()}
    out = []
    for t in lines:
        e = tk(t, return_tensors="np", truncation=True, max_length=128)
        ids = s.run(None, {k: v.astype(np.int64) for k, v in e.items() if k in names})[0].argmax(-1)[0].tolist()
        toks = tk.convert_ids_to_tokens(e["input_ids"][0])
        out.append(spans([[x, LABELS[i]] for x, i in zip(toks, ids) if x not in ("<s>", "</s>")]))
    return out

def torch_fp32():
    md = AutoModelForTokenClassification.from_pretrained("models/phobert-ner-address-pytorch").eval()
    out = []
    for t in lines:
        e = tk(t, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): ids = md(**e).logits.argmax(-1)[0].tolist()
        toks = tk.convert_ids_to_tokens(e["input_ids"][0])
        out.append(spans([[x, LABELS[i]] for x, i in zip(toks, ids) if x not in ("<s>", "</s>")]))
    return out

o, p = onnx_fp32(), torch_fp32()
json.dump([{"i": i, "text": lines[i], "onnx_fp32": o[i], "pytorch_fp32": p[i]}
           for i in range(len(lines))], OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
same = sum(1 for a, b in zip(o, p) if a == b)
print("da ghi %s (%d dong) | ONNX FP32 giong PyTorch FP32: %d/%d dong" % (OUT, len(lines), same, len(lines)))
