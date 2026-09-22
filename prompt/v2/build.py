"""Nối các module trong prompt/v2 thành một file system prompt hoàn chỉnh.

Sửa nội dung ở MODULE NGUỒN (0X_*.txt), không sửa trực tiếp file trong compiled/ —
file đó bị ghi đè mỗi lần chạy lại script này.

Dùng:
    python build.py                      # module 01-08 (không có partial-input)
    python build.py --with-partial-input # module 01-09 (có partial-input)
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "compiled"

CORE_MODULES = [
    "01_role_and_task.txt",
    "02_label_definitions.txt",
    "03_boundary_and_bio.txt",
    "04_ambiguity_and_fallback.txt",
    "05_decision_process.txt",
    "06_io_format.txt",
    "07_output_checklist.txt",
    "08_examples_core.txt",
]
PARTIAL_INPUT_MODULE = "09_examples_partial_input.txt"


def build(with_partial_input: bool) -> str:
    modules = list(CORE_MODULES)
    if with_partial_input:
        modules.append(PARTIAL_INPUT_MODULE)

    parts = []
    for name in modules:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Thiếu module: {name}")
        parts.append(path.read_text(encoding="utf-8").strip())

    return "\n\n---\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-partial-input", action="store_true",
                         help="Gộp thêm module 09 (ví dụ input cắt dở/autocomplete)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compiled = build(args.with_partial_input)

    out_name = "system_prompt_v2_with_partial_input.txt" if args.with_partial_input else "system_prompt_v2.txt"
    out_path = OUT_DIR / out_name
    out_path.write_text(compiled, encoding="utf-8")

    n_modules = len(CORE_MODULES) + (1 if args.with_partial_input else 0)
    print(f"Đã nối {n_modules} module -> {out_path} ({len(compiled)} ký tự)")


if __name__ == "__main__":
    main()
