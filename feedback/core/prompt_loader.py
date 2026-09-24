"""Nạp prompt lõi, tự build khi stale và trả nội dung kèm sha."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from feedback.core.schemas import RunMeta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_V2_DIR = PROJECT_ROOT / "prompt" / "v2"
COMPILED_DIR = PROMPT_V2_DIR / "compiled"

VARIANTS: dict[str, tuple[str, list[str]]] = {
    "v2": ("system_prompt_v2.txt", []),
    "v2_with_partial_input": (
        "system_prompt_v2_with_partial_input.txt",
        ["--with-partial-input"],
    ),
}

_CACHE: dict[str, tuple[str, str]] = {}


def _variant_config(variant: str) -> tuple[str, list[str]]:
    """Lấy cấu hình variant hoặc báo danh sách variant hợp lệ."""
    if variant not in VARIANTS:
        valid = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Variant {variant!r} không tồn tại; hợp lệ: {valid}")
    return VARIANTS[variant]


def _compiled_path(variant: str) -> Path:
    """Trả đường dẫn file compiled của variant."""
    filename, _ = _variant_config(variant)
    return COMPILED_DIR / filename


def _source_modules(variant: str) -> list[Path]:
    """Danh sách module nguồn thực sự tạo nên variant này.

    Đọc trực tiếp CORE_MODULES/PARTIAL_INPUT_MODULE từ build.py thay vì glob
    ``0*_*.txt``: glob sẽ quét nhầm cả 00_manifest.txt (không nằm trong bản
    build) và module 09 (chỉ thuộc variant partial-input), khiến variant bị coi
    là stale sai. Lấy từ build.py thì danh sách không bao giờ lệch.
    """
    _, arguments = _variant_config(variant)
    build_path = PROMPT_V2_DIR / "build.py"
    spec = importlib.util.spec_from_file_location("_prompt_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không nạp được {build_path}")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    names = list(build.CORE_MODULES)
    if "--with-partial-input" in arguments:
        names.append(build.PARTIAL_INPUT_MODULE)
    return [PROMPT_V2_DIR / name for name in names]


def is_stale(variant: str) -> bool:
    """True khi compiled thiếu hoặc cũ hơn một module nguồn của chính variant."""
    compiled_path = _compiled_path(variant)
    if not compiled_path.exists():
        return True
    compiled_mtime = compiled_path.stat().st_mtime
    return any(
        module.stat().st_mtime > compiled_mtime
        for module in _source_modules(variant)
        if module.exists()
    )


def rebuild(variant: str) -> None:
    """Chạy prompt/v2/build.py mà không sửa nội dung module prompt."""
    _, arguments = _variant_config(variant)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [sys.executable, "build.py", *arguments],
        cwd=PROMPT_V2_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    compiled_path = _compiled_path(variant)
    if process.returncode != 0 or not compiled_path.exists() or compiled_path.stat().st_size == 0:
        stderr = process.stderr.strip() or "không có stderr"
        raise RuntimeError(f"Không build được prompt {variant!r}: {stderr}")


def load_prompt(variant: str = "v2_with_partial_input") -> tuple[str, str]:
    """Trả về (nội dung prompt, sha256 8 ký tự đầu).

    sha băm CHUỖI đã đọc, không phải byte trên đĩa, nên sẽ không khớp với
    ``sha256sum`` của file: build.py ghi CRLF trên Windows còn LF trên Linux.
    Băm chuỗi cho cùng một sha trên mọi nền tảng khi nội dung prompt giống
    nhau — đó mới là thứ cần truy vết, vì đó là nội dung gửi cho LLM.
    """
    stale = is_stale(variant)
    if stale:
        _CACHE.pop(variant, None)
        rebuild(variant)
    elif variant in _CACHE:
        return _CACHE[variant]

    content = _compiled_path(variant).read_text(encoding="utf-8")
    if not content:
        raise RuntimeError(f"Prompt compiled của variant {variant!r} đang rỗng")
    prompt_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    result = (content, prompt_sha)
    _CACHE[variant] = result
    return result


TASK_PROMPT_DIR = PROJECT_ROOT / "feedback" / "prompts"


def load_task_prompt(name: str) -> tuple[str, str]:
    """Nạp ``feedback/prompts/<name>.txt``, bỏ khối chú thích ``#`` ở đầu file; trả (nội dung, sha8).

    Khối chú thích là các dòng ``#`` liền nhau từ dòng đầu tới dòng trống đầu tiên — tiêu đề Markdown
    phía sau dòng trống đó được giữ nguyên.
    """
    lines = (TASK_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        index += 1
    content = "\n".join(lines[index:]).strip() + "\n"
    return content, hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def build_run_meta(variant: str = "v2_with_partial_input") -> RunMeta:
    """Tạo metadata truy vết prompt, commit Git và thời điểm chạy."""
    _, prompt_sha = load_prompt(variant)
    git_commit = "unknown"
    try:
        process = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode == 0 and process.stdout.strip():
            git_commit = process.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return RunMeta(
        prompt_sha=prompt_sha,
        prompt_variant=variant,
        git_commit=git_commit,
        run_at=datetime.now(timezone.utc).isoformat(),
    )
