"""Test build, cache, sha và metadata của prompt lõi."""

import hashlib
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from feedback.core import prompt_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PROMPT_DIR = PROJECT_ROOT / "prompt" / "v2"


@pytest.fixture()
def isolated_prompt_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sao chép nguồn prompt sang tmp_path để test không chạm compiled thật."""
    prompt_dir = tmp_path / "v2"
    prompt_dir.mkdir()
    shutil.copy2(REAL_PROMPT_DIR / "build.py", prompt_dir / "build.py")
    for source in REAL_PROMPT_DIR.glob("0*_*.txt"):
        shutil.copy2(source, prompt_dir / source.name)

    compiled_dir = prompt_dir / "compiled"
    monkeypatch.setattr(prompt_loader, "PROMPT_V2_DIR", prompt_dir)
    monkeypatch.setattr(prompt_loader, "COMPILED_DIR", compiled_dir)
    prompt_loader._CACHE.clear()
    yield prompt_dir
    prompt_loader._CACHE.clear()


def test_returns_content_and_sha(isolated_prompt_tree: Path) -> None:
    content, prompt_sha = prompt_loader.load_prompt()
    assert content
    assert re.fullmatch(r"[0-9a-f]{8}", prompt_sha)


def test_sha_deterministic(isolated_prompt_tree: Path) -> None:
    first = prompt_loader.load_prompt()
    second = prompt_loader.load_prompt()
    assert first[1] == second[1]


def test_sha_matches_content(isolated_prompt_tree: Path) -> None:
    content, prompt_sha = prompt_loader.load_prompt()
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    assert prompt_sha == expected


def test_autobuild_when_missing(isolated_prompt_tree: Path) -> None:
    compiled = prompt_loader.COMPILED_DIR / "system_prompt_v2_with_partial_input.txt"
    assert not compiled.exists()
    content, _ = prompt_loader.load_prompt()
    assert compiled.exists()
    assert content


def test_stale_detection(isolated_prompt_tree: Path) -> None:
    prompt_loader.load_prompt()
    compiled = prompt_loader.COMPILED_DIR / "system_prompt_v2_with_partial_input.txt"
    source = isolated_prompt_tree / "03_boundary_and_bio.txt"
    older = source.stat().st_mtime - 10
    os.utime(compiled, (older, older))
    assert prompt_loader.is_stale("v2_with_partial_input") is True


def test_rebuild_on_stale(isolated_prompt_tree: Path) -> None:
    prompt_loader.load_prompt()
    compiled = prompt_loader.COMPILED_DIR / "system_prompt_v2_with_partial_input.txt"
    source = isolated_prompt_tree / "03_boundary_and_bio.txt"
    older = source.stat().st_mtime - 10
    os.utime(compiled, (older, older))
    old_mtime = compiled.stat().st_mtime

    content, _ = prompt_loader.load_prompt()

    assert content
    assert compiled.stat().st_mtime > old_mtime
    assert prompt_loader.is_stale("v2_with_partial_input") is False


def test_unknown_variant(isolated_prompt_tree: Path) -> None:
    with pytest.raises(ValueError, match="v2_with_partial_input"):
        prompt_loader.load_prompt("khong_ton_tai")


def test_tag_system_is_L(isolated_prompt_tree: Path) -> None:
    content, _ = prompt_loader.load_prompt()
    assert "B-L5" in content
    assert content.count("B-STREET") == 1


def test_run_meta(isolated_prompt_tree: Path) -> None:
    meta = prompt_loader.build_run_meta()
    assert re.fullmatch(r"[0-9a-f]{8}", meta.prompt_sha)
    assert meta.prompt_variant == "v2_with_partial_input"
    assert meta.git_commit
    assert datetime.fromisoformat(meta.run_at).tzinfo is not None


def _touch_after(target: Path, reference: Path) -> None:
    """Đặt mtime của target muộn hơn reference.

    Mốc phải tính từ file compiled: fixture dùng shutil.copy2 nên module nguồn
    giữ nguyên mtime gốc trong repo, vốn đã cũ hơn file vừa build.
    """
    later = reference.stat().st_mtime + 60
    os.utime(target, (later, later))


def test_stale_ignores_manifest(isolated_prompt_tree: Path) -> None:
    """00_manifest.txt không nằm trong bản build nên không làm compiled stale."""
    prompt_loader.load_prompt()
    compiled = prompt_loader.COMPILED_DIR / "system_prompt_v2_with_partial_input.txt"
    _touch_after(isolated_prompt_tree / "00_manifest.txt", compiled)
    assert prompt_loader.is_stale("v2_with_partial_input") is False


def test_stale_ignores_module_outside_variant(isolated_prompt_tree: Path) -> None:
    """Module 09 chỉ thuộc variant partial-input, không ảnh hưởng variant v2."""
    prompt_loader.load_prompt("v2")
    prompt_loader.load_prompt("v2_with_partial_input")
    compiled_core = prompt_loader.COMPILED_DIR / "system_prompt_v2.txt"
    compiled_partial = prompt_loader.COMPILED_DIR / "system_prompt_v2_with_partial_input.txt"
    newest = max(compiled_core, compiled_partial, key=lambda f: f.stat().st_mtime)
    _touch_after(isolated_prompt_tree / "09_examples_partial_input.txt", newest)

    assert prompt_loader.is_stale("v2") is False
    assert prompt_loader.is_stale("v2_with_partial_input") is True


def test_stale_still_detects_own_module(isolated_prompt_tree: Path) -> None:
    """Module thuộc variant vẫn phải làm compiled stale sau khi sửa."""
    prompt_loader.load_prompt("v2")
    compiled = prompt_loader.COMPILED_DIR / "system_prompt_v2.txt"
    _touch_after(isolated_prompt_tree / "03_boundary_and_bio.txt", compiled)
    assert prompt_loader.is_stale("v2") is True
