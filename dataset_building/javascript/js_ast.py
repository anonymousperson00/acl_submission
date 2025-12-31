import re
from pathlib import Path
from typing import Optional, Tuple

from git_ops import run


def is_js_like(path: str) -> bool:
    p = (path or "").lower()
    return any(p.endswith(x) for x in [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"])


def is_sfc(path: str) -> bool:
    p = (path or "").lower()
    return p.endswith(".svelte") or p.endswith(".vue")


def should_skip(path: str, dir_keywords: list[str], file_suffixes: list[str]) -> bool:
    p = (path or "").lower()
    if any(k in p for k in dir_keywords):
        return True
    if any(p.endswith(s) for s in file_suffixes):
        return True
    return False


def extract_script_block(code: str) -> str:
    if not code:
        return ""
    m = re.search(r"<script[^>]*>(.*?)</script>", code, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def parse_hunks(patch: str) -> Optional[Tuple[int, int, int, int]]:
    if not patch:
        return None
    hunk_re = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")
    old_s, old_e, new_s, new_e = [], [], [], []
    for line in patch.splitlines():
        if not line.startswith("@@"):
            continue
        m = hunk_re.match(line)
        if not m:
            continue
        o1 = int(m.group(1))
        ol = int(m.group(2) or "1")
        n1 = int(m.group(3))
        nl = int(m.group(4) or "1")
        old_s.append(o1)
        old_e.append(o1 + ol - 1)
        new_s.append(n1)
        new_e.append(n1 + nl - 1)
    if not old_s:
        return None
    return min(old_s), max(old_e), min(new_s), max(new_e)


def snippet(code: str, start: int, end: int, ctx: int) -> str:
    if not code:
        return ""
    lines = code.splitlines()
    n = len(lines)
    s = max(1, start - ctx)
    e = min(n, end + ctx)
    return "\n".join(lines[s - 1 : e])


def is_ast_valid(code: str, node_bin: str, parser_script: Path) -> bool:
    if not isinstance(code, str) or not code.strip():
        return False
    if not parser_script.exists():
        raise FileNotFoundError(f"missing parser script: {parser_script}")
    p = run([node_bin, str(parser_script)], inp=code)
    return p.returncode == 0
