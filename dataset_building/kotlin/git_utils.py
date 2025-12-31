
import subprocess
from pathlib import Path
from typing import List, Optional

def sh(cmd: List[str], cwd: Optional[Path] = None, ok: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if ok and p.returncode != 0:
        raise RuntimeError(
            f"cmd: {' '.join(cmd)}\n"
            f"out:\n{p.stdout}\n"
            f"err:\n{p.stderr}"
        )
    return p

def git(args: List[str], cwd: Path) -> str:
    p = sh(["git"] + args, cwd=cwd, ok=False)
    return p.stdout if p.returncode == 0 else ""

def git_ok() -> None:
    sh(["git", "--version"], ok=True)

def longpaths() -> None:
    try:
        sh(["git", "config", "--global", "core.longpaths", "true"], ok=False)
    except Exception:
        pass
