import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def run(cmd: list[str], cwd: Optional[Path] = None, inp: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=inp,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def ensure_git() -> None:
    p = run(["git", "--version"])
    if p.returncode != 0:
        raise RuntimeError("git not found in PATH")


def clone_repo(owner: str, repo: str, cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / f"{owner}__{repo}"
    if (repo_dir / ".git").exists():
        return repo_dir

    url = f"https://github.com/{owner}/{repo}.git"
    p = run(["git", "clone", "--quiet", url, str(repo_dir)])
    if p.returncode != 0:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return None

    return repo_dir


def remove_repo(owner: str, repo: str, cache_dir: Path) -> None:
    repo_dir = cache_dir / f"{owner}__{repo}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)


def parent_sha(repo_dir: Path, sha: str) -> Optional[str]:
    p = run(["git", "-C", str(repo_dir), "rev-list", "--parents", "-n", "1", sha])
    if p.returncode != 0:
        return None
    parts = p.stdout.strip().split()
    return parts[1] if len(parts) >= 2 else None


def changed_files(repo_dir: Path, sha: str) -> List[Dict[str, Any]]:
    p = run(["git", "-C", str(repo_dir), "show", "--name-status", "--format=", sha])
    if p.returncode != 0:
        return []

    out: List[Dict[str, Any]] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R") and len(parts) >= 3:
            out.append({"status": "renamed", "path": parts[2], "prev": parts[1]})
            continue

        if len(parts) >= 2:
            mapped = {"M": "modified", "A": "added", "D": "removed"}.get(status, "other")
            out.append({"status": mapped, "path": parts[1], "prev": None})

    return out


def file_text(repo_dir: Path, ref: str, path: str) -> str:
    p = run(["git", "-C", str(repo_dir), "show", f"{ref}:{path}"])
    return p.stdout if p.returncode == 0 else ""


def file_patch(repo_dir: Path, parent: str, sha: str, path: str) -> str:
    p = run(["git", "-C", str(repo_dir), "diff", parent, sha, "--", path])
    return p.stdout if p.returncode == 0 else ""
