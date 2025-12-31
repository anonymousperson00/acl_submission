# mine.py
import csv
import json
from pathlib import Path

from config import (
    repos_dir, commits_csv, files_csv, patches_jsonl,
    since, no_merges, keep_patch, patch_max, sec_re
)
from git_utils import sh, git_ok, longpaths

commits_cols = [
    "repo", "sha", "parents", "author_name", "author_email", "date",
    "message", "files_changed", "insertions", "deletions", "flag_security"
]
files_cols = ["repo", "sha", "file", "ext", "added", "deleted"]

def load_seen(path: Path) -> set:
    out = set()
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.add(r["sha"])
    return out

def iter_git_log(repo: Path) -> str:
    rs = "\x1e"
    us = "\x1f"
    fmt = f"__c__{rs}%H{us}%P{us}%an{us}%ae{us}%ad{us}%B"

    cmd = ["git", "log", "--all", "--date=iso-strict", f"--pretty=format:{fmt}", "--numstat"]
    if no_merges:
        cmd.insert(2, "--no-merges")
    if since:
        cmd.insert(2, f"--since={since}")

    return (sh(cmd, cwd=repo, ok=False).stdout or "")

def mine_repo(repo: Path, seen: set, w_commits: csv.DictWriter, w_files: csv.DictWriter) -> None:
    text = iter_git_log(repo)
    if not text.strip():
        return

    cur = None
    cur_files = []

    def flush():
        nonlocal cur, cur_files
        if not cur:
            return

        sha = cur["sha"]
        if sha in seen:
            cur, cur_files = None, []
            return

        insertions = sum(int(a) if a.isdigit() else 0 for a, _, _ in cur_files)
        deletions = sum(int(d) if d.isdigit() else 0 for _, d, _ in cur_files)
        message = cur["message"]

        flag = bool(sec_re.search(message)) or any(sec_re.search(p) for _, _, p in cur_files)

        w_commits.writerow({
            "repo": repo.name,
            "sha": sha,
            "parents": cur["parents"],
            "author_name": cur["author_name"],
            "author_email": cur["author_email"],
            "date": cur["date"],
            "message": message.replace("\r", " ").replace("\n", " ").strip(),
            "files_changed": len(cur_files),
            "insertions": insertions,
            "deletions": deletions,
            "flag_security": int(flag),
        })

        for a, d, p in cur_files:
            w_files.writerow({
                "repo": repo.name,
                "sha": sha,
                "file": p,
                "ext": Path(p).suffix.lower(),
                "added": int(a) if a.isdigit() else 0,
                "deleted": int(d) if d.isdigit() else 0,
            })

        if keep_patch and flag:
            patch = sh(
                ["git", "show", "--no-color", "--no-prefix", "-U0", "--format=", sha, "--", "."],
                cwd=repo, ok=False
            ).stdout or ""
            if len(patch) > patch_max:
                patch = patch[:patch_max] + "\n... [truncated] ..."

            with patches_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "repo": repo.name,
                    "sha": sha,
                    "date": cur["date"],
                    "message": cur["message"],
                    "files": [p for _, _, p in cur_files],
                    "patch_u0": patch,
                }, ensure_ascii=False) + "\n")

        seen.add(sha)
        cur, cur_files = None, []

    head = "__c__\x1e"

    for line in text.splitlines():
        if line.startswith(head):
            flush()
            payload = line[len(head):]
            parts = payload.split("\x1f")
            cur = {
                "sha": parts[0],
                "parents": parts[1] if len(parts) > 1 else "",
                "author_name": parts[2] if len(parts) > 2 else "",
                "author_email": parts[3] if len(parts) > 3 else "",
                "date": parts[4] if len(parts) > 4 else "",
                "message": parts[5] if len(parts) > 5 else "",
            }
            cur_files = []
            continue

        if "\t" in line:
            try:
                a, d, p = line.split("\t", 2)
                cur_files.append((a, d, p))
            except ValueError:
                pass

    flush()

def main() -> None:
    longpaths()
    git_ok()

    seen = load_seen(commits_csv)

    new_commits = not commits_csv.exists()
    new_files = not files_csv.exists()

    with commits_csv.open("a", newline="", encoding="utf-8") as f1, \
         files_csv.open("a", newline="", encoding="utf-8") as f2:

        w_commits = csv.DictWriter(f1, fieldnames=commits_cols)
        w_files = csv.DictWriter(f2, fieldnames=files_cols)

        if new_commits:
            w_commits.writeheader()
        if new_files:
            w_files.writeheader()

        for d in sorted(repos_dir.iterdir(), key=lambda p: p.name.lower()):
            if not d.is_dir():
                continue
            if not (d / ".git").exists():
                continue
            try:
                mine_repo(d, seen, w_commits, w_files)
                f1.flush()
                f2.flush()
            except Exception as e:
                print(f"[mine] {d.name} error: {e}")

if __name__ == "__main__":
    main()
