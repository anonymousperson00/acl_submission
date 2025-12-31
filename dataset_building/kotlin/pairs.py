# pairs.py
import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

from config import (
    repos_dir, commits_csv, files_csv, pairs_csv, pairs_jsonl,
    only_exts, max_file_lines, max_file_touch, only_flagged,
    cve_re, cls_re
)
from git_utils import git

pairs_cols = [
    "repo", "sha", "file", "ext", "author_name", "author_email", "date", "message",
    "flag_security", "cve_ids", "total_insertions", "total_deletions",
    "file_added", "file_deleted", "changed_before_lines", "changed_after_lines",
    "class_name", "before_s", "before_e", "after_s", "after_e", "before_code", "after_code"
]

def to_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def load_meta():
    commits = {}
    files_by_commit = defaultdict(list)

    with commits_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            commits[(r["repo"], r["sha"])] = r

    with files_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            files_by_commit[(r["repo"], r["sha"])].append(r)

    return commits, files_by_commit

def hunk_lines(patch: str):
    before, after = set(), set()
    h = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    old_ln = new_ln = None

    for line in (patch or "").splitlines():
        m = h.match(line)
        if m:
            old_ln, new_ln = int(m.group(1)), int(m.group(3))
            continue
        if old_ln is None or new_ln is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            after.add(new_ln)
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            before.add(old_ln)
            old_ln += 1
        else:
            old_ln += 1
            new_ln += 1

    return before, after

def class_span(text: str, line_no: int):
    if not text or not line_no:
        return None

    lines = text.splitlines()
    if line_no < 1 or line_no > len(lines):
        return None

    idx = line_no - 1
    start = None
    header = None

    for i in range(idx, -1, -1):
        m = cls_re.search(lines[i])
        if m:
            start = i
            header = m
            break

    if start is None:
        return None

    depth = 0
    opened = False
    end = len(lines) - 1

    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if "{" in lines[i]:
            opened = True
        if opened and depth <= 0:
            end = i
            break

    return {
        "name": header.group("name") if header else None,
        "s": start + 1,
        "e": end + 1,
        "code": "\n".join(lines[start:end + 1]),
    }

def cves(*texts):
    out = set()
    for t in texts:
        if not t:
            continue
        for m in cve_re.finditer(t):
            v = m.group(0).upper().replace(" ", "")
            if not v.startswith("CVE-"):
                v = v.replace("CVE", "CVE-", 1)
            out.add(v)
    return sorted(out)

def main():
    commits, files_by_commit = load_meta()

    file_freq = Counter()
    for (repo, _sha), rows in files_by_commit.items():
        for r in rows:
            p = r["file"]
            if Path(p).suffix.lower() in only_exts:
                file_freq[(repo, p)] += 1

    with pairs_csv.open("w", newline="", encoding="utf-8-sig") as fcsv, \
         pairs_jsonl.open("w", encoding="utf-8") as fjson:

        w = csv.DictWriter(fcsv, fieldnames=pairs_cols, quoting=csv.QUOTE_ALL)
        w.writeheader()

        for (repo, sha), meta in sorted(commits.items(), key=lambda x: (x[0][0], x[0][1])):
            if only_flagged and str(meta.get("flag_security", "0")) not in {"1", "true", "True"}:
                continue

            repo_dir = repos_dir / repo
            if not (repo_dir / ".git").exists():
                continue

            rows = files_by_commit.get((repo, sha), [])
            if not rows:
                continue

            for fr in rows:
                path = fr["file"]
                ext = Path(path).suffix.lower()
                if ext not in only_exts:
                    continue

                if file_freq[(repo, path)] > max_file_touch:
                    continue

                added = to_int(fr.get("added", 0))
                deleted = to_int(fr.get("deleted", 0))
                if added + deleted == 0 or added + deleted > max_file_lines:
                    continue

                before_text = git(["show", f"{sha}^:{path}"], cwd=repo_dir)
                after_text = git(["show", f"{sha}:{path}"], cwd=repo_dir)
                patch_text = git(["show", "--no-color", "-U0", sha, "--", path], cwd=repo_dir)

                changed_before, changed_after = hunk_lines(patch_text)
                rep_before = min(changed_before) if changed_before else None
                rep_after = min(changed_after) if changed_after else None

                before_cls = class_span(before_text, rep_before) if rep_before else None
                after_cls = class_span(after_text, rep_after) if rep_after else None

                if not before_cls and before_text:
                    before_cls = {"name": None, "s": 1, "e": len(before_text.splitlines()), "code": before_text}
                if not after_cls and after_text:
                    after_cls = {"name": None, "s": 1, "e": len(after_text.splitlines()), "code": after_text}

                if not before_cls and not after_cls:
                    continue

                cve_ids = cves(meta.get("message", ""), patch_text)
                class_name = (after_cls or {}).get("name") or (before_cls or {}).get("name") or ""

                row = {
                    "repo": repo,
                    "sha": sha,
                    "file": path,
                    "ext": ext,
                    "author_name": meta.get("author_name", ""),
                    "author_email": meta.get("author_email", ""),
                    "date": meta.get("date", ""),
                    "message": meta.get("message", "").replace("\r", " ").replace("\n", " "),
                    "flag_security": meta.get("flag_security", ""),
                    "cve_ids": ";".join(cve_ids),
                    "total_insertions": meta.get("insertions", ""),
                    "total_deletions": meta.get("deletions", ""),
                    "file_added": added,
                    "file_deleted": deleted,
                    "changed_before_lines": ",".join(map(str, sorted(changed_before))),
                    "changed_after_lines": ",".join(map(str, sorted(changed_after))),
                    "class_name": class_name,
                    "before_s": (before_cls or {}).get("s", ""),
                    "before_e": (before_cls or {}).get("e", ""),
                    "after_s": (after_cls or {}).get("s", ""),
                    "after_e": (after_cls or {}).get("e", ""),
                    "before_code": (before_cls or {}).get("code", ""),
                    "after_code": (after_cls or {}).get("code", ""),
                }

                w.writerow(row)
                fjson.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
