from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from git_ops import (
    ensure_git,
    clone_repo,
    remove_repo,
    parent_sha,
    changed_files,
    file_text,
    file_patch,
)
from js_ast import (
    is_js_like,
    is_sfc,
    should_skip,
    extract_script_block,
    parse_hunks,
    snippet,
    is_ast_valid,
)

root = Path(__file__).resolve().parent

input_csv = root / "npm_vuln_metadata.csv"
output_csv = root / "npm_js_dataset.csv"

parser_script = root / "parse_check.js"
node_bin = "node"

cache_dir = root / "_cache" / "repos"

context_lines = 3
max_rows: Optional[int] = None

skip_dir_keywords = ["node_modules"]
skip_file_suffixes = [".min.js"]

dedup_by_snippet = False


def parse_repo_url(url: str) -> Optional[Tuple[str, str]]:
    if not url or "github.com" not in url:
        return None
    try:
        part = url.split("github.com/")[-1].strip("/").replace(".git", "")
        owner, repo = part.split("/", 1)
        return owner, repo
    except Exception:
        return None


def parse_commit_sha(url: str) -> Optional[str]:
    if not url or "/commit/" not in url:
        return None
    try:
        return url.split("/commit/")[-1].split("?")[0].strip()
    except Exception:
        return None


def build_records(row: pd.Series, owner: str, repo: str, repo_dir: Path) -> List[Dict[str, Any]]:
    sha = parse_commit_sha(str(row.get("github_commit", "")))
    if not sha:
        return []

    psha = parent_sha(repo_dir, sha)
    if not psha:
        return []

    files = changed_files(repo_dir, sha)
    if not files:
        return []

    records: List[Dict[str, Any]] = []

    for f in files:
        status = f["status"]
        if status not in {"modified", "renamed"}:
            continue

        path = f["path"]
        prev = f["prev"] or path
        if not path:
            continue

        if should_skip(path, skip_dir_keywords, skip_file_suffixes):
            continue

        if not (is_js_like(path) or is_sfc(path)):
            continue

        patch = file_patch(repo_dir, psha, sha, path)
        if not patch.strip() and prev != path:
            patch = file_patch(repo_dir, psha, sha, prev)
        if not patch.strip():
            continue

        hunk = parse_hunks(patch)
        if not hunk:
            continue

        old_s, old_e, new_s, new_e = hunk

        before = file_text(repo_dir, psha, prev)
        after = file_text(repo_dir, sha, path)
        if not before or not after:
            continue

        if is_sfc(path):
            before_ast = extract_script_block(before)
            after_ast = extract_script_block(after)
            if not before_ast or not after_ast:
                continue
        else:
            before_ast, after_ast = before, after

        full_ok = False
        try:
            full_ok = is_ast_valid(before_ast, node_bin, parser_script) and is_ast_valid(after_ast, node_bin, parser_script)
        except Exception:
            full_ok = False

        before_snip = snippet(before, old_s, old_e, context_lines)
        after_snip = snippet(after, new_s, new_e, context_lines)
        if not before_snip.strip() or not after_snip.strip():
            continue

        if not full_ok:
            if is_sfc(path):
                b = extract_script_block(before_snip)
                a = extract_script_block(after_snip)
            else:
                b, a = before_snip, after_snip

            if not b or not a:
                continue

            try:
                if not (is_ast_valid(b, node_bin, parser_script) and is_ast_valid(a, node_bin, parser_script)):
                    continue
            except Exception:
                continue

        records.append({
            "package_name": row.get("package_name", ""),
            "ecosystem": row.get("ecosystem", ""),
            "ghsa_id": row.get("ghsa_id", ""),
            "cve_ids": row.get("cve_ids", ""),
            "cwe_ids": row.get("cwe_ids", ""),
            "summary": row.get("summary", ""),
            "severity": row.get("severity", ""),
            "vulnerable_version_range": row.get("vulnerable_version_range", ""),
            "fixed_version": row.get("fixed_version", ""),
            "github_repo_url": row.get("github_repo", ""),
            "owner": owner,
            "repo": repo,
            "commit_sha": sha,
            "parent_sha": psha,
            "file_path": path,
            "status": status,
            "old_start_line": old_s,
            "old_end_line": old_e,
            "new_start_line": new_s,
            "new_end_line": new_e,
            "file_code_before": before,
            "file_code_after": after,
            "vuln_code_before": before_snip,
            "fixed_code_after": after_snip,
            "label_before": 1,
            "label_after": 0,
            "patch": patch,
        })

    return records


def main() -> None:
    ensure_git()
    df = pd.read_csv(input_csv)

    df = df[df["github_repo"].notna() & df["github_commit"].notna()]
    df = df[df["github_repo"].astype(str).str.contains("github.com", na=False)]
    df = df[df["github_commit"].astype(str).str.contains("/commit/", na=False)]

    parsed = df["github_repo"].astype(str).apply(parse_repo_url)
    df["owner"] = parsed.apply(lambda x: x[0] if x else None)
    df["repo_name"] = parsed.apply(lambda x: x[1] if x else None)
    df = df[df["owner"].notna() & df["repo_name"].notna()]

    before_rows = len(df)
    df = df.drop_duplicates(subset=["ghsa_id", "github_repo", "github_commit"])

    if max_rows is not None:
        df = df.head(max_rows)

    print(f"rows_in={before_rows} rows_unique={len(df)}")

    all_rows: List[Dict[str, Any]] = []

    for (owner, repo), group in df.groupby(["owner", "repo_name"], sort=False):
        repo_dir = clone_repo(owner, repo, cache_dir)
        if repo_dir is None:
            print(f"skip_repo={owner}/{repo}")
            continue

        try:
            for _, row in group.iterrows():
                all_rows.extend(build_records(row, owner, repo, repo_dir))
        finally:
            remove_repo(owner, repo, cache_dir)

    if not all_rows:
        print("no_records")
        return

    out = pd.DataFrame(all_rows)
    before_dedup = len(out)
    out = out.drop_duplicates()
    after_dedup = len(out)

    if dedup_by_snippet and "vuln_code_before" in out.columns:
        out = out.drop_duplicates(
            subset=["owner", "repo", "file_path", "commit_sha", "old_start_line", "old_end_line"]
        ).reset_index(drop=True)

    out.to_csv(output_csv, index=False, encoding="utf-8")



if __name__ == "__main__":
    main()
