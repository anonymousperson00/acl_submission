import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


root = Path(__file__).resolve().parent
output_csv = root / "npm_vuln_metadata.csv"

api_url = "https://api.github.com/graphql"
token = os.getenv("github_token") or os.getenv("gh_token") or os.getenv("gh_pat") or os.getenv("gq_token") or os.getenv("github_api_token") or os.getenv("GITHUB_TOKEN")

max_advisories = 15000
page_size = 100

ecosystem = "NPM"

session = requests.Session()
session.headers.update({"Accept": "application/vnd.github+json"})
if token:
    session.headers.update({"Authorization": f"Bearer {token}"})


query = """
query($after: String, $first: Int!) {
  securityAdvisories(first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ghsaId
      summary
      severity
      identifiers { type value }
      vulnerabilities(first: 100) {
        nodes {
          package { ecosystem name }
          vulnerableVersionRange
          firstPatchedVersion { identifier }
        }
      }
      references { url }
    }
  }
}
"""


def graphql(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = session.post(api_url, json=payload, timeout=60)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        wait_for_rate_limit()
        r = session.post(api_url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def rate_limit() -> Tuple[int, int, int]:
    url = "https://api.github.com/rate_limit"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return 0, 0, 0
    data = r.json() or {}
    core = (data.get("resources") or {}).get("core") or {}
    remaining = int(core.get("remaining") or 0)
    reset = int(core.get("reset") or 0)
    limit = int(core.get("limit") or 0)
    return remaining, reset, limit


def wait_for_rate_limit() -> None:
    remaining, reset, _ = rate_limit()
    if remaining > 0:
        return
    now = int(time.time())
    sleep_s = max(5, reset - now + 5)
    ts = datetime.fromtimestamp(reset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[gh] rate limit reached, sleeping {sleep_s}s (reset at {ts})")
    time.sleep(sleep_s)


def pick_ids(identifiers: List[Dict[str, Any]]) -> Tuple[str, str]:
    cve_ids: List[str] = []
    other_ids: List[str] = []
    for x in identifiers or []:
        t = (x.get("type") or "").upper()
        v = (x.get("value") or "").strip()
        if not v:
            continue
        if t == "CVE" or v.upper().startswith("CVE-"):
            cve_ids.append(v)
        else:
            other_ids.append(v)
    return ";".join(sorted(set(cve_ids))), ";".join(sorted(set(other_ids)))


def best_github_urls(refs: List[Dict[str, Any]]) -> Tuple[str, str]:
    repo_url = ""
    commit_url = ""

    for r in refs or []:
        url = (r.get("url") or "").strip()
        if not url:
            continue

        if "github.com" not in url:
            continue

        if "/commit/" in url and not commit_url:
            commit_url = url

        if not repo_url:
            parts = url.split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                repo_url = "https://github.com/" + "/".join(parts[:2]).replace(".git", "")

        if repo_url and commit_url:
            break

    return repo_url, commit_url


def write_rows(rows: List[Dict[str, Any]], out_path: Path, write_header: bool) -> None:
    cols = [
        "ecosystem",
        "package_name",
        "ghsa_id",
        "summary",
        "severity",
        "cve_ids",
        "cwe_ids",
        "vulnerable_version_range",
        "fixed_version",
        "github_repo",
        "github_commit",
    ]
    mode = "w" if write_header else "a"
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    if not token:
        print("[warn] no github token found; you may hit rate limits quickly")

    after: Optional[str] = None
    advisories_seen = 0
    rows_written = 0
    page = 0

    if output_csv.exists():
        output_csv.unlink()

    while advisories_seen < max_advisories:
        wait_for_rate_limit()

        page += 1
        payload = {"query": query, "variables": {"after": after, "first": page_size}}
        data = graphql(payload)
        nodes = (((data.get("data") or {}).get("securityAdvisories") or {}).get("nodes") or [])

        if not nodes:
            break

        page_rows: List[Dict[str, Any]] = []

        for adv in nodes:
            if advisories_seen >= max_advisories:
                break

            ghsa_id = adv.get("ghsaId", "")
            summary = adv.get("summary", "")
            sev = adv.get("severity", "")

            cve_ids, other_ids = pick_ids(adv.get("identifiers") or [])
            cwe_ids = other_ids

            repo_url, commit_url = best_github_urls(adv.get("references") or [])

            vulns = ((adv.get("vulnerabilities") or {}).get("nodes") or [])
            npm_vulns = [v for v in vulns if ((v.get("package") or {}).get("ecosystem") == ecosystem)]

            if not npm_vulns:
                advisories_seen += 1
                continue

            for v in npm_vulns:
                pkg = (v.get("package") or {}).get("name", "")
                vrange = v.get("vulnerableVersionRange") or ""
                fpv = (v.get("firstPatchedVersion") or {}).get("identifier") or ""

                page_rows.append({
                    "ecosystem": ecosystem,
                    "package_name": pkg,
                    "ghsa_id": ghsa_id,
                    "summary": summary,
                    "severity": sev,
                    "cve_ids": cve_ids,
                    "cwe_ids": cwe_ids,
                    "vulnerable_version_range": vrange,
                    "fixed_version": fpv,
                    "github_repo": repo_url,
                    "github_commit": commit_url,
                })

            advisories_seen += 1

        write_rows(page_rows, output_csv, write_header=(page == 1))
        rows_written += len(page_rows)

        page_info = ((data.get("data") or {}).get("securityAdvisories") or {}).get("pageInfo") or {}
        after = page_info.get("endCursor")
        has_next = bool(page_info.get("hasNextPage"))

        print(f"[ghsa] page={page} advisories={advisories_seen} rows={rows_written}")

        if not has_next:
            break

    print(f"saved {rows_written} rows to {output_csv.name}")


if __name__ == "__main__":
    main()
