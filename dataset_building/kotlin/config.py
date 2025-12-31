
from pathlib import Path
import os
import re

root = Path(__file__).resolve().parent

repos_dir = root / "repos"
out_dir = root / "out"
meta_dir = out_dir / "meta"
pair_dir = out_dir / "pairs"

for d in (repos_dir, out_dir, meta_dir, pair_dir):
    d.mkdir(parents=True, exist_ok=True)

commits_csv = meta_dir / "commits.csv"
files_csv = meta_dir / "files.csv"
patches_jsonl = meta_dir / "flagged_patches.jsonl"

pairs_csv = pair_dir / "kotlin_pairs.csv"
pairs_jsonl = pair_dir / "kotlin_pairs.jsonl"

repos = [
    "shadowsocks/shadowsocks-android",
    "guardianproject/orbot-android",
]

token = os.getenv("github_token") or os.getenv("GITHUB_TOKEN")

partial = False
tags = False

since = None
no_merges = True

only_exts = {".kt", ".kts"}

keep_patch = True
patch_max = 200_000

max_file_lines = 10
max_file_touch = 10
only_flagged = True

sec_keys = [
    r"\bsec(urity|harden(ing)?)\b",
    r"\bvuln(erab(ility|le))?\b",
    r"\bcve[- ]?\d{4}-\d{3,7}\b",
    r"\bcwe[- ]?\d{2,4}\b",
    r"\bexploit\b",
    r"\battack\b",
    r"\bmitigat(e|ion)\b",
    r"auth(enticat(e|ion)|oriz(e|ation))",
    r"\bcsrf\b",
    r"\bssrf\b",
    r"\bxss\b",
    r"\bxxe\b",
    r"\brce\b",
    r"\boverflow\b",
    r"\boob\b",
    r"\btls\b",
    r"\bssl\b",
    r"\bcertificate\b",
    r"\bpinning\b",
    r"\bprivacy\b",
    r"\btracking\b",
    r"\bleak(s|age)?\b",
    r"\bexpos(e|ed|ure)\b",
    r"\binfo(?:rmation)?[- ]?disclosure\b",
    r"\bsanitize(d|s|r|ion)?\b",
    r"\bvalidate(d|s|ion)?\b",
    r"\bwebview\b",
    r"\bcleartext\b",
    r"\bnetwork[- ]?security[- ]?config\b",
    r"\bexported\b",
    r"\bpermission(s)?\b",
]
sec_re = re.compile("|".join(sec_keys), re.IGNORECASE)

cve_re = re.compile(r"\bcve[- ]?\d{4}-\d{3,7}\b", re.IGNORECASE)

cls_re = re.compile(
    r"^\s*(?:public|private|protected|internal|open|abstract|sealed|data|final|enum|inline|"
    r"value|external|static|\s)*"
    r"(?:class|object|interface|enum\s+class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
