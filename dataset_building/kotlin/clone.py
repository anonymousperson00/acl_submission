# clone.py
from config import repos, repos_dir, token, partial, tags
from git_utils import sh, git_ok, longpaths

def clone_url(slug: str) -> str:
    if token:
        return f"https://x-access-token:{token}@github.com/{slug}.git"
    return f"https://github.com/{slug}.git"

def dest_dir(slug: str):
    owner, name = slug.split("/")
    return repos_dir / f"{owner}__{name}"

def sync_repo(slug: str) -> None:
    dst = dest_dir(slug)
    url = clone_url(slug)

    if not dst.exists():
        cmd = ["git", "clone", "--no-single-branch"]
        if partial:
            cmd = ["git", "clone", "--filter=blob:none", "--no-single-branch"]
        if not tags:
            cmd.insert(2, "--no-tags")
        sh(cmd + [url, str(dst)])
    else:
        sh(["git", "remote", "set-url", "origin", url], cwd=dst, ok=False)
        fetch_cmd = ["git", "fetch", "origin", "--prune"]
        if tags:
            fetch_cmd.append("--tags")
        sh(fetch_cmd, cwd=dst, ok=False)

    sh(["git", "remote", "set-head", "origin", "--auto"], cwd=dst, ok=False)

def main() -> None:
    longpaths()
    git_ok()

    for slug in repos:
        try:
            sync_repo(slug)
        except Exception as e:
            print(f"[clone] {slug}  error: {e}")

if __name__ == "__main__":
    main()
