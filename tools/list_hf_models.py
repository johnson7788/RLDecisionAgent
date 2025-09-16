#!/usr/bin/env python3
"""
列出本地的huggingface模型

- Uses huggingface_hub.scan_cache_dir() when available (recommended).
- Falls back to scanning ~/.cache/huggingface/hub/models--* if needed.
- Works cross‑platform and respects the HF_HOME env var if set.
- Optional: pass --cache-dir to point at a custom cache location.
- Optional: pass --json for machine-readable output.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    s = float(n)
    for u in units:
        if s < 1024.0 or u == units[-1]:
            return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{n} B"


def get_default_cache_dir() -> Path:
    # Honor HF_HOME; otherwise use ~/.cache/huggingface
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").expanduser().resolve()


def scan_with_hf_hub(cache_dir: Path) -> Optional[List[Dict]]:
    """
    Try to use huggingface_hub.scan_cache_dir for an accurate listing.
    Returns a list of dicts or None if huggingface_hub is unavailable.
    """
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore
    except Exception:
        return None

    # Some versions accept cache_dir, others use default; handle both.
    try:
        scan = scan_cache_dir(cache_dir=cache_dir)  # type: ignore[call-arg]
    except TypeError:
        scan = scan_cache_dir()  # type: ignore[call-arg]

    models: List[Dict] = []
    repos = getattr(scan, "repos", [])

    for repo in repos:
        # repo_type may be an enum or string
        repo_type = str(getattr(repo, "repo_type", "model"))
        if "model" not in repo_type.lower():
            continue

        rid = getattr(repo, "repo_id", None)
        # Size attribute name differs across versions
        size = getattr(repo, "size_on_disk", None)
        if size is None:
            size = getattr(repo, "repo_size_on_disk", None)
        rdir = getattr(repo, "repo_path", None) or getattr(repo, "repo_dir", None)

        # Collect revision info if available
        revs_out: List[Dict] = []
        for rev in getattr(repo, "revisions", []) or []:
            sha = getattr(rev, "commit_hash", None) or getattr(rev, "revision", None) or getattr(rev, "sha", None)
            rsize = getattr(rev, "size_on_disk", None)
            accessed = getattr(rev, "last_accessed", None) or getattr(rev, "last_modified", None)
            revs_out.append(
                {
                    "sha": sha,
                    "size_on_disk": int(rsize) if isinstance(rsize, (int, float)) else rsize,
                    "last_accessed": str(accessed) if accessed is not None else None,
                }
            )

        models.append(
            {
                "repo_id": rid,
                "repo_dir": str(rdir) if rdir else None,
                "size_on_disk": int(size) if isinstance(size, (int, float)) else size,
                "revisions": revs_out,
                "source": "huggingface_hub.scan_cache_dir",
            }
        )

    return models


def decode_repo_id_from_dirname(name: str) -> str:
    """
    Convert a directory like 'models--google--t5-small' to 'google/t5-small'.
    This mirrors how HF encodes repo IDs for filesystem safety.
    """
    rest = name[len("models--") :]
    return rest.replace("--", "/")


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except (OSError, PermissionError):
            # Skip files we can't stat
            pass
    return total


def scan_manually(cache_dir: Path) -> List[Dict]:
    hub_dir = cache_dir / "hub"
    out: List[Dict] = []
    if not hub_dir.exists():
        return out

    for entry in hub_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            rid = decode_repo_id_from_dirname(entry.name)
            try:
                size = dir_size(entry)
            except Exception:
                size = None
            out.append(
                {
                    "repo_id": rid,
                    "repo_dir": str(entry),
                    "size_on_disk": size,
                    "revisions": [],
                    "source": "manual-scan",
                }
            )
    return out


def merge_by_repo_id(primary: List[Dict], fallback: List[Dict]) -> List[Dict]:
    """
    Merge two lists of model dicts by repo_id, preferring entries in 'primary'.
    """
    seen = {m.get("repo_id"): m for m in primary if m.get("repo_id")}
    for m in fallback:
        rid = m.get("repo_id")
        if not rid or rid in seen:
            continue
        seen[rid] = m
    return sorted(seen.values(), key=lambda x: (x.get("repo_id") or "").lower())


def print_table(models: List[Dict]) -> None:
    if not models:
        print("No cached Hugging Face models were found.")
        return

    # Compute column widths
    id_width = max(10, min(60, max(len(m.get("repo_id") or "") for m in models)))
    size_width = 10
    header = f"{'#':>3}  {'Model ID':<{id_width}}  {'Size':>{size_width}}  Local Path"
    print(header)
    print("-" * len(header))

    for idx, m in enumerate(models, 1):
        repo_id = m.get("repo_id") or "-"
        size = human_bytes(m.get("size_on_disk"))
        path = m.get("repo_dir") or "-"
        print(f"{idx:>3}  {repo_id:<{id_width}}  {size:>{size_width}}  {path}")


def main():
    parser = argparse.ArgumentParser(description="List locally cached Hugging Face models.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to the HF cache directory (defaults to HF_HOME or ~/.cache/huggingface).")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a table.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else get_default_cache_dir()

    models = scan_with_hf_hub(cache_dir) or []
    manual = scan_manually(cache_dir)
    combined = merge_by_repo_id(models, manual)

    if args.json:
        print(json.dumps(combined, ensure_ascii=False, indent=2))
    else:
        print(f"Cache dir: {cache_dir}")
        print_table(combined)


if __name__ == "__main__":
    main()
