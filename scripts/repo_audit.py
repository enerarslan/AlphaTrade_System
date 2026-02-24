"""Repository source-tracking audit utilities.

Ensures critical Python source directories are tracked by Git and not masked
by broad ignore rules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )


def _check_not_ignored(project_root: Path, rel_path: str) -> tuple[bool, str]:
    result = _run_git(["check-ignore", "-v", rel_path], project_root)
    if result.returncode == 0:
        details = result.stdout.strip() or result.stderr.strip()
        return False, f"{rel_path} is ignored by Git ({details})"
    return True, f"{rel_path} is not ignored"


def _list_untracked_py(project_root: Path, rel_dir: str) -> list[str]:
    result = _run_git(
        ["ls-files", "--others", "--exclude-standard", "--", f"{rel_dir}/*.py"],
        project_root,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_repo_audit(project_root: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    ignore_file = project_root / ".gitignore"
    if not ignore_file.exists():
        return False, [".gitignore not found"]

    content = ignore_file.read_text(encoding="utf-8")
    if "/data/" not in content:
        ok = False
        messages.append("Missing `/data/` ignore rule for root runtime data folder.")

    check_ok, check_msg = _check_not_ignored(project_root, "quant_trading_system/data/loader.py")
    if not check_ok:
        ok = False
    messages.append(check_msg)

    untracked = _list_untracked_py(project_root, "quant_trading_system/data")
    if untracked:
        ok = False
        messages.append(
            "Untracked python sources under quant_trading_system/data: "
            + ", ".join(sorted(untracked))
        )
    else:
        messages.append("All python sources under quant_trading_system/data are tracked.")

    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Repository source-tracking audit")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root directory (default: repository root)",
    )
    args = parser.parse_args()

    ok, messages = run_repo_audit(args.project_root)
    for message in messages:
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
