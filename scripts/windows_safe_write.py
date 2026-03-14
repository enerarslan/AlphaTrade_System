"""
Write large text payloads to disk without putting them on the Windows command line.

Usage:
    Get-Content big.txt | python scripts/windows_safe_write.py path/to/output.txt
    python scripts/windows_safe_write.py path/to/output.txt --input-file big.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Destination file path")
    parser.add_argument("--input-file", help="Read payload from file instead of stdin")
    parser.add_argument("--append", action="store_true", help="Append instead of overwrite")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding")
    return parser.parse_args()


def read_payload(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding=args.encoding)
    return sys.stdin.read()


def main() -> int:
    args = parse_args()
    destination = Path(args.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = read_payload(args)

    if args.append and destination.exists():
        with destination.open("a", encoding=args.encoding, newline="\n") as handle:
            handle.write(payload)
        return 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=args.encoding,
        newline="\n",
        delete=False,
        dir=destination.parent,
    ) as handle:
        handle.write(payload)
        temp_name = handle.name

    os.replace(temp_name, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
