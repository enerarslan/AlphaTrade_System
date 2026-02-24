#!/usr/bin/env python3
"""Repository secret scan command for local/CI policy gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.security.secret_scanner import SecretScanner, SecretScannerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Scan repository for accidental secret exposures")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Path to scan (repeatable). Defaults to repository root.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--max-file-size-kb",
        type=int,
        default=1024,
        help="Skip files larger than this size in KB",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute secret scanner and return shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    scan_paths = [Path(path).resolve() for path in args.path] if args.path else [PROJECT_ROOT]
    scanner = SecretScanner(
        SecretScannerConfig(max_file_size_bytes=max(1, args.max_file_size_kb) * 1024)
    )
    findings = scanner.scan_paths(scan_paths)

    if args.format == "json":
        payload = [
            {
                "path": finding.path,
                "line": finding.line,
                "rule_id": finding.rule_id,
                "severity": finding.severity,
                "message": finding.message,
                "snippet": finding.snippet,
            }
            for finding in findings
        ]
        print(json.dumps(payload, indent=2))
    else:
        if not findings:
            print("Secret scan passed: no findings.")
        for finding in findings:
            print(
                f"[{finding.severity}] {finding.path}:{finding.line} "
                f"{finding.rule_id} - {finding.message}"
            )
            print(f"  {finding.snippet}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
