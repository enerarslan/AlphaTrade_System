"""Repository secret scanner with policy-oriented detection rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SecretFinding:
    """A single potential secret exposure finding."""

    path: str
    line: int
    rule_id: str
    severity: str
    message: str
    snippet: str


@dataclass
class SecretScannerConfig:
    """Scanner configuration."""

    max_file_size_bytes: int = 1_000_000
    exclude_dirs: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "logs",
        "data",
        "models_artifacts",
    )
    include_extensions: tuple[str, ...] = (
        ".env",
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".py",
        ".sh",
        ".ps1",
        ".js",
        ".ts",
        ".tsx",
    )
    placeholder_tokens: tuple[str, ...] = (
        "your_",
        "example",
        "sample",
        "dummy",
        "changeme",
        "replace_me",
        "test",
        "fake",
        "mock",
    )


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    severity: str
    message: str
    pattern: re.Pattern[str]


class SecretScanner:
    """Scans filesystem paths for accidental secret disclosure."""

    def __init__(self, config: SecretScannerConfig | None = None) -> None:
        self.config = config or SecretScannerConfig()
        self._rules: tuple[_Rule, ...] = (
            _Rule(
                rule_id="private_key_block",
                severity="critical",
                message="Private key material detected",
                pattern=re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----"),
            ),
            _Rule(
                rule_id="aws_access_key_id",
                severity="high",
                message="AWS access key ID detected",
                pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            ),
            _Rule(
                rule_id="jwt_token",
                severity="high",
                message="JWT token-like value detected",
                pattern=re.compile(
                    r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b"
                ),
            ),
            _Rule(
                rule_id="generic_secret_assignment",
                severity="high",
                message="Credential-like assignment detected",
                pattern=re.compile(
                    r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|password|passwd)\b"
                    r"\s*[:=]\s*['\"]([a-zA-Z0-9/_+=-]{20,})['\"]"
                ),
            ),
            _Rule(
                rule_id="slack_webhook",
                severity="high",
                message="Slack webhook URL detected",
                pattern=re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
            ),
        )

    def scan_paths(self, paths: list[Path]) -> list[SecretFinding]:
        """Scan explicit paths and return collected findings."""
        findings: list[SecretFinding] = []
        for path in paths:
            if path.is_dir():
                for file_path in self._iter_files(path):
                    findings.extend(self._scan_file(file_path, base_dir=path))
                continue
            findings.extend(self._scan_file(path, base_dir=path.parent))
        return findings

    def _iter_files(self, root: Path) -> list[Path]:
        """Iterate candidate files recursively under root."""
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if self._is_excluded(path, root):
                continue
            if not self._is_supported_extension(path):
                continue
            if path.stat().st_size > self.config.max_file_size_bytes:
                continue
            files.append(path)
        return files

    def _scan_file(self, file_path: Path, base_dir: Path) -> list[SecretFinding]:
        """Scan one text file for all configured rules."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        findings: list[SecretFinding] = []
        rel_path = str(file_path.resolve().relative_to(base_dir.resolve()))
        for line_number, line in enumerate(content.splitlines(), start=1):
            for rule in self._rules:
                for match in rule.pattern.finditer(line):
                    if self._should_ignore_match(rule.rule_id, line, match):
                        continue
                    findings.append(
                        SecretFinding(
                            path=rel_path,
                            line=line_number,
                            rule_id=rule.rule_id,
                            severity=rule.severity,
                            message=rule.message,
                            snippet=self._safe_snippet(line, match.group(0)),
                        )
                    )
        return findings

    def _should_ignore_match(
        self,
        rule_id: str,
        line: str,
        match: re.Match[str],
    ) -> bool:
        """Filter obvious placeholders and non-secret template values."""
        line_lower = line.lower()
        if "secret-scan:ignore" in line_lower:
            return True
        if any(token in line_lower for token in self.config.placeholder_tokens):
            return True

        if rule_id == "generic_secret_assignment":
            value = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            value_lower = value.lower()
            if any(token in value_lower for token in self.config.placeholder_tokens):
                return True
            if not any(char.isdigit() for char in value):
                return True

        return False

    def _is_excluded(self, path: Path, root: Path) -> bool:
        """Return True if a file path should be excluded from scanning."""
        relative_parts = path.resolve().relative_to(root.resolve()).parts
        return any(part in self.config.exclude_dirs for part in relative_parts)

    def _is_supported_extension(self, path: Path) -> bool:
        """Return True if file extension is in configured allowlist."""
        if path.name.startswith(".env"):
            return True
        return path.suffix.lower() in self.config.include_extensions

    @staticmethod
    def _safe_snippet(line: str, matched_value: str) -> str:
        """Return a redacted line snippet for reporting."""
        redacted = line.replace(matched_value, "<redacted>")
        return redacted.strip()
