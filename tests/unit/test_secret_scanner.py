"""Unit tests for repository secret scanner."""

from pathlib import Path

from quant_trading_system.security.secret_scanner import SecretScanner


class TestSecretScanner:
    """Tests for SecretScanner detection rules."""

    def test_detects_private_key_block(self, tmp_path: Path) -> None:
        """Private key block markers must be flagged."""
        test_file = tmp_path / "bad.pem"
        test_file.write_text("-----BEGIN PRIVATE KEY-----\nabcdef\n-----END PRIVATE KEY-----\n")

        scanner = SecretScanner()
        findings = scanner.scan_paths([tmp_path])

        assert len(findings) == 1
        assert findings[0].rule_id == "private_key_block"

    def test_ignores_placeholder_values(self, tmp_path: Path) -> None:
        """Template/example secrets should not trip the scanner."""
        test_file = tmp_path / ".env.example"
        test_file.write_text("API_SECRET=your_api_secret_here\n")

        scanner = SecretScanner()
        findings = scanner.scan_paths([tmp_path])

        assert findings == []

    def test_detects_generic_secret_assignment(self, tmp_path: Path) -> None:
        """Realistic long credential assignments should be detected."""
        test_file = tmp_path / "config.py"
        test_file.write_text(
            "API_SECRET = 'prod_secret_1234567890abcdef1234567890abcdef'\n"
        )

        scanner = SecretScanner()
        findings = scanner.scan_paths([tmp_path])

        assert len(findings) == 1
        assert findings[0].rule_id == "generic_secret_assignment"

    def test_excluded_directories_are_skipped(self, tmp_path: Path) -> None:
        """Files under excluded directories should not be scanned."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        test_file = git_dir / "config"
        test_file.write_text("API_SECRET=prod_secret_1234567890abcdef1234567890abcdef\n")

        scanner = SecretScanner()
        findings = scanner.scan_paths([tmp_path])

        assert findings == []
