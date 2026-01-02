"""Secure Configuration Manager for AlphaTrade System.

This module provides secure credential management with support for:
- Environment variables (development)
- AWS Secrets Manager (production)
- HashiCorp Vault (production)

SECURITY NOTES:
- Never log credential values
- Use secure memory handling where possible
- Implement credential rotation support
- Audit all credential access
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class SecretsBackend(str, Enum):
    """Supported secrets management backends."""

    ENVIRONMENT = "environment"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    HASHICORP_VAULT = "hashicorp_vault"


@dataclass
class CredentialAccessLog:
    """Audit log entry for credential access."""

    timestamp: datetime
    credential_name: str
    accessor: str
    action: str
    success: bool
    masked_value: str | None = None
    error: str | None = None


@dataclass
class SecureCredential:
    """Container for a secure credential with metadata."""

    name: str
    value: str
    source: SecretsBackend
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    rotation_due: datetime | None = None

    def __repr__(self) -> str:
        """Secure repr that never shows credential value."""
        return f"SecureCredential(name={self.name!r}, source={self.source}, masked=***)"

    def __str__(self) -> str:
        """Secure str that never shows credential value."""
        return f"[CREDENTIAL:{self.name}]"

    @property
    def masked_value(self) -> str:
        """Return masked version of credential for logging."""
        if len(self.value) <= 8:
            return "***"
        return f"{self.value[:4]}...{self.value[-4:]}"

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def get_secrets(self, keys: list[str]) -> dict[str, str | None]:
        """Retrieve multiple secrets."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """Secrets provider using environment variables."""

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable."""
        return os.environ.get(key)

    def get_secrets(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple secrets from environment."""
        return {key: os.environ.get(key) for key in keys}

    def is_available(self) -> bool:
        """Environment is always available."""
        return True


class AWSSecretsManagerProvider(SecretsProvider):
    """Secrets provider using AWS Secrets Manager."""

    def __init__(
        self,
        region: str | None = None,
        secret_name: str | None = None,
    ):
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.secret_name = secret_name or os.environ.get(
            "AWS_SECRETS_NAME", "alphatrade/production"
        )
        self._client = None
        self._cached_secrets: dict[str, str] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl_seconds = 300  # 5 minutes

    def _get_client(self):
        """Lazy initialization of boto3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client(
                    "secretsmanager",
                    region_name=self.region,
                )
            except ImportError:
                logger.error("boto3 not installed. Run: pip install boto3")
                raise
            except Exception as e:
                logger.error(f"Failed to create AWS Secrets Manager client: {e}")
                raise
        return self._client

    def _load_secrets(self) -> dict[str, str]:
        """Load all secrets from AWS Secrets Manager."""
        now = datetime.now(timezone.utc)
        if (
            self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self._cache_ttl_seconds
        ):
            return self._cached_secrets

        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=self.secret_name)
            secret_string = response.get("SecretString", "{}")
            self._cached_secrets = json.loads(secret_string)
            self._cache_time = now
            logger.info(f"Loaded secrets from AWS Secrets Manager: {self.secret_name}")
            return self._cached_secrets
        except Exception as e:
            logger.error(f"Failed to load secrets from AWS: {e}")
            return {}

    def get_secret(self, key: str) -> str | None:
        """Get a single secret from AWS."""
        secrets = self._load_secrets()
        return secrets.get(key)

    def get_secrets(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple secrets from AWS."""
        secrets = self._load_secrets()
        return {key: secrets.get(key) for key in keys}

    def is_available(self) -> bool:
        """Check if AWS Secrets Manager is available."""
        try:
            self._get_client()
            return True
        except Exception:
            return False


class VaultSecretsProvider(SecretsProvider):
    """Secrets provider using HashiCorp Vault."""

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        secret_path: str | None = None,
    ):
        self.addr = addr or os.environ.get("VAULT_ADDR", "")
        self.token = token or os.environ.get("VAULT_TOKEN", "")
        self.secret_path = secret_path or os.environ.get(
            "VAULT_SECRET_PATH", "secret/data/alphatrade"
        )
        self._client = None
        self._cached_secrets: dict[str, str] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl_seconds = 300

    def _get_client(self):
        """Lazy initialization of hvac client."""
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(url=self.addr, token=self.token)
                if not self._client.is_authenticated():
                    raise ValueError("Vault authentication failed")
            except ImportError:
                logger.error("hvac not installed. Run: pip install hvac")
                raise
            except Exception as e:
                logger.error(f"Failed to create Vault client: {e}")
                raise
        return self._client

    def _load_secrets(self) -> dict[str, str]:
        """Load all secrets from Vault."""
        now = datetime.now(timezone.utc)
        if (
            self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self._cache_ttl_seconds
        ):
            return self._cached_secrets

        try:
            client = self._get_client()
            response = client.secrets.kv.v2.read_secret_version(
                path=self.secret_path.replace("secret/data/", "")
            )
            self._cached_secrets = response.get("data", {}).get("data", {})
            self._cache_time = now
            logger.info(f"Loaded secrets from Vault: {self.secret_path}")
            return self._cached_secrets
        except Exception as e:
            logger.error(f"Failed to load secrets from Vault: {e}")
            return {}

    def get_secret(self, key: str) -> str | None:
        """Get a single secret from Vault."""
        secrets = self._load_secrets()
        return secrets.get(key)

    def get_secrets(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple secrets from Vault."""
        secrets = self._load_secrets()
        return {key: secrets.get(key) for key in keys}

    def is_available(self) -> bool:
        """Check if Vault is available."""
        if not self.addr or not self.token:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False


class SecureConfigManager:
    """Centralized secure configuration manager.

    Supports multiple backends with automatic fallback:
    1. AWS Secrets Manager (production)
    2. HashiCorp Vault (production)
    3. Environment variables (development/fallback)

    Usage:
        config = SecureConfigManager.get_instance()
        api_key = config.get_credential("ALPACA_API_KEY")
    """

    _instance: SecureConfigManager | None = None
    _lock = None  # Will be initialized in __new__

    def __new__(cls) -> SecureConfigManager:
        """Singleton pattern for config manager."""
        if cls._instance is None:
            import threading

            if cls._lock is None:
                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Initialize the config manager."""
        if self._initialized:
            return

        self._providers: dict[SecretsBackend, SecretsProvider] = {}
        self._access_log: list[CredentialAccessLog] = []
        self._credentials_cache: dict[str, SecureCredential] = {}
        self._primary_backend = SecretsBackend.ENVIRONMENT

        # Initialize providers
        self._init_providers()
        self._initialized = True

    def _init_providers(self):
        """Initialize all available secrets providers."""
        # Environment is always available
        self._providers[SecretsBackend.ENVIRONMENT] = EnvironmentSecretsProvider()

        # Determine primary backend from environment
        backend_name = os.environ.get("SECRETS_BACKEND", "environment").lower()
        try:
            self._primary_backend = SecretsBackend(backend_name)
        except ValueError:
            logger.warning(
                f"Unknown SECRETS_BACKEND: {backend_name}, using environment"
            )
            self._primary_backend = SecretsBackend.ENVIRONMENT

        # Try to initialize AWS Secrets Manager
        if self._primary_backend == SecretsBackend.AWS_SECRETS_MANAGER:
            try:
                provider = AWSSecretsManagerProvider()
                if provider.is_available():
                    self._providers[SecretsBackend.AWS_SECRETS_MANAGER] = provider
                    logger.info("AWS Secrets Manager provider initialized")
            except Exception as e:
                logger.warning(f"AWS Secrets Manager not available: {e}")

        # Try to initialize Vault
        if self._primary_backend == SecretsBackend.HASHICORP_VAULT:
            try:
                provider = VaultSecretsProvider()
                if provider.is_available():
                    self._providers[SecretsBackend.HASHICORP_VAULT] = provider
                    logger.info("HashiCorp Vault provider initialized")
            except Exception as e:
                logger.warning(f"HashiCorp Vault not available: {e}")

    @classmethod
    def get_instance(cls) -> SecureConfigManager:
        """Get the singleton instance."""
        return cls()

    def _log_access(
        self,
        credential_name: str,
        action: str,
        success: bool,
        masked_value: str | None = None,
        error: str | None = None,
    ):
        """Log credential access for auditing."""
        import traceback

        # Get caller information
        stack = traceback.extract_stack()
        accessor = "unknown"
        for frame in reversed(stack[:-2]):
            if "secure_config.py" not in frame.filename:
                accessor = f"{frame.filename}:{frame.lineno}:{frame.name}"
                break

        log_entry = CredentialAccessLog(
            timestamp=datetime.now(timezone.utc),
            credential_name=credential_name,
            accessor=accessor,
            action=action,
            success=success,
            masked_value=masked_value,
            error=error,
        )
        self._access_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

        # Log at appropriate level
        if success:
            logger.debug(f"Credential access: {credential_name} by {accessor}")
        else:
            logger.warning(
                f"Failed credential access: {credential_name} by {accessor}: {error}"
            )

    def get_credential(
        self,
        key: str,
        required: bool = True,
        default: str | None = None,
    ) -> str | None:
        """Get a credential securely.

        Args:
            key: The credential key/name
            required: If True, raises error when not found
            default: Default value if not found and not required

        Returns:
            The credential value or default

        Raises:
            ValueError: If required credential is not found
        """
        # Check cache first
        if key in self._credentials_cache:
            cached = self._credentials_cache[key]
            if not cached.is_expired():
                self._log_access(key, "get_cached", True, cached.masked_value)
                return cached.value

        # Try primary backend first
        value = None
        source = None

        if self._primary_backend in self._providers:
            provider = self._providers[self._primary_backend]
            value = provider.get_secret(key)
            if value is not None:
                source = self._primary_backend

        # Fallback to environment if primary failed
        if value is None and self._primary_backend != SecretsBackend.ENVIRONMENT:
            provider = self._providers[SecretsBackend.ENVIRONMENT]
            value = provider.get_secret(key)
            if value is not None:
                source = SecretsBackend.ENVIRONMENT

        # Handle result
        if value is not None:
            credential = SecureCredential(name=key, value=value, source=source)
            self._credentials_cache[key] = credential
            self._log_access(key, "get", True, credential.masked_value)
            return value
        elif required:
            self._log_access(key, "get", False, error="Required credential not found")
            raise ValueError(
                f"Required credential '{key}' not found in any secrets backend. "
                f"Please set it in your .env file or secrets manager."
            )
        else:
            self._log_access(key, "get", True, error="Using default")
            return default

    def get_credentials(
        self,
        keys: list[str],
        required: bool = True,
    ) -> dict[str, str | None]:
        """Get multiple credentials at once."""
        return {key: self.get_credential(key, required=required) for key in keys}

    def validate_credential_format(
        self,
        key: str,
        value: str,
        pattern: str | None = None,
    ) -> bool:
        """Validate credential format.

        Args:
            key: Credential name (used to infer pattern if not provided)
            value: The credential value to validate
            pattern: Optional regex pattern to match

        Returns:
            True if valid, False otherwise
        """
        if pattern is None:
            # Infer pattern from key name
            patterns = {
                "ALPACA_API_KEY": r"^[A-Z0-9]{20,}$",
                "ALPACA_API_SECRET": r"^[A-Za-z0-9]{30,}$",
                "DATABASE__PASSWORD": r"^.{8,}$",  # At least 8 chars
                "SLACK_WEBHOOK_URL": r"^https://hooks\.slack\.com/.*$",
                "PAGERDUTY_SERVICE_KEY": r"^[a-f0-9]{32}$",
            }
            pattern = patterns.get(key, r"^.+$")  # Default: non-empty

        try:
            return bool(re.match(pattern, value))
        except re.error:
            logger.error(f"Invalid regex pattern for {key}: {pattern}")
            return False

    def verify_alpaca_credentials(self) -> bool:
        """Verify Alpaca API credentials are valid.

        Returns:
            True if credentials are valid and working
        """
        try:
            api_key = self.get_credential("ALPACA_API_KEY", required=False)
            api_secret = self.get_credential("ALPACA_API_SECRET", required=False)

            if not api_key or not api_secret:
                logger.warning("Alpaca credentials not configured")
                return False

            # Validate format
            if not self.validate_credential_format("ALPACA_API_KEY", api_key):
                logger.error("Alpaca API key has invalid format")
                return False

            if not self.validate_credential_format("ALPACA_API_SECRET", api_secret):
                logger.error("Alpaca API secret has invalid format")
                return False

            # Try to connect (optional - requires alpaca-py)
            try:
                from alpaca.trading.client import TradingClient

                client = TradingClient(api_key, api_secret, paper=True)
                account = client.get_account()
                logger.info(
                    f"Alpaca credentials verified. Account status: {account.status}"
                )
                return True
            except ImportError:
                logger.info(
                    "alpaca-py not installed, skipping live credential verification"
                )
                return True
            except Exception as e:
                logger.error(f"Alpaca credential verification failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Error verifying Alpaca credentials: {e}")
            return False

    def get_access_log(
        self,
        limit: int = 100,
        credential_name: str | None = None,
    ) -> list[CredentialAccessLog]:
        """Get credential access audit log.

        Args:
            limit: Maximum number of entries to return
            credential_name: Filter by credential name

        Returns:
            List of access log entries
        """
        logs = self._access_log
        if credential_name:
            logs = [log for log in logs if log.credential_name == credential_name]
        return logs[-limit:]

    def clear_cache(self):
        """Clear the credentials cache."""
        self._credentials_cache.clear()
        logger.info("Credentials cache cleared")

    def get_configuration_status(self) -> dict[str, Any]:
        """Get configuration status summary."""
        return {
            "primary_backend": self._primary_backend.value,
            "available_backends": [
                backend.value for backend in self._providers.keys()
            ],
            "cached_credentials": len(self._credentials_cache),
            "access_log_entries": len(self._access_log),
            "alpaca_configured": bool(
                self.get_credential("ALPACA_API_KEY", required=False)
            ),
            "database_configured": bool(
                self.get_credential("DATABASE__PASSWORD", required=False)
            ),
        }


# Convenience function for getting credentials
@lru_cache(maxsize=128)
def get_secure_config() -> SecureConfigManager:
    """Get the secure config manager instance."""
    return SecureConfigManager.get_instance()


def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
    """Mask a sensitive value for safe logging.

    Args:
        value: The sensitive value to mask
        show_chars: Number of characters to show at start and end

    Returns:
        Masked string like "ABCD...WXYZ"
    """
    if not value:
        return "***"
    if len(value) <= show_chars * 2:
        return "***"
    return f"{value[:show_chars]}...{value[-show_chars:]}"
