"""Programmatic Alembic migration runner for production schema management."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from quant_trading_system.config.settings import Settings, get_settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_alembic_config(
    *,
    settings: Settings | None = None,
    database_url: str | None = None,
) -> Config:
    """Build an Alembic configuration pinned to the current repository."""
    repo_root = _repo_root()
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(repo_root / "quant_trading_system" / "database" / "migrations"),
    )

    resolved_url = str(database_url or (settings or get_settings()).database.connection_string)
    config.attributes["database_url"] = resolved_url
    config.set_main_option("sqlalchemy.url", resolved_url)
    return config


def run_database_migrations(
    *,
    revision: str = "head",
    settings: Settings | None = None,
    database_url: str | None = None,
) -> None:
    """Upgrade the database schema to the requested Alembic revision."""
    config = build_alembic_config(settings=settings, database_url=database_url)
    command.upgrade(config, revision)
