from __future__ import annotations

from scripts.deploy import DatabaseManager, DeployConfig


def test_deploy_database_manager_uses_alembic_runner(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "quant_trading_system.database.migration_runner.run_database_migrations",
        lambda *, database_url: captured.setdefault("database_url", database_url),
    )
    monkeypatch.setenv("DATABASE__USER", "postgres")
    monkeypatch.setenv("DATABASE__PASSWORD", "secret")

    manager = DatabaseManager(
        DeployConfig(
            db_host="db.internal",
            db_port=5433,
            db_name="alphatrade_prod",
        )
    )

    assert manager.migrate() is True
    assert (
        captured["database_url"] == "postgresql://postgres:secret@db.internal:5433/alphatrade_prod"
    )
