"""
Alembic migration environment configuration.

This module configures the Alembic environment for running migrations
with our SQLAlchemy models and database connection.
"""

# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv(override=True)

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import the SQLAlchemy models metadata
from quant_trading_system.database.models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = Base.metadata


def get_url():
    """Get database URL from environment, alembic.ini, or settings."""
    import os

    # First check environment variable
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # Check alembic.ini config
    try:
        config_url = config.get_main_option("sqlalchemy.url")
        if config_url:
            return config_url
    except Exception:
        pass

    # Fall back to settings
    try:
        from quant_trading_system.config.settings import get_settings
        settings = get_settings()
        return settings.database.url
    except Exception:
        # Default for development
        return "postgresql://postgres:postgres@localhost:5432/quant_trading"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
