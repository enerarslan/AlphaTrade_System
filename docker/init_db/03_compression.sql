-- =============================================================================
-- AlphaTrade System - TimescaleDB Compression Placeholder
-- =============================================================================
-- Compression policies are handled by Alembic migrations.
-- This file is kept for compatibility but does not configure compression.
--
-- See: quant_trading_system/database/migrations/versions/002_timescaledb_hypertables.py
--
-- Run order: 03 (after hypertables are created)
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Compression policies will be handled by Alembic migrations';
END $$;
