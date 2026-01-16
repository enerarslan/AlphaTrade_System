-- =============================================================================
-- AlphaTrade System - TimescaleDB Retention Placeholder
-- =============================================================================
-- Retention policies are handled by Alembic migrations.
-- This file is kept for compatibility but does not configure retention.
--
-- Note: Retention policies are only applicable to hypertables.
-- Only ohlcv_bars and features are converted to hypertables.
--
-- See: quant_trading_system/database/migrations/versions/002_timescaledb_hypertables.py
--
-- Run order: 04 (after compression policies)
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Retention policies will be handled by Alembic migrations';
END $$;
