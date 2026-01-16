-- =============================================================================
-- AlphaTrade System - TimescaleDB Hypertable Placeholder
-- =============================================================================
-- Hypertable conversion is handled by Alembic migrations.
-- This file is kept for compatibility but does not perform any conversions.
--
-- See: quant_trading_system/database/migrations/versions/002_timescaledb_hypertables.py
--
-- Run order: 02 (after extensions are enabled)
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Hypertable conversion will be handled by Alembic migrations';
END $$;
