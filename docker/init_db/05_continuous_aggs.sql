-- =============================================================================
-- AlphaTrade System - TimescaleDB Continuous Aggregates Placeholder
-- =============================================================================
-- Continuous aggregates are created after tables exist and data is available.
-- This file is kept for compatibility but does not create aggregates.
--
-- Continuous aggregates will be created later via a separate migration or script
-- once the system has data to aggregate.
--
-- Available aggregate types (to be created later):
-- - ohlcv_hourly: Hourly OHLCV bars from 15-minute data
-- - ohlcv_daily: Daily OHLCV bars from 15-minute data
-- - feature_daily_stats: Daily feature statistics per symbol
--
-- Run order: 05 (after all tables and policies are configured)
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Continuous aggregates will be created after data is available';
END $$;
