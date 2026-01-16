-- =============================================================================
-- AlphaTrade System - Database Extensions
-- =============================================================================
-- This script enables required PostgreSQL extensions for TimescaleDB and
-- time-series data management.
--
-- Run order: 01 (first)
-- =============================================================================

-- Enable TimescaleDB extension (required for hypertables)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID generation functions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable btree_gist for exclusion constraints and range queries
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Log successful extension creation
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB and required extensions enabled successfully';
END $$;
