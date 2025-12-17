-- Database initialization script for production deployment
-- This script sets up the initial database configuration

-- Create database if it doesn't exist (handled by Docker)
-- CREATE DATABASE IF NOT EXISTS drawings;

-- Set database configuration for optimal performance
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application user with limited privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_user_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE drawings TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT CREATE ON SCHEMA public TO app_user;

-- Create indexes for better performance (will be created by Alembic migrations)
-- These are just placeholders for reference

-- Performance monitoring view
CREATE OR REPLACE VIEW performance_stats AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public';

-- Backup and maintenance functions
CREATE OR REPLACE FUNCTION cleanup_old_analyses(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean up old analysis results older than specified days
    DELETE FROM interpretability_results 
    WHERE created_timestamp < NOW() - INTERVAL '1 day' * days_old;
    
    DELETE FROM anomaly_analyses 
    WHERE analysis_timestamp < NOW() - INTERVAL '1 day' * days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log the cleanup
    INSERT INTO maintenance_log (operation, affected_rows, timestamp)
    VALUES ('cleanup_old_analyses', deleted_count, NOW());
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create maintenance log table
CREATE TABLE IF NOT EXISTS maintenance_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(100) NOT NULL,
    affected_rows INTEGER,
    timestamp TIMESTAMP DEFAULT NOW(),
    details TEXT
);

-- Create function to update statistics
CREATE OR REPLACE FUNCTION update_table_statistics()
RETURNS VOID AS $$
BEGIN
    ANALYZE drawings;
    ANALYZE drawing_embeddings;
    ANALYZE anomaly_analyses;
    ANALYZE interpretability_results;
    ANALYZE age_group_models;
    
    INSERT INTO maintenance_log (operation, timestamp)
    VALUES ('update_statistics', NOW());
END;
$$ LANGUAGE plpgsql;

-- Set up automatic statistics updates
-- This would typically be done via cron or pg_cron extension