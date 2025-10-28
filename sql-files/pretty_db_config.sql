-- Pretty Database Configuration for SQLite
-- Run this to set up pretty formatting: sqlite3 tech4hse_results.db < pretty_db_config.sql

-- Set pretty table formatting
.mode table
.headers on

-- Set column widths for better readability
.width 12 8 12 12 12 12 12

-- Create a view for easy access to pretty PII results
CREATE VIEW IF NOT EXISTS pretty_pii_results AS
SELECT 
    mechanism as 'Mechanism',
    epsilon as 'Epsilon',
    printf('%.1f%%', overall_protection * 100) as 'Overall %',
    printf('%.1f%%', email_protection * 100) as 'Email %',
    printf('%.1f%%', phone_protection * 100) as 'Phone %',
    printf('%.1f%%', address_protection * 100) as 'Address %',
    printf('%.1f%%', name_protection * 100) as 'Name %',
    num_samples as 'Samples'
FROM pii_protection_results;

-- Create a view for MedQA results with pretty formatting
CREATE VIEW IF NOT EXISTS pretty_medqa_results AS
SELECT 
    mechanism as 'Mechanism',
    epsilon as 'Epsilon',
    COUNT(*) as 'Total',
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as 'Correct',
    printf('%.1f%%', AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100) as 'Accuracy %'
FROM medqa_results
GROUP BY mechanism, epsilon;

-- Create a view for experiment summary
CREATE VIEW IF NOT EXISTS pretty_experiments AS
SELECT 
    id as 'ID',
    experiment_type as 'Type',
    substr(description, 1, 50) || '...' as 'Description',
    datetime(created_at) as 'Created'
FROM experiments;

-- Show the pretty views
SELECT 'PII Protection Results (All)' as 'View';
SELECT * FROM pretty_pii_results LIMIT 5;

SELECT 'Best Mechanisms (ε=2.0)' as 'View';
SELECT * FROM pretty_pii_results WHERE 'Epsilon' = 2.0 ORDER BY 'Overall %' DESC;

SELECT 'Experiments Summary' as 'View';
SELECT * FROM pretty_experiments;
