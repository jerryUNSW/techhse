-- Pretty SQLite Queries for Tech4HSE Results Database
-- Run these with: sqlite3 tech4hse_results.db < pretty_sqlite_queries.sql

-- Set pretty formatting
.mode table
.headers on
.width 12 8 12 12 12 12 12

-- 1. PII Protection Results with Percentage Formatting
SELECT 
    mechanism as 'Mechanism',
    epsilon as 'Epsilon',
    printf('%.1f%%', overall_protection * 100) as 'Overall %',
    printf('%.1f%%', email_protection * 100) as 'Email %',
    printf('%.1f%%', phone_protection * 100) as 'Phone %',
    printf('%.1f%%', address_protection * 100) as 'Address %',
    printf('%.1f%%', name_protection * 100) as 'Name %'
FROM pii_protection_results 
ORDER BY mechanism, epsilon;

-- 2. Best Performing Mechanisms (Epsilon 2.0)
SELECT 
    mechanism as 'Mechanism',
    printf('%.1f%%', overall_protection * 100) as 'Overall %',
    printf('%.1f%%', email_protection * 100) as 'Email %',
    printf('%.1f%%', phone_protection * 100) as 'Phone %',
    printf('%.1f%%', address_protection * 100) as 'Address %',
    printf('%.1f%%', name_protection * 100) as 'Name %',
    num_samples as 'Samples'
FROM pii_protection_results 
WHERE epsilon = 2.0
ORDER BY overall_protection DESC;

-- 3. MedQA Results (when available) with Pretty Formatting
SELECT 
    mechanism as 'Mechanism',
    epsilon as 'Epsilon',
    total_questions as 'Total',
    correct_answers as 'Correct',
    printf('%.1f%%', accuracy * 100) as 'Accuracy %'
FROM (
    SELECT mechanism, epsilon, 
           COUNT(*) as total_questions,
           SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
           AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy
    FROM medqa_results
    GROUP BY mechanism, epsilon
    ORDER BY mechanism, epsilon
);

-- 4. Experiment Summary
SELECT 
    id as 'ID',
    experiment_type as 'Type',
    description as 'Description',
    created_at as 'Created'
FROM experiments
ORDER BY created_at DESC;

-- 5. Protection Rate Comparison (All Epsilon Values)
SELECT 
    epsilon as 'Epsilon',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'PhraseDP' THEN overall_protection ELSE NULL END) * 100) as 'PhraseDP %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'InferDPT' THEN overall_protection ELSE NULL END) * 100) as 'InferDPT %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'SANTEXT+' THEN overall_protection ELSE NULL END) * 100) as 'SANTEXT+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CusText+' THEN overall_protection ELSE NULL END) * 100) as 'CusText+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CluSanT' THEN overall_protection ELSE NULL END) * 100) as 'CluSanT %'
FROM pii_protection_results
GROUP BY epsilon
ORDER BY epsilon;

-- 6. PII Type Performance Comparison (Epsilon 2.0)
SELECT 
    'Email' as 'PII Type',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'PhraseDP' THEN email_protection ELSE NULL END) * 100) as 'PhraseDP %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'InferDPT' THEN email_protection ELSE NULL END) * 100) as 'InferDPT %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'SANTEXT+' THEN email_protection ELSE NULL END) * 100) as 'SANTEXT+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CusText+' THEN email_protection ELSE NULL END) * 100) as 'CusText+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CluSanT' THEN email_protection ELSE NULL END) * 100) as 'CluSanT %'
FROM pii_protection_results WHERE epsilon = 2.0
UNION ALL
SELECT 
    'Phone' as 'PII Type',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'PhraseDP' THEN phone_protection ELSE NULL END) * 100) as 'PhraseDP %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'InferDPT' THEN phone_protection ELSE NULL END) * 100) as 'InferDPT %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'SANTEXT+' THEN phone_protection ELSE NULL END) * 100) as 'SANTEXT+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CusText+' THEN phone_protection ELSE NULL END) * 100) as 'CusText+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CluSanT' THEN phone_protection ELSE NULL END) * 100) as 'CluSanT %'
FROM pii_protection_results WHERE epsilon = 2.0
UNION ALL
SELECT 
    'Address' as 'PII Type',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'PhraseDP' THEN address_protection ELSE NULL END) * 100) as 'PhraseDP %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'InferDPT' THEN address_protection ELSE NULL END) * 100) as 'InferDPT %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'SANTEXT+' THEN address_protection ELSE NULL END) * 100) as 'SANTEXT+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CusText+' THEN address_protection ELSE NULL END) * 100) as 'CusText+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CluSanT' THEN address_protection ELSE NULL END) * 100) as 'CluSanT %'
FROM pii_protection_results WHERE epsilon = 2.0
UNION ALL
SELECT 
    'Name' as 'PII Type',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'PhraseDP' THEN name_protection ELSE NULL END) * 100) as 'PhraseDP %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'InferDPT' THEN name_protection ELSE NULL END) * 100) as 'InferDPT %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'SANTEXT+' THEN name_protection ELSE NULL END) * 100) as 'SANTEXT+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CusText+' THEN name_protection ELSE NULL END) * 100) as 'CusText+ %',
    printf('%.1f%%', AVG(CASE WHEN mechanism = 'CluSanT' THEN name_protection ELSE NULL END) * 100) as 'CluSanT %'
FROM pii_protection_results WHERE epsilon = 2.0;

-- 7. Database Statistics
SELECT 
    'experiments' as 'Table',
    COUNT(*) as 'Records',
    'Experiment metadata' as 'Description'
FROM experiments
UNION ALL
SELECT 
    'medqa_results',
    COUNT(*),
    'Individual MedQA question results'
FROM medqa_results
UNION ALL
SELECT 
    'pii_protection_results',
    COUNT(*),
    'Aggregated PII protection rates'
FROM pii_protection_results
UNION ALL
SELECT 
    'pii_protection_samples',
    COUNT(*),
    'Individual PII protection examples'
FROM pii_protection_samples;
