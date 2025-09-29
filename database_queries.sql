-- Useful SQLite queries for tech4hse_results.db

-- 1. View all experiments
SELECT id, experiment_type, description, created_at FROM experiments;

-- 2. View PII protection results summary
SELECT mechanism, epsilon, overall_protection, email_protection, phone_protection, address_protection, name_protection
FROM pii_protection_results 
ORDER BY mechanism, epsilon;

-- 3. View best performing mechanisms for PII protection (epsilon 2.0)
SELECT mechanism, overall_protection, email_protection, phone_protection, address_protection, name_protection
FROM pii_protection_results 
WHERE epsilon = 2.0
ORDER BY overall_protection DESC;

-- 4. View MedQA results (when available)
SELECT mechanism, epsilon, COUNT(*) as total_questions,
       SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
       ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 3) as accuracy
FROM medqa_results
GROUP BY mechanism, epsilon
ORDER BY mechanism, epsilon;

-- 5. View sample PII protection examples
SELECT p.mechanism, p.epsilon, s.row_index, 
       substr(s.original_text, 1, 100) || '...' as original_preview,
       substr(s.sanitized_text, 1, 100) || '...' as sanitized_preview
FROM pii_protection_samples s
JOIN pii_protection_results p ON s.protection_result_id = p.id
WHERE p.mechanism = 'PhraseDP' AND p.epsilon = 2.0
LIMIT 5;

-- 6. Count total records in each table
SELECT 'experiments' as table_name, COUNT(*) as record_count FROM experiments
UNION ALL
SELECT 'medqa_results', COUNT(*) FROM medqa_results
UNION ALL
SELECT 'pii_protection_results', COUNT(*) FROM pii_protection_results
UNION ALL
SELECT 'pii_protection_samples', COUNT(*) FROM pii_protection_samples;

-- 7. View protection rates by PII type (epsilon 2.0)
SELECT mechanism,
       email_protection,
       phone_protection, 
       address_protection,
       name_protection,
       overall_protection
FROM pii_protection_results 
WHERE epsilon = 2.0
ORDER BY overall_protection DESC;

-- 8. Find mechanisms with best email protection
SELECT mechanism, epsilon, email_protection
FROM pii_protection_results 
ORDER BY email_protection DESC
LIMIT 10;

-- 9. View experiment details with mechanism counts
SELECT e.id, e.experiment_type, e.description,
       CASE 
         WHEN e.experiment_type = 'medqa_ume' THEN (SELECT COUNT(*) FROM medqa_results WHERE experiment_id = e.id)
         WHEN e.experiment_type = 'pii_protection' THEN (SELECT COUNT(*) FROM pii_protection_results WHERE experiment_id = e.id)
       END as result_count
FROM experiments e;

-- 10. Insert a new MedQA result (example)
-- INSERT INTO medqa_results (experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer)
-- VALUES (2, 1, 'PhraseDP', 2.0, 1, 'A', 'A');
