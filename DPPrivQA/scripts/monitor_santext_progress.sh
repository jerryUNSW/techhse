#!/bin/bash
# Monitor SANTEXT+ experiment progress

echo "SANTEXT+ Experiment Progress Monitor"
echo "===================================="
echo ""

cd /Users/jerry/techhse/DPPrivQA

# Check running processes
echo "Running Processes:"
ps aux | grep "run_mmlu_inferdpt_santext.*santext" | grep -v grep | wc -l | xargs echo "  Active experiments:"
echo ""

# Check database progress
echo "Database Progress (ε=2.0):"
echo "-------------------------"

for dataset in professional_law professional_medicine clinical_knowledge college_medicine; do
    case $dataset in
        professional_law) expected=200 ;;
        professional_medicine) expected=272 ;;
        clinical_knowledge) expected=265 ;;
        college_medicine) expected=173 ;;
    esac
    
    count=$(sqlite3 exp-results/results.db "SELECT COUNT(*) FROM mmlu_${dataset}_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0;" 2>/dev/null || echo "0")
    correct=$(sqlite3 exp-results/results.db "SELECT SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) FROM mmlu_${dataset}_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0;" 2>/dev/null || echo "0")
    
    if [ "$count" = "" ]; then count=0; fi
    if [ "$correct" = "" ]; then correct=0; fi
    
    percent=$((count * 100 / expected))
    accuracy="N/A"
    if [ "$count" -gt 0 ]; then
        accuracy=$(echo "scale=1; $correct * 100 / $count" | bc)
    fi
    
    echo "  ${dataset}: ${count}/${expected} (${percent}%) - Accuracy: ${accuracy}%"
done

echo ""
echo "Total Progress:"
total=$(sqlite3 exp-results/results.db "SELECT COUNT(*) FROM (SELECT COUNT(*) FROM mmlu_professional_law_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0 UNION ALL SELECT COUNT(*) FROM mmlu_professional_medicine_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0 UNION ALL SELECT COUNT(*) FROM mmlu_clinical_knowledge_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0 UNION ALL SELECT COUNT(*) FROM mmlu_college_medicine_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0);" 2>/dev/null || echo "0")
echo "  Total completed: ${total}/910"

echo ""
echo "Recent Activity (last 5 results per dataset):"
for dataset in professional_law professional_medicine clinical_knowledge college_medicine; do
    echo "  ${dataset}:"
    sqlite3 exp-results/results.db "SELECT question_idx, is_correct, datetime(created_at, 'localtime') FROM mmlu_${dataset}_epsilon_dependent_results WHERE mechanism='santext' AND epsilon=2.0 ORDER BY created_at DESC LIMIT 5;" 2>/dev/null | awk -F'|' '{printf "    Q%d: %s at %s\n", $1, ($2==1 ? "✓" : "✗"), $3}' || echo "    No results yet"
done

