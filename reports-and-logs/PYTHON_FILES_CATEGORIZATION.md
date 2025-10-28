# Python Files Categorization - Root Directory

## 📊 CORE EXPERIMENT SCRIPTS (Keep - Active)

### PII Protection Experiments
- `create_results_database.py` - Creates SQLite database for results ✅
- `query_results_database.py` - Query interface for database ✅
- `update_database_with_clusant.py` - Update DB with CluSanT results ✅
- `db_browser.py` - Database browser interface ✅
- `download_pii_dataset.py` - Download PII dataset ✅

### Sanitization Methods (Core Implementations)
- `inferdpt.py` - InferDPT implementation ✅
- `santext_integration.py` - SANTEXT+ integration ✅
- `utils.py` - Utility functions (PhraseDP, etc.) ✅
- `dp_sanitizer.py` - DP sanitizer interface ✅

### Main QA Testing
- `test-medqa-usmle-4-options.py` - MedQA-UME main experiment ✅
- `test-medmcqa.py` - MedMCQA main experiment ✅

### Plotting (Final)
- `generate_epsilon_line_plots.py` - Epsilon trend line plots ✅
- `generate_unified_ppi_plots.py` - Unified PPI plots ✅
- `update_plots_pastel_colors.py` - Plot styling updates ✅
- `make_fig1_text_larger.py` - Figure 1 text sizing ✅

### BibTeX Management
- `fetch_bibtex_final.py` - Final BibTeX fetcher ✅

---

## 🧪 TEST & DEBUG FILES (Review/Remove)

### Old Test Files (Likely Outdated)
- `test.py` - Generic test file ❌
- `simple_debug.py` - Debug script ❌
- `test_fixed_batch.py` - Batch test ❌
- `debug_batch_perturb.py` - Debug perturbation ❌
- `debug_custext_comprehensive.py` - CusText debug ❌
- `debug_nebius_models.py` - Nebius API debug ❌

### Mechanism Testing (Old)
- `test_phrase_dp.py` - PhraseDP test ❌
- `test_old_phrasedp.py` - Old PhraseDP test ❌
- `test_old_phrasedp_integration.py` - Old integration test ❌
- `test_openai_phrasedp.py` - OpenAI PhraseDP test ❌
- `test_phrase_dp_comparison.py` - PhraseDP comparison ❌
- `test_inferdpt.py` - InferDPT test ❌
- `test_santext_demo.py` - SANTEXT demo ❌
- `test_clusant_fix.py` - CluSanT fix test ❌

### Sanitization Method Tests (Multiple Versions)
- `test_5_sanitization_methods.py` ❌
- `test_5_correct_sanitization_methods.py` ❌
- `test_5_official_sanitization_methods.py` ❌
- `test_5_real_sanitization_methods.py` ❌
- `sanitization_methods.py` - May be outdated ⚠️

### Epsilon Testing (Old Experiments)
- `epsilon_experiment.py` ❌
- `run_epsilon_test.py` ❌
- `run_scaled_epsilon_test.py` ❌
- `test_epsilon_comparison.py` ❌
- `test_epsilon_comparison_scaled.py` ❌
- `test_extended_epsilon_comparison.py` ❌
- `efficient_epsilon_test.py` ❌
- `compare_epsilon_experiments.py` ❌
- `ten_question_epsilon_test.py` ⚠️ (May be used for analysis)

### Scenario Testing
- `run_scenario_test.py` ❌
- `test_scenario_3_2_only.py` ❌
- `test-gpt5-mini-scenarios-2-4.py` ❌

---

## 📈 ANALYSIS & MONITORING (Review)

### Progress Monitoring (Outdated if experiments complete)
- `monitor_progress.py` ❌
- `monitor_progress_2.py` ❌
- `monitor_epsilon2_progress.py` ❌
- `hourly_progress_report.py` ❌
- `daily_email_summary.py` ⚠️ (May be useful)
- `check_hse_bench_progress.py` ❌
- `check_hse_bench_progress_simple.py` ❌

### Analysis Scripts (Check if still used)
- `experiment_analysis.py` ⚠️
- `analyze_medqa_patterns.py` ⚠️
- `epsilon_trend_investigation_report.py` ⚠️
- `combined_experiment_report.py` ⚠️
- `semantic_similarity_analysis.py` ⚠️
- `exponential_mechanism_analysis.py` ⚠️
- `exponential_mechanism_simple.py` ⚠️

### Data Extraction
- `extract_phrase_dp_data.py` ❌
- `extract_17_questions_results.py` ❌

---

## 🔍 EVALUATION SCRIPTS (Review)

### Privacy Evaluation
- `privacy_evaluation.py` ⚠️
- `realistic_privacy_evaluation.py` ⚠️
- `ner_pii_privacy_evaluation.py` ⚠️
- `privacy_visualization.py` ⚠️
- `unified_gpt_inference_attack.py` ⚠️

### BERT Similarity
- `bert_similarity_evaluation.py` ⚠️

### PII Protection Experiments (Old Versions)
- `pii_protection_experiment.py` ❌ (Superseded by version in experiment_results/ppi-protection/)
- `pii_protection_experiment_row_by_row.py` ❌
- `pii_protection_experiment_remaining_tmp.py` ❌
- `cus_text_ppi_protection_experiment.py` ⚠️ (May contain useful functions)

---

## 📊 CANDIDATE GENERATION & SAMPLING (Review)

- `one_question_equal_band_email.py` ⚠️
- `one_question_lowband_test.py` ❌
- `fixed_pool_sampling_analysis.py` ⚠️
- `show_selected_candidates.py` ⚠️

---

## 📚 OTHER QA TESTS (Review)

- `test-hotpot-QA.py` ❌
- `testing_medical_qa.py` ❌
- `main_qa.py` ⚠️
- `test-hse-bench-gpt.py` ❌
- `test-hse-bench-deepseek.py` ❌

---

## 🗄️ OLD DATABASE QUERIES (Review)

- `query_updated_database.py` ❌ (Likely duplicate of query_results_database.py)
- `simple_database_query.py` ❌

---

## 🎨 PLOTTING (Old Versions)

- `create_ppi_comparison_plots.py` ⚠️ (Check if superseded)
- `update_fig1_van_gogh_colors.py` ⚠️
- `update_fig1_larger_text_height.py` ⚠️

---

## 🔧 UTILITY SCRIPTS

- `imports_and_init.py` ⚠️
- `prompt_loader.py` ⚠️
- `convert_pdf_to_txt.py` ⚠️

---

## 📊 CALCULATION SCRIPTS

- `calculate_corrected_results.py` ⚠️
- `calculate_quota_unaffected_results.py` ⚠️
- `analyze_quota_unaffected_mechanisms.py` ⚠️

---

## 🔄 COMPARISON SCRIPTS

- `run_comparison.py` ❌
- `privacy_mechanisms_comparison.py` ⚠️
- `comprehensive_method_test.py` ❌

---

## 📋 BIBTEX (Old Versions)

- `fetch_bibtex_entries.py` ❌ (Superseded by fetch_bibtex_final.py)
- `fetch_bibtex_entries_improved.py` ❌ (Superseded by fetch_bibtex_final.py)

---

## 🎯 SANTEXT ANALYSIS

- `explain_santext_randomness.py` ⚠️

