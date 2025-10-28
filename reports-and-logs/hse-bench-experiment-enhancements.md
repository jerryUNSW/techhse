# HSE-bench Experiment Enhancements

## Overview
This document outlines planned enhancements for the HSE-bench privacy-preserving mechanism experiments, including detailed data storage and task-type analysis capabilities.

## Current Status
- **Current Experiment**: Running regulation category (448 questions) with 6 mechanisms across 3 epsilon values
- **Current Data Storage**: Summary statistics only (accuracy counts, total questions)
- **Current Analysis**: Basic accuracy comparison across mechanisms

## Planned Enhancements

### 1. Enhanced Data Storage in JSON Files

#### Current JSON Structure (Summary Only)
```json
{
  "experiment_type": "HSE-bench",
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "remote_model": "deepseek-chat",
  "num_samples": -1,
  "epsilon_values": [1.0, 2.0, 3.0],
  "start_time": "2025-09-29 20:50:07.641387",
  "results": {
    "1.0": {
      "old_phrase_dp_local_cot_correct": 15,
      "inferdpt_local_cot_correct": 15,
      "santext_local_cot_correct": 14,
      "total_questions": 20
    }
  },
  "shared_results": {
    "local_alone_correct": 15,
    "non_private_cot_correct": 16,
    "purely_remote_correct": 16,
    "total_questions": 20
  }
}
```

#### Enhanced JSON Structure (Detailed Data)
```json
{
  "experiment_type": "HSE-bench",
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "remote_model": "deepseek-chat",
  "num_samples": -1,
  "epsilon_values": [1.0, 2.0, 3.0],
  "start_time": "2025-09-29 20:50:07.641387",
  "category": "regulation",
  "task_types": ["rule_recall", "rule_application", "issue_spotting", "rule_conclusion"],
  "detailed_results": [
    {
      "question_id": 1,
      "task_type": "rule_recall",
      "original_question": "During a routine inspection at a manufacturing plant...",
      "options": ["A. 29 CFR Part 1910.212", "B. 29 CFR Part 1926", "C. OSHA Act, Section 5(a)(1)", "D. NIOSH guidelines"],
      "correct_answer": "A",
      "reference": "US - Code of Federal Regulations Title 29 Labor Volume Subtitle B Regulations Relating to Labor",
      "scenarios": {
        "shared_results": {
          "local_alone": {
            "response": "A",
            "is_correct": true,
            "response_time": 2.3
          },
          "non_private_cot": {
            "cot_guidance": "To answer this question, I need to consider the specific regulation that applies to machine guarding requirements...",
            "response": "A", 
            "is_correct": true,
            "response_time": 4.1
          },
          "purely_remote": {
            "response": "A",
            "is_correct": true,
            "response_time": 1.8
          }
        },
        "epsilon_results": {
          "1.0": {
            "old_phrase_dp_local_cot": {
              "perturbed_question": "During a routine inspection at a manufacturing facility...",
              "cot_guidance": "To answer this question, I need to consider the specific regulation that applies to machine safety requirements...",
              "response": "A",
              "is_correct": true,
              "response_time": 4.5
            },
            "inferdpt_local_cot": {
              "perturbed_question": "During a routine inspection at a manufacturing plant...",
              "cot_guidance": "To answer this question, I need to consider the specific regulation that applies to equipment maintenance requirements...",
              "response": "A",
              "is_correct": true,
              "response_time": 4.2
            },
            "santext_local_cot": {
              "perturbed_question": "During a routine inspection at a manufacturing facility...",
              "cot_guidance": "To answer this question, I need to consider the specific regulation that applies to machine safety requirements...",
              "response": "A",
              "is_correct": true,
              "response_time": 4.3
            }
          },
          "2.0": { /* Similar structure for epsilon 2.0 */ },
          "3.0": { /* Similar structure for epsilon 3.0 */ }
        }
      }
    }
  ],
  "summary_results": {
    "shared_results": { /* Current summary structure */ },
    "epsilon_results": { /* Current summary structure */ }
  }
}
```

### 2. Task-Type Analysis Capabilities

#### Planned Analysis Framework
```python
def analyze_by_task_type(detailed_results):
    """Analyze mechanism performance by IRAC task type."""
    
    task_type_analysis = {
        "rule_recall": {
            "total_questions": 112,
            "mechanisms": {
                "local_alone": {"accuracy": 0.0, "correct": 0, "total": 0},
                "non_private_cot": {"accuracy": 0.0, "correct": 0, "total": 0},
                "purely_remote": {"accuracy": 0.0, "correct": 0, "total": 0},
                "phrasedp_1.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "phrasedp_2.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "phrasedp_3.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "inferdpt_1.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "inferdpt_2.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "inferdpt_3.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "santext_1.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "santext_2.0": {"accuracy": 0.0, "correct": 0, "total": 0},
                "santext_3.0": {"accuracy": 0.0, "correct": 0, "total": 0}
            }
        },
        "rule_application": { /* Similar structure */ },
        "issue_spotting": { /* Similar structure */ },
        "rule_conclusion": { /* Similar structure */ }
    }
    
    return task_type_analysis
```

#### Planned Visualizations
1. **Task-Type Performance Heatmap**
   - X-axis: Mechanisms (Local, CoT, PhraseDP, InferDPT, SANTEXT+, Remote)
   - Y-axis: Task Types (rule_recall, rule_application, issue_spotting, rule_conclusion)
   - Color: Accuracy percentage

2. **Epsilon vs Task-Type Analysis**
   - X-axis: Epsilon values (1.0, 2.0, 3.0)
   - Y-axis: Task Types
   - Lines: Different privacy mechanisms
   - Shows privacy-utility trade-off by task complexity

3. **Perturbation Quality Analysis**
   - Compare original vs perturbed questions
   - Analyze CoT quality across different mechanisms
   - Measure semantic similarity preservation

### 3. Implementation Plan

#### Phase 1: Enhanced Data Storage (Future)
- Modify `test-hse-bench.py` to store detailed results
- Add question metadata (task_type, reference, etc.)
- Store perturbed questions and CoT responses
- Include response times and model confidence scores

#### Phase 2: Task-Type Analysis (Future)
- Create analysis scripts for task-type breakdown
- Generate task-specific performance reports
- Create visualizations for mechanism comparison by task type

#### Phase 3: Advanced Analytics (Future)
- Perturbation quality metrics
- CoT effectiveness analysis
- Privacy-utility trade-off analysis by task complexity

### 4. Data Storage Requirements

#### What to Store (Future Implementation)
- ✅ **Original questions** (full text)
- ✅ **Perturbed questions** (privacy-sanitized versions)
- ✅ **CoT responses** (Chain of Thought guidance)
- ✅ **Model responses** (final answers)
- ✅ **Task type** (rule_recall, rule_application, issue_spotting, rule_conclusion)
- ✅ **Question metadata** (reference, category, etc.)
- ✅ **Response times** (performance metrics)
- ✅ **Confidence scores** (if available)
- ✅ **Perturbation quality metrics** (semantic similarity, etc.)

#### Current Implementation (Running Experiment)
- ✅ **Summary statistics only** (accuracy counts, totals)
- ❌ **No detailed question data**
- ❌ **No perturbed questions**
- ❌ **No CoT responses**
- ❌ **No task-type breakdown**

### 5. Analysis Capabilities (Future)

#### Task-Type Performance Analysis
```python
# Example analysis functions to implement
def get_task_type_performance(detailed_results, task_type):
    """Get performance metrics for a specific task type."""
    pass

def compare_mechanisms_by_task_type(detailed_results):
    """Compare all mechanisms across all task types."""
    pass

def analyze_perturbation_quality(detailed_results):
    """Analyze quality of privacy perturbations."""
    pass

def generate_task_type_report(detailed_results):
    """Generate comprehensive task-type analysis report."""
    pass
```

#### Expected Insights
1. **Which mechanisms work best for which task types?**
   - Rule recall vs rule application vs issue spotting vs rule conclusion
   
2. **How does privacy affect different types of legal reasoning?**
   - Epsilon impact on task-specific performance
   
3. **What are the characteristics of effective perturbations?**
   - Quality of CoT guidance across mechanisms
   - Semantic preservation in perturbed questions

### 6. File Structure (Future)
```
QA-results/hse-bench/
├── hse_bench_detailed_results_regulation_20250929.json  # Enhanced JSON
├── task_type_analysis_regulation_20250929.json         # Task-type breakdown
├── perturbation_quality_analysis_20250929.json         # Perturbation metrics
└── plots/
    ├── task_type_performance_heatmap.png
    ├── epsilon_vs_task_type.png
    └── perturbation_quality_analysis.png
```

## Notes
- **Current experiment continues with existing summary-only storage**
- **Enhanced storage will be implemented in future experiments**
- **Task-type analysis capabilities will be added post-experiment**
- **All enhancements are planned for future iterations, not current running experiment**

