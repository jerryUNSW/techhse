# MedMCQA Experiment Parallelization Analysis

## Current Execution Flow (Sequential)

### Per Question Processing:
1. **Epsilon-Independent Scenarios** (run once, shared across all epsilon):
   - Scenario 1: Purely Local Model
   - Scenario 2: Non-Private Local + Remote CoT  
   - Scenario 4: Purely Remote Model

2. **Epsilon-Dependent Scenarios** (run for each epsilon: 1.0, 2.0, 3.0):
   - Scenario 3.0: PhraseDP (Old)
   - Scenario 3.2: InferDPT
   - Scenario 3.3: SANTEXT+

## Parallelization Opportunities

### 🚀 **Level 1: Epsilon-Independent Scenarios (High Impact)**

**Current**: Sequential execution
```python
scenario_1_result = run_scenario_1_purely_local(...)
scenario_2_result = run_scenario_2_non_private_cot(...)  
scenario_4_result = run_scenario_4_purely_remote(...)
```

**Parallelized**: All 3 scenarios can run simultaneously
- **Scenario 1**: Uses local model only
- **Scenario 2**: Uses local model + remote CoT
- **Scenario 4**: Uses remote model only
- **No Dependencies**: Each scenario is independent

**Time Savings**: ~66% reduction (3 scenarios → 1 time unit)

### 🚀 **Level 2: Epsilon-Dependent Scenarios (Medium Impact)**

**Current**: Sequential execution for each epsilon
```python
for epsilon in [1.0, 2.0, 3.0]:
    scenario_3_0_result = run_scenario_3_private_local_cot_with_epsilon(..., epsilon)
    scenario_3_2_result = run_scenario_3_private_local_cot_with_epsilon(..., epsilon)  
    scenario_3_3_result = run_scenario_3_private_local_cot_with_epsilon(..., epsilon)
```

**Parallelized**: All 3 privacy mechanisms can run simultaneously for each epsilon
- **Scenario 3.0**: PhraseDP (Old)
- **Scenario 3.2**: InferDPT
- **Scenario 3.3**: SANTEXT+
- **No Dependencies**: Each mechanism is independent

**Time Savings**: ~66% reduction (3 mechanisms → 1 time unit)

### 🚀 **Level 3: Cross-Epsilon Parallelization (High Impact)**

**Current**: Sequential execution across epsilon values
```python
for epsilon in [1.0, 2.0, 3.0]:
    # Run all scenarios for this epsilon
```

**Parallelized**: All epsilon values can be processed simultaneously
- **Epsilon 1.0**: All scenarios
- **Epsilon 2.0**: All scenarios  
- **Epsilon 3.0**: All scenarios
- **No Dependencies**: Different epsilon values are independent

**Time Savings**: ~66% reduction (3 epsilon values → 1 time unit)

### 🚀 **Level 4: Question-Level Parallelization (Maximum Impact)**

**Current**: Sequential processing of questions
```python
for question in questions:
    # Process all scenarios for this question
```

**Parallelized**: Multiple questions can be processed simultaneously
- **Batch Size**: 5-10 questions in parallel
- **Resource Management**: Balance CPU, memory, and API limits
- **Dependencies**: Each question is independent

**Time Savings**: ~80-90% reduction (10 questions → 1 time unit)

## Implementation Strategy

### **Phase 1: Epsilon-Independent Parallelization**
```python
import concurrent.futures
import threading

def run_epsilon_independent_scenarios_parallel(local_client, model_name, remote_client, question, options, correct_answer):
    """Run scenarios 1, 2, 4 in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three scenarios
        future_1 = executor.submit(run_scenario_1_purely_local, local_client, model_name, question, options, correct_answer)
        future_2 = executor.submit(run_scenario_2_non_private_cot, local_client, model_name, remote_client, question, options, correct_answer)
        future_4 = executor.submit(run_scenario_4_purely_remote, remote_client, question, options, correct_answer)
        
        # Collect results
        scenario_1_result = future_1.result()
        scenario_2_result = future_2.result()
        scenario_4_result = future_4.result()
        
    return scenario_1_result, scenario_2_result, scenario_4_result
```

### **Phase 2: Epsilon-Dependent Parallelization**
```python
def run_epsilon_dependent_scenarios_parallel(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, epsilon):
    """Run scenarios 3.0, 3.2, 3.3 in parallel for a single epsilon."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three privacy mechanisms
        future_3_0 = executor.submit(run_scenario_3_private_local_cot_with_epsilon, local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'phrasedp', True, epsilon)
        future_3_2 = executor.submit(run_scenario_3_private_local_cot_with_epsilon, local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'inferdpt', False, epsilon)
        future_3_3 = executor.submit(run_scenario_3_private_local_cot_with_epsilon, local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'santext', False, epsilon)
        
        # Collect results
        scenario_3_0_result = future_3_0.result()
        scenario_3_2_result = future_3_2.result()
        scenario_3_3_result = future_3_3.result()
        
    return scenario_3_0_result, scenario_3_2_result, scenario_3_3_result
```

### **Phase 3: Cross-Epsilon Parallelization**
```python
def run_all_epsilon_scenarios_parallel(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, epsilon_values):
    """Run all epsilon-dependent scenarios in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all epsilon values
        futures = {}
        for epsilon in epsilon_values:
            future = executor.submit(run_epsilon_dependent_scenarios_parallel, local_client, model_name, remote_client, sbert_model, question, options, correct_answer, epsilon)
            futures[epsilon] = future
        
        # Collect results
        results = {}
        for epsilon, future in futures.items():
            results[epsilon] = future.result()
            
    return results
```

### **Phase 4: Question-Level Parallelization**
```python
def run_questions_parallel(questions, batch_size=5):
    """Run multiple questions in parallel."""
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all questions in batch
            futures = []
            for question in batch:
                future = executor.submit(process_single_question, question)
                futures.append(future)
            
            # Collect results
            batch_results = [future.result() for future in futures]
            results.extend(batch_results)
    
    return results
```

## Expected Performance Improvements

### **Current Timeline (500 questions):**
- **Per Question**: ~4 minutes
- **Total Time**: ~33 hours
- **Completion**: Tomorrow evening

### **With Level 1+2 Parallelization:**
- **Per Question**: ~1.5 minutes (66% reduction)
- **Total Time**: ~12 hours
- **Completion**: Tonight

### **With Level 1+2+3 Parallelization:**
- **Per Question**: ~0.5 minutes (87% reduction)
- **Total Time**: ~4 hours
- **Completion**: This evening

### **With All Levels (1+2+3+4):**
- **Per Question**: ~0.1 minutes (97% reduction)
- **Total Time**: ~1 hour
- **Completion**: This afternoon

## Resource Considerations

### **API Rate Limits:**
- **DeepSeek**: Monitor API rate limits
- **Nebius**: Monitor local model rate limits
- **Threading**: Use ThreadPoolExecutor for I/O bound operations

### **Memory Usage:**
- **Current**: ~21GB for single process
- **Parallel**: ~50-100GB for 5-10 parallel processes
- **Management**: Implement memory monitoring and cleanup

### **Error Handling:**
- **Retry Logic**: Implement exponential backoff
- **Failure Isolation**: One failed scenario shouldn't stop others
- **Progress Tracking**: Maintain incremental writes during parallel execution

## Implementation Priority

1. **Phase 1** (Epsilon-Independent): **Immediate** - Easy to implement, high impact
2. **Phase 2** (Epsilon-Dependent): **Next** - Medium complexity, good impact  
3. **Phase 3** (Cross-Epsilon): **Later** - Higher complexity, excellent impact
4. **Phase 4** (Question-Level): **Future** - Complex, maximum impact

## Conclusion

**Immediate 66% speedup** is achievable with minimal code changes by parallelizing epsilon-independent scenarios. This would reduce your 500-sample experiment from ~33 hours to ~12 hours, completing tonight instead of tomorrow evening.
