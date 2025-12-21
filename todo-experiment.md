# Experiment TODO - MedQA-USMLE Dataset

## Missing Results Summary

### PhraseDP (ε=1.0)
- Questions Missing: 400
- Indices: 100-499

### PhraseDP (ε=2.0)
- Questions Missing: 400
- Indices: 100-499

### PhraseDP (ε=3.0)
- Questions Missing: 400
- Indices: 100-499

### PhraseDP+ (ε=1.0)
- Questions Missing: 142
- Indices: 358-499

### PhraseDP+ (ε=3.0)
- Questions Missing: 142
- Indices: 358-499

## Comprehensive Logging Requirements

### Logging Strategy

All experiments must record detailed workflow processes to enable:
- Reproducibility
- Debugging and error analysis
- Progress tracking
- Performance monitoring
- Failure recovery

### What to Log

#### 1. Experiment Metadata (Per Run)
- **Timestamp**: Start time, end time, duration
- **Configuration**: 
  - Dataset: MedQA-USMLE
  - Epsilon values tested
  - Local model name
  - Remote model name
  - Question index range (start_index, num_samples)
  - Mechanisms tested
- **Environment**: 
  - Python version
  - Package versions (transformers, sentence-transformers, etc.)
  - API endpoints used (Nebius, OpenAI)
  - System information

#### 2. Per-Question Logging
For each question processed, log:
- **Question ID**: Dataset index
- **Question Text**: Full question text
- **Options**: All multiple choice options
- **Correct Answer**: Ground truth
- **MetaMap Phrases**: Extracted phrases (if applicable)
- **Processing Steps**:
  - Sanitization input/output (for privacy mechanisms)
  - CoT generation (if applicable)
  - Local model inference
  - Remote model inference (if applicable)
- **Results**:
  - Predicted answer
  - Is correct (boolean)
  - Response text
  - Processing time per step
- **Errors**: Any exceptions or failures with full stack traces

#### 3. Mechanism-Specific Logging

**PhraseDP:**
- Sanitized question text
- Candidate paraphrases generated
- Similarity scores
- Selected candidate
- Epsilon value used
- Number of API calls made

**PhraseDP+:**
- All PhraseDP logging plus:
- MetaMap phrases used
- Medical mode settings
- Few-shot examples (if used)

**Local Model:**
- Model name
- Input prompt
- Response text
- Token usage
- Inference time

**Remote Model:**
- Model name
- CoT prompt
- CoT response
- Final answer prompt
- Final answer response
- Token usage
- API latency

#### 4. Progress Tracking
- Questions completed / total
- Current accuracy (running total)
- Time elapsed
- Estimated time remaining
- Questions per hour rate
- Error rate

#### 5. Error Handling
- Failed questions with error messages
- Retry attempts
- API failures (rate limits, timeouts, etc.)
- Model failures
- Network issues

### Logging Implementation

#### File-Based Logging
- **JSON Results File**: Structured results per question
  - Location: `exp/new-exp/medqa_usmle_*.json`
  - Format: Per-question results with all metadata
- **Text Log File**: Human-readable workflow log
  - Location: `exp/new-exp/logs/medqa_usmle_*.log`
  - Format: Timestamped entries for each step
- **Database**: Structured storage
  - Location: `data/tech4hse_results.db`
  - Tables: `medqa_results`, `medqa_detailed_results`

#### Log Levels
- **INFO**: Normal workflow steps, progress updates
- **DEBUG**: Detailed intermediate values, API requests/responses
- **WARNING**: Non-fatal issues, retries, fallbacks
- **ERROR**: Failures, exceptions, critical issues

### Workflow Process Documentation

#### Pre-Experiment Setup
1. Verify API keys (NEBIUS, OPENAI_API_KEY) in `.env`
2. Check model availability
3. Initialize database connection
4. Create output directories
5. Load configuration from `config.yaml`
6. Initialize privacy mechanisms (if needed)
7. Load Sentence-BERT model
8. Verify dataset access

#### Per-Question Workflow
1. **Load Question**
   - Load from dataset
   - Extract question text, options, correct answer
   - Extract MetaMap phrases (if applicable)
   - Log question metadata

2. **Scenario 1: Local Model (if needed)**
   - Format prompt with question and options
   - Call local model API
   - Extract answer
   - Log: input, output, time, tokens

3. **Scenario 2: Local + CoT (if needed)**
   - Generate CoT from remote model (original question)
   - Format prompt with question, options, and CoT
   - Call local model API
   - Extract answer
   - Log: CoT text, input, output, time

4. **Scenario 3: Privacy Mechanisms (PhraseDP/PhraseDP+)**
   - Apply sanitization to question
   - Log: original, sanitized, candidates, similarity scores
   - Generate CoT from remote model (sanitized question)
   - Log: CoT text
   - Format prompt with original question, options, and CoT
   - Call local model API
   - Extract answer
   - Log: input, output, time

5. **Scenario 4: Remote Model (if needed)**
   - Format prompt with question and options
   - Call remote model API
   - Extract answer
   - Log: input, output, time, tokens

6. **Save Results**
   - Write to JSON file (append)
   - Write to database
   - Update progress counters
   - Log completion status

#### Post-Experiment
1. Calculate final statistics
2. Generate summary report
3. Save final JSON file
4. Update database with summary
5. Log experiment completion
6. Print summary to console

### Example Log Entry Format

```json
{
  "timestamp": "2025-12-21T15:30:45.123456",
  "question_id": 358,
  "question": "...",
  "correct_answer": "A",
  "scenarios": {
    "local": {
      "input": "...",
      "output": "...",
      "predicted": "A",
      "is_correct": true,
      "processing_time": 1.23
    },
    "phrasedp_eps1.0": {
      "sanitized_question": "...",
      "cot_text": "...",
      "predicted": "B",
      "is_correct": false,
      "processing_time": 5.67
    }
  },
  "errors": []
}
```

### Monitoring During Execution

- **Real-time Progress**: Print every N questions (e.g., every 10)
- **Checkpointing**: Save intermediate results every M questions (e.g., every 50)
- **Error Recovery**: Log failed questions for later retry
- **Resource Monitoring**: Track API usage, token consumption, costs

### Best Practices

1. **Always log before and after critical operations**
2. **Include timestamps for all log entries**
3. **Log full error messages with stack traces**
4. **Save intermediate results frequently (checkpointing)**
5. **Use structured logging (JSON) for machine-readable logs**
6. **Use human-readable logs for debugging**
7. **Log API responses (sanitized if containing sensitive data)**
8. **Track resource usage (tokens, API calls, time)**

