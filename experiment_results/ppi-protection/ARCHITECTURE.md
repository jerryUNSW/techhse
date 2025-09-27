# PII Protection Experiment - Technical Architecture

This document provides a detailed technical overview of the PII protection experiment framework architecture, optimizations, and implementation details.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Main Experiment │────│  Results Output │
│                 │    │     Controller   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                │
                    ┌───────────────────────┐
                    │   EmbeddingCache      │
                    │   - SBERT Model       │
                    │   - CusText+ Vectors  │
                    │   - InferDPT Data     │
                    │   - CluSanT Clusters  │
                    │   - PhraseDP Cache    │
                    └───────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Privacy     │    │   Incremental   │    │   Resume        │
│   Mechanisms  │    │   Saving        │    │   System        │
│               │    │                 │    │                 │
└───────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. EmbeddingCache Class

**Purpose**: Centralized embedding management with lazy loading and caching.

```python
class EmbeddingCache:
    def __init__(self):
        self.sbert_model = None          # SentenceTransformer model
        self.ct_embeddings = None        # CusText+ counter-fitting vectors
        self.inferdpt_embeddings = None  # InferDPT token embeddings + metadata
        self.clusant_embeddings = None   # CluSanT cluster embeddings
        self.phrase_cache = {}           # PhraseDP candidates per row
```

**Key Methods**:

#### `get_sbert_model()`
- **Lazy Loading**: Only loads SentenceTransformer when first requested
- **Singleton Pattern**: Returns same instance across all calls
- **Memory Impact**: ~500MB RAM, loaded once for entire experiment

#### `get_custext_embeddings()`
- **Dependencies**: Requires CusText+ vectors file and NLTK stopwords
- **Returns**: `(emb_matrix, idx2word, word2idx, stop_set)` tuple
- **Fallback**: Returns `(None, None, None, set())` on failure
- **Memory Impact**: ~200MB for counter-fitting vectors

#### `get_inferdpt_embeddings()`
- **Expensive Operation**: Loads 11K token embeddings and distance matrices
- **Dependencies**: InferDPT data directory with precomputed files
- **Returns**: `(token_to_vector_dict, sorted_distance_data, delta_f_new)` tuple
- **Optimization**: Passes epsilon=1.0 to avoid epsilon-dependent initialization
- **Memory Impact**: ~1GB for full embedding matrices

#### `get_clusant_embeddings()`
- **Complex Setup**: Requires CluSanT directory structure and embedding handler
- **Side Effect**: Changes working directory to CluSanT root
- **Returns**: `(clus_embeddings, clusant_root)` tuple
- **Dependency Management**: Handles dynamic path and import additions

### 2. Incremental Saving System

**Purpose**: Prevent data loss and enable long-running experiments.

#### `save_incremental_results(results, results_file)`
- **Frequency**: Called after each mechanism/epsilon completion
- **Atomicity**: Uses JSON write to ensure data consistency
- **Directory Creation**: Automatically creates parent directories
- **Error Handling**: Graceful degradation with warnings

#### File Structure
```json
{
  "PhraseDP": {
    "1.0": {
      "overall": 0.85,
      "emails": 0.90,
      "phones": 0.80,
      "addresses": 0.85,
      "names": 0.90,
      "samples": [
        {"row": 0, "original": "John's email is john@example.com", "sanitized": "Person's email is contact@company.com"}
      ]
    },
    "1.5": { ... }
  },
  "InferDPT": { ... }
}
```

### 3. Resume System

**Purpose**: Enable experiment continuation after interruption.

#### `load_existing_results(results_file)`
- **Input Validation**: Handles missing files and invalid JSON
- **Backward Compatibility**: Works with results from previous versions
- **Error Recovery**: Returns empty dict on any loading failure

#### Resume Logic
```python
for mechanism_name in ['PhraseDP', 'InferDPT', ...]:
    for epsilon in epsilon_values:
        # Skip if already completed
        if mechanism_name in results and epsilon in results[mechanism_name]:
            continue
        # Process this combination
        ...
```

## Privacy Mechanism Integration

### 1. PhraseDP Integration

```python
# Candidate generation (cached per row)
cache_key = int(idx)
if cache_key in embedding_cache.phrase_cache:
    candidates, candidate_embeddings = embedding_cache.phrase_cache[cache_key]
else:
    # Generate via Nebius API
    candidates = generate_sentence_replacements_with_nebius_diverse(...)
    # Encode with cached SBERT model
    sbert_model = embedding_cache.get_sbert_model()
    candidate_embeddings = {c: sbert_model.encode(c) for c in candidates}
    # Cache for reuse across epsilons
    embedding_cache.phrase_cache[cache_key] = (candidates, candidate_embeddings)

# DP selection with cached embeddings
sanitized_text = differentially_private_replacement(
    target_phrase=original_text,
    epsilon=epsilon,
    candidate_phrases=candidates,
    candidate_embeddings=candidate_embeddings,
    sbert_model=sbert_model
)
```

### 2. InferDPT Integration

```python
# Use cached embeddings to avoid repeated initialization
token_to_vector_dict, sorted_distance_data, delta_f_new = embedding_cache.get_inferdpt_embeddings()

if token_to_vector_dict is not None:
    sanitized_text = perturb_sentence(
        original_text,
        epsilon,
        token_to_vector_dict=token_to_vector_dict,
        sorted_distance_data=sorted_distance_data,
        delta_f_new=delta_f_new
    )
```

**Key Optimization**: By passing cached embeddings to `perturb_sentence()`, we avoid the expensive `initialize_embeddings()` call on every invocation.

### 3. SANTEXT+ Integration

```python
# Global mechanism reused across all rows/epsilons
if santext_global is None:
    santext_global = create_santext_mechanism(epsilon=epsilon, p=0.1)
    # Load vocabulary once
    santext_global.load_vocabulary_from_files(...)

# Reuse for all sanitizations
sanitized_text = santext_global.sanitize_text(original_text)
```

### 4. CusText+ Integration

```python
# Cached embeddings loaded once
ct_emb_matrix, ct_idx2word, ct_word2idx, ct_stop_set = embedding_cache.get_custext_embeddings()

# Use cached embeddings for sanitization
sanitized_text = sanitize_with_custext(
    original_text,
    epsilon=epsilon,
    emb_matrix=ct_emb_matrix,  # Cached
    idx2word=ct_idx2word,      # Cached
    word2idx=ct_word2idx,      # Cached
    stop_set=ct_stop_set       # Cached
)
```

### 5. CluSanT Integration

```python
# Cached embeddings and environment setup
clus_embeddings, clusant_root = embedding_cache.get_clusant_embeddings()

# Create mechanism instance with cached embeddings
clus = CluSanT(
    embedding_file='all-MiniLM-L6-v2',
    embeddings=clus_embeddings,  # Cached
    epsilon=epsilon
)
```

## Performance Optimizations

### Memory Management

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| SBERT Model | 500MB × 5 methods | 500MB × 1 | 75% |
| InferDPT Embeddings | 1GB × 5 epsilons | 1GB × 1 | 80% |
| CusText+ Vectors | 200MB × 5 epsilons | 200MB × 1 | 80% |
| **Total Peak** | **~8.5GB** | **~1.7GB** | **80%** |

### I/O Optimizations

1. **Incremental Saving**: Prevents loss of hours of computation
2. **Lazy Loading**: Only load embeddings when method is used
3. **File Caching**: PhraseDP candidates cached per row across epsilons
4. **Skip Logic**: Resume skips completed work

### Runtime Optimizations

```python
# Before: Linear time with repeated initialization
for method in methods:
    for epsilon in epsilons:
        for row in rows:
            initialize_embeddings()  # O(n) each time
            process_row()

# After: Constant time initialization
cache = EmbeddingCache()
for method in methods:
    cache.load_method_embeddings()  # O(1) amortized
    for epsilon in epsilons:
        if already_completed(method, epsilon):  # Skip optimization
            continue
        for row in rows:
            process_row_with_cache()  # O(1) embedding access
        save_incremental_results()  # Fault tolerance
```

## Error Handling & Fault Tolerance

### Graceful Degradation

```python
try:
    # Attempt optimal path with cached embeddings
    embeddings = cache.get_method_embeddings()
    result = method_with_cache(text, embeddings)
except Exception as e:
    print(f"Warning: {method} failed with cache: {e}")
    # Fallback to original implementation
    result = method_without_cache(text)
```

### Recovery Mechanisms

1. **Embedding Loading**: Each method has fallback paths
2. **API Failures**: PhraseDP falls back to original text
3. **File I/O**: Incremental saves with error reporting
4. **Resume Logic**: Validates existing results before using

### State Management

```python
# CluSanT working directory management
clus_saved_cwd = None
try:
    clus_saved_cwd = os.getcwd()
    os.chdir(clusant_root)
    # ... CluSanT operations
finally:
    if clus_saved_cwd:
        os.chdir(clus_saved_cwd)  # Always restore
```

## Data Flow

### Input Processing

```
Raw Dataset Row
      ↓
Extract PII spans using BIO labels
      ↓
Reconstruct text with tokens + whitespace
      ↓
Filter to PII-containing sentences
      ↓
Apply privacy mechanism with cached embeddings
      ↓
Evaluate protection (leak detection)
      ↓
Store results + sample texts
```

### Output Generation

```
Aggregated Results
      ↓
Calculate protection rates per PII type
      ↓
Generate visualization plots
      ↓
Save JSON results + PNG plots
      ↓
Send email notification (if configured)
```

## Threading & Concurrency

**Current Implementation**: Single-threaded with sequential processing
- **Rationale**: GPU/API rate limits, memory constraints
- **Future Enhancement**: Could parallelize row processing within epsilon

**Potential Parallelization Points**:
```python
# Row-level parallelization (within epsilon)
with multiprocessing.Pool() as pool:
    results = pool.map(process_row_with_cache, rows)

# Method-level parallelization (independent methods)
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(run_method, m): m for m in methods}
```

## Configuration Management

### Environment Variables
- `NEBIUS_API`: API key for PhraseDP candidate generation
- `NEBIUS_MODEL`: Model name (default: Meta-Llama-3.1-8B-Instruct)

### File Paths
- Dataset: `/home/yizhang/tech4HSE/pii_external_dataset.csv`
- CusText+ vectors: `/home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt`
- CluSanT root: `/home/yizhang/tech4HSE/CluSanT`
- InferDPT data: `InferDPT/data/` (relative to working directory)

### Extensibility Points

**Adding New Privacy Methods**:
1. Add to `EmbeddingCache` if requires embeddings
2. Add method name to mechanisms list
3. Add initialization logic in mechanism setup
4. Add processing logic in main loop
5. Ensure proper error handling

**Modifying Evaluation Metrics**:
1. Update `calculate_pii_protection_rate()` function
2. Modify aggregation logic in main loop
3. Update plotting functions for new metrics
4. Adjust email summary generation

## Testing & Validation

### Unit Testing Approach
```python
# Test embedding caching
cache = EmbeddingCache()
model1 = cache.get_sbert_model()
model2 = cache.get_sbert_model()
assert model1 is model2  # Same instance

# Test incremental saving
save_incremental_results(test_data, test_file)
loaded = load_existing_results(test_file)
assert loaded == test_data

# Test resume logic
results['completed_method'] = {'1.0': {'overall': 0.8}}
# Should skip completed_method/1.0 combination
```

### Integration Testing
```bash
# Small dataset test
python pii_protection_experiment.py --rows 2

# Resume test
python pii_protection_experiment.py --rows 2
# Interrupt with Ctrl+C
python pii_protection_experiment.py --rows 2 --resume results_file.json
```

This architecture provides a robust, efficient, and extensible framework for large-scale privacy mechanism evaluation with minimal resource usage and maximum fault tolerance.