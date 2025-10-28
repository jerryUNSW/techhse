# Summary of Previous Findings - Tech4HSE Project

## Overview
This document summarizes the key findings from our extensive experimentation with privacy-preserving multi-hop question answering systems, differential privacy mechanisms, and model comparisons.

## 1. Model Performance Comparisons

### Local vs Remote Model Performance
**Key Finding**: Remote models consistently outperform local models on multi-hop reasoning tasks.

**Evidence from txt-files/res.txt and txt-files/phi-2-deep-seek.txt:**
- **Local Model (microsoft/phi-2)**: Poor performance on multi-hop questions
- **Remote Model (GPT-4o)**: Significantly better accuracy
- **Pattern**: Local models struggle with complex reasoning, remote models excel

**Example Results:**
- Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
  - Local: "The film was called 'The Devil's Doll'" (Incorrect)
  - Remote: "Yes, both Scott Derrickson and Ed Wood were American" (Correct)

### Model-Specific Performance
**Qwen/Qwen2.5-Coder-7B-fast Performance:**
- Used for phrase-level differential privacy experiments
- Generated diverse paraphrases with varying similarity scores
- Successfully created candidates ranging from 0.17 to 0.97 similarity

## 2. Differential Privacy Mechanisms

### Phrase-Level DP (Custom Implementation)
**Key Achievements:**
- **Diverse Candidate Generation**: Successfully generated candidates with similarity scores ranging from 17% to 97%
- **Semantic Coherence**: Maintained meaningful paraphrases while providing privacy protection
- **Epsilon Analysis**: Demonstrated clear privacy-utility trade-offs across epsilon values (1.0 to 5.0)

**Technical Improvements:**
- **Multiple API Calls**: 5 API calls × 5 candidates = 25 total candidates
- **Prompt Engineering**: Externalized prompts for better maintainability
- **Parsing Logic**: Fixed to handle LLM responses correctly
- **Quality Control**: Eliminated tautological/nonsensical questions

### InferDPT Integration
**Key Findings:**
- **Token-Level Perturbation**: Creates semantically incoherent text
- **High Privacy, Low Utility**: Provides maximum privacy but minimal semantic coherence
- **Similarity Scores**: Very low (0.05-0.13 range, some negative correlations)
- **Epsilon Impact**: No clear relationship between epsilon and similarity

**Test Results:**
- **Accuracy**: 66.67% (2/3 correct) on 3 questions
- **Perturbation Quality**: Completely nonsensical output
- **Local Model Compensation**: Local model partially compensates for bad CoT

## 3. Multi-Hop Question Answering Framework

### Experimental Scenarios
**Scenario 1**: Purely Local Model (Baseline)
**Scenario 2**: Non-Private Local Model + Remote CoT
**Scenario 2.5**: Non-Private Local Model + Local CoT
**Scenario 3.1**: Private Local Model + CoT (Phrase DP)
**Scenario 3.2**: Private Local Model + CoT (InferDPT)
**Scenario 4**: Purely Remote Model

### Key Insights
**Privacy-Utility Trade-off:**
- **Phrase DP**: Good balance between privacy and utility
- **InferDPT**: Maximum privacy, minimum utility
- **Local Model Dependency**: InferDPT performance heavily depends on local model quality

## 4. Epsilon Parameter Analysis

### Phrase DP Epsilon Results
**Low Epsilon (1.0)**: High privacy protection, low similarity (0.24-0.32)
**High Epsilon (5.0)**: Low privacy protection, high similarity (0.80-0.95)
**Optimal Range**: Epsilon 2.0-3.0 provides good balance

### InferDPT Epsilon Results
**No Clear Pattern**: Similarity scores don't show consistent relationship with epsilon
**All Values Low**: Similarity range -0.0471 to 0.1339 across all epsilon values

## 5. Technical Challenges and Solutions

### Data Path Issues
**Problem**: InferDPT had hardcoded absolute paths
**Solution**: Fixed to use relative paths (`InferDPT/data/`)

### Candidate Generation Issues
**Problem**: Initially generating ~200 candidates instead of 25
**Solution**: Fixed parsing logic to extract only one valid paraphrase per completion

### Prompt Engineering
**Problem**: Narrow similarity range (25-35%)
**Solution**: Implemented spectrum-based generation with varied generalization levels

## 6. Quality Improvements

### Semantic Coherence
- **Before**: Aggressive generalization led to narrow similarity range
- **After**: Balanced approach with varied generalization levels
- **Result**: Wide similarity range (17-97%) for better exponential mechanism performance

### Candidate Diversity
- **Before**: All candidates had similar similarity scores
- **After**: Spectrum of candidates from high specificity to maximum privacy
- **Result**: Better privacy-utility trade-offs

## 7. Key Experimental Results

### Phrase DP Performance
- **Success Rate**: High (successful perturbations)
- **Similarity Range**: 0.17-0.97 (excellent diversity)
- **Semantic Quality**: High (maintains meaning)
- **Privacy Level**: Moderate to High

### InferDPT Performance
- **Success Rate**: 100% (all perturbations successful)
- **Similarity Range**: -0.05-0.13 (very low)
- **Semantic Quality**: Very Low (nonsensical)
- **Privacy Level**: Very High

### Multi-Hop Accuracy
- **Scenario 3.2 (InferDPT)**: 66.67% (2/3 correct)
- **Local Model Compensation**: Critical factor in InferDPT performance
- **Context Utilization**: Local model relies heavily on context when CoT is poor

## 8. Recommendations

### For Production Use
1. **Phrase DP**: Better choice for practical applications requiring semantic coherence
2. **InferDPT**: Suitable only for maximum privacy scenarios where utility can be sacrificed
3. **Local Model Selection**: Critical for InferDPT performance

### For Research
1. **Model Comparison**: Test with different local models to understand performance dependencies
2. **Epsilon Optimization**: Further explore epsilon values for optimal privacy-utility balance
3. **Hybrid Approaches**: Consider combining phrase-level and token-level techniques

## 9. Files and Documentation

### Key Output Files
- `testing-phraseDP.txt`: Comprehensive phrase DP results
- `testing-inferdpt.txt`: InferDPT analysis results
- `epsilon_experiment_results.txt`: Epsilon parameter analysis
- `multi_hop_experiment_results_*.txt`: Full multi-hop experiment results

### Configuration Files
- `config.yaml`: Centralized configuration
- `prompts/system_prompt.txt`: External system prompt
- `prompts/user_prompt_template.txt`: External user prompt template

## 10. Future Directions

### Immediate Next Steps
1. **Model Comparison**: Test InferDPT with different local models
2. **Hybrid DP**: Explore combinations of phrase and token-level techniques
3. **Real-world Evaluation**: Test on more diverse question types

### Long-term Research
1. **Adaptive Epsilon**: Dynamic epsilon selection based on question complexity
2. **Semantic Preservation**: Better techniques for maintaining meaning under high privacy
3. **Multi-modal Privacy**: Extend to image and audio privacy preservation

---

*This summary represents the comprehensive findings from our extensive experimentation with privacy-preserving multi-hop question answering systems.*
