# Critical Epsilon Configuration Issue Report

**Date**: 2025-10-02  
**Issue**: All experiments labeled as epsilon 2.0 and 3.0 were actually running with epsilon 1.0  
**Severity**: CRITICAL - Results are not comparable as intended  

## Problem Summary

All experiment scripts were reading `config['epsilon']` from `config.yaml` instead of using command-line arguments or proper epsilon overrides. Since `config.yaml` has `epsilon: 1.0`, **ALL experiments ran with epsilon 1.0**, regardless of their intended epsilon values.

## Affected Experiments

### ❌ INCORRECTLY RAN WITH EPSILON 1.0:

| Experiment | Intended Epsilon | Actual Epsilon | Script Used | Status |
|------------|------------------|----------------|-------------|---------|
| **Next 500 Questions - Epsilon 1.0** | 1.0 | 1.0 | `test-medqa-usmle-next-500-simple.py` | ✅ CORRECT |
| **Next 500 Questions - Epsilon 2.0** | 2.0 | 1.0 | `test-medqa-usmle-epsilon2-phrasedp-only.py` | ❌ WRONG |
| **Next 500 Questions - Epsilon 3.0** | 3.0 | 1.0 | `test-medqa-usmle-next-500-simple.py` | ❌ WRONG |

### 📊 Current Database Status:
- **Epsilon 1.0**: Contains correct data (actually ran with epsilon 1.0)
- **Epsilon 2.0**: Contains incorrect data (actually ran with epsilon 1.0)  
- **Epsilon 3.0**: Contains incorrect data (actually ran with epsilon 1.0)

## Root Cause Analysis

### Scripts with Config Dependencies:
1. **`test-medqa-usmle-next-500-simple.py`** - Reads `config['epsilon']` throughout
2. **`test-medqa-usmle-epsilon2-phrasedp-only.py`** - Reads `config['epsilon']` throughout

### Evidence:
- All JSON result files lack epsilon information (scripts didn't save epsilon values)
- All scripts use `config['epsilon']` in multiple places
- Config.yaml consistently shows `epsilon: 1.0`

## Experiments That Need to be Rerun

### 🔴 HIGH PRIORITY - RERUN REQUIRED:

#### 1. **Next 500 Questions - Epsilon 2.0**
- **Script**: `test-medqa-usmle-epsilon2-phrasedp-only.py` (✅ FIXED)
- **Mechanisms**: PhraseDP (Normal Mode) vs PhraseDP+ (Medical Mode)
- **Questions**: 500-999 (indices 500-999)
- **Status**: Script fixed to use epsilon 2.0, needs rerun

#### 2. **Next 500 Questions - Epsilon 3.0**
- **Script**: `test-medqa-usmle-next-500-simple.py` (❌ NEEDS FIXING)
- **Mechanisms**: Local Model, PhraseDP+, Local + CoT
- **Questions**: 500-999 (indices 500-999)
- **Status**: Script needs fixing, then rerun

### 🟡 MEDIUM PRIORITY - VERIFY NEED:

#### 3. **First 500 Questions - PhraseDP+ (Medical Mode)**
- **Script**: `test-medqa-first-500-phrasedp-plus.py` (✅ CORRECT)
- **Mechanisms**: PhraseDP+ (Medical Mode) only
- **Questions**: 0-499 (indices 0-499)
- **Status**: Script is correct, ready to run for epsilon 1.0, 2.0, 3.0

## Required Actions

### Immediate Actions:
1. **Fix `test-medqa-usmle-next-500-simple.py`** to accept epsilon as command-line argument
2. **Rerun Epsilon 2.0 experiment** (script already fixed)
3. **Rerun Epsilon 3.0 experiment** (after fixing script)
4. **Update database** with correct epsilon values

### Script Fixes Needed:
1. **`test-medqa-usmle-next-500-simple.py`**:
   - Add `--epsilon` command-line argument
   - Replace all `config['epsilon']` with the argument value
   - Save epsilon value in JSON results

2. **`test-medqa-usmle-epsilon2-phrasedp-only.py`**:
   - ✅ Already fixed to override `config['epsilon'] = 2.0`

### Database Updates:
1. **Remove incorrect epsilon 2.0 and 3.0 entries**
2. **Load correct epsilon 2.0 and 3.0 results** after rerunning

## Commands to Rerun Experiments

### Epsilon 2.0 (Next 500 Questions):
```bash
conda run -n priv-env python test-medqa-usmle-epsilon2-phrasedp-only.py
```

### Epsilon 3.0 (Next 500 Questions) - After fixing script:
```bash
conda run -n priv-env python test-medqa-usmle-next-500-simple.py --epsilon 3.0
```

### First 500 Questions - PhraseDP+ (All Epsilons):
```bash
# Epsilon 1.0
conda run -n priv-env python test-medqa-first-500-phrasedp-plus.py --epsilon 1.0

# Epsilon 2.0  
conda run -n priv-env python test-medqa-first-500-phrasedp-plus.py --epsilon 2.0

# Epsilon 3.0
conda run -n priv-env python test-medqa-first-500-phrasedp-plus.py --epsilon 3.0
```

## Impact Assessment

### Data Integrity:
- **3 experiments** need to be rerun
- **Database contains incorrect epsilon labels**
- **Comparison analysis** between epsilon values is invalid

### Research Impact:
- **Epsilon trend analysis** is currently meaningless
- **Privacy-accuracy tradeoff** conclusions are invalid
- **Paper results** would be incorrect if published as-is

### Timeline Impact:
- **~6-8 hours** of computation time needed for reruns
- **Database cleanup** and reload required
- **Analysis scripts** may need updates

## Prevention Measures

### ✅ Implemented:
1. **`test-medqa-first-500-phrasedp-plus.py`** properly handles epsilon via command-line
2. **`test-medqa-usmle-epsilon2-phrasedp-only.py`** fixed to override config

### 🔄 In Progress:
1. **Fix `test-medqa-usmle-next-500-simple.py`** to accept epsilon argument
2. **Add epsilon validation** to all scripts
3. **Save epsilon values** in all result files

### 📋 Recommendations:
1. **Always use command-line arguments** for epsilon in new scripts
2. **Never rely on config.yaml** for critical experiment parameters
3. **Add validation** to ensure intended epsilon is used
4. **Include epsilon** in all result file names and JSON data

---

**Next Steps**: Fix the remaining script and rerun the affected experiments to ensure data integrity and valid research results.

