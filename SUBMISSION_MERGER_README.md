# Submission Merger Scripts

## Overview
Two scripts for merging complete submission .npy files with different approaches:

1. **`merge_submissions.py`** - Full merger with LINKS evaluation and Pareto extraction
2. **`merge_submissions_simple.py`** - Simple merger without evaluation (faster, structure-only)

## Scripts Description

### 1. Full Merger (`merge_submissions.py`)

**Features:**
- ✅ Loads two complete submission files
- ✅ Evaluates all mechanisms using LINKS Tools
- ✅ Extracts Pareto frontier for each problem
- ✅ Creates comparison plots
- ✅ Calculates hypervolume improvements
- ✅ Generates detailed analysis reports

**Requirements:**
- JAX/LINKS environment
- Target curves available
- Longer runtime (full evaluation)

**Usage:**
```bash
python merge_submissions.py submission1.npy submission2.npy [output_name]
```

**Outputs:**
- `{output_name}.npy` - Final optimized submission
- `submission_merge_comparison.png` - Before/after plots for all 6 problems
- `hypervolume_comparison.png` - HV comparison chart
- `merge_summary.txt` - Detailed text report

### 2. Simple Merger (`merge_submissions_simple.py`)

**Features:**
- ✅ Loads two complete submission files
- ✅ Combines mechanisms using simple strategies
- ✅ Enforces 1000 mechanism limit per problem
- ✅ Validates final structure
- ✅ Fast execution (no evaluation)

**Merging Strategy:**
- If combined count ≤ 1000: Take all mechanisms
- If combined count > 1000: Proportional distribution with minimum representation
- Maintains structural integrity

**Usage:**
```bash
python merge_submissions_simple.py submission1.npy submission2.npy [output_name]
```

**Outputs:**
- `{output_name}.npy` - Merged submission
- `{output_name}_summary.json` - Merge statistics

## Example Usage

### Quick Test Merge
```bash
python merge_submissions_simple.py overnight_run.npy optimized_results.npy quick_merge
```

### Full Optimization Merge
```bash
python merge_submissions.py overnight_run.npy optimized_results.npy final_optimized
```

### Using Absolute Paths
```bash
python merge_submissions_simple.py /path/to/submission1.npy /path/to/submission2.npy final_submission
```

## Test Results

**Example merge of overnight run + optimized curve 6:**
```
Problem      Sub1   Sub2   Merged   Strategy
-------------------------------------------------------
Problem 1    175    22     197      All combined
Problem 2    191    115    306      All combined
Problem 3    1000   15     1000     Proportional limit
Problem 4    761    12     773      All combined
Problem 5    610    17     627      All combined
Problem 6    250    111    361      All combined
-------------------------------------------------------
TOTAL        2987   292    3264
```

## When to Use Which Script

### Use Simple Merger When:
- ✅ Quick structure validation needed
- ✅ No JAX/LINKS environment available
- ✅ Just want to combine mechanisms without optimization
- ✅ Testing or preliminary merging

### Use Full Merger When:
- ✅ Final competition submission preparation
- ✅ Want true Pareto optimization
- ✅ Need hypervolume analysis
- ✅ Have full LINKS environment available

## File Structure Validation

Both scripts validate that the merged submission contains:
- ✅ All 6 problems (Problem 1 through Problem 6)
- ✅ Each problem has a list of mechanism dictionaries
- ✅ Each mechanism has required keys: `x0`, `edges`, `fixed_joints`, `motor`, `target_joint`
- ✅ Proper data types and structure

## Error Handling

Both scripts include comprehensive error handling for:
- Missing input files
- Corrupted .npy files
- Invalid submission structure
- Evaluation failures (full merger only)
- File I/O errors

## Tips

1. **Always backup original files** before merging
2. **Use simple merger first** to validate structure and strategy
3. **Use full merger for final submission** to get optimal results
4. **Check the summary files** to understand what was merged
5. **Validate mechanism counts** don't exceed 1000 per problem (LINKS limit)