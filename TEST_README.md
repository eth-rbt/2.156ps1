# Test File: test_saving_2_iterations.py

## Purpose
This file tests the submission saving functionality with reduced parameters to validate that the mechanism data is correctly saved in the proper format.

## Changes from Original
- **Iterations per run**: 2 (instead of 12)
- **Population size**: 50 (instead of 100/200)
- **Generations**: 50 (instead of 200)
- **Mechanisms sampled**: 1000 (instead of 8000)
- **Output directory**: `TEST_saving_2iter_TIMESTAMP`

## Expected Runtime
5-15 minutes (instead of 24+ hours)

## What to Check After Running
1. **Submission file**: `TEST_saving_2iter_TIMESTAMP/submission.npy`
2. **Analysis file**: `TEST_saving_2iter_TIMESTAMP/submission_analysis.json`
3. **Console output**: Look for "✅ Processing Problem X with actual mechanism data"
4. **Mechanism counts**: Should show actual mechanisms (not placeholders)

## Run Command
```bash
python test_saving_2_iterations.py
```

## Expected Output Structure
```
submission.npy contains:
{
  'Problem 1': [list of mechanism dicts with x0, edges, fixed_joints, motor, target_joint],
  'Problem 2': [list of mechanism dicts],
  ...
  'Problem 6': [list of mechanism dicts]
}
```

Each mechanism dict should have actual data, not placeholders.