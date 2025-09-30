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
3. **Pareto frontier plot**: `TEST_saving_2iter_TIMESTAMP/submission_pareto_frontiers.png`
4. **Console output**: Look for:
   - "✅ Processing Problem X - collecting from all nodes/iterations"
   - "Found X mechanisms from Y nodes, iter Z"
   - "✅ Extracted X Pareto frontier mechanisms for Problem Y"
   - "🎯 PLOTTING 6 PARETO FRONTIERS FROM SUBMISSION DATA"
   - "Evaluating X mechanisms for Problem Y..."
5. **Mechanism counts**: Should show actual Pareto frontier mechanisms per curve
6. **No JSON errors**: All numpy types should be converted to JSON-serializable types
7. **Pareto plot validation**: 2×3 subplot grid showing each curve's Pareto frontier with hypervolume scores

## Run Command
```bash
python test_saving_2_iterations.py
```

## Expected Output Structure
```
submission.npy contains:
{
  'Problem 1': [Pareto frontier mechanisms from all nodes/iterations],
  'Problem 2': [Pareto frontier mechanisms from all nodes/iterations],
  ...
  'Problem 6': [Pareto frontier mechanisms from all nodes/iterations]
}
```

Each mechanism dict should have:
- **x0**: Joint positions (list of [x, y] coordinates)
- **edges**: Connectivity matrix (list of [node1, node2] pairs)
- **fixed_joints**: Fixed node indices (list of integers)
- **motor**: Motor configuration (list like [0, 1])
- **target_joint**: Target joint index (integer, not numpy.int64)

## Key Features
- ✅ **Per-curve Pareto extraction**: Each curve gets its own Pareto frontier from all collected mechanisms
- ✅ **Multi-node aggregation**: Combines mechanisms from 5, 8, and 9 node configurations
- ✅ **Multi-iteration aggregation**: Combines mechanisms from 2 iterations per configuration
- ✅ **JSON-safe**: All numpy types converted to standard Python types
- ✅ **Feasibility filtering**: Only includes mechanisms with distance ≤ 0.75, material ≤ 10.0