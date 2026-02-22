# Knapsack Heuristics Evolution

This example demonstrates how OpenEvolve can discover sophisticated heuristics for the 0/1 Knapsack problem starting from a simple greedy ratio approach.

## Problem Description

The [0/1 Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) involves selecting a subset of items, each with a weight and a value, such that the total weight does not exceed a given capacity and the total value is maximized.

While the problem is NP-hard, many heuristics exist. This example challenges the LLM to find novel heuristics that might outperform basic greedy approaches on various problem instances, including "traps" designed to exploit simple greedy logic.

## Getting Started

To run this evolution:

```bash
# From the project root
python openevolve-run.py examples/knapsack_heuristics/initial_program.py \
  examples/knapsack_heuristics/evaluator.py \
  --config examples/knapsack_heuristics/config.yaml
```

## Structure

- `initial_program.py`: A basic greedy heuristic that picks items based on their value-to-weight ratio.
- `evaluator.py`: Tests the heuristics on a suite of instances, measuring value achievement (`combined_score`) and constraint satisfaction (`correctness`).
- `config.yaml`: Configuration for the evolution process, including cascade evaluation settings.
- `requirements.txt`: Dependencies for the heuristics (e.g., `numpy`).

## Metrics

- `combined_score`: Normalized value achieved across all test instances (0.0 to 1.0). Solutions that exceed capacity for an instance receive zero score for that instance.
- `correctness`: The percentage of instances solved without exceeding the knapsack capacity.

## Evaluator Features

The evaluator implements several advanced features for robust evolution:
1. **Cascade Evaluation**: A fast stage 1 check on a subset of instances to quickly filter out failing programs.
2. **Timeout Protection**: Safeguards against infinite loops in evolved code using a 2-second timeout per trial.
3. **Artifact Generation**: Provides detailed feedback to the LLM, including instance-by-instance scores and specific error suggestions.
4. **Diverse Test Suite**: Includes uncorrelated, strongly correlated, and "trap" instances where simple greedy fails significantly.
