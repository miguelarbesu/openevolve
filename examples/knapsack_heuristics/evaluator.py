import importlib.util
import sys
import random
import concurrent.futures
import traceback
import yaml
from pathlib import Path
from openevolve.evaluation_result import EvaluationResult


def _load_config():
    """Loads settings from config.yaml in the same directory."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def solve_knapsack_dp(items, capacity):
    """Solves the 0/1 knapsack problem using dynamic programming to find the optimal value."""
    n = len(items)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i - 1]["weight"] <= w:
                dp[i][w] = max(
                    items[i - 1]["value"] + dp[i - 1][w - items[i - 1]["weight"]],
                    dp[i - 1][w],
                )
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


def generate_extreme_trap(n):
    """
    Generates a trap where a tiny high-density item blocks a massive optimal item.
    This exploits the fact that greedy commits to high density without looking ahead at capacity.
    """
    capacity = 1000

    # 1. Optimal Item: Weight=Capacity, Value=Capacity. Density=1.0.
    # Takes full capacity.
    opt_item = {"value": capacity, "weight": capacity}

    # 2. Trap Item: Tiny weight, density > 1.0.
    # Weight 1, Value 2. Density 2.0.
    trap_item = {"value": 2, "weight": 1}

    items = [opt_item, trap_item]

    # 3. Filler items that are too big to fit in the gap (capacity - 1 = 999)
    # or have very poor value.
    for _ in range(n - 2):
        items.append({"value": 1, "weight": capacity})

    optimal_value = solve_knapsack_dp(items, capacity)
    return {"capacity": capacity, "items": items, "optimal_value": optimal_value}


def generate_generalized_trap(n):
    """
    Generates a scalable trap instance.
    General idea:
    1 item with slightly better density but large weight.
    2 items with slightly worse density that fill capacity perfectly.
    """
    # Scale parameters
    k = random.randint(10, 100)  # multiplier

    # Capacity
    capacity = 2 * k

    # Item A: Weight k+1, Value k+2. Density = (k+2)/(k+1) > 1
    item_a = {"value": k + 2, "weight": k + 1}

    # Item B, C: Weight k, Value k. Density = 1
    item_b = {"value": k, "weight": k}
    item_c = {"value": k, "weight": k}

    items = [item_a, item_b, item_c]

    # Add noise items (very low value/weight)
    for _ in range(n - 3):
        items.append({"value": 0, "weight": 1000})  # Useless

    optimal_value = solve_knapsack_dp(items, capacity)

    return {"capacity": capacity, "items": items, "optimal_value": optimal_value}


def generate_trap_instance():
    # Keep the original fixed one as it's a good baseline unit test
    items = [
        {"value": 52, "weight": 51},
        {"value": 50, "weight": 50},
        {"value": 50, "weight": 50},
        {"value": 1, "weight": 1},
        {"value": 1, "weight": 1},
    ]
    capacity = 100
    optimal_value = 100
    return {"capacity": capacity, "items": items, "optimal_value": optimal_value}


def generate_difficult_instance(n, weight_range=(10, 100), correlation="strong"):
    """
    Generates a difficult knapsack instance.
    """
    items = []
    total_weight = 0
    for _ in range(n):
        w = random.randint(*weight_range)
        if correlation == "strong":
            v = w + 10
        elif correlation == "weak":
            v = max(1, w + random.randint(-5, 5))
        elif correlation == "uncorrelated":
            v = random.randint(10, 100)
        elif correlation == "inverse":
            # Higher weight, higher value, but ratio might be tricky
            v = int(w * (0.8 + random.random() * 0.4))
        else:
            v = random.randint(10, 100)
        items.append({"value": v, "weight": w})
        total_weight += w

    capacity = int(total_weight * 0.3)  # Even tighter capacity
    optimal_value = solve_knapsack_dp(items, capacity)
    return {"capacity": capacity, "items": items, "optimal_value": optimal_value}


def get_test_instances(seed=None):
    """Returns a list of difficult knapsack problem instances."""
    # Seed for reproducibility
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(42)  # Default fallback

    instances = [
        # 1. Tricky small instance (Greedy trap)
        {
            "name": "greedy_trap_small",
            "capacity": 100,
            "items": [
                {"value": 10, "weight": 60},
                {"value": 10, "weight": 60},
                {"value": 12, "weight": 100},
            ],
            "optimal_value": 12,
        },
        # 2. Uncorrelated (Medium)
        {
            **generate_difficult_instance(30, correlation="uncorrelated"),
            "name": "uncorrelated_medium",
        },
        # 3. Inverse correlated (Hard for greedy)
        {
            **generate_difficult_instance(40, correlation="inverse"),
            "name": "inverse_correlated_hard",
        },
        # 4. Strongly correlated (classic hard)
        {
            **generate_difficult_instance(20, correlation="strong"),
            "name": "strongly_correlated_hard",
        },
        # 5. Large scale uncorrelated
        {
            **generate_difficult_instance(
                200, weight_range=(5, 50), correlation="uncorrelated"
            ),
            "name": "large_scale_unclorrelated",
        },
        # 6. Another greedy trap
        {
            "name": "greedy_trap_medium",
            "capacity": 50,
            "items": [
                {"value": 31, "weight": 30},
                {"value": 20, "weight": 25},
                {"value": 20, "weight": 25},
            ],
            "optimal_value": 40,
        },
        # 7. Specific Greedy Trap Hard
        {
            **generate_trap_instance(),
            "name": "greedy_trap_fixed",
        },
        # 8. Extreme Greedy Trap (Tiny Item blocks Huge Item)
        {
            **generate_extreme_trap(10),
            "name": "greedy_trap_extreme",
        },
        # 9. Generalized Trap
        {
            **generate_generalized_trap(10),
            "name": "greedy_trap_generalized",
        },
    ]
    return [i for i in instances if i["optimal_value"] > 0]


def evaluate(module_path):
    """
    Evaluates the solve_knapsack function in the given module.
    """
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("evolved_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "solve_knapsack"):
            error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Program is missing required 'solve_knapsack' function",
                "suggestion": "Make sure your program includes a function named 'solve_knapsack' that takes (items, capacity)",
            }
            return EvaluationResult(
                metrics={
                    "combined_score": 0.0,
                    "correctness": 0.0,
                    "error": "Function solve_knapsack not found",
                },
                artifacts=error_artifacts,
            )

        config = _load_config()
        eval_seed = config.get("evaluator", {}).get("random_seed") or config.get(
            "random_seed", 42
        )
        test_instances = get_test_instances(seed=eval_seed)
        total_score = 0
        correct_count = 0
        instance_scores = {}

        for instance in test_instances:
            instance_score = 0.0
            try:
                # Run with timeout to prevent infinite loops in evolved code
                selected_indices = run_with_timeout(
                    module.solve_knapsack,
                    args=(instance["items"], instance["capacity"]),
                    timeout_seconds=2,
                )

                total_value = 0
                total_weight = 0
                valid_indices = True

                if not isinstance(selected_indices, (list, tuple)):
                    valid_indices = False
                elif len(set(selected_indices)) != len(selected_indices):
                    # Duplicate indices selected
                    valid_indices = False

                if valid_indices:
                    for idx in selected_indices:
                        if not isinstance(idx, int) or not (
                            0 <= idx < len(instance["items"])
                        ):
                            valid_indices = False
                            break
                        total_value += instance["items"][idx]["value"]
                        total_weight += instance["items"][idx]["weight"]

                if valid_indices and total_weight <= instance["capacity"]:
                    correct_count += 1
                    # Normalized by the actual optimal value found by DP
                    instance_score = min(1.0, total_value / instance["optimal_value"])
                else:
                    instance_score = 0.0

            except Exception as e:
                # Failed to execute trial
                print(f"Trial failed: {str(e)}")
                instance_score = 0.0

            total_score += instance_score
            instance_scores[instance["name"]] = instance_score

        num_instances = len(test_instances)
        combined_score = total_score / num_instances if num_instances > 0 else 0
        correctness = correct_count / num_instances if num_instances > 0 else 0

        artifacts = {
            "num_test_instances": num_instances,
            "correctness_rate": f"{correctness:.2%}",
            "average_optimality": f"{combined_score:.4f}",
            "instance_breakdown": instance_scores,
        }

        return EvaluationResult(
            metrics={
                "combined_score": combined_score,
                "correctness": correctness,
            },
            artifacts=artifacts,
        )

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())

        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code",
        }

        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "correctness": 0.0,
                "error": str(e),
            },
            artifacts=error_artifacts,
        )


def evaluate_stage1(program_path):
    """First stage evaluation with a subset of trials for speed"""
    # Just run the first 3 instances as a quick check
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("evolved_module", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "solve_knapsack"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "runs_successfully": 0.0}
            )

        config = _load_config()
        eval_seed = config.get("evaluator", {}).get("random_seed") or config.get(
            "random_seed", 42
        )
        test_instances = get_test_instances(seed=eval_seed)[:3]
        total_score = 0
        correct_count = 0
        success = True

        for instance in test_instances:
            try:
                selected_indices = run_with_timeout(
                    module.solve_knapsack,
                    args=(instance["items"], instance["capacity"]),
                    timeout_seconds=1,
                )

                total_weight = sum(
                    instance["items"][i]["weight"]
                    for i in selected_indices
                    if 0 <= i < len(instance["items"])
                )
                total_value = sum(
                    instance["items"][i]["value"]
                    for i in selected_indices
                    if 0 <= i < len(instance["items"])
                )

                if total_weight <= instance["capacity"]:
                    total_score += total_value / instance["optimal_value"]
                    correct_count += 1
            except Exception:
                success = False
                break

        if not success:
            return EvaluationResult(
                metrics={
                    "combined_score": 0.0,
                    "correctness": 0.0,
                    "runs_successfully": 0.0,
                }
            )

        return EvaluationResult(
            metrics={
                "combined_score": total_score / 3.0,
                "correctness": correct_count / 3.0,
                "runs_successfully": 1.0,
            }
        )
    except Exception:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "correctness": 0.0,
                "runs_successfully": 0.0,
            }
        )


def evaluate_stage2(program_path):
    """Second stage evaluation: full test suite"""
    return evaluate(program_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <module_path>")
        sys.exit(1)

    result = evaluate(sys.argv[1])
    metrics = result.metrics
    print(
        f"combined_score={metrics['combined_score']:.4f}, correctness={metrics['correctness']:.4f}"
    )
    print(f"artifacts={result.artifacts}")
