import importlib.util
import os
import sys
import random


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
    # We want to ensure greedy doesn't find anything else to pick.
    for _ in range(n - 2):
        # Make them heavy so they don't fit with the trap
        # Or make them very low value so picking them doesn't help score much.
        # Let's make them fit but be terrible value.
        # Weight 999, Value 1.
        items.append({"value": 1, "weight": capacity})

    # Greedy order:
    # 1. Trap (Ratio 2.0). Picks it. Rem Cap 999.
    # 2. Opt (Ratio 1.0). Weight 1000. Fails.
    # 3. Fillers (Ratio ~0). Weight 1000. Fail.
    # Result: Value 2.

    # Optimal order:
    # 1. Opt (Ratio 1.0). Picks it. Rem Cap 0.
    # Result: Value 1000.

    # Score: 2/1000 = 0.002.

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
    # Should be 2k (Item B + Item C) vs k+2 (Item A)

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
    "strong": value = weight + 10 (hard for some, but greedy ratio still okay)
    "weak": value = weight + random(-5, 5)
    "uncorrelated": value and weight independent
    "inverse": large weights have slightly better value but worse ratio
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


def get_test_instances():
    """Returns a list of difficult knapsack problem instances."""
    # Seed for reproducibility
    random.seed(42)

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
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("evolved_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "solve_knapsack"):
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "error": "Function solve_knapsack not found",
        }

    test_instances = get_test_instances()
    total_score = 0
    correct_count = 0
    instance_scores = {}
    for instance in test_instances:
        instance_score = 0.0
        try:
            selected_indices = module.solve_knapsack(
                instance["items"], instance["capacity"]
            )

            total_value = 0
            total_weight = 0
            valid_indices = True

            if not isinstance(selected_indices, (list, tuple)):
                valid_indices = False
            elif len(set(selected_indices)) != len(selected_indices):
                valid_indices = False

            if valid_indices:
                for idx in selected_indices:
                    if not (0 <= idx < len(instance["items"])):
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

        except Exception:
            # Failed to execute
            instance_score = 0.0

        total_score += instance_score
        instance_scores[instance["name"]] = instance_score

    num_instances = len(test_instances)
    combined_score = total_score / num_instances if num_instances > 0 else 0
    correctness = correct_count / num_instances if num_instances > 0 else 0

    return {
        "combined_score": combined_score,
        "correctness": correctness,
        "instance_scores": instance_scores,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <module_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])
    print(
        f"combined_score={metrics['combined_score']:.4f}, correctness={metrics['correctness']:.4f}"
    )
    print(f"instance_scores={metrics['instance_scores']}")
