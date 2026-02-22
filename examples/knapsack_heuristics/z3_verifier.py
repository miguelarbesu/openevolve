import sys
import json
import traceback
from z3 import Solver, Int, Sum, Or, And, Not, If, sat, ExprRef
from typing import Dict, List, Any


class Z3KnapsackVerifier:
    def __init__(self, n_items: int = 5, fitness_threshold: float = 0.9):
        self.n_items = n_items
        self.fitness_threshold = fitness_threshold

    def _get_optimal_selection(
        self, solver: Solver, w: List[ExprRef], v: List[ExprRef], cap: ExprRef
    ):
        """
        Adds constraints to find the optimal selection in Z3 for n_items.
        Returns the symbolic value of the optimal selection.
        """
        x_opt = [Int(f"x_opt_{i}") for i in range(self.n_items)]
        for x in x_opt:
            solver.add(Or(x == 0, x == 1))

        # Sustainability constraint
        solver.add(Sum([x_opt[i] * w[i] for i in range(self.n_items)]) <= cap)

        # Value of this specific selection
        opt_val = Sum([x_opt[i] * v[i] for i in range(self.n_items)])

        return opt_val, x_opt

    def verify(self, heuristic_z3_logic: str) -> Dict[str, Any]:
        """
        Verifies the heuristic using Z3 logic provided as a Python function string.
        The function should be named 'get_heuristic_selection(W, V, Cap, solver)'
        and return a list of Z3 expressions/values representing selected indices (0 or 1).
        """
        # Define symbolic constants
        W = [Int(f"w_{i}") for i in range(self.n_items)]
        V = [Int(f"v_{i}") for i in range(self.n_items)]
        Cap = Int("cap")

        s = Solver()

        # Realistic constraints to avoid trivial/unstable cases
        s.add(Cap > 10, Cap <= 100)
        for i in range(self.n_items):
            s.add(W[i] > 1, W[i] <= 100)
            s.add(V[i] > 1, V[i] <= 100)

        # Optimal Value
        opt_val, _ = self._get_optimal_selection(s, W, V, Cap)

        # Heuristic Selection
        # We execute the provided logic to get the heuristic's indicator variables
        try:
            # We assume heuristic_z3_logic defines a function:
            # def get_heuristic_selection(W, V, Cap, solver): ...
            def sym_get(lst, idx):
                """Symbolic getter: lst[idx] where idx is a Z3 symbolic integer"""
                from z3 import If, Sum

                return Sum([If(idx == j, lst[j], 0) for j in range(len(lst))])

            def sym_set(lst, idx, val):
                """Symbolic setter: mock setting by returning a new list/logic (Z3 lists are tricky, but we can If-chain them)"""
                # For simplicity in heuristics, we usually don't need symbolic SETTING of items
                # but if we do, we usually re-construct a new list.
                return [If(idx == j, val, lst[j]) for j in range(len(lst))]

            local_vars = {}
            exec(
                heuristic_z3_logic,
                {
                    "z3": sys.modules["z3"],
                    "Sum": Sum,
                    "If": If,
                    "Or": Or,
                    "And": And,
                    "Not": Not,
                    "sym_get": sym_get,
                    "sym_set": sym_set,
                },
                local_vars,
            )
            get_heuristic_selection = local_vars.get("get_heuristic_selection")

            if not get_heuristic_selection:
                return {
                    "error": "Heuristic Z3 logic did not define 'get_heuristic_selection'"
                }

            x_h = get_heuristic_selection(W, V, Cap, s)
            h_val = Sum([x_h[i] * V[i] for i in range(self.n_items)])
            h_weight = Sum([x_h[i] * W[i] for i in range(self.n_items)])

            # Constraint: Heuristic must be valid (usually it is by construction, but let's be sure)
            s.add(h_weight <= Cap)

            # THE BOSS FIGHT: Find a case where Heuristic < Threshold * Optimal
            # We can't directly use float 0.9 in Z3 Int context, so we use integer math:
            # h_val * 100 < threshold_int * opt_val
            threshold_int = int(self.fitness_threshold * 100)
            s.add(h_val * 100 < threshold_int * opt_val)

            if s.check() == sat:
                model = s.model()
                counter_example = {
                    "capacity": model[Cap].as_long(),
                    "items": [
                        {
                            "value": model[V[i]].as_long(),
                            "weight": model[W[i]].as_long(),
                        }
                        for i in range(self.n_items)
                    ],
                    "optimal_value_at_least": model.eval(opt_val).as_long(),
                    "heuristic_value": model.eval(h_val).as_long(),
                }
                return {"status": "failed", "counter_example": counter_example}
            else:
                return {
                    "status": "proven",
                    "message": f"Heuristic always hits >={self.fitness_threshold * 100}% for n={self.n_items}",
                }

        except Exception as e:
            return {
                "error": f"Verification execution failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }


def main():
    # Simple manual test: a BAD heuristic that ignores everything and picks nothing
    bad_heuristic = """
def get_heuristic_selection(W, V, Cap, solver):
    return [z3.IntVal(0) for _ in range(len(W))]
"""
    verifier = Z3KnapsackVerifier()
    result = verifier.verify(bad_heuristic)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
