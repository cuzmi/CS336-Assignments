from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def analyze_isoflops() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "isoflops_curves.json"
    runs = json.loads(data_path.read_text(encoding="utf-8"))

    # Step 1: data
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["compute_budget"]].append(run)

    # Step 2: 遭到固定budget下的最小loss
    optimal_points = []
    for budget, budget_runs in sorted(grouped.items()):
        best = min(budget_runs, key=lambda run: run["final_loss"])
        n_opt = best["parameters"]
        d_opt = budget / (6 * n_opt)  # Use C = 6ND to recover the optimal token count.
        optimal_points.append((budget, n_opt, d_opt, best["final_loss"]))

    budgets = [point[0] for point in optimal_points]
    optimal_params = [point[1] for point in optimal_points]
    optimal_tokens = [point[2] for point in optimal_points]

    # Step 3: log线性fit
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("This script needs numpy. Install it first.") from exc

    param_slope, param_intercept = np.polyfit(np.log(budgets), np.log(optimal_params), 1)
    token_slope, token_intercept = np.polyfit(np.log(budgets), np.log(optimal_tokens), 1)

    # Step 4: 外推
    for target_budget in [1e23, 1e24]:
        pred_n = float(np.exp(param_intercept) * target_budget**param_slope)
        pred_d = float(np.exp(token_intercept) * target_budget**token_slope)
        print(f"C = {target_budget:.0e}")
        print(f"  predicted N_opt = {pred_n:.6e}")
        print(f"  predicted D_opt = {pred_d:.6e}")

    print("\nOptimal points used for fitting:")
    for budget, n_opt, d_opt, loss in optimal_points:
        print(
            f"C={budget:.0e}, N_opt={n_opt:.6e}, D_opt={d_opt:.6e}, loss={loss:.6f}"
        )

    print("\nFitted scaling laws:")
    print(f"N_opt(C) = {np.exp(param_intercept):.6e} * C^{param_slope:.6f}")
    print(f"D_opt(C) = {np.exp(token_intercept):.6e} * C^{token_slope:.6f}")


if __name__ == "__main__":
    analyze_isoflops()
