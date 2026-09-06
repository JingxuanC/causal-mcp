"""
causal_graph.py — Causal graph structure learning (NOTEARS / PC algorithm)

用途: 从价格/因子数据中自动发现因果图结构，补充人工知识图谱。

用法: POST /learn-graph
{
  "data": {
    "columns": ["sh600519", "sz000858", "sh600036", "sz002415"],
    "rows": [[1700, 185, 35.5, 42.3], [1715, 187, 36.0, 43.1], ...],
    "dates": ["2024-01-01", "2024-01-02", ...]
  },
  "method": "pc",  // "pc" or "notears"
  "alpha": 0.05    // significance level for PC algorithm
}

响应:
{
  "graph": {
    "edges": [
      {"from": "sh600519", "to": "sz000858", "weight": 0.85, "direction": "→"},
      {"from": "sh600519", "to": "sh600036", "weight": 0.62, "direction": "→"},
      {"from": "sz002415", "to": "sh600036", "weight": 0.45, "direction": "→"}
    ],
    "nodes": ["sh600519", "sz000858", "sh600036", "sz002415"],
    "method": "pc"
  },
  "metrics": {
    "n_edges": 3,
    "sparsity": 0.5,
    "avg_weight": 0.64
  }
}
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats


def partial_corr(data: np.ndarray, i: int, j: int, cond_set: list) -> float:
    """Compute partial correlation between i and j given conditioning set."""
    n = data.shape[0]
    if not cond_set:
        return float(np.corrcoef(data[:, i], data[:, j])[0, 1])

    # Regress both vars on conditioning set, correlate residuals
    X_cond = np.column_stack([data[:, k] for k in cond_set])
    X_cond = np.column_stack([np.ones(n), X_cond])

    for target_col in [i, j]:
        y = data[:, target_col]
        beta = np.linalg.lstsq(X_cond, y, rcond=None)[0]
        resid = y - X_cond @ beta
        if target_col == i:
            resid_i = resid
        else:
            resid_j = resid

    corr = np.corrcoef(resid_i, resid_j)[0, 1]
    return float(corr if not np.isnan(corr) else 0.0)


def pc_algorithm(data: np.ndarray, columns: list, alpha: float = 0.05) -> dict:
    """
    PC algorithm (Peter & Clark) for causal graph structure learning.
    Start with fully connected graph, iteratively remove edges when
    variables are conditionally independent.
    """
    n_vars = data.shape[1]
    n_samples = data.shape[0]

    # Initialize fully connected undirected graph
    edges = set()
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            edges.add((i, j))

    # Also track adjacency
    adj = {i: set() for i in range(n_vars)}
    for i, j in list(edges):
        adj[i].add(j)
        adj[j].add(i)

    max_cond = min(n_vars - 2, 3)  # limit conditioning set size

    for k in range(max_cond + 1):
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in list(adj[i]):
                    if j <= i:
                        continue
                    # Find neighbors of i (excluding j)
                    neighbors = sorted(adj[i] - {j})
                    if len(neighbors) < k:
                        continue
                    # Try all conditioning sets of size k
                    from itertools import combinations
                    for cond in combinations(neighbors, k):
                        cond_list = list(cond)
                        pc = partial_corr(data, i, j, cond_list)
                        # Fisher z-test for significance
                        z = 0.5 * np.log((1 + pc) / (1 - pc + 1e-10))
                        se = 1.0 / np.sqrt(n_samples - len(cond_list) - 3)
                        p_val = 2 * (1 - stats.norm.cdf(abs(z / se)))
                        if p_val > alpha:
                            edges.discard((min(i, j), max(i, j)))
                            adj[i].discard(j)
                            adj[j].discard(i)
                            changed = True
                            break

    # Build result
    result_edges = []
    for (i, j) in edges:
        corr = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
        # Determine direction by correlation sign
        direction = "→" if abs(corr) > 0.1 else "—"
        result_edges.append({
            "from": columns[i],
            "to": columns[j],
            "weight": round(abs(corr) if not np.isnan(corr) else 0, 4),
            "correlation": round(float(corr) if not np.isnan(corr) else 0, 4),
            "direction": direction,
        })

    return {
        "edges": sorted(result_edges, key=lambda e: e["weight"], reverse=True),
        "nodes": columns,
        "method": "pc",
        "alpha": alpha,
    }


def note_ars_algorithm(data: np.ndarray, columns: list, lambda1: float = 0.1) -> dict:
    """
    NOTEARS: Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian
    for Structure learning. Simplified implementation using regression + threshold.

    Full NOTEARS requires solving constrained optimization with matrix exponential.
    This simplified version uses Lasso regression to identify potential parents.
    """
    n_vars = data.shape[1]
    edges = []

    for i in range(n_vars):
        y = data[:, i]
        X = np.delete(data, i, axis=1)
        other_cols = [c for idx, c in enumerate(columns) if idx != i]

        # Ridge regression to get weights (simplified from actual NOTEARS)
        from sklearn.linear_model import Ridge
        try:
            model = Ridge(alpha=lambda1)
            model.fit(X, y)
            coefs = model.coef_
        except Exception:
            # Fallback: simple correlation
            coefs = np.array([float(np.corrcoef(y, X[:, j])[0, 1])
                             for j in range(X.shape[1])])

        for j, coef in enumerate(coefs):
            if abs(coef) > 0.05:  # threshold weak edges
                edges.append({
                    "from": other_cols[j],
                    "to": columns[i],
                    "weight": round(abs(float(coef)), 4),
                    "direction": "→",
                })

    return {
        "edges": sorted(edges, key=lambda e: e["weight"], reverse=True),
        "nodes": columns,
        "method": "notears_ridge",
    }


def learn_graph(body: dict) -> dict:
    """Main entry: learn causal graph from tabular data."""
    raw = body.get("data", {})
    columns = raw.get("columns", [])
    rows = raw.get("rows", [])
    method = body.get("method", "pc")
    alpha = body.get("alpha", 0.05)

    if len(columns) < 2 or len(rows) < 5:
        return {"error": "Need at least 2 columns and 5 rows of data"}

    data = np.array(rows, dtype=np.float64)

    if method == "notears":
        result = note_ars_algorithm(data, columns)
    else:
        result = pc_algorithm(data, columns, alpha)

    # Add metrics
    n_edges = len(result["edges"])
    n_possible = len(columns) * (len(columns) - 1) // 2
    weights = [e["weight"] for e in result["edges"]]

    result["metrics"] = {
        "n_edges": n_edges,
        "sparsity": round(1 - n_edges / max(n_possible, 1), 4),
        "avg_weight": round(float(np.mean(weights)) if weights else 0, 4),
    }

    return result


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    result = learn_graph(data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
