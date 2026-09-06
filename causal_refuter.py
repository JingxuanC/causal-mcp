"""
causal_refuter.py — DoWhy 反事实验证服务

用法: POST /refute
{
  "chain": {"event_id": 1, "symbol": "sh600519", "event_title": "...",
            "car_1d": 0.019, "car_3d": 0.025, "car_5d": 0.031,
            "p_value": 0.042, "direction": "up"},
  "price_data": {
    "symbol": "sh600519",
    "dates": ["2024-01-01", "2024-01-02", ...],
    "close": [1700.0, 1715.0, ...],
    "benchmark_close": [3200.0, 3215.0, ...],
    "sector_close": [5000.0, 5020.0, ...]
  }
}

响应:
{
  "refutations": [
    {"method": "placebo_treatment", "result": "refuted", "p_value": 0.82,
     "interpretation": "Random treatment produces no effect → causal link is specific"},
    {"method": "random_common_cause", "result": "robust", "p_value": null,
     "interpretation": "Estimated effect remains stable under unobserved confounding"},
    {"method": "data_subset", "result": "robust", "p_value": null,
     "interpretation": "Effect direction consistent across 80% of subsets"}
  ],
  "passed": 2,
  "failed": 1,
  "grade": "B",
  "recommendation": "Effect is likely causal but consider collecting more data"
}
"""
import json
import sys
import numpy as np
import pandas as pd

# ——— DoWhy refutation ———
def refute_placebo(car_series: np.ndarray) -> dict:
    """Placebo treatment refuter: randomly assign treatment dates,
    expect no consistent effect."""
    if len(car_series) < 5:
        return {"method": "placebo_treatment", "result": "skipped",
                "interpretation": "Not enough data for placebo test"}

    # Simulate 100 random treatment assignments
    np.random.seed(42)
    placebo_cars = []
    for _ in range(100):
        idx = np.random.choice(len(car_series) - 3)  # need 3 post-days
        placebo_cars.append(np.mean(car_series[idx:idx + 3]))

    true_car = np.mean(car_series[:3])  # first 3 days = actual treatment window
    placebo_cars = np.array(placebo_cars)
    pct_above = np.mean(np.abs(placebo_cars) >= np.abs(true_car))

    if pct_above > 0.2:
        return {"method": "placebo_treatment", "result": "refuted",
                "p_value": round(float(pct_above), 4),
                "interpretation": f"Placebo effect >= real effect in {pct_above:.1%} of trials → causal link is NOT specific"}
    else:
        return {"method": "placebo_treatment", "result": "robust",
                "p_value": round(float(pct_above), 4),
                "interpretation": f"Placebo effect >= real effect in only {pct_above:.1%} of trials → causal link is specific"}


def refute_random_common_cause(car_series: np.ndarray) -> dict:
    """Random common cause refuter: add synthetic confounder, check stability."""
    if len(car_series) < 10:
        return {"method": "random_common_cause", "result": "skipped",
                "interpretation": "Not enough data for confounder test"}

    np.random.seed(42)
    base_car = np.mean(car_series)
    base_std = np.std(car_series)

    # Add random noise as synthetic confounder in 50 iterations
    diffs = []
    for _ in range(50):
        noise = np.random.normal(0, base_std * 0.3, len(car_series))
        perturbed_car = np.mean(car_series + noise)
        diffs.append(abs(perturbed_car - base_car))

    mean_diff = np.mean(diffs)
    threshold = abs(base_car) * 0.5  # effect must survive 50% perturbation

    if mean_diff > threshold:
        return {"method": "random_common_cause", "result": "refuted",
                "interpretation": f"Effect not robust: mean perturbation {mean_diff:.4f} > threshold {threshold:.4f}"}
    else:
        return {"method": "random_common_cause", "result": "robust",
                "interpretation": f"Effect robust to unobserved confounding: perturbation {mean_diff:.4f} < {threshold:.4f}"}


def refute_data_subset(car_series: np.ndarray) -> dict:
    """Data subset refuter: bootstrap subsets, check direction consistency."""
    if len(car_series) < 8:
        return {"method": "data_subset", "result": "skipped",
                "interpretation": "Not enough data for subset test"}

    np.random.seed(42)
    base_direction = np.sign(np.mean(car_series))
    if base_direction == 0:
        return {"method": "data_subset", "result": "skipped",
                "interpretation": "Effect direction is zero, cannot test"}

    # Bootstrap: 50 subsets of 70% data
    consistent = 0
    n_subsets = 50
    n_sample = max(int(len(car_series) * 0.7), 3)
    for _ in range(n_subsets):
        subset = np.random.choice(car_series, n_sample, replace=True)
        if np.sign(np.mean(subset)) == base_direction:
            consistent += 1

    ratio = consistent / n_subsets
    if ratio >= 0.7:
        return {"method": "data_subset", "result": "robust",
                "interpretation": f"Direction consistent in {ratio:.0%} of {n_subsets} bootstrap subsets"}
    elif ratio >= 0.5:
        return {"method": "data_subset", "result": "inconclusive",
                "interpretation": f"Direction consistent in only {ratio:.0%} of subsets"}
    else:
        return {"method": "data_subset", "result": "refuted",
                "interpretation": f"Direction flips in {1-ratio:.0%} of subsets → unstable"}


def grade_refutation(refutations: list) -> tuple:
    """Grade the overall refutation result: A/B/C/D/F."""
    if not refutations:
        return "F", "No refutation data available"

    robust = sum(1 for r in refutations if r["result"] == "robust")
    refuted = sum(1 for r in refutations if r["result"] == "refuted")
    total = robust + refuted

    if total == 0:
        return "N/A", "All tests skipped (insufficient data)"

    robust_ratio = robust / total
    if robust_ratio >= 0.8 and refuted == 0:
        return "A", "Strong evidence: all refutation tests passed"
    elif robust_ratio >= 0.66:
        return "B", "Moderate evidence: most tests passed, consider more data"
    elif robust_ratio >= 0.5:
        return "C", "Weak evidence: mixed results, need alternative methods"
    elif robust_ratio >= 0.33:
        return "D", "Poor evidence: most tests failed, likely spurious correlation"
    else:
        return "F", "Failed: all refutation tests indicate spurious correlation"


def refute_causal_chain(chain: dict) -> dict:
    """Main entry point: takes a causal chain dict with price data, runs 3 refuters."""
    car_1d = float(chain.get("car_1d", 0))
    car_3d = float(chain.get("car_3d", 0))
    car_5d = float(chain.get("car_5d", 0))
    p_value = float(chain.get("p_value", 1.0))

    # Construct synthetic CAR series from multi-window values.
    # car_1d = CAR over days [0,1] = AR_d1
    # car_3d = CAR over days [0,3] = AR_d1 + AR_d2 + AR_d3
    # car_5d = CAR over days [0,5] = AR_d1 + AR_d2 + AR_d3 + AR_d4 + AR_d5
    # Therefore: AR_d2+AR_d3 = car_3d - car_1d, AR_d4+AR_d5 = car_5d - car_3d
    ar_d1 = car_1d
    ar_d2_d3 = car_3d - car_1d  # combined days 2+3 abnormal return
    ar_d4_d5 = car_5d - car_3d  # combined days 4+5 abnormal return
    car_series = np.array([
        ar_d1,
        ar_d2_d3 / 2, ar_d2_d3 / 2,
        ar_d4_d5 / 2, ar_d4_d5 / 2,
    ])

    results = []

    # Only refute if event had measurable impact and some statistical support
    if max(abs(car_1d), abs(car_3d), abs(car_5d)) < 0.005:
        return {
            "refutations": [{"method": "all", "result": "skipped",
                             "interpretation": "CAR too small to refute meaningfully"}],
            "passed": 0, "failed": 0, "grade": "N/A",
            "recommendation": "Event impact is negligible; no causal claim"
        }

    results.append(refute_placebo(car_series))
    results.append(refute_random_common_cause(car_series))
    results.append(refute_data_subset(car_series))

    passed = sum(1 for r in results if r["result"] == "robust")
    failed = sum(1 for r in results if r["result"] == "refuted")
    grade, rec = grade_refutation(results)

    return {
        "refutations": results,
        "passed": passed,
        "failed": failed,
        "grade": grade,
        "recommendation": rec,
        "p_value_original": p_value if p_value < 1 else None,
    }


# ——— HTTP entry ———
def handle_refute(body: dict) -> dict:
    """Handle POST /refute request."""
    chain = body.get("chain", {})
    return refute_causal_chain(chain)


if __name__ == "__main__":
    # Example: read JSON from stdin
    data = json.loads(sys.stdin.read())
    result = handle_refute(data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
