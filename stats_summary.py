import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class ConditionStats:
    # Process: Aggregate Stats Schema
    # Purpose: Store per-condition summary metrics for export and reporting.
    condition_id: str
    family: str
    severity: str
    samples: int
    success_rate: float
    within_tolerance_rate: float
    exact_match_rate: float
    fallback_rate: float
    mean_hour_error: float
    mean_minute_error: float
    mean_second_error: float


# Process: Boolean Parsing Utility
# Purpose: Normalize common truthy string forms to bool.
def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# Process: Numeric Parsing Utility
# Purpose: Convert scalar text to float with NaN for missing values.
def _to_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


# Process: Mean with NaN Guard
# Purpose: Compute mean after filtering missing numeric values.
def _safe_mean(values: Sequence[float]) -> float:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return float("nan")
    return sum(filtered) / len(filtered)


# Process: Metrics Ingestion
# Purpose: Load row-wise experiment metrics from CSV.
def read_metrics_rows(metrics_csv: Path) -> List[Dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Process: Condition-Level Aggregation
# Purpose: Group rows by condition and compute summary rates/errors.
def compute_condition_stats(rows: Iterable[Dict[str, str]]) -> List[ConditionStats]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_id"]].append(row)

    stats: List[ConditionStats] = []
    for condition_id, group in sorted(grouped.items()):
        family = group[0].get("family", "unknown")
        severity = group[0].get("severity", "unknown")
        samples = len(group)

        success_rate = sum(_to_bool(r.get("success", "false")) for r in group) / samples
        within_rate = sum(_to_bool(r.get("within_tolerance", "false")) for r in group) / samples
        exact_rate = sum(_to_bool(r.get("exact_time_match", "false")) for r in group) / samples
        fallback_rate = sum(_to_bool(r.get("fallback_used", "false")) for r in group) / samples

        hour_err = _safe_mean([_to_float(r.get("hour_error", "")) for r in group])
        minute_err = _safe_mean([_to_float(r.get("minute_error", "")) for r in group])
        second_err = _safe_mean([_to_float(r.get("second_error", "")) for r in group])

        stats.append(
            ConditionStats(
                condition_id=condition_id,
                family=family,
                severity=severity,
                samples=samples,
                success_rate=success_rate,
                within_tolerance_rate=within_rate,
                exact_match_rate=exact_rate,
                fallback_rate=fallback_rate,
                mean_hour_error=hour_err,
                mean_minute_error=minute_err,
                mean_second_error=second_err,
            )
        )

    return stats


# Process: Metric Extraction
# Purpose: Pull one metric vector for a specific condition ID.
def _extract_metric(rows: Sequence[Dict[str, str]], condition_id: str, key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if row.get("condition_id") != condition_id:
            continue
        if key in {"success", "within_tolerance", "exact_time_match", "fallback_used"}:
            values.append(1.0 if _to_bool(row.get(key, "false")) else 0.0)
        else:
            values.append(_to_float(row.get(key, "")))
    return [v for v in values if not math.isnan(v)]


# Process: Permutation Significance Test
# Purpose: Estimate p-value for absolute mean difference between two samples.
def permutation_test_mean_diff(
    a: Sequence[float], b: Sequence[float], iterations: int, rng: random.Random
) -> float:
    if not a or not b:
        return float("nan")

    observed = abs((sum(a) / len(a)) - (sum(b) / len(b)))
    pooled = list(a) + list(b)
    n_a = len(a)
    count = 0

    for _ in range(iterations):
        rng.shuffle(pooled)
        a_s = pooled[:n_a]
        b_s = pooled[n_a:]
        diff = abs((sum(a_s) / len(a_s)) - (sum(b_s) / len(b_s)))
        if diff >= observed:
            count += 1

    return (count + 1) / (iterations + 1)


# Process: Bootstrap Confidence Interval
# Purpose: Estimate 95% CI for mean delta via resampling.
def bootstrap_delta_ci(
    a: Sequence[float], b: Sequence[float], iterations: int, rng: random.Random
) -> Tuple[float, float]:
    if not a or not b:
        return (float("nan"), float("nan"))

    deltas: List[float] = []
    for _ in range(iterations):
        a_s = [a[rng.randrange(len(a))] for _ in range(len(a))]
        b_s = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append((sum(a_s) / len(a_s)) - (sum(b_s) / len(b_s)))

    deltas.sort()
    lo_idx = int(0.025 * (len(deltas) - 1))
    hi_idx = int(0.975 * (len(deltas) - 1))
    return deltas[lo_idx], deltas[hi_idx]


# Process: CSV Summary Export
# Purpose: Write per-condition aggregate statistics to CSV.
def write_condition_stats_csv(stats: Sequence[ConditionStats], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition_id",
                "family",
                "severity",
                "samples",
                "success_rate",
                "within_tolerance_rate",
                "exact_match_rate",
                "fallback_rate",
                "mean_hour_error",
                "mean_minute_error",
                "mean_second_error",
            ]
        )
        for item in stats:
            writer.writerow(
                [
                    item.condition_id,
                    item.family,
                    item.severity,
                    item.samples,
                    f"{item.success_rate:.6f}",
                    f"{item.within_tolerance_rate:.6f}",
                    f"{item.exact_match_rate:.6f}",
                    f"{item.fallback_rate:.6f}",
                    f"{item.mean_hour_error:.6f}",
                    f"{item.mean_minute_error:.6f}",
                    f"{item.mean_second_error:.6f}",
                ]
            )


# Process: Markdown Report Generation
# Purpose: Build overview and baseline-comparison statistical report.
def generate_stats_report(
    metrics_csv: Path,
    output_md: Path,
    output_csv: Path,
    baseline_condition: str,
    iterations: int,
    seed: int,
) -> None:
    # Process: Data Preparation
    # Purpose: Read metrics, compute aggregates, and persist CSV summary.
    rows = read_metrics_rows(metrics_csv)
    stats = compute_condition_stats(rows)
    write_condition_stats_csv(stats, output_csv)

    # Process: Report Setup
    # Purpose: Initialize RNG, output folder, and baseline metric vector.
    rng = random.Random(seed)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    baseline_within = _extract_metric(rows, baseline_condition, "within_tolerance")

    # Process: Markdown Assembly
    # Purpose: Compose section headers and condition overview table.
    lines: List[str] = []
    lines.append("# Statistical Summary")
    lines.append("")
    lines.append(f"Baseline condition: {baseline_condition}")
    lines.append(f"Permutation iterations: {iterations}")
    lines.append("")
    lines.append("## Condition Overview")
    lines.append("")
    lines.append("| condition_id | family | severity | n | success | within_tol | exact_match | fallback | mean_hour_err | mean_minute_err | mean_second_err |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for item in stats:
        lines.append(
            "| "
            f"{item.condition_id} | {item.family} | {item.severity} | {item.samples} | "
            f"{item.success_rate:.3f} | {item.within_tolerance_rate:.3f} | {item.exact_match_rate:.3f} | {item.fallback_rate:.3f} | "
            f"{item.mean_hour_error:.3f} | {item.mean_minute_error:.3f} | {item.mean_second_error:.3f} |"
        )

    # Process: Baseline Pairwise Analysis
    # Purpose: Compare within-tolerance rates against baseline with CI and p-value.
    lines.append("")
    lines.append("## Baseline vs Condition (within_tolerance)")
    lines.append("")
    lines.append("| condition_id | delta_mean (baseline - condition) | 95% CI | permutation p-value |")
    lines.append("|---|---:|---|---:|")

    for item in stats:
        if item.condition_id == baseline_condition:
            continue

        cond_within = _extract_metric(rows, item.condition_id, "within_tolerance")
        if not baseline_within or not cond_within:
            lines.append(f"| {item.condition_id} | n/a | n/a | n/a |")
            continue

        delta = (sum(baseline_within) / len(baseline_within)) - (sum(cond_within) / len(cond_within))
        ci_low, ci_high = bootstrap_delta_ci(baseline_within, cond_within, iterations, rng)
        p_value = permutation_test_mean_diff(baseline_within, cond_within, iterations, rng)

        lines.append(
            f"| {item.condition_id} | {delta:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {p_value:.4f} |"
        )

    # Process: Report Persist
    # Purpose: Write completed markdown report to disk.
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Process: CLI Entrypoint
# Purpose: Parse arguments and execute stats report generation.
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stats summary from metrics_by_condition.csv")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to metrics_by_condition.csv")
    parser.add_argument("--output-md", type=Path, default=Path("results/stats_summary.md"), help="Output markdown summary")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stats_by_condition.csv"),
        help="Output CSV with per-condition aggregate stats",
    )
    parser.add_argument(
        "--baseline-condition",
        default="baseline_clean",
        help="Condition ID used as baseline in pairwise comparisons",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="Permutation/bootstrap iterations")
    parser.add_argument("--seed", type=int, default=20260721, help="Random seed")
    args = parser.parse_args()

    generate_stats_report(
        metrics_csv=args.metrics_csv,
        output_md=args.output_md,
        output_csv=args.output_csv,
        baseline_condition=args.baseline_condition,
        iterations=args.iterations,
        seed=args.seed,
    )

    print(f"[OK] Wrote stats markdown: {args.output_md}")
    print(f"[OK] Wrote stats csv: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
