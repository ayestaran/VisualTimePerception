import argparse
import csv
import json
import math
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from symbolic_clock import SymbolicDetectionError, interpret_clock_image
from stats_summary import generate_stats_report


@dataclass(frozen=True)
class Condition:
    condition_id: str
    family: str
    severity: str
    description: str
    transform: Callable[[np.ndarray, random.Random], Tuple[np.ndarray, bool]]


def angle_to_point(center: Tuple[int, int], length: int, angle_deg: float) -> Tuple[int, int]:
    theta = np.deg2rad(angle_deg)
    x = int(round(center[0] + np.sin(theta) * length))
    y = int(round(center[1] - np.cos(theta) * length))
    return x, y


def make_synthetic_clock_image(hour: int, minute: int, second: int, size: int) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    cv2.circle(image, center, radius, (30, 95, 139), 8, lineType=cv2.LINE_AA)

    minute_angle = ((minute % 60) + (second % 60) / 60.0) * 6.0
    hour_angle = ((hour % 12) + (minute % 60) / 60.0 + (second % 60) / 3600.0) * 30.0
    second_angle = (second % 60) * 6.0

    minute_tip = angle_to_point(center, int(radius * 0.72), minute_angle)
    hour_tip = angle_to_point(center, int(radius * 0.48), hour_angle)
    second_tip = angle_to_point(center, int(radius * 0.78), second_angle)

    cv2.line(image, center, hour_tip, (20, 40, 65), 10, lineType=cv2.LINE_AA)
    cv2.line(image, center, minute_tip, (18, 62, 107), 7, lineType=cv2.LINE_AA)
    cv2.line(image, center, second_tip, (40, 40, 200), 3, lineType=cv2.LINE_AA)
    cv2.circle(image, center, 8, (20, 40, 65), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, center, 3, (40, 40, 200), -1, lineType=cv2.LINE_AA)

    return image


def cyclic_minute_delta(a: int, b: int) -> int:
    raw = abs((a % 60) - (b % 60))
    return min(raw, 60 - raw)


def cyclic_hour_delta(a: int, b: int) -> int:
    raw = abs((a % 12) - (b % 12))
    return min(raw, 12 - raw)


def cyclic_second_delta(a: int, b: int) -> int:
    raw = abs((a % 60) - (b % 60))
    return min(raw, 60 - raw)


def _identity(image: np.ndarray, _rng: random.Random) -> Tuple[np.ndarray, bool]:
    return image.copy(), False


def _rotate(degrees: float) -> Callable[[np.ndarray, random.Random], Tuple[np.ndarray, bool]]:
    def _inner(image: np.ndarray, _rng: random.Random) -> Tuple[np.ndarray, bool]:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return rotated, False

    return _inner


def _gaussian_blur(ksize: int) -> Callable[[np.ndarray, random.Random], Tuple[np.ndarray, bool]]:
    def _inner(image: np.ndarray, _rng: random.Random) -> Tuple[np.ndarray, bool]:
        return cv2.GaussianBlur(image, (ksize, ksize), 0), False

    return _inner


def _noise(std_dev: float) -> Callable[[np.ndarray, random.Random], Tuple[np.ndarray, bool]]:
    def _inner(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool]:
        noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        noise = noise_rng.normal(0.0, std_dev, image.shape).astype(np.float32)
        noised = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noised, False

    return _inner


def _distractor_line(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool]:
    out = image.copy()
    h, w = out.shape[:2]
    x1 = rng.randint(0, w - 1)
    y1 = rng.randint(0, h - 1)
    x2 = rng.randint(0, w - 1)
    y2 = rng.randint(0, h - 1)
    cv2.line(out, (x1, y1), (x2, y2), (25, 25, 25), 2, lineType=cv2.LINE_AA)
    return out, False


def _mirror_tracked(image: np.ndarray, _rng: random.Random) -> Tuple[np.ndarray, bool]:
    return cv2.flip(image, 1), True


def _mirror_untracked(image: np.ndarray, _rng: random.Random) -> Tuple[np.ndarray, bool]:
    return cv2.flip(image, 1), False


def _occlusion_patch(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool]:
    out = image.copy()
    h, w = out.shape[:2]
    patch_w = max(12, int(w * 0.14))
    patch_h = max(12, int(h * 0.08))
    x = rng.randint(int(w * 0.2), int(w * 0.8) - patch_w)
    y = rng.randint(int(h * 0.2), int(h * 0.8) - patch_h)
    cv2.rectangle(out, (x, y), (x + patch_w, y + patch_h), (240, 240, 240), -1)
    return out, False


def _combined_stress(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool]:
    rotated, _ = _rotate(12.0)(image, rng)
    blurred, _ = _gaussian_blur(5)(rotated, rng)
    noised, _ = _noise(10.0)(blurred, rng)
    lined, _ = _distractor_line(noised, rng)
    return lined, False


def build_conditions() -> List[Condition]:
    return [
        Condition("baseline_clean", "baseline", "none", "Clean synthetic render", _identity),
        Condition("rotate_8deg", "rotation", "low", "Rotate clock image by 8 degrees", _rotate(8.0)),
        Condition("rotate_16deg", "rotation", "medium", "Rotate clock image by 16 degrees", _rotate(16.0)),
        Condition("gaussian_blur_5", "blur", "medium", "Gaussian blur kernel 5", _gaussian_blur(5)),
        Condition("noise_std_8", "noise", "medium", "Gaussian noise std=8", _noise(8.0)),
        Condition("distractor_line", "artifact", "medium", "Add extra dark line artifact", _distractor_line),
        Condition("mirror_tracked", "orientation", "low", "Mirrored image with correct metadata", _mirror_tracked),
        Condition("mirror_untracked", "orientation", "high", "Mirrored image with incorrect metadata", _mirror_untracked),
        Condition("occlusion_patch", "occlusion", "medium", "Local bright rectangular occlusion", _occlusion_patch),
        Condition("combined_stress", "combined", "high", "Rotation + blur + noise + line artifact", _combined_stress),
    ]


def get_condition_preview_images(
    hour: int = 10,
    minute: int = 10,
    second: int = 30,
    image_size: int = 200,
    seed: int = 42,
) -> List[dict]:
    """Return one base64 PNG thumbnail per condition for UI display."""
    import base64 as _base64

    rng = random.Random(seed)
    conditions = build_conditions()
    base_image = make_synthetic_clock_image(hour, minute, second, image_size)
    previews = []
    for condition in conditions:
        transformed, _ = condition.transform(base_image, rng)
        mime_type = "image/png"
        ok, buf = cv2.imencode(".png", transformed)
        if not ok:
            mime_type = "image/jpeg"
            ok, buf = cv2.imencode(".jpg", transformed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            continue
        b64 = _base64.b64encode(buf.tobytes()).decode("ascii")
        previews.append(
            {
                "condition_id": condition.condition_id,
                "description": condition.description,
                "family": condition.family,
                "severity": condition.severity,
                "imageData": f"data:{mime_type};base64,{b64}",
            }
        )
    return previews


def write_condition_matrix(path: Path, conditions: List[Condition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition_id", "family", "severity", "description"])
        for condition in conditions:
            writer.writerow([condition.condition_id, condition.family, condition.severity, condition.description])


def classify_failure(exc: Exception) -> str:
    if isinstance(exc, SymbolicDetectionError):
        return "symbolic_detection_error"
    if isinstance(exc, RuntimeError):
        return "runtime_error"
    return "unexpected_error"


def run_experiments(
    output_dir: Path,
    samples_per_condition: int,
    seed: int,
    image_size: int,
    hour_tol: int,
    minute_tol: int,
    second_tol: int,
    with_stats: bool,
    stats_iterations: int,
) -> Tuple[Path, Path, Path]:
    rng = random.Random(seed)
    conditions = build_conditions()

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_matrix_path = output_dir / "condition_matrix.csv"
    metrics_path = output_dir / "metrics_by_condition.csv"
    stats_md_path = output_dir / "stats_summary.md"
    stats_csv_path = output_dir / "stats_by_condition.csv"

    write_condition_matrix(condition_matrix_path, conditions)

    fieldnames = [
        "run_id",
        "run_timestamp_utc",
        "seed",
        "condition_id",
        "family",
        "severity",
        "sample_index",
        "expected_hour",
        "expected_minute",
        "expected_second",
        "pred_hour",
        "pred_minute",
        "pred_second",
        "pred_time",
        "success",
        "within_tolerance",
        "exact_time_match",
        "hour_error",
        "minute_error",
        "second_error",
        "fallback_used",
        "fallback_mode",
        "candidate_count",
        "color_hour_detected",
        "color_minute_detected",
        "color_second_detected",
        "failure_type",
        "failure_message",
    ]

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for condition in conditions:
            for sample_idx in range(samples_per_condition):
                expected_hour = rng.randint(1, 12)
                expected_minute = rng.randint(0, 59)
                expected_second = rng.randint(0, 59)
                base_image = make_synthetic_clock_image(
                    hour=expected_hour,
                    minute=expected_minute,
                    second=expected_second,
                    size=image_size,
                )

                transformed, was_flipped = condition.transform(base_image, rng)

                row = {
                    "run_id": run_id,
                    "run_timestamp_utc": run_timestamp,
                    "seed": seed,
                    "condition_id": condition.condition_id,
                    "family": condition.family,
                    "severity": condition.severity,
                    "sample_index": sample_idx,
                    "expected_hour": expected_hour,
                    "expected_minute": expected_minute,
                    "expected_second": expected_second,
                    "pred_hour": "",
                    "pred_minute": "",
                    "pred_second": "",
                    "pred_time": "",
                    "success": False,
                    "within_tolerance": False,
                    "exact_time_match": False,
                    "hour_error": "",
                    "minute_error": "",
                    "second_error": "",
                    "fallback_used": False,
                    "fallback_mode": "",
                    "candidate_count": "",
                    "color_hour_detected": "",
                    "color_minute_detected": "",
                    "color_second_detected": "",
                    "failure_type": "",
                    "failure_message": "",
                }

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)

                try:
                    if not cv2.imwrite(str(temp_path), transformed):
                        raise RuntimeError("Failed to write temporary experiment image")

                    symbolic = interpret_clock_image(temp_path, was_flipped=was_flipped)
                    pred_hour = int(symbolic.get("HOUR_HAND", -1))
                    pred_minute = int(symbolic.get("MINUTE_HAND", -1))
                    pred_second = int(symbolic.get("SECOND_HAND", -1))

                    hour_error = cyclic_hour_delta(pred_hour, expected_hour)
                    minute_error = cyclic_minute_delta(pred_minute, expected_minute)
                    second_error = cyclic_second_delta(pred_second, expected_second)

                    within_tol = (
                        hour_error <= hour_tol
                        and minute_error <= minute_tol
                        and second_error <= second_tol
                    )

                    exact_match = (
                        pred_hour == expected_hour
                        and pred_minute == expected_minute
                        and pred_second == expected_second
                    )

                    debug_metrics = symbolic.get("debug_metrics") or {}

                    row.update(
                        {
                            "pred_hour": pred_hour,
                            "pred_minute": pred_minute,
                            "pred_second": pred_second,
                            "pred_time": symbolic.get("TIME", ""),
                            "success": True,
                            "within_tolerance": within_tol,
                            "exact_time_match": exact_match,
                            "hour_error": hour_error,
                            "minute_error": minute_error,
                            "second_error": second_error,
                            "fallback_used": bool(debug_metrics.get("fallback_used", False)),
                            "fallback_mode": debug_metrics.get("fallback_mode", "none"),
                            "candidate_count": debug_metrics.get("candidate_count", ""),
                            "color_hour_detected": bool(debug_metrics.get("color_hour_detected", False)),
                            "color_minute_detected": bool(debug_metrics.get("color_minute_detected", False)),
                            "color_second_detected": bool(debug_metrics.get("color_second_detected", False)),
                        }
                    )

                except Exception as exc:
                    row.update(
                        {
                            "failure_type": classify_failure(exc),
                            "failure_message": str(exc).replace("\n", " ")[:240],
                        }
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

                writer.writerow(row)

    if with_stats:
        generate_stats_report(
            metrics_csv=metrics_path,
            output_md=stats_md_path,
            output_csv=stats_csv_path,
            baseline_condition="baseline_clean",
            iterations=stats_iterations,
            seed=seed,
        )

    return condition_matrix_path, metrics_path, stats_md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible robustness sweeps for symbolic clock extraction")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for generated artifacts")
    parser.add_argument("--samples-per-condition", type=int, default=40, help="Number of synthetic samples per condition")
    parser.add_argument("--seed", type=int, default=20260721, help="Random seed")
    parser.add_argument("--image-size", type=int, default=420, help="Synthetic image size in pixels")
    parser.add_argument("--hour-tol", type=int, default=1, help="Allowed hour delta")
    parser.add_argument("--minute-tol", type=int, default=1, help="Allowed minute delta")
    parser.add_argument("--second-tol", type=int, default=2, help="Allowed second delta")
    parser.add_argument("--no-stats", action="store_true", help="Skip generation of stats_summary.md")
    parser.add_argument("--stats-iterations", type=int, default=2000, help="Permutation/bootstrap iterations for stats")
    args = parser.parse_args()

    condition_matrix_path, metrics_path, stats_md_path = run_experiments(
        output_dir=args.output_dir,
        samples_per_condition=args.samples_per_condition,
        seed=args.seed,
        image_size=args.image_size,
        hour_tol=args.hour_tol,
        minute_tol=args.minute_tol,
        second_tol=args.second_tol,
        with_stats=not args.no_stats,
        stats_iterations=args.stats_iterations,
    )

    print(f"[OK] Condition matrix: {condition_matrix_path}")
    print(f"[OK] Metrics CSV: {metrics_path}")
    if not args.no_stats:
        print(f"[OK] Stats summary: {stats_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
