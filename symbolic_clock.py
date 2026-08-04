import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from probabilistic_logic_reasoner import ProbabilisticLogicClockReasoner


LineCandidate = Dict[str, float]

ColorRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


HOUR_HAND_COLOR_RANGES: Tuple[ColorRange, ...] = (
    ((96, 60, 55), (116, 255, 98)),
    ((72, 90, 50), (86, 160, 86)),
    ((34, 95, 55), (48, 160, 85)),
    ((5, 100, 65), (20, 160, 96)),
    ((160, 120, 65), (179, 200, 96)),
)

MINUTE_HAND_COLOR_RANGES: Tuple[ColorRange, ...] = (
    ((94, 55, 96), (116, 255, 165)),
    ((72, 90, 88), (86, 160, 118)),
    ((34, 95, 88), (48, 160, 124)),
    ((5, 100, 96), (20, 165, 132)),
    ((160, 110, 100), (179, 190, 140)),
)

SECOND_HAND_COLOR_RANGES: Tuple[ColorRange, ...] = (
    ((0, 150, 150), (20, 255, 255)),
    ((96, 170, 145), (106, 210, 220)),
    ((36, 160, 160), (48, 210, 210)),
)


class SymbolicDetectionError(RuntimeError):
    def __init__(self, message: str, debug_metrics: Optional[Dict[str, Union[int, bool, str, List[float]]]] = None):
        super().__init__(message)
        self.debug_metrics = debug_metrics or {}


def _clock_angle_from_tip(center: Tuple[int, int], tip: Tuple[int, int]) -> float:
    dx = tip[0] - center[0]
    dy = center[1] - tip[1]
    angle = math.degrees(math.atan2(dx, dy)) % 360.0
    return angle


def _endpoint_farthest_from_center(
    center: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> Tuple[int, int]:
    d1 = math.hypot(p1[0] - center[0], p1[1] - center[1])
    d2 = math.hypot(p2[0] - center[0], p2[1] - center[1])
    return p1 if d1 >= d2 else p2


def _distance_point_to_segment(
    point: Tuple[int, int],
    a: Tuple[int, int],
    b: Tuple[int, int],
) -> float:
    px, py = point
    ax, ay = a
    bx, by = b

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay

    denom = abx * abx + aby * aby
    if denom == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _detect_clock_circle(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    h, w = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=120,
        param2=30,
        minRadius=int(min(h, w) * 0.25),
        maxRadius=int(min(h, w) * 0.48),
    )

    if circles is not None and len(circles[0]) > 0:
        x, y, r = circles[0][0]
        return (int(x), int(y)), int(r)

    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.45)
    return center, radius


def _extract_hand_candidates(
    gray: np.ndarray,
    center: Tuple[int, int],
    radius: int,
) -> List[LineCandidate]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    cv2.circle(mask, center, radius, 255, -1)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=int(radius * 0.2),
        maxLineGap=14,
    )

    if lines is None:
        return []

    candidates: List[LineCandidate] = []
    for ln in lines:
        vals = np.ravel(ln)
        if vals.size < 4:
            continue
        x1, y1, x2, y2 = vals[:4]
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        d1 = math.hypot(p1[0] - center[0], p1[1] - center[1])
        d2 = math.hypot(p2[0] - center[0], p2[1] - center[1])

        length = math.hypot(x2 - x1, y2 - y1)
        if length < radius * 0.18:
            continue

        # Hands pivot at the center, so at least one endpoint should stay near it.
        if min(d1, d2) > radius * 0.2:
            continue

        dist_to_center = _distance_point_to_segment(center, p1, p2)
        if dist_to_center > radius * 0.18:
            continue

        tip = _endpoint_farthest_from_center(center, p1, p2)
        tip_dist = math.hypot(tip[0] - center[0], tip[1] - center[1])
        if tip_dist < radius * 0.2:
            continue

        # Prefer long, center-anchored, dark strokes (clock hands) over dial edges.
        line_mask = np.zeros_like(gray)
        cv2.line(line_mask, p1, p2, 255, 3)
        mean_gray = cv2.mean(gray, mask=line_mask)[0]
        darkness = 1.0 - (mean_gray / 255.0)
        center_anchor = 1.0 - min(1.0, min(d1, d2) / (radius * 0.2))
        tip_reach = min(1.0, tip_dist / (radius * 0.85))
        length_norm = min(1.0, length / (radius * 0.85))
        score = (0.45 * length_norm) + (0.25 * center_anchor) + (0.2 * darkness) + (0.1 * tip_reach)

        candidates.append(
            {
                "length": float(length),
                "angle": float(_clock_angle_from_tip(center, tip)),
                "score": float(score),
            }
        )

    candidates.sort(key=lambda c: (c.get("score", 0.0), c["length"]), reverse=True)
    return candidates


def _color_mask_hsv(
    hsv: np.ndarray,
    lower: Tuple[int, int, int],
    upper: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _multi_color_mask_hsv(hsv: np.ndarray, ranges: Tuple[ColorRange, ...]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, _color_mask_hsv(hsv, lower, upper))
    return mask


def _hex_color_to_hsv(hex_value: str) -> Optional[Tuple[int, int, int]]:
    if not isinstance(hex_value, str):
        return None
    cleaned = hex_value.strip()
    if not cleaned.startswith("#") or len(cleaned) != 7:
        return None
    try:
        rgb = tuple(int(cleaned[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return None
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _ranges_from_hsv_hint(
    hsv_hint: Tuple[int, int, int],
    hue_tol: int,
    sat_floor: int,
    val_floor: int,
    sat_pad: int,
    val_pad: int,
) -> Tuple[ColorRange, ...]:
    h, s, v = hsv_hint
    lower_s = max(sat_floor, s - sat_pad)
    upper_s = min(255, s + sat_pad)
    lower_v = max(val_floor, v - val_pad)
    upper_v = min(255, v + val_pad)

    low_h = h - hue_tol
    high_h = h + hue_tol
    if low_h >= 0 and high_h <= 179:
        return (((low_h, lower_s, lower_v), (high_h, upper_s, upper_v)),)

    wrapped: List[ColorRange] = []
    if low_h < 0:
        wrapped.append(((0, lower_s, lower_v), (high_h, upper_s, upper_v)))
        wrapped.append(((179 + low_h, lower_s, lower_v), (179, upper_s, upper_v)))
    else:
        wrapped.append(((low_h, lower_s, lower_v), (179, upper_s, upper_v)))
        wrapped.append(((0, lower_s, lower_v), (high_h - 179, upper_s, upper_v)))
    return tuple(wrapped)


def _color_ranges_with_hint(
    base_ranges: Tuple[ColorRange, ...],
    color_hint_hex: Optional[str],
    hue_tol: int,
    sat_floor: int,
    val_floor: int,
    sat_pad: int,
    val_pad: int,
) -> Tuple[ColorRange, ...]:
    hint_hsv = _hex_color_to_hsv(color_hint_hex) if color_hint_hex else None
    if hint_hsv is None:
        return base_ranges
    # When a concrete UI hand color is provided, prioritize that color family only.
    # Mixing broad fallback palettes with explicit hints can pull detection to wrong strokes.
    # Use tighter pads than legacy defaults to avoid overlap between nearby blues.
    hint_hue_tol = max(4, min(hue_tol, 8))
    hint_sat_pad = max(20, min(sat_pad, 48))
    hint_val_pad = max(20, min(val_pad, 48))
    return _ranges_from_hsv_hint(
        hint_hsv,
        hue_tol=hint_hue_tol,
        sat_floor=sat_floor,
        val_floor=val_floor,
        sat_pad=hint_sat_pad,
        val_pad=hint_val_pad,
    )


def _center_connected_component(mask: np.ndarray, center: Tuple[int, int], probe_radius: int) -> np.ndarray:
    if probe_radius < 3:
        probe_radius = 3

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    clean = cv2.dilate(clean, np.ones((3, 3), dtype=np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((clean > 0).astype(np.uint8))
    if num_labels <= 1:
        return np.zeros_like(mask)

    probe = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(probe, center, probe_radius, 255, -1)
    touching_labels = np.unique(labels[(probe > 0) & (labels > 0)])
    if touching_labels.size == 0:
        return np.zeros_like(mask)

    best_label = int(touching_labels[0])
    best_area = int(stats[best_label, cv2.CC_STAT_AREA])
    for lb in touching_labels[1:]:
        area = int(stats[int(lb), cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = int(lb)

    component = np.zeros_like(mask, dtype=np.uint8)
    component[labels == best_label] = 255
    return component


def _angle_from_mask_tip(
    mask: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    min_dist_ratio: float,
) -> Optional[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    dist = np.sqrt(dx * dx + dy * dy)

    min_dist = radius * min_dist_ratio
    max_dist = radius * 0.95
    keep = (dist >= min_dist) & (dist <= max_dist)
    if not np.any(keep):
        return None

    kx = xs[keep]
    ky = ys[keep]
    kd = dist[keep]
    max_d = float(np.max(kd))
    tip_keep = kd >= max_d - 2.0
    tip_x = int(round(float(np.mean(kx[tip_keep]))))
    tip_y = int(round(float(np.mean(ky[tip_keep]))))
    return _clock_angle_from_tip(center, (tip_x, tip_y))


def _detect_hands_by_color(
    bgr: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    hand_color_hints: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], Optional[float], Dict[str, Union[int, bool]]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hints = hand_color_hints or {}
    hour_ranges = _color_ranges_with_hint(
        HOUR_HAND_COLOR_RANGES,
        hints.get("hour") if isinstance(hints, dict) else None,
        hue_tol=14,
        sat_floor=40,
        val_floor=25,
        sat_pad=115,
        val_pad=85,
    )
    minute_ranges = _color_ranges_with_hint(
        MINUTE_HAND_COLOR_RANGES,
        hints.get("minute") if isinstance(hints, dict) else None,
        hue_tol=14,
        sat_floor=40,
        val_floor=30,
        sat_pad=105,
        val_pad=80,
    )

    hour_mask = _multi_color_mask_hsv(hsv, hour_ranges)
    minute_mask = _multi_color_mask_hsv(hsv, minute_ranges)

    hour_component = _center_connected_component(hour_mask, center, probe_radius=int(radius * 0.08))
    minute_component = _center_connected_component(minute_mask, center, probe_radius=int(radius * 0.08))

    hour_angle = _angle_from_mask_tip(hour_component, center, radius, min_dist_ratio=0.18)
    minute_angle = _angle_from_mask_tip(minute_component, center, radius, min_dist_ratio=0.2)
    metrics: Dict[str, Union[int, bool]] = {
        "hour_mask_pixels": int(np.count_nonzero(hour_mask)),
        "minute_mask_pixels": int(np.count_nonzero(minute_mask)),
        "hour_component_pixels": int(np.count_nonzero(hour_component)),
        "minute_component_pixels": int(np.count_nonzero(minute_component)),
        "color_hour_detected": bool(hour_angle is not None),
        "color_minute_detected": bool(minute_angle is not None),
        "used_hour_color_hint": bool(hints.get("hour")) if isinstance(hints, dict) else False,
        "used_minute_color_hint": bool(hints.get("minute")) if isinstance(hints, dict) else False,
    }
    return hour_angle, minute_angle, metrics


def _dedupe_by_angle(candidates: List[LineCandidate], min_sep_deg: float = 8.0) -> List[LineCandidate]:
    deduped: List[LineCandidate] = []
    for cand in sorted(candidates, key=lambda c: (c.get("score", 0.0), c["length"]), reverse=True):
        angle = cand["angle"]
        too_close = False
        for picked in deduped:
            delta = abs(angle - picked["angle"])
            delta = min(delta, 360.0 - delta)
            if delta < min_sep_deg:
                too_close = True
                break
        if not too_close:
            deduped.append(cand)
    return deduped


def _detect_red_second_hand_angle(
    bgr: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    second_color_hint: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Union[int, bool]]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    second_ranges = _color_ranges_with_hint(
        SECOND_HAND_COLOR_RANGES,
        second_color_hint,
        hue_tol=16,
        sat_floor=40,
        val_floor=35,
        sat_pad=120,
        val_pad=95,
    )
    mask = _multi_color_mask_hsv(hsv, second_ranges)

    circle_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    mask = cv2.bitwise_and(mask, circle_mask)
    mask_pixels = int(np.count_nonzero(mask))

    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    second_angle = _angle_from_mask_tip(mask, center, radius, min_dist_ratio=0.22)
    metrics: Dict[str, Union[int, bool]] = {
        "second_mask_pixels": mask_pixels,
        "color_second_detected": bool(second_angle is not None),
        "used_second_color_hint": bool(second_color_hint),
    }
    return second_angle, metrics


def _angles_to_symbolic(
    hour_angle: float,
    minute_angle: float,
    second_angle: Optional[float],
) -> Dict[str, Union[int, str]]:
    minute_progress = (minute_angle % 360.0) / 6.0
    minute = int(math.floor(minute_progress)) % 60

    # Compensate hour-hand position by minute progress before snapping to nearest hour.
    hour_value = (hour_angle / 30.0) % 12.0
    hour_adjusted = (hour_value - (minute_progress / 60.0)) % 12.0
    hour = int(round(hour_adjusted)) % 12
    if hour == 0:
        hour = 12

    second = 0
    if second_angle is not None:
        second = int(round((second_angle % 360.0) / 6.0)) % 60

    return {
        "HOUR_HAND": hour,
        "MINUTE_HAND": minute,
        "SECOND_HAND": second,
        "TIME": f"{hour:02d}:{minute:02d}:{second:02d}",
    }


def interpret_clock_image(
    image_path: Path,
    was_flipped: bool = False,
    hand_color_hints: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[int, str, float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Could not load saved PNG for symbolic processing.")

    if was_flipped:
        image = cv2.flip(image, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center, radius = _detect_clock_circle(gray)
    hour_angle, minute_angle, color_metrics = _detect_hands_by_color(
        image,
        center,
        radius,
        hand_color_hints=hand_color_hints,
    )
    second_hint = hand_color_hints.get("second") if isinstance(hand_color_hints, dict) else None
    second_angle, second_metrics = _detect_red_second_hand_angle(
        image,
        center,
        radius,
        second_color_hint=second_hint,
    )
    all_candidates = _dedupe_by_angle(_extract_hand_candidates(gray, center, radius))
    fallback_used = False
    fallback_mode = "none"

    def _debug_metrics_payload(fallback_used_flag: bool, fallback_mode_value: str) -> Dict[str, Union[int, bool, str, List[float]]]:
        return {
            "pipeline": "color-mask + hough-fallback",
            "fallback_used": fallback_used_flag,
            "fallback_mode": fallback_mode_value,
            "candidate_count": int(len(all_candidates)),
            "candidate_angles_deg": [round(float(c["angle"]), 2) for c in all_candidates[:24]],
            **color_metrics,
            **second_metrics,
        }

    # Some plain themes use very similar hour/minute hues. If both color masks collapse
    # onto the same stroke, recover hour/minute from line geometry.
    if hour_angle is not None and minute_angle is not None and len(all_candidates) >= 2:
        overlap_delta = abs(hour_angle - minute_angle)
        overlap_delta = min(overlap_delta, 360.0 - overlap_delta)
        if overlap_delta < 6.0:
            candidates = list(all_candidates)
            if second_angle is not None:
                candidates = [
                    c
                    for c in candidates
                    if min(abs(c["angle"] - second_angle), 360.0 - abs(c["angle"] - second_angle)) > 8.0
                ]
            candidates = _dedupe_by_angle(candidates)
            if len(candidates) >= 2:
                top_two = sorted(candidates[:4], key=lambda c: c["length"], reverse=True)[:2]
                minute_cand = max(top_two, key=lambda c: c["length"])
                hour_cand = min(top_two, key=lambda c: c["length"])
                minute_angle = minute_cand["angle"]
                hour_angle = hour_cand["angle"]
                fallback_used = True
                fallback_mode = "resolve-hour-minute-overlap"

    # Fallback when color masks fail in non-canonical snapshots.
    if hour_angle is None or minute_angle is None:
        fallback_used = True
        candidates = list(all_candidates)
        if len(candidates) < 2:
            raise SymbolicDetectionError(
                "Unable to detect enough clock hands in the snapshot.",
                _debug_metrics_payload(True, "candidate-shortage"),
            )

        # Keep the color-detected hand when available, and only infer the missing one.
        if hour_angle is not None and minute_angle is None:
            fallback_mode = "infer-minute-from-hour"
            filtered = [
                c for c in candidates
                if min(abs(c["angle"] - hour_angle), 360.0 - abs(c["angle"] - hour_angle)) > 8.0
            ]
            if second_angle is not None:
                filtered = [
                    c for c in filtered
                    if min(abs(c["angle"] - second_angle), 360.0 - abs(c["angle"] - second_angle)) > 8.0
                ]
            if len(filtered) == 0:
                raise SymbolicDetectionError(
                    "Unable to detect enough clock hands in the snapshot.",
                    _debug_metrics_payload(True, fallback_mode),
                )
            minute_angle = max(filtered, key=lambda c: c["length"])["angle"]

        elif hour_angle is None and minute_angle is not None:
            fallback_mode = "infer-hour-from-minute"
            filtered = [
                c for c in candidates
                if min(abs(c["angle"] - minute_angle), 360.0 - abs(c["angle"] - minute_angle)) > 8.0
            ]
            if second_angle is not None:
                filtered = [
                    c for c in filtered
                    if min(abs(c["angle"] - second_angle), 360.0 - abs(c["angle"] - second_angle)) > 8.0
                ]
            if len(filtered) == 0:
                raise SymbolicDetectionError(
                    "Unable to detect enough clock hands in the snapshot.",
                    _debug_metrics_payload(True, fallback_mode),
                )
            hour_angle = min(filtered, key=lambda c: c["length"])["angle"]

        else:
            fallback_mode = "infer-both-from-lines"
            if second_angle is not None:
                filtered = [
                    c for c in candidates
                    if min(abs(c["angle"] - second_angle), 360.0 - abs(c["angle"] - second_angle)) > 8.0
                ]
                filtered = _dedupe_by_angle(filtered)
                if len(filtered) >= 2:
                    candidates = filtered

            top_two = sorted(candidates[:4], key=lambda c: c["length"], reverse=True)[:2]
            if len(top_two) < 2:
                raise SymbolicDetectionError(
                    "Unable to detect enough clock hands in the snapshot.",
                    _debug_metrics_payload(True, fallback_mode),
                )

            minute_cand = max(top_two, key=lambda c: c["length"])
            hour_cand = min(top_two, key=lambda c: c["length"])
            minute_angle = minute_cand["angle"]
            hour_angle = hour_cand["angle"]

    if hour_angle is None or minute_angle is None:
        raise SymbolicDetectionError(
            "Unable to resolve hour/minute hands from the snapshot.",
            _debug_metrics_payload(fallback_used, fallback_mode),
        )

    symbolic = _angles_to_symbolic(hour_angle, minute_angle, second_angle)

    probabilistic_reasoning: Dict[str, Union[str, float, int, dict, list]]
    try:
        reasoner = ProbabilisticLogicClockReasoner(
            observed_hour_angle=hour_angle,
            observed_minute_angle=minute_angle,
            observed_second_angle=second_angle,
            candidate_angles_deg=[float(c["angle"]) for c in all_candidates[:24]],
        )
        probabilistic_reasoning = reasoner.infer()
    except Exception as exc:
        probabilistic_reasoning = {
            "method": "weighted-model-counting-enumeration",
            "error": f"probabilistic reasoning unavailable: {exc}",
        }

    symbolic.update(
        {
            "hour_angle": round(hour_angle, 2),
            "minute_angle": round(minute_angle, 2),
            "second_angle": round(second_angle, 2) if second_angle is not None else None,
            "center_x": int(center[0]),
            "center_y": int(center[1]),
            "radius": int(radius),
            "debug_metrics": {
                **_debug_metrics_payload(fallback_used, fallback_mode),
            },
            "probabilistic_reasoning": probabilistic_reasoning,
        }
    )
    return symbolic
