import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from typing import Dict, Tuple

import cv2
import numpy as np


def angle_to_point(center: Tuple[int, int], length: int, angle_deg: float) -> Tuple[int, int]:
    theta = np.deg2rad(angle_deg)
    x = int(round(center[0] + np.sin(theta) * length))
    y = int(round(center[1] - np.cos(theta) * length))
    return x, y


def make_synthetic_clock_png(hour: int = 9, minute: int = 0, second: int = 0, size: int = 420) -> bytes:
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

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode synthetic PNG")

    return encoded.tobytes()


def post_json(url: str, payload: Dict) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body)


def cyclic_minute_delta(a: int, b: int) -> int:
    raw = abs((a % 60) - (b % 60))
    return min(raw, 60 - raw)


def cyclic_hour_delta(a: int, b: int) -> int:
    raw = abs((a % 12) - (b % 12))
    return min(raw, 12 - raw)


def cyclic_second_delta(a: int, b: int) -> int:
    raw = abs((a % 60) - (b % 60))
    return min(raw, 60 - raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end symbolic clock self-test")
    parser.add_argument("--host", default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5014, help="Flask port")
    parser.add_argument("--hour", type=int, default=9, help="Expected synthetic hour (1-12)")
    parser.add_argument("--minute", type=int, default=0, help="Expected synthetic minute (0-59)")
    parser.add_argument("--second", type=int, default=30, help="Expected synthetic second (0-59)")
    parser.add_argument("--hour-tol", type=int, default=1, help="Allowed hour delta")
    parser.add_argument("--minute-tol", type=int, default=1, help="Allowed minute delta")
    parser.add_argument("--second-tol", type=int, default=2, help="Allowed second delta")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    expected_hour = ((args.hour - 1) % 12) + 1
    expected_minute = args.minute % 60
    expected_second = args.second % 60

    try:
        png_bytes = make_synthetic_clock_png(
            hour=expected_hour,
            minute=expected_minute,
            second=expected_second,
        )
    except Exception as exc:
        print(f"[FAIL] Could not build synthetic image: {exc}")
        return 1

    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    synthetic_hand_hints = {
        # Synthetic frame uses BGR (20,40,65), (18,62,107), (40,40,200)
        # which convert to RGB hex values below.
        "hour": "#412814",
        "minute": "#6b3e12",
        "second": "#c82828",
    }

    try:
        response_payload = post_json(
            f"{base_url}/vision/clock",
            {
                "image": data_url,
                "save_image": True,
                "flipped": False,
                "include_processing_stages": False,
                "realtime_mode": False,
                "hand_colors": synthetic_hand_hints,
            },
        )
        symbolic_response = response_payload.get("symbolic", response_payload)
    except urllib.error.URLError as exc:
        print(f"[FAIL] Could not reach Flask backend at {base_url}: {exc}")
        print("Start the app first with: python app.py")
        return 1
    except Exception as exc:
        print(f"[FAIL] HTTP test failed: {exc}")
        return 1

    got_hour = int(symbolic_response.get("HOUR_HAND", -1))
    got_minute = int(symbolic_response.get("MINUTE_HAND", -1))
    got_second = int(symbolic_response.get("SECOND_HAND", -1))

    hour_error = cyclic_hour_delta(got_hour, expected_hour)
    minute_error = cyclic_minute_delta(got_minute, expected_minute)
    second_error = cyclic_second_delta(got_second, expected_second)

    print("vision-clock response:")
    print(json.dumps(response_payload, indent=2))
    print("symbolic-clock response:")
    print(json.dumps(symbolic_response, indent=2))

    if (
        hour_error <= args.hour_tol
        and minute_error <= args.minute_tol
        and second_error <= args.second_tol
    ):
        print(
            f"[PASS] Symbolic conversion is within tolerance. "
            f"expected={expected_hour:02d}:{expected_minute:02d}:{expected_second:02d}, "
            f"got={got_hour:02d}:{got_minute:02d}:{got_second:02d}"
        )
        return 0

    print(
        f"[FAIL] Symbolic conversion mismatch. "
        f"expected={expected_hour:02d}:{expected_minute:02d}:{expected_second:02d}, "
        f"got={got_hour:02d}:{got_minute:02d}:{got_second:02d}, "
        f"hour_error={hour_error}, minute_error={minute_error}, second_error={second_error}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
