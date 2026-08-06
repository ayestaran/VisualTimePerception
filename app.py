import base64
import binascii
import html
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from run_experiments import get_condition_preview_images, run_experiments
from stats_summary import generate_stats_report
from symbolic_clock import SymbolicDetectionError, interpret_clock_image

# Process: Filesystem Configuration
# Title: Define project paths and ensure runtime artifact directories exist.
BASE_DIR = Path(__file__).resolve().parent
README_FILE = BASE_DIR / "README.md"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_FILE = SNAPSHOT_DIR / "clock_snapshot.png"
META_FILE = SNAPSHOT_DIR / "snapshot_meta.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMAGE_FILE = STATIC_DIR / "processed_clock.png"
PROCESSED_STAGES_DATA_JSON_FILE = STATIC_DIR / "processed_clock_data.json"
PROCESSED_IMAGE_ARTIFACT_URL = "/artifacts/processed-image.png"
PROCESSED_STAGES_ARTIFACT_URL = "/artifacts/processed-stages.json"

ALLOWED_ARTIFACTS = {
    "condition_matrix.csv",
    "metrics_by_condition.csv",
    "stats_by_condition.csv",
    "stats_summary.md",
}

# Process: Concurrency Primitives
# Title: Protect shared run state and in-memory artifacts across threads.
RUN_LOCK = threading.Lock()
SWEEP_RUN_LOCK = threading.Lock()
SWEEP_STATE_LOCK = threading.Lock()
PROCESSING_ARTIFACTS_LOCK = threading.Lock()

# Process: In-Memory Processing Artifacts
# Title: Cache generated image/stage artifacts for API retrieval.
PROCESSING_ARTIFACTS = {
    "processed_image_png": None,
    "processed_stages_payload": None,
    "processed_stages_json": None,
}

# Process: Sweep State Store
# Title: Track long-running experiment status for progress polling.
SWEEP_STATE = {
    "runId": None,
    "running": False,
    "completed": False,
    "failed": False,
    "stage": "idle",
    "percent": 0,
    "message": "Idle",
    "error": None,
    "startedAt": None,
    "finishedAt": None,
    "summaryPreview": "",
}

# Process: Flask App Initialization
# Title: Create the web application with template/static roots.
app = Flask(__name__, template_folder="templates", static_folder="static")


# Process: Environment Parsing Utility
# Title: Read positive integer config from environment variables.
def _read_positive_int_env(name: str) -> Optional[int]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


REALTIME_CAPTURE_MAX_SECONDS = _read_positive_int_env("REALTIME_CAPTURE_MAX_SECONDS")


# Process: Static Directory Guard
# Title: Ensure static directory exists before file operations.
def _ensure_static_dir() -> Path:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    return STATIC_DIR


# Process: Image Payload Decoding
# Title: Convert base64 payload into an OpenCV BGR image array.
def _decode_image_payload(image_payload: str) -> np.ndarray:
    if not isinstance(image_payload, str) or not image_payload.strip():
        raise ValueError("No image payload provided")

    encoded = image_payload
    if image_payload.startswith("data:image/"):
        encoded = image_payload.split("base64,", 1)[1]

    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload") from exc

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data")
    return image


# Process: Frame Encoding Utility
# Title: Encode an OpenCV frame to base64 PNG text.
def _encode_frame_to_base64(frame: np.ndarray) -> str:
    success, encoded_buffer = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Failed to encode OpenCV frame to PNG")
    return base64.b64encode(encoded_buffer).decode("ascii")


# Process: Lightweight Markdown Rendering
# Title: Render README markdown subset into safe HTML.
def _render_readme_markdown(markdown_text: str) -> str:
    """Render a limited markdown subset used by README.md without extra dependencies."""
    lines = markdown_text.splitlines()
    html_parts = []
    list_type = None
    in_code_block = False
    code_lines = []
    line_index = 0

    def close_list() -> None:
        nonlocal list_type
        if list_type is not None:
            html_parts.append(f"</{list_type}>")
            list_type = None

    def render_inline(text: str) -> str:
        inline_parts = []
        cursor = 0
        token_pattern = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)")

        for token in token_pattern.finditer(text):
            if token.start() > cursor:
                inline_parts.append(html.escape(text[cursor:token.start()]))

            code_text = token.group(1)
            link_label = token.group(2)
            link_target = token.group(3)

            if code_text is not None:
                inline_parts.append(f"<code>{html.escape(code_text)}</code>")
            else:
                normalized_target = (link_target or "").strip()
                if normalized_target and not re.match(r"^(?:[a-z]+:|/|#)", normalized_target, flags=re.IGNORECASE):
                    markdown_target = normalized_target.split("#", 1)[0]
                    if markdown_target.lower().endswith(".md"):
                        normalized_target = f"/docs/{normalized_target}"

                safe_target = html.escape(normalized_target, quote=True)
                safe_label = html.escape(link_label or "")
                inline_parts.append(f'<a href="{safe_target}">{safe_label}</a>')

            cursor = token.end()

        if cursor < len(text):
            inline_parts.append(html.escape(text[cursor:]))

        return "".join(inline_parts)

    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.strip()

        if in_code_block:
            if stripped.startswith("```"):
                html_parts.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(code_lines))))
                code_lines = []
                in_code_block = False
            else:
                code_lines.append(line)
            line_index += 1
            continue

        if stripped.startswith("```"):
            close_list()
            in_code_block = True
            code_lines = []
            line_index += 1
            continue

        if not stripped:
            close_list()
            line_index += 1
            continue

        if len(stripped) >= 3 and set(stripped) <= {"-"}:
            line_index += 1
            continue

        if len(stripped) >= 3 and set(stripped) <= {"="}:
            line_index += 1
            continue

        if line_index + 1 < len(lines):
            underline = lines[line_index + 1].strip()
            if underline and set(underline) <= {"="}:
                close_list()
                html_parts.append(f"<h1>{render_inline(stripped)}</h1>")
                line_index += 2
                continue
            if underline and set(underline) <= {"-"}:
                close_list()
                html_parts.append(f"<h2>{render_inline(stripped)}</h2>")
                line_index += 2
                continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            close_list()
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            html_parts.append(f"<h{heading_level}>{render_inline(heading_text)}</h{heading_level}>")
            line_index += 1
            continue

        unordered_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        if unordered_match:
            if list_type != "ul":
                close_list()
                html_parts.append("<ul>")
                list_type = "ul"
            html_parts.append(f"<li>{render_inline(unordered_match.group(1).strip())}</li>")
            line_index += 1
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            if list_type != "ol":
                close_list()
                html_parts.append("<ol>")
                list_type = "ol"
            html_parts.append(f"<li>{render_inline(ordered_match.group(1).strip())}</li>")
            line_index += 1
            continue

        close_list()
        paragraph_lines = [stripped]
        line_index += 1
        while line_index < len(lines):
            candidate = lines[line_index].strip()
            if not candidate:
                break
            if candidate.startswith("```"):
                break
            if re.match(r"^(#{1,6})\s+", candidate):
                break
            if re.match(r"^[-*+]\s+", candidate):
                break
            if re.match(r"^\d+\.\s+", candidate):
                break
            if line_index + 1 < len(lines):
                next_candidate = lines[line_index + 1].strip()
                if next_candidate and set(next_candidate) <= {"="}:
                    break
                if next_candidate and set(next_candidate) <= {"-"}:
                    break
            paragraph_lines.append(candidate)
            line_index += 1

        html_parts.append(f"<p>{render_inline(' '.join(paragraph_lines))}</p>")

    close_list()
    if in_code_block:
        html_parts.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(code_lines))))

    return "\n".join(html_parts)


# Process: README Title Extraction
# Title: Determine display title from markdown heading/content.
def _extract_readme_title(readme_text: str) -> str:
    for raw_line in readme_text.splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^#\s+(.*)$", stripped)
        if match:
            return match.group(1).strip()
        if stripped and not set(stripped) <= {"="}:
            return stripped
    return "Project README"


# Process: README Section Parsing
# Title: Split README into structured sections for template rendering.
def _build_readme_sections(readme_text: str) -> tuple[str, str, list[dict[str, str]]]:
    title = _extract_readme_title(readme_text)
    lines = readme_text.splitlines()

    sections = []
    current_title = None
    current_lines = []
    subtitle = ""

    def flush_section() -> None:
        nonlocal current_title, current_lines
        if not current_title:
            return
        section_body = "\n".join(current_lines).strip()
        section_html = _render_readme_markdown(section_body)
        sections.append({"title": current_title, "html": section_html})
        current_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()
        heading_match = re.match(r"^##\s+(.*)$", stripped)
        if heading_match:
            if current_title is not None:
                flush_section()

            current_title = heading_match.group(1).strip()
            continue

        if current_title is not None:
            current_lines.append(raw_line)

    flush_section()

    if not sections:
        sections = [{"title": "README", "html": _render_readme_markdown(readme_text)}]

    return title, subtitle, sections


# Process: Stage Entry Construction
# Title: Build encoded frame metadata and optional dense pixel data.
def _build_stage_entry(frame: np.ndarray, encode_frame=None, include_dense_data: bool = True) -> dict:
    frame_for_encoding = encode_frame if encode_frame is not None else frame
    numeric_frame = np.asarray(frame)
    min_value = float(np.min(numeric_frame)) if numeric_frame.size else 0.0
    max_value = float(np.max(numeric_frame)) if numeric_frame.size else 0.0
    entry = {
        "image": _encode_frame_to_base64(frame_for_encoding),
        "shape": list(numeric_frame.shape),
        "dtype": str(numeric_frame.dtype),
        "min": min_value,
        "max": max_value,
    }
    if include_dense_data:
        entry["data"] = numeric_frame.tolist()
    return entry


# Process: Preprocessing Payload Generation
# Title: Compute intermediate CV stages, vectors, masks, and debug info.
def _build_preprocessing_stage_payload(
    image_payload: str,
    flipped: bool = False,
    include_dense_data: bool = True,
) -> dict:
    # Process: Source Decode and Orientation
    # Title: Decode incoming image and apply mirror correction when requested.
    source_bgr = _decode_image_payload(image_payload)
    if flipped:
        source_bgr = cv2.flip(source_bgr, 1)

    # Process: Core CV Pipeline
    # Title: Create RGB, grayscale, blur, threshold, and edge representations.
    rgb_frame = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    threshold_value, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_low = 50
    canny_high = 150
    edges_frame = cv2.Canny(gray_frame, canny_low, canny_high)

    contours_frame = rgb_frame.copy()
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray_frame, dtype=np.uint8)
    contour_vectors = []
    bounding_box_vectors = []
    crop_box = [0, 0, rgb_frame.shape[1], rgb_frame.shape[0]]
    clock_center_vector = [rgb_frame.shape[1] / 2.0, rgb_frame.shape[0] / 2.0]

    # Process: Dominant Contour Selection
    # Title: Identify primary clock region and derive crop/center metadata.
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        crop_box = [int(x), int(y), int(w), int(h)]
        clock_center_vector = [float(x + (w / 2.0)), float(y + (h / 2.0))]
        if w > 20 and h > 20:
            cv2.rectangle(contours_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.drawContours(contours_frame, [contour], -1, (0, 255, 0), 2)

        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    # Process: Contour Feature Extraction
    # Title: Derive symbolic contour and bounding-box feature vectors.
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        moments = cv2.moments(contour)
        if moments.get("m00", 0.0):
            centroid_x = float(moments["m10"] / moments["m00"])
            centroid_y = float(moments["m01"] / moments["m00"])
        else:
            centroid_x = 0.0
            centroid_y = 0.0
        contour_vectors.append([area, perimeter, centroid_x, centroid_y])

        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect = float(bw / bh) if bh else 0.0
        bounding_box_vectors.append([int(bx), int(by), int(bw), int(bh), area, aspect])

    # Process: Clock Crop Computation
    # Title: Produce padded crop around largest detected contour.
    cropped_frame = rgb_frame.copy()
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        pad = max(8, int(max(w, h) * 0.08))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(rgb_frame.shape[1], x + w + pad)
        y1 = min(rgb_frame.shape[0], y + h + pad)
        cropped_region = rgb_frame[y0:y1, x0:x1]
        if cropped_region.size > 0:
            cropped_frame = cropped_region

    # Process: Crop Mask Construction
    # Title: Build a binary mask of the selected crop region.
    crop_mask = np.zeros_like(gray_frame, dtype=np.uint8)
    crop_x, crop_y, crop_w, crop_h = crop_box
    crop_x1 = max(0, min(gray_frame.shape[1], crop_x + crop_w))
    crop_y1 = max(0, min(gray_frame.shape[0], crop_y + crop_h))
    if crop_w > 0 and crop_h > 0 and crop_x < crop_x1 and crop_y < crop_y1:
        crop_mask[crop_y:crop_y1, crop_x:crop_x1] = 255

    # Process: Hand Candidate Extraction
    # Title: Use Hough lines near center to estimate hand angles/lengths.
    lines = cv2.HoughLinesP(edges_frame, 1, np.pi / 180.0, threshold=42, minLineLength=26, maxLineGap=8)
    hand_angle_vectors = []
    if lines is not None:
        center_x, center_y = clock_center_vector
        for line in lines[:60]:
            line_values = np.asarray(line).reshape(-1)
            if line_values.size < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in line_values[:4]]
            d1 = float(np.hypot(x1 - center_x, y1 - center_y))
            d2 = float(np.hypot(x2 - center_x, y2 - center_y))
            if d1 <= d2:
                near_x, near_y, far_x, far_y = x1, y1, x2, y2
            else:
                near_x, near_y, far_x, far_y = x2, y2, x1, y1
            if float(np.hypot(near_x - center_x, near_y - center_y)) > 24.0:
                continue
            angle = float((np.degrees(np.arctan2(center_y - far_y, far_x - center_x)) + 360.0) % 360.0)
            length = float(np.hypot(far_x - near_x, far_y - near_y))
            hand_angle_vectors.append([angle, length])

            # Process: Stage Payload Assembly
            # Title: Package intermediate frame outputs for downstream inspection.
    stage_payload = {
        "rgb": _build_stage_entry(
            rgb_frame,
            encode_frame=cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
            include_dense_data=include_dense_data,
        ),
        "grayscale": _build_stage_entry(gray_frame, include_dense_data=include_dense_data),
        "threshold": _build_stage_entry(thresh_frame, include_dense_data=include_dense_data),
        "blur": _build_stage_entry(blur_frame, include_dense_data=include_dense_data),
        "edges": _build_stage_entry(edges_frame, include_dense_data=include_dense_data),
        "contours": _build_stage_entry(
            contours_frame,
            encode_frame=cv2.cvtColor(contours_frame, cv2.COLOR_RGB2BGR),
            include_dense_data=include_dense_data,
        ),
        "cropped": _build_stage_entry(
            cropped_frame,
            encode_frame=cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR),
            include_dense_data=include_dense_data,
        ),
    }

    # Process: Symbolic and Debug Assembly
    # Title: Attach feature vectors and diagnostic metadata to payload.
    payload = {
        **stage_payload,
        "symbolic_vectors": {
            "hand_angles": hand_angle_vectors,
            "contours": contour_vectors,
            "bounding_boxes": bounding_box_vectors,
            "clock_center": [float(clock_center_vector[0]), float(clock_center_vector[1])],
        },
        "debug": {
            "num_contours": int(len(contours)),
            "threshold_value": float(threshold_value),
            "canny_low": int(canny_low),
            "canny_high": int(canny_high),
            "crop_box": [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])],
            "frame_shapes": {name: value["shape"] for name, value in stage_payload.items()},
            "frame_dtypes": {name: value["dtype"] for name, value in stage_payload.items()},
            "frame_minmax": {name: {"min": value["min"], "max": value["max"]} for name, value in stage_payload.items()},
            "dense_data_included": bool(include_dense_data),
        },
    }

    # Process: Optional Dense Masks
    # Title: Include mask arrays only when dense export is enabled.
    if include_dense_data:
        payload["masks"] = {
            "otsu": thresh_frame.tolist(),
            "edges": edges_frame.tolist(),
            "contours": contour_mask.tolist(),
            "crop": crop_mask.tolist(),
        }
        payload["debug"]["mask_shapes"] = {
            "otsu": list(thresh_frame.shape),
            "edges": list(edges_frame.shape),
            "contours": list(contour_mask.shape),
            "crop": list(crop_mask.shape),
        }

    return payload


# Process: Stage-Inclusion Policy
# Title: Resolve whether preprocessing stages should be included in response.
def _should_include_processing_stages(include_processing_stages: Optional[bool]) -> bool:
    if include_processing_stages is None:
        return True
    return bool(include_processing_stages)


# Process: Stage Artifact Caching
# Title: Build preprocessing payload and cache JSON bytes in memory.
def _save_processed_stages_data_json(
    image_payload: str,
    flipped: bool = False,
    include_dense_data: bool = True,
) -> Path:
    stages_payload = _build_preprocessing_stage_payload(
        image_payload,
        flipped=flipped,
        include_dense_data=include_dense_data,
    )
    stages_json = json.dumps(stages_payload).encode("utf-8")
    with PROCESSING_ARTIFACTS_LOCK:
        PROCESSING_ARTIFACTS["processed_stages_payload"] = stages_payload
        PROCESSING_ARTIFACTS["processed_stages_json"] = stages_json
    return PROCESSED_STAGES_DATA_JSON_FILE


# Process: Processed Preview Rendering
# Title: Create a compact OpenCV visualization of key detection cues.
def _build_opencv_processed_frame(image_payload: str, flipped: bool = False) -> np.ndarray:
    source_image = _decode_image_payload(image_payload)
    if flipped:
        source_image = cv2.flip(source_image, 1)

    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 1)

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    processed_cv_frame = cv2.addWeighted(display_frame, 0.7, edges_color, 0.3, 0)
    processed_cv_frame = cv2.resize(processed_cv_frame, (420, 420), interpolation=cv2.INTER_AREA)
    return processed_cv_frame


# Process: Processed Image Caching
# Title: Encode processed preview frame and store PNG bytes in memory.
def _save_processed_image(image_payload: str, flipped: bool = False) -> Path:
    processed_cv_frame = _build_opencv_processed_frame(image_payload, flipped=flipped)
    success, encoded = cv2.imencode(".png", processed_cv_frame)
    if not success:
        raise RuntimeError("Failed to save processed OpenCV image")
    with PROCESSING_ARTIFACTS_LOCK:
        PROCESSING_ARTIFACTS["processed_image_png"] = encoded.tobytes()
    return PROCESSED_IMAGE_FILE


# Process: Symbolic Response Construction
# Title: Run symbolic inference from payload and shape API response object.
def _build_symbolic_response_from_payload(
    image_payload: str,
    save_image: bool,
    flipped: bool = False,
    include_processing_stages: Optional[bool] = None,
    hand_color_hints: Optional[dict] = None,
    realtime_mode: bool = False,
) -> dict:
    # Process: Input Decode and Orientation
    # Title: Decode payload image and apply optional mirroring.
    source_image = _decode_image_payload(image_payload)
    if flipped:
        source_image = cv2.flip(source_image, 1)

    if _should_include_processing_stages(include_processing_stages):
        _save_processed_stages_data_json(
            image_payload,
            flipped=flipped,
            include_dense_data=not realtime_mode,
        )
    if save_image:
        _save_processed_image(image_payload, flipped=flipped)

    # Process: Temporary Image Bridge
    # Title: Persist decoded image to temp file for symbolic engine input.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    cv2.imwrite(str(temp_path), source_image)

    try:
        # Process: Symbolic Interpretation
        # Title: Execute clock hand/time extraction with optional color hints.
        symbolic = interpret_clock_image(
            temp_path,
            was_flipped=False,
            hand_color_hints=hand_color_hints if isinstance(hand_color_hints, dict) else None,
        )
    except SymbolicDetectionError as exc:
        raise exc
    except RuntimeError as exc:
        raise exc
    finally:
        # Process: Temporary Cleanup
        # Title: Remove temporary file even when inference fails.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Process: Orientation Annotation
    # Title: Attach mirror/orientation metadata to symbolic output.
    symbolic["was_mirrored"] = bool(flipped)
    symbolic["image_orientation"] = "MIRRORED" if flipped else "NORMAL"

    response_payload = {"symbolic": symbolic}
    if save_image:
        response_payload["saved_image_url"] = PROCESSED_IMAGE_ARTIFACT_URL
    else:
        response_payload["saved_image_url"] = None
    if _should_include_processing_stages(include_processing_stages):
        response_payload["processed_data_url"] = PROCESSED_STAGES_ARTIFACT_URL
    else:
        response_payload["processed_data_url"] = None
    return response_payload


# Process: Snapshot Metadata Persistence
# Title: Store flip/orientation metadata for saved snapshots.
def _save_snapshot_metadata(flipped: bool) -> None:
    META_FILE.write_text(json.dumps({"flipped": bool(flipped)}), encoding="utf-8")


# Process: Snapshot Metadata Loading
# Title: Read saved snapshot metadata with safe fallback defaults.
def _load_snapshot_metadata() -> dict:
    if not META_FILE.exists():
        return {"flipped": False}

    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"flipped": False}


# Process: Request Field Extraction
# Title: Normalize symbolic endpoint request payload fields.
def _extract_symbolic_request_fields(
    payload: dict,
) -> tuple[str, bool, bool, Optional[bool], Optional[dict], bool]:
    image_payload = payload.get("image", "")
    save_image = bool(payload.get("save_image", False))
    flipped = bool(payload.get("flipped", False))
    include_processing_stages = payload.get("include_processing_stages")
    hand_color_hints = payload.get("hand_colors")
    realtime_mode = bool(payload.get("realtime_mode", False))
    return image_payload, save_image, flipped, include_processing_stages, hand_color_hints, realtime_mode


# Process: Processing Error Mapping
# Title: Convert processing exceptions to structured HTTP responses.
def _processing_error_response(exc: Exception, allow_value_error: bool = False):
    if isinstance(exc, SymbolicDetectionError):
        return jsonify({
            "error": str(exc),
            "debug_metrics": exc.debug_metrics,
        }), 422
    if isinstance(exc, RuntimeError):
        return jsonify({"error": str(exc)}), 422
    if allow_value_error and isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 400
    return jsonify({"error": f"Unexpected processing error: {exc}"}), 500


# Process: Filesystem Removal Utility
# Title: Remove a file or directory when it exists.
def _remove_generated_path(path: Path) -> bool:
    if not path.exists():
        return False

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    return True


# Process: Global Artifact Cleanup
# Title: Clear generated files, caches, and optionally sweep state.
def _clear_generated_artifacts(preserve_experiment_data: bool = False) -> dict:
    removed: list[str] = []

    if not preserve_experiment_data:
        for artifact_name in ALLOWED_ARTIFACTS:
            if _remove_generated_path(RESULTS_DIR / artifact_name):
                removed.append(f"results/{artifact_name}")

    if _remove_generated_path(SNAPSHOT_FILE):
        removed.append("snapshots/clock_snapshot.png")
    if _remove_generated_path(META_FILE):
        removed.append("snapshots/snapshot_meta.json")

    # Clear processed artifacts used by realtime preview/stages endpoints.
    with PROCESSING_ARTIFACTS_LOCK:
        PROCESSING_ARTIFACTS["processed_image_png"] = None
        PROCESSING_ARTIFACTS["processed_stages_payload"] = None
        PROCESSING_ARTIFACTS["processed_stages_json"] = None

    # Remove legacy static artifacts when present.
    if _remove_generated_path(PROCESSED_IMAGE_FILE):
        removed.append("static/processed_clock.png")
    if _remove_generated_path(PROCESSED_STAGES_DATA_JSON_FILE):
        removed.append("static/processed_clock_data.json")

    if not preserve_experiment_data:
        _set_sweep_state(
            runId=None,
            running=False,
            completed=False,
            failed=False,
            stage="idle",
            percent=0,
            message="Idle",
            error=None,
            startedAt=None,
            finishedAt=None,
            summaryPreview="",
        )

    return {
        "status": "ok",
        "removed": removed,
        "message": "Generated data and processed artifacts cleared.",
    }


# Process: Experiment-Only Cleanup
# Title: Remove sweep outputs and reset sweep state only.
def _clear_experiment_artifacts() -> dict:
    removed: list[str] = []

    for artifact_name in ALLOWED_ARTIFACTS:
        if _remove_generated_path(RESULTS_DIR / artifact_name):
            removed.append(f"results/{artifact_name}")

    _set_sweep_state(
        runId=None,
        running=False,
        completed=False,
        failed=False,
        stage="idle",
        percent=0,
        message="Idle",
        error=None,
        startedAt=None,
        finishedAt=None,
        summaryPreview="",
    )

    return {
        "status": "ok",
        "removed": removed,
        "message": "Experiment sweep data and artifacts cleared.",
    }


# Process: CORS Middleware
# Title: Add permissive cross-origin headers to every response.
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# Process: Home Page Endpoint
# Title: Render main UI with runtime configuration.
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        runtime_config={
            "realtime_capture_max_seconds": REALTIME_CAPTURE_MAX_SECONDS,
        },
    )


# Process: Project README Endpoint
# Title: Render README content as sectioned HTML view.
@app.route("/project-readme", methods=["GET"])
def project_readme():
    try:
        readme_text = README_FILE.read_text(encoding="utf-8")
    except OSError:
        readme_text = "README.md could not be loaded."

    hero_title, hero_subtitle, readme_sections = _build_readme_sections(readme_text)

    return render_template(
        "project_readme.html",
        hero_title=hero_title,
        hero_subtitle=hero_subtitle,
        readme_sections=readme_sections,
    )


# Process: Markdown Document Endpoint
# Title: Safely render markdown files constrained to project root.
@app.route("/docs/<path:doc_name>", methods=["GET"])
def readme_document(doc_name: str):
    if not doc_name.lower().endswith(".md"):
        abort(404)

    candidate = (BASE_DIR / doc_name).resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        abort(404)

    if not candidate.exists() or not candidate.is_file():
        abort(404)

    try:
        doc_text = candidate.read_text(encoding="utf-8")
    except OSError:
        abort(404)

    doc_title = _extract_readme_title(doc_text)
    _, _, doc_sections = _build_readme_sections(doc_text)

    return render_template(
        "project_readme.html",
        hero_title=doc_title,
        hero_subtitle="",
        readme_sections=doc_sections,
    )


# Process: Legacy README Alias Endpoint
# Title: Serve custom project markdown document alias.
@app.route("/CUSTON-MINI-PROJECT.md", methods=["GET"])
def custon_mini_project_document():
    return readme_document("CUSTON-MINI-PROJECT.md")


# Process: Legacy Assessment Alias Endpoint
# Title: Serve assessment markdown document alias.
@app.route("/ASSESSMENT-COMPONENTS..md", methods=["GET"])
def assessment_components_document():
    return readme_document("ASSESSMENT-COMPONENTS..md")


# Process: Processed Image Artifact Endpoint
# Title: Return latest cached processed image preview.
@app.route(PROCESSED_IMAGE_ARTIFACT_URL, methods=["GET"])
def processed_image_artifact():
    with PROCESSING_ARTIFACTS_LOCK:
        image_bytes = PROCESSING_ARTIFACTS.get("processed_image_png")

    if not image_bytes:
        return jsonify({"error": "No processed image available yet."}), 404

    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="processed_clock.png",
    )


# Process: Processed Stages Artifact Endpoint
# Title: Return latest cached preprocessing stages JSON.
@app.route(PROCESSED_STAGES_ARTIFACT_URL, methods=["GET"])
def processed_stages_artifact():
    with PROCESSING_ARTIFACTS_LOCK:
        stages_json = PROCESSING_ARTIFACTS.get("processed_stages_json")

    if not stages_json:
        return jsonify({"error": "No processed stages available yet."}), 404

    return app.response_class(stages_json, mimetype="application/json")


# Process: Snapshot Save Endpoint
# Title: Persist captured image bytes and mirror metadata.
@app.route("/save-image", methods=["POST", "OPTIONS"])
def save_image():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("imageData", "")
    flipped = payload.get("flipped", False)

    if not isinstance(image_data, str) or "base64," not in image_data:
        return jsonify({"error": "Invalid PNG payload"}), 400

    try:
        encoded = image_data.split("base64,", 1)[1]
        raw_bytes = base64.b64decode(encoded)
        SNAPSHOT_FILE.write_bytes(raw_bytes)
        _save_snapshot_metadata(bool(flipped))
    except (ValueError, OSError) as exc:
        return jsonify({"error": f"Failed to save image: {exc}"}), 500

    return jsonify({
        "status": "saved",
        "file": str(SNAPSHOT_FILE.name),
        "flipped": bool(flipped),
    })


# Process: Symbolic Clock Endpoint
# Title: Resolve symbolic time from payload image or saved snapshot.
@app.route("/symbolic-clock", methods=["POST", "OPTIONS"])
def symbolic_clock():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    image_payload, save_image, flipped, include_processing_stages, hand_color_hints, realtime_mode = _extract_symbolic_request_fields(payload)

    if isinstance(image_payload, str) and image_payload:
        try:
            response_payload = _build_symbolic_response_from_payload(
                image_payload,
                save_image=save_image,
                flipped=flipped,
                include_processing_stages=include_processing_stages,
                hand_color_hints=hand_color_hints,
                realtime_mode=realtime_mode,
            )
            return jsonify(response_payload)
        except Exception as exc:
            return _processing_error_response(exc)

    if not SNAPSHOT_FILE.exists():
        return jsonify({"error": "No saved snapshot found. Capture a snapshot first."}), 404

    metadata = _load_snapshot_metadata()
    flipped = bool(metadata.get("flipped", False))

    try:
        symbolic = interpret_clock_image(SNAPSHOT_FILE, was_flipped=flipped)
    except Exception as exc:
        return _processing_error_response(exc)

    symbolic["was_mirrored"] = flipped
    symbolic["image_orientation"] = "MIRRORED" if flipped else "NORMAL"

    return jsonify(symbolic)


# Process: Vision Clock Endpoint
# Title: Require payload image and return structured symbolic response.
@app.route("/vision/clock", methods=["POST", "OPTIONS"])
def vision_clock():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    image_payload, save_image, flipped, include_processing_stages, hand_color_hints, realtime_mode = _extract_symbolic_request_fields(payload)

    if not isinstance(image_payload, str) or not image_payload:
        return jsonify({"error": "No image payload provided"}), 400

    try:
        response_payload = _build_symbolic_response_from_payload(
            image_payload,
            save_image=save_image,
            flipped=flipped,
            include_processing_stages=include_processing_stages,
            hand_color_hints=hand_color_hints,
            realtime_mode=realtime_mode,
        )
        return jsonify(response_payload)
    except Exception as exc:
        return _processing_error_response(exc, allow_value_error=True)


# Process: Experiment Sweep Start Endpoint
# Title: Validate parameters, enqueue worker, and return run token.
@app.route("/run-experiments", methods=["POST", "OPTIONS"])
def run_experiments_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}

    samples_per_condition = int(payload.get("samplesPerCondition", 40))
    stats_iterations = int(payload.get("statsIterations", 2000))
    seed = int(payload.get("seed", 20260721))
    image_size = int(payload.get("imageSize", 420))
    hour_tol = int(payload.get("hourTol", 1))
    minute_tol = int(payload.get("minuteTol", 1))
    second_tol = int(payload.get("secondTol", 2))

    if samples_per_condition < 1 or samples_per_condition > 500:
        return jsonify({"error": "samplesPerCondition must be between 1 and 500"}), 400
    if stats_iterations < 100 or stats_iterations > 50000:
        return jsonify({"error": "statsIterations must be between 100 and 50000"}), 400
    if image_size < 200 or image_size > 1200:
        return jsonify({"error": "imageSize must be between 200 and 1200"}), 400

    locked = SWEEP_RUN_LOCK.acquire(blocking=False)
    if not locked:
        return jsonify({"error": "An experiment run is already in progress."}), 409

    run_id = uuid.uuid4().hex[:12]
    _set_sweep_state(
        runId=run_id,
        running=True,
        completed=False,
        failed=False,
        stage="queued",
        percent=1,
        message="Queued",
        error=None,
        startedAt=int(time.time()),
        finishedAt=None,
        summaryPreview="",
    )

    thread = threading.Thread(
        target=_sweep_worker,
        kwargs={
            "run_id": run_id,
            "samples_per_condition": samples_per_condition,
            "stats_iterations": stats_iterations,
            "seed": seed,
            "image_size": image_size,
            "hour_tol": hour_tol,
            "minute_tol": minute_tol,
            "second_tol": second_tol,
        },
        daemon=True,
    )
    thread.start()

    previews = []
    try:
        previews = get_condition_preview_images(seed=seed)
    except Exception:
        # Thumbnail previews are best-effort and should not block run start.
        previews = []

    return jsonify(
        {
            "status": "accepted",
            "runId": run_id,
            "message": "Experiment sweep started.",
            "conditions": previews,
        }
    ), 202


# Process: Sweep State Mutation Helper
# Title: Apply atomic updates to shared sweep state.
def _set_sweep_state(**kwargs) -> None:
    with SWEEP_STATE_LOCK:
        SWEEP_STATE.update(kwargs)


    # Process: Background Sweep Worker
    # Title: Execute experiments, run stats, and publish completion state.
def _sweep_worker(
    run_id: str,
    samples_per_condition: int,
    stats_iterations: int,
    seed: int,
    image_size: int,
    hour_tol: int,
    minute_tol: int,
    second_tol: int,
) -> None:
    try:
        # Process: Experiment Phase
        # Title: Generate condition matrix and per-sample metrics.
        _set_sweep_state(stage="running", percent=10, message="Generating condition samples")
        condition_matrix_path, metrics_path, stats_path = run_experiments(
            output_dir=RESULTS_DIR,
            samples_per_condition=samples_per_condition,
            seed=seed,
            image_size=image_size,
            hour_tol=hour_tol,
            minute_tol=minute_tol,
            second_tol=second_tol,
            with_stats=False,
            stats_iterations=stats_iterations,
        )

        # Process: Statistical Phase
        # Title: Compute summary statistics from generated metrics.
        _set_sweep_state(stage="stats", percent=70, message="Running statistical tests")
        stats_md_path = RESULTS_DIR / "stats_summary.md"
        stats_csv_path = RESULTS_DIR / "stats_by_condition.csv"
        generate_stats_report(
            metrics_csv=metrics_path,
            output_md=stats_md_path,
            output_csv=stats_csv_path,
            baseline_condition="baseline_clean",
            iterations=stats_iterations,
            seed=seed,
        )

        # Process: Completion Publishing
        # Title: Mark run successful and expose summary preview.
        summary_preview = stats_md_path.read_text(encoding="utf-8") if stats_md_path.exists() else ""

        _set_sweep_state(
            runId=run_id,
            running=False,
            completed=True,
            failed=False,
            stage="completed",
            percent=100,
            message="Completed",
            error=None,
            finishedAt=int(time.time()),
            summaryPreview=summary_preview,
        )
    except Exception as exc:
        # Process: Failure Publishing
        # Title: Record worker failure details for client polling.
        _set_sweep_state(
            runId=run_id,
            running=False,
            completed=False,
            failed=True,
            stage="failed",
            message="Failed",
            error=str(exc),
            finishedAt=int(time.time()),
        )
    finally:
        # Process: Lock Release
        # Title: Always release exclusive sweep execution lock.
        SWEEP_RUN_LOCK.release()


# Process: Condition Preview Endpoint
# Title: Return synthetic thumbnails for all sweep conditions.
@app.route("/sweep-condition-previews", methods=["GET"])
def sweep_condition_previews():
    previews = get_condition_preview_images()
    return jsonify({"conditions": previews})


# Process: Sweep Progress Endpoint
# Title: Return current state of background experiment run.
@app.route("/sweep-progress", methods=["GET"])
def sweep_progress():
    with SWEEP_STATE_LOCK:
        state = dict(SWEEP_STATE)
    return jsonify(state)


# Process: Artifact Inventory Endpoint
# Title: Report existence and size for downloadable experiment artifacts.
@app.route("/experiment-artifacts", methods=["GET"])
def experiment_artifacts():
    artifacts = []
    for name in sorted(ALLOWED_ARTIFACTS):
        path = RESULTS_DIR / name
        artifacts.append(
            {
                "name": name,
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
        )
    return jsonify({"artifacts": artifacts})


# Process: Summary Endpoint
# Title: Return markdown summary text if available.
@app.route("/experiment-summary", methods=["GET"])
def experiment_summary():
    summary_path = RESULTS_DIR / "stats_summary.md"
    if not summary_path.exists():
        return jsonify({"summary": "No stats summary yet. Run an experiment sweep first."})

    return jsonify({"summary": summary_path.read_text(encoding="utf-8")})


# Process: Global Clear Endpoint
# Title: Clear generated runtime artifacts with optional sweep preservation.
@app.route("/clear-generated-data", methods=["POST", "OPTIONS"])
def clear_generated_data():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(silent=True) or {}
        preserve_experiment_data = bool(payload.get("preserve_experiment_data", False))

        return jsonify(_clear_generated_artifacts(preserve_experiment_data=preserve_experiment_data))
    except Exception as exc:
        return jsonify({"error": f"Failed to clear generated data: {exc}"}), 500


# Process: Experiment Clear Endpoint
# Title: Clear experiment outputs and reset sweep lifecycle state.
@app.route("/clear-experiment-data", methods=["POST", "OPTIONS"])
def clear_experiment_data():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        return jsonify(_clear_experiment_artifacts())
    except Exception as exc:
        return jsonify({"error": f"Failed to clear experiment data: {exc}"}), 500


# Process: Artifact Download Endpoint
# Title: Serve whitelisted experiment artifacts as attachments.
@app.route("/experiment-artifact/<path:artifact_name>", methods=["GET"])
def experiment_artifact_download(artifact_name: str):
    if artifact_name not in ALLOWED_ARTIFACTS:
        return jsonify({"error": "Artifact not allowed."}), 404

    artifact_path = RESULTS_DIR / artifact_name
    if not artifact_path.exists():
        return jsonify({"error": "Artifact not found. Run experiments first."}), 404

    return send_file(artifact_path, as_attachment=True, download_name=artifact_name)


# Process: Application Entrypoint
# Title: Start Flask server with optional debug/reloader mode.
if __name__ == "__main__":
    debug_mode = os.environ.get("CLOCK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5014, debug=debug_mode, use_reloader=debug_mode)
