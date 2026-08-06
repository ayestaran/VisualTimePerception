import json
from pathlib import Path
from typing import Any, Iterable, List, Optional


# Process Title: Read JSON With Safe Fallback
# Purpose: Load JSON from disk and return a caller-provided default on missing/invalid files.
def read_json_file(path: Path, default: Optional[Any] = None) -> Any:
    # Process Title: Missing File Guard
    # Purpose: Exit early when the target JSON file does not exist.
    if not path.exists():
        return default

    # Process Title: Parse JSON Payload
    # Purpose: Decode UTF-8 content and parse JSON safely.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Process Title: Error Fallback
        # Purpose: Keep callers resilient by returning the default when parsing fails.
        return default


# Process Title: Write JSON Atomically Per Path
# Purpose: Ensure parent directory exists, then persist JSON content as UTF-8 text.
def write_json_file(path: Path, payload: Any) -> None:
    # Process Title: Ensure Output Directory
    # Purpose: Create parent directories before writing file content.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Process Title: Serialize JSON Payload
    # Purpose: Encode payload with ASCII-safe JSON for deterministic output.
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


# Process Title: Read JSONL Records
# Purpose: Stream line-delimited JSON and keep only dictionary rows.
def read_jsonl_file(path: Path) -> List[dict]:
    rows: List[dict] = []

    # Process Title: Missing File Guard
    # Purpose: Return an empty collection when no JSONL file exists.
    if not path.exists():
        return rows

    # Process Title: Iterate And Parse Lines
    # Purpose: Skip blanks, parse each JSON line, and collect dictionary objects.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


# Process Title: Write JSONL Records
# Purpose: Serialize dictionary rows into one JSON object per line.
def write_jsonl_file(path: Path, rows: Iterable[dict]) -> None:
    # Process Title: Ensure Output Directory
    # Purpose: Create parent directories before writing file content.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Process Title: Serialize Row Stream
    # Purpose: Write each row as compact ASCII-safe JSON followed by a newline.
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
