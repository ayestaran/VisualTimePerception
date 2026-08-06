#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "Error: mmdc is not installed or not in PATH." >&2
  echo "Install with: npm install -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

if [[ -n "${MMDC_BROWSER_BIN:-}" ]]; then
  BROWSER_BIN="$MMDC_BROWSER_BIN"
else
  if [[ -x "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" ]]; then
    BROWSER_BIN="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
  elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
    BROWSER_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  else
    echo "Error: No supported browser found for mmdc (Edge or Chrome)." >&2
    echo "Set MMDC_BROWSER_BIN to an executable Chromium-based browser path." >&2
    exit 1
  fi
fi

CONFIG_FILE="$(mktemp -t mmdc-config).json"
cleanup() {
  rm -f "$CONFIG_FILE"
}
trap cleanup EXIT

cat > "$CONFIG_FILE" <<EOF
{
  "executablePath": "$BROWSER_BIN",
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
EOF

shopt -s nullglob
files=(*_flowchart.mmd)

if (( ${#files[@]} == 0 )); then
  echo "No *_flowchart.mmd files found in $SCRIPT_DIR"
  exit 0
fi

for input in "${files[@]}"; do
  output="${input%.mmd}.svg"
  echo "Rendering $input -> $output"
  mmdc -p "$CONFIG_FILE" -i "$input" -o "$output"
done

echo "Done. Rendered ${#files[@]} flowchart file(s)."
