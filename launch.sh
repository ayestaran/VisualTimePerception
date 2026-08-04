#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/.flask_clock.pid"
LOG_FILE="$ROOT_DIR/flask_clock.log"
SERVER_PORT="5014"
REALTIME_CAPTURE_MAX_SECONDS=""

is_positive_integer() {
  local value="$1"
  [[ "$value" =~ ^[1-9][0-9]*$ ]]
}

parse_runtime_options() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --realtime-max-seconds|--max-realtime-seconds)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for $1"
          usage
          exit 1
        fi
        if ! is_positive_integer "$2"; then
          echo "Invalid realtime max seconds: $2"
          echo "Value must be a positive integer (for example: 60)."
          exit 1
        fi
        REALTIME_CAPTURE_MAX_SECONDS="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

get_python_cmd() {
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "$ROOT_DIR/.venv/bin/python"
  else
    command -v python3
  fi
}

is_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

read_pid_file() {
  cat "$PID_FILE" 2>/dev/null || true
}

get_port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -tiTCP:"$SERVER_PORT" -sTCP:LISTEN 2>/dev/null || true
  fi
}

collect_server_pids() {
  local -a found=()

  local file_pid
  file_pid="$(read_pid_file)"
  if is_running "$file_pid"; then
    found+=("$file_pid")
  fi

  local port_pid
  while IFS= read -r port_pid; do
    [[ -z "$port_pid" ]] && continue
    if is_running "$port_pid"; then
      found+=("$port_pid")
    fi
  done < <(get_port_pids)

  # Print unique PIDs, one per line.
  if [[ -n "${found[*]-}" ]]; then
    printf '%s\n' "${found[@]}" | awk 'NF && !seen[$0]++'
  fi
}

start_server() {
  local existing_pid
  existing_pid="$(read_pid_file)"
  if [[ -n "$existing_pid" ]] && is_running "$existing_pid"; then
    echo "Server already running (pid=$existing_pid)."
    return 0
  fi
  if [[ -f "$PID_FILE" ]]; then
    rm -f "$PID_FILE"
  fi

  local py_cmd
  py_cmd="$(get_python_cmd)"

  echo "Starting Flask clock server..."
  (
    cd "$ROOT_DIR"
    local -a env_args=("CLOCK_DEBUG=0")
    if [[ -n "$REALTIME_CAPTURE_MAX_SECONDS" ]]; then
      env_args+=("REALTIME_CAPTURE_MAX_SECONDS=$REALTIME_CAPTURE_MAX_SECONDS")
      echo "Realtime capture max duration: ${REALTIME_CAPTURE_MAX_SECONDS}s"
    else
      echo "Realtime capture max duration: unlimited"
    fi
    nohup env "${env_args[@]}" "$py_cmd" app.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )

  local pid
  pid="$(cat "$PID_FILE")"
  sleep 1
  if is_running "$pid"; then
    echo "Started (pid=$pid)."
    echo "URL: http://127.0.0.1:5014"
    echo "Log: $LOG_FILE"
  else
    echo "Failed to start server. Check log: $LOG_FILE"
    rm -f "$PID_FILE"
    return 1
  fi
}

stop_server() {
  local -a pids=()
  local pid

  # Fast path: trust PID file first to avoid potential lsof stalls on some systems.
  local file_pid
  file_pid="$(read_pid_file)"
  if is_running "$file_pid"; then
    pids+=("$file_pid")
  fi

  if [[ -z "${pids[*]-}" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(collect_server_pids)
  fi

  if [[ -z "${pids[*]-}" ]]; then
    rm -f "$PID_FILE"
    echo "Server is not running."
    return 0
  fi

  echo "Stopping server (pid(s)=${pids[*]})..."
  kill "${pids[@]}" 2>/dev/null || true

  for _ in {1..20}; do
    local -a alive=()
    for pid in "${pids[@]-}"; do
      if is_running "$pid"; then
        alive+=("$pid")
      fi
    done
    if [[ -z "${alive[*]-}" ]]; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    pids=("${alive[@]}")
    sleep 0.2
  done

  echo "Graceful stop timed out; forcing kill for pid(s)=${pids[*]}..."
  kill -9 "${pids[@]}" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped (forced)."
  return 0
}

status_server() {
  local -a pids=()
  local pid
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(collect_server_pids)

  if [[ -n "${pids[*]-}" ]]; then
    if [[ "${pids[0]}" != "$(read_pid_file)" ]]; then
      printf '%s\n' "${pids[0]}" > "$PID_FILE"
    fi
    echo "Server running (pid(s)=${pids[*]})."
    echo "URL: http://127.0.0.1:$SERVER_PORT"
    echo "Log: $LOG_FILE"
    return 0
  fi

  rm -f "$PID_FILE"
  echo "Server not running."
  return 1
}

usage() {
  echo "Usage: ./launch.sh {start|stop|force-stop|restart|status} [--realtime-max-seconds <positive-int>]"
  echo "Examples:"
  echo "  ./launch.sh start --realtime-max-seconds 120"
  echo "  ./launch.sh restart --realtime-max-seconds 45"
}

cmd="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  start)
    parse_runtime_options "$@"
    start_server
    ;;
  stop)
    stop_server
    ;;
  force-stop)
    stop_server
    ;;
  restart)
    parse_runtime_options "$@"
    stop_server || true
    start_server
    ;;
  status)
    status_server
    ;;
  *)
    usage
    exit 1
    ;;
esac
