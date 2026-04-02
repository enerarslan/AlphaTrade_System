#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <session-name> <log-file> <command> [args...]" >&2
  exit 1
fi

session_name="$1"
log_file="$2"
shift 2

mkdir -p "$(dirname "$log_file")"

run_logged_command() {
  "$@" 2>&1 | tee "$log_file"
  return "${PIPESTATUS[0]}"
}

if [[ "${ALPHATRADE_FORCE_FOREGROUND:-0}" == "1" ]] || [[ -n "${TMUX:-}" ]]; then
  run_logged_command "$@"
  exit $?
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; running in foreground." >&2
  run_logged_command "$@"
  exit $?
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name"
  echo "Attach with: tmux attach -t $session_name"
  echo "Resume after disconnect: rerun the same launch script or attach to the existing session."
  exit 0
fi

printf -v cwd_text '%q' "$PWD"
printf -v command_text '%q ' "$@"
printf -v log_text '%q' "$log_file"
tmux_payload="set -euo pipefail; cd ${cwd_text}; ${command_text}2>&1 | tee ${log_text}; exit \${PIPESTATUS[0]}"

tmux new-session -d -s "$session_name" "bash -lc $(printf '%q' "$tmux_payload")"

echo "Started tmux session: $session_name"
echo "Log file: $log_file"
echo "Attach with: tmux attach -t $session_name"
echo "Resume after interruption: rerun the same launch script with the same model name."
