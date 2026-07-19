#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/run_tests.sh [all|fast|unit|lint|typecheck|typecheck-release|format|format-all|maintainability] [args...]

Modes:
  all          Run the pytest suite.
  fast, unit   Run the pytest suite; aliases kept for shared CI ergonomics.
  lint         Run Ruff lint across the repository.
  typecheck, typecheck-release
               Run ty over the configured typed surface.
  format       Check Ruff formatting across the repository.
  format-all   Alias for format.
  maintainability
               Validate static maintainability metrics against the current ratchet.
EOF
}

run_ruff() {
  if [[ -x .venv/bin/ruff ]]; then
    .venv/bin/ruff "$@"
    return
  fi
  uv run --no-project --with ruff ruff "$@"
}

run_ty() {
  if [[ "${PORTFOLIO_BACKTESTER_NO_PROJECT_TOOLS:-0}" == "1" ]]; then
    uv run --no-project --with "ty>=0.0.55" ty check --extra-search-path typings "$@"
    return
  fi
  uv run --extra dev ty check "$@"
}

mode="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$mode" in
  all | fast | unit)
    exec uv run python -m pytest "$@"
    ;;
  lint)
    run_ruff check . "$@"
    ;;
  typecheck | typecheck-release)
    echo "Running ty typed surface from pyproject.toml."
    run_ty --error-on-warning "$@"
    ;;
  format | format-all)
    run_ruff format --check . "$@"
    ;;
  maintainability)
    python scripts/dev/maintainability_metrics.py --ratchet "$@"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
