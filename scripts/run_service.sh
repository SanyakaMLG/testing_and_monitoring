#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${DEFAULT_RUN_ID:-}" ]]; then
  echo "DEFAULT_RUN_ID is not set" >&2
  exit 1
fi

sanitized_user="$(whoami | tr -c '[:alnum:]' '_')"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-${sanitized_user}_testing_and_monitoring}"
export APP_PORT="${APP_PORT:-8890}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://158.160.2.37:5000/}"
export EVIDENTLY_URL="${EVIDENTLY_URL:-http://158.160.2.37:8000/}"
export EVIDENTLY_PROJECT_ID="${EVIDENTLY_PROJECT_ID:-019d3a33-9c54-7f59-be64-59e8fc864a1c}"
export EVIDENTLY_BATCH_SIZE="${EVIDENTLY_BATCH_SIZE:-20}"
export EVIDENTLY_REPORT_INTERVAL_SECONDS="${EVIDENTLY_REPORT_INTERVAL_SECONDS:-30}"
export EVIDENTLY_ENABLED="${EVIDENTLY_ENABLED:-true}"

echo "COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME"
echo "APP_PORT=$APP_PORT"

docker compose up --build -d
