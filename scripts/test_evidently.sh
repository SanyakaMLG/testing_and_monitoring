#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8890}"
BATCH_SIZE="${BATCH_SIZE:-20}"
REQUEST_COUNT=""
POLL_INTERVAL="${POLL_INTERVAL:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

usage() {
  cat <<'EOF'
Usage:
  scripts/test_evidently.sh [options]

Options:
  --base-url URL          Service base URL. Default: http://localhost:8890
  --batch-size N          Evidently batch size. Default: 20
  --request-count N       Number of valid predict requests to send.
                          Default: 2 * batch_size + 2
  --poll-interval SEC     Metrics polling interval. Default: 5
  --timeout SEC           How long to wait for a successful Evidently report. Default: 120
  --help                  Show this help

The script:
  1. checks /health
  2. records current Evidently counters from /metrics
  3. sends enough valid /predict requests to produce two full batches
  4. waits until ml_service_evidently_report_total{status="success"} increases
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --request-count)
      REQUEST_COUNT="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REQUEST_COUNT" ]]; then
  REQUEST_COUNT=$((BATCH_SIZE * 2 + 2))
fi

if (( REQUEST_COUNT < BATCH_SIZE * 2 )); then
  echo "request-count should be at least 2 * batch-size to trigger a report" >&2
  exit 1
fi

metric_value() {
  local metric_name="$1"
  local value
  value="$(
    curl -fsS "$BASE_URL/metrics" |
      awk -v metric_name="$metric_name" '$1 == metric_name { print $2; found=1; exit } END { if (!found) print "0" }'
  )"
  echo "$value"
}

float_gt() {
  local left="$1"
  local right="$2"
  awk -v left="$left" -v right="$right" 'BEGIN { exit !(left > right) }'
}

health_json="$(curl -fsS "$BASE_URL/health")"
echo "health: $health_json"

success_before="$(metric_value 'ml_service_evidently_report_total{status="success"}')"
failure_before="$(metric_value 'ml_service_evidently_report_total{status="failure"}')"
pending_before="$(metric_value 'ml_service_evidently_pending_events')"

echo "before: success=$success_before failure=$failure_before pending=$pending_before"
echo "sending $REQUEST_COUNT valid predict requests to $BASE_URL"

payloads=(
'{"age":22,"workclass":"Private","fnlwgt":85000,"education":"HS-grad","education.num":9,"marital.status":"Never-married","occupation":"Other-service","relationship":"Own-child","race":"White","sex":"Female","capital.gain":0,"capital.loss":0,"hours.per.week":20,"native.country":"United-States"}'
'{"age":37,"workclass":"Private","fnlwgt":120000,"education":"Bachelors","education.num":13,"marital.status":"Married-civ-spouse","occupation":"Sales","relationship":"Husband","race":"White","sex":"Male","capital.gain":1,"capital.loss":0,"hours.per.week":45,"native.country":"Canada"}'
'{"age":52,"workclass":"Self-emp-inc","fnlwgt":240000,"education":"Masters","education.num":14,"marital.status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Asian-Pac-Islander","sex":"Male","capital.gain":5000,"capital.loss":200,"hours.per.week":60,"native.country":"India"}'
'{"age":67,"workclass":"Federal-gov","fnlwgt":420000,"education":"Doctorate","education.num":16,"marital.status":"Divorced","occupation":"Prof-specialty","relationship":"Not-in-family","race":"Black","sex":"Female","capital.gain":20000,"capital.loss":2500,"hours.per.week":55,"native.country":"Germany"}'
)

status_200=0
for ((i = 0; i < REQUEST_COUNT; i++)); do
  payload_index=$((i % ${#payloads[@]}))
  status_code="$(curl -s -o /dev/null -w '%{http_code}' \
    -X POST "$BASE_URL/predict" \
    -H 'Content-Type: application/json' \
    -d "${payloads[$payload_index]}")"
  if [[ "$status_code" == "200" ]]; then
    status_200=$((status_200 + 1))
  else
    echo "unexpected status code from /predict: $status_code" >&2
  fi
done

echo "predict requests with 200 status: $status_200"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  success_now="$(metric_value 'ml_service_evidently_report_total{status="success"}')"
  failure_now="$(metric_value 'ml_service_evidently_report_total{status="failure"}')"
  pending_now="$(metric_value 'ml_service_evidently_pending_events')"
  echo "poll: success=$success_now failure=$failure_now pending=$pending_now"
  if float_gt "$success_now" "$success_before"; then
    echo "Evidently success counter increased: $success_before -> $success_now"
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done

echo "Timed out waiting for Evidently success counter to increase" >&2
success_final="$(metric_value 'ml_service_evidently_report_total{status="success"}')"
failure_final="$(metric_value 'ml_service_evidently_report_total{status="failure"}')"
pending_final="$(metric_value 'ml_service_evidently_pending_events')"
echo "final: success=$success_final failure=$failure_final pending=$pending_final" >&2
exit 1
