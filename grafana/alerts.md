# Grafana Alerts

Ниже приведены Prometheus-выражения и пороги для alert rules.

Contact point в Grafana отправляет сообщения в Telegram contact point `ML Service Telegram`.

Практика настройки:

- для содержательных метрик используется `noDataState = OK`, чтобы алерты не срабатывали только из-за отсутствия трафика;
- доступность сервиса можно контролировать отдельным алертом по `up == 0`.

## 1. High p95 Service Latency

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(ml_service_http_request_duration_seconds_bucket{route="/predict"}[5m])
  )
) > 0.5
```

Порог: `0.5s` в течение `5m`.

## 2. High p95 Inference Latency

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(ml_service_inference_duration_seconds_bucket[5m])
  )
) > 0.2
```

Порог: `0.2s` в течение `5m`.

## 3. 5XX Responses Detected

```promql
sum(rate(ml_service_http_requests_total{status_code=~"5.."}[5m])) > 0
```

Порог: любое ненулевое значение в течение `5m`.

## 4. Sustained High Memory Usage

```promql
(
  avg_over_time(ml_service_process_memory_bytes{instance="158.160.71.215:8890"}[15m]) > 1.2e+09
)
and
(
  clamp_min(delta(ml_service_process_memory_bytes{instance="158.160.71.215:8890"}[15m]), 0) > 2e+08
)
```

Порог: 15-минутное среднее RSS выше `1.2 GB`, и за те же `15m` память выросла ещё минимум на `200 MB`. Это не алертит на просто «липкий» RSS после нагрузки, а ловит продолжающийся рост.

## 5. Prediction Distribution Shift

```promql
(
  sum(rate(ml_service_prediction_total{prediction="1"}[15m]))
  /
  clamp_min(sum(rate(ml_service_prediction_total[15m])), 0.001)
) > 0.95
or
(
  sum(rate(ml_service_prediction_total{prediction="1"}[15m]))
  /
  clamp_min(sum(rate(ml_service_prediction_total[15m])), 0.001)
) < 0.05
```

Порог: почти все ответы ушли в один класс в течение `15m`.

## 6. Probability Distribution Shift

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(ml_service_prediction_probability_bucket[15m])
  )
) > 0.98
```

Порог: верхний хвост вероятностей стал слишком концентрированным.

## 7. Missing Feature Spike

```promql
sum(increase(ml_service_input_missing_feature_total[15m])) > 5
```

Порог: более `5` пропусков за `15m`.
