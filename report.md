# Отчёт по ДЗ

В работе доработан шаблонный FastAPI сервис с моделью из MLflow, добавлены тесты, мониторинг в Prometheus/Grafana, алертинг и drift monitoring через Evidently.

## 1. Доработки надёжности и отказоустойчивости

Добавлена обработка нескольких проблемных ситуаций, чтобы сервис не падал и отвечал предсказуемо:

1. Если модель не загружена, `/predict` возвращает `503`, а `/health` показывает деградированное состояние вместо падения приложения.
2. Обновление модели сделано атомарным: если новая модель не загрузилась, сервис продолжает работать со старой.
3. `run_id` валидируется на уровне схемы запроса, поэтому неверный формат отсекается до обращения к MLflow.
4. Если `run_id` не существует или недоступен, `/updateModel` возвращает `404`.
5. Добавлена проверка загруженного артефакта: сервис отклоняет модель без `predict_proba` и без `feature_names_in_`.
6. Если для активной модели не хватает обязательных фичей, `/predict` возвращает `422` и перечисляет отсутствующие поля.
7. Лишние поля и некорректные диапазоны (`age`, `hours.per.week`, `capital.*`) отсекаются схемой Pydantic.
8. Ошибки инференса и невалидные вероятности модели обрабатываются без краша всего приложения.
9. Сервис стартует в деградированном режиме, если `DEFAULT_RUN_ID` не задан или стартовая модель недоступна.
10. Повторный `/updateModel` на тот же `run_id` не перезагружает модель лишний раз.
11. Для drift monitoring ограничена in-memory очередь событий и ускорена её разгрузка, чтобы backlog не рос бесконечно под высокой нагрузкой.

## 2. Добавленные тесты

Сейчас в проекте `16` тестов.

Покрыты четыре уровня:

1. Препроцессинг:
   - выбор только нужных колонок;
   - корректная работа alias-полей;
   - ошибка при отсутствии обязательных фичей;
   - ошибка при попытке использовать неподдерживаемые фичи модели.
2. Работа контейнера модели:
   - атомарность замены модели;
   - успешное получение вероятности и класса;
   - отклонение невалидного артефакта;
   - отклонение некорректного ответа модели;
   - отсутствие повторной загрузки при том же `run_id`.
3. Хэндлеры и валидация:
   - успешный запуск сервиса и получение предсказания;
   - `422` при неполном наборе признаков;
   - `422` при лишних полях;
   - `404` при обновлении на несуществующий `run_id`;
   - `422` при невалидном формате `run_id`;
   - деградированный запуск без стартовой модели.
4. Drift monitor:
   - bounded queue для Evidently;
   - обработка нескольких батчей за один цикл фоновой задачи.

Команда запуска:

```bash
pytest -q
```

## 3. Логируемые метрики

### Технические метрики сервиса

- `ml_service_http_requests_total{method,route,status_code}` — число запросов и коды ответа.
- `ml_service_http_request_duration_seconds` — время ответа сервиса.
- `ml_service_process_cpu_percent` — загрузка CPU процесса.
- `ml_service_process_memory_bytes` — RSS-память процесса.
- `ml_service_process_thread_count` — число потоков.

Для latency-метрик рассчитываются `p75`, `p90`, `p95`, `p99`, `p99.9` через `histogram_quantile(...)`.

### Метрики входных данных

- `ml_service_preprocessing_duration_seconds` — время предобработки.
- `ml_service_input_numeric_feature_values{feature}` — распределение числовых фичей.
- `ml_service_input_categorical_feature_total{feature,value}` — частоты категориальных значений.
- `ml_service_input_missing_feature_total{feature}` — число пропусков по входным полям.

### Метрики работы модели

- `ml_service_inference_duration_seconds` — время инференса.
- `ml_service_prediction_probability` — распределение вероятностей.
- `ml_service_prediction_total{prediction}` — распределение классов предсказаний.

### Метрики по модели и её обновлению

- `ml_service_model_load_duration_seconds{status}` — время загрузки модели.
- `ml_service_model_update_total{status}` — число попыток обновления модели.
- `ml_service_current_model_info{run_id,model_type}` — активный `run_id` и тип модели.
- `ml_service_current_model_feature{run_id,feature}` — набор обязательных фичей активной модели.
- `ml_service_last_model_update_timestamp_seconds` — время последнего успешного обновления.

### Метрики Evidently

- `ml_service_evidently_report_total{status}` — успешные и неуспешные отправки drift-отчётов.
- `ml_service_evidently_pending_events` — размер очереди накопленных событий.
- `ml_service_last_evidently_report_timestamp_seconds` — время последней успешной отправки отчёта.

## 4. Дашборд Grafana

Импортируемый JSON лежит в [grafana/dashboard.json](grafana/dashboard.json).

Панели разделены на четыре группы:

1. Технические метрики:
   - request rate;
   - latency quantiles;
   - HTTP status codes;
   - CPU usage;
   - memory usage.
2. Метрики данных:
   - preprocessing latency;
   - missing input features;
   - top categorical values;
   - numeric feature distribution.
3. Метрики модели:
   - inference latency quantiles;
   - prediction classes;
   - probability distribution.
4. Метрики активной модели и мониторинга:
   - current model info;
   - current model features;
   - model updates;
   - pending drift events;
   - evidently reports.

## 5. Алертинг

Набор выражений хранится в [grafana/alerts.md](grafana/alerts.md). В Grafana настроены `8` alert rules:

1. `Student1 Instance Down`
2. `High p95 Service Latency`
3. `High p95 Inference Latency`
4. `HTTP 5xx Responses`
5. `Sustained High Memory Usage`
6. `Prediction Distribution Shift`
7. `Probability Distribution Shift`
8. `Missing Feature Spike`

Уведомления отправляются в Telegram через contact point `ML Service Telegram`.

## 6. Evidently и Drift Monitoring

Фоновый drift monitoring устроен так:

1. После успешного `predict` сервис сохраняет используемые моделью признаки, вероятность и итоговый класс.
2. События копятся в памяти батчами размера `EVIDENTLY_BATCH_SIZE`.
3. Первый полный батч после загрузки модели становится `reference dataset`.
4. Каждый следующий батч сравнивается с reference через Evidently:
   - `DataDriftPreset()`;
   - drift-метрика для `prediction`;
   - drift-метрика для `probability`.
5. Отчёт отправляется в удалённый workspace Evidently.
6. После смены модели reference и очередь событий сбрасываются.

Дополнительно очередь ограничена через `EVIDENTLY_MAX_PENDING_EVENTS`, а за один проход фоновой задачи разрешена обработка нескольких батчей через `EVIDENTLY_MAX_BATCHES_PER_INTERVAL`. Это помогает быстрее разгребать backlog после всплесков нагрузки.

Для ручной e2e-проверки отправки отчётов добавлен скрипт [scripts/test_evidently.sh](scripts/test_evidently.sh). Он отправляет достаточное число валидных запросов в `/predict`, затем опрашивает `/metrics` и проверяет рост `ml_service_evidently_report_total{status="success"}`.

## 7. Полезные файлы

- [README.md](README.md) — общее описание проекта и примеры запуска.
- [grafana/dashboard.json](grafana/dashboard.json) — дашборд Grafana.
- [grafana/alerts.md](grafana/alerts.md) — alert-выражения и пороги.
- [scripts/run_service.sh](scripts/run_service.sh) — запуск сервиса на ВМ.
- [scripts/load_test.py](scripts/load_test.py) — генератор нагрузки.
- [scripts/test_evidently.sh](scripts/test_evidently.sh) — проверка отправки Evidently-отчётов.
